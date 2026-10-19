/**
 * @file signal_escalation.hpp
 * @brief "First Ctrl-C asks, a later Ctrl-C terminates" for every binary.
 *
 * A graceful-shutdown handler only sets a flag; the program then finishes
 * what it is doing (in-flight frames, a report, a performance summary) and
 * returns. If that wind-down itself hangs, the user's next Ctrl-C used to
 * set the same flag again and do nothing else, so only SIGKILL could stop
 * the process.
 *
 * ONE KEYPRESS IS OFTEN SEVERAL SIGNALS. The terminal sends SIGINT to the
 * whole foreground process group, and a wrapper in that group forwards it
 * again: coreutils `timeout` signals its child and then the child's process
 * group, so one Ctrl-C on `timeout 20 ./runner` reaches the runner 2-3
 * times within microseconds. Escalating on the literal second delivery
 * turned that single keypress into an immediate kill with no summary.
 *
 * So a request is escalated by TIME, not by count: noteInterruptRequest()
 * records when the first request arrived and treats every repeat within
 * kInterruptCoalesceMs of it as the same request. Only a repeat later than
 * that is "the user pressed Ctrl-C again"; the handler then calls
 * terminateBySignal(), which restores SIG_DFL and re-raises.
 *
 * kInterruptCoalesceMs = 200: forwarded copies arrive within microseconds
 * (well under 1 ms even on a loaded machine), while a deliberate second
 * keypress by a person is slower than that. A double-tap faster than
 * 200 ms counts as one request - the next press terminates.
 *
 * Async-signal-safety: noteInterruptRequest() uses clock_gettime() and a
 * lock-free atomic compare-exchange (a failed clock_gettime() falls back to
 * detail::brokenClockMs(), a lock-free counter); restoreDefaultSignal() uses
 * sigaction(); terminateBySignal() adds raise(). POSIX lists all of them as
 * async-signal-safe. The signal being handled is blocked while its handler
 * runs, so the re-raised copy is delivered - with the default action - as
 * the handler returns.
 *
 * Handlers are installed persistently (installPersistentHandler) with
 * SA_RESTART, which is what glibc's std::signal() did before, so blocking
 * calls interrupted by a graceful request behave as they did.
 *
 * Windows (NOT BUILT ON WINDOWS HERE - no MSVC on the development host; this
 * branch is compiled and driven on Linux against a stub <windows.h> by
 * tests/scripts/test_win32_console_ctrl.py): the CRT's signal() resets a
 * handler to SIG_DFL before calling it, so it can neither coalesce nor stay
 * installed. SIGINT is therefore taken from the console:
 * installPersistentHandler(SIGINT, h) registers one console control routine
 * (SetConsoleCtrlHandler) that runs h on Ctrl-C - on the console's own
 * thread - with the same 200 ms rule on GetTickCount64(). When h escalates
 * (terminateBySignal), the routine hands the event to the default routine,
 * which ends the process with STATUS_CONTROL_C_EXIT, as an unhandled Ctrl-C
 * does. Ctrl-Break, close, logoff and shutdown always go to the default
 * routine: Ctrl-Break stays the immediate kill. SIGTERM keeps the CRT
 * signal() (Windows never sends it from outside).
 */

#ifndef DXAPP_SIGNAL_ESCALATION_HPP
#define DXAPP_SIGNAL_ESCALATION_HPP

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
// Every C++ example TU reaches this header (run_dir.hpp, the runners, the
// graph CLI): no min/max macros (NOMINMAX, also set by the top-level CMake
// but not by a consumer's own build), and the lean header - no rpcndr.h
// `small` or `hyper`. WIN32_LEAN_AND_MEAN is undefined again afterwards if
// it was set here; a TU that needs the full <windows.h> includes it first.
// minwindef.h's `far` and `near` (defined empty, always) cannot be avoided:
// tests/scripts/test_windows_macro_names.py keeps them out of src/.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#define DXAPP_SIGNAL_ESCALATION_LEAN
#endif
#include <windows.h>
#ifdef DXAPP_SIGNAL_ESCALATION_LEAN
#undef WIN32_LEAN_AND_MEAN
#undef DXAPP_SIGNAL_ESCALATION_LEAN
#endif
#else
#include <signal.h>
#include <time.h>
#endif

namespace dxapp {

/// Handler signature accepted by installPersistentHandler().
typedef void (*SignalHandlerFn)(int);

/// Repeats of an interrupt within this many milliseconds of the first are
/// the same request (see the file comment for why 200).
const long long kInterruptCoalesceMs = 200;

#ifndef _WIN32
static_assert(ATOMIC_LLONG_LOCK_FREE == 2,
              "the interrupt timestamp must be a lock-free atomic to be "
              "touched from a signal handler");
#endif

namespace detail {

/// wincon.h's CTRL_C_EVENT and CTRL_BREAK_EVENT, spelled here so the
/// dispatch below compiles - and is unit-tested - off Windows too.
const unsigned long kConsoleCtrlC = 0;
const unsigned long kConsoleCtrlBreak = 1;

/// The SIGINT handler the console routine runs (Windows; null: none).
inline std::atomic<SignalHandlerFn>& consoleInterruptHandler() {
    static std::atomic<SignalHandlerFn> handler{nullptr};
    return handler;
}

/// Set by terminateBySignal(SIGINT) inside the console routine: the
/// routine then hands the event to the default routine.
inline std::atomic<bool>& consoleTerminateRequested() {
    static std::atomic<bool> requested{false};
    return requested;
}

/// True on the thread running the console routine, while it runs the handler.
inline bool& insideConsoleRoutine() {
    static thread_local bool inside = false;
    return inside;
}

/**
 * @brief What the console control routine does with one event.
 * @return true: handled, the process keeps running (Ctrl-C, a graceful
 *         request); false: the default routine runs, which ends the process
 *         (Ctrl-C after the handler escalated, Ctrl-Break, close, logoff,
 *         shutdown, or no handler installed).
 */
inline bool dispatchConsoleCtrl(unsigned long ctrl_type) {
    if (ctrl_type != kConsoleCtrlC) return false;
    const SignalHandlerFn handler = consoleInterruptHandler().load();
    if (handler == nullptr) return false;
    insideConsoleRoutine() = true;
    handler(SIGINT);
    insideConsoleRoutine() = false;
    return !consoleTerminateRequested().load();
}

#ifdef _WIN32
static_assert(kConsoleCtrlC == CTRL_C_EVENT, "wincon.h CTRL_C_EVENT");
static_assert(kConsoleCtrlBreak == CTRL_BREAK_EVENT, "wincon.h CTRL_BREAK_EVENT");

inline BOOL WINAPI consoleCtrlRoutine(DWORD ctrl_type) {
    return dispatchConsoleCtrl(ctrl_type) ? TRUE : FALSE;
}

/// Register consoleCtrlRoutine once per process.
inline bool registerConsoleCtrlRoutineOnce() {
    static std::atomic<bool> registered{false};
    if (registered.load()) return true;
    if (!SetConsoleCtrlHandler(&consoleCtrlRoutine, TRUE)) return false;
    registered.store(true);
    return true;
}
#endif

}  // namespace detail

/**
 * @brief Put `sig` back to its default action. Async-signal-safe.
 */
inline void restoreDefaultSignal(int sig) {
#ifdef _WIN32
    std::signal(sig, SIG_DFL);
#else
    struct sigaction action = {};
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(sig, &action, nullptr);
#endif
}

/**
 * @brief Terminate the process by `sig`'s default action. Async-signal-safe.
 *
 * Called from inside `sig`'s own handler: `sig` is blocked there, so the
 * raised copy stays pending and is delivered (SIG_DFL: terminate) when the
 * handler returns. The exit status is "killed by `sig`", as for a process
 * that never installed a handler.
 */
inline void terminateBySignal(int sig) {
#ifdef _WIN32
    if (sig == SIGINT && detail::insideConsoleRoutine()) {
        // The console routine hands the event on; the default routine ends
        // the process (see the file comment).
        detail::consoleTerminateRequested().store(true);
        return;
    }
#endif
    restoreDefaultSignal(sig);
    raise(sig);
}

/**
 * @brief The coalescing rule on a caller-supplied clock: record an interrupt
 *        request made at `now_ms` (monotonic milliseconds, any origin).
 * @return true when `now_ms` is more than kInterruptCoalesceMs after the
 *         first recorded request; false for the first request and for
 *         coalesced repeats. Lock-free; async-signal-safe.
 */
inline bool noteInterruptRequestAt(std::atomic<long long>& first_ms,
                                   long long now_ms) {
    // +1 so that a (theoretical) clock value of 0 still reads as "set".
    const long long stamp = now_ms + 1;
    long long first = 0;
    if (first_ms.compare_exchange_strong(first, stamp)) return false;  // first
    return stamp - first > kInterruptCoalesceMs;
}

#ifndef _WIN32
namespace detail {

/// The clock noteInterruptRequest() uses when CLOCK_MONOTONIC fails: every
/// call reads kInterruptCoalesceMs + 1 ms after the previous one, so each
/// repeat of a request counts as "later" and escalates - the count-based
/// rule from before coalescing, the only safe one without a clock (U-42).
/// A constant-initialised lock-free atomic: async-signal-safe.
inline long long brokenClockMs() {
    static std::atomic<long long> now{0};
    return now.fetch_add(kInterruptCoalesceMs + 1) + kInterruptCoalesceMs + 1;
}

}  // namespace detail
#endif

/// Monotonic milliseconds: clock_gettime(CLOCK_MONOTONIC) on POSIX
/// (async-signal-safe; detail::brokenClockMs() if that call fails),
/// GetTickCount64() on Windows.
inline long long monotonicMs() {
#ifdef _WIN32
    return static_cast<long long>(GetTickCount64());
#else
    struct timespec now_ts = {};
    if (clock_gettime(CLOCK_MONOTONIC, &now_ts) != 0) return detail::brokenClockMs();
    return static_cast<long long>(now_ts.tv_sec) * 1000 + now_ts.tv_nsec / 1000000;
#endif
}

/**
 * @brief Record an interrupt request; say whether it is a LATER repeat.
 * @param first_ms  per-handler state, 0 until the first request.
 * @return true when this delivery arrives more than kInterruptCoalesceMs
 *         after the first recorded request (the caller should escalate);
 *         false for the first request and for coalesced repeats.
 * Async-signal-safe.
 */
inline bool noteInterruptRequest(std::atomic<long long>& first_ms) {
    return noteInterruptRequestAt(first_ms, monotonicMs());
}

/**
 * @brief Install `handler` for `sig` for every delivery (SA_RESTART).
 * @return true on success.
 */
inline bool installPersistentHandler(int sig, SignalHandlerFn handler) {
#ifdef _WIN32
    if (sig == SIGINT) {
        detail::consoleInterruptHandler().store(handler);
        return detail::registerConsoleCtrlRoutineOnce();
    }
    return std::signal(sig, handler) != SIG_ERR;
#else
    struct sigaction action = {};
    action.sa_handler = handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    return sigaction(sig, &action, nullptr) == 0;
#endif
}

/**
 * @brief installPersistentHandler(), plus one warning on stderr if it
 *        fails. Never aborts: without the handler the program still runs,
 *        and the signal simply keeps its default action.
 * @param name  shown in the warning, e.g. "SIGINT".
 * @return true on success.
 */
inline bool installHandlerOrWarn(int sig, SignalHandlerFn handler,
                                 const char* name) {
    if (installPersistentHandler(sig, handler)) return true;
#ifdef _WIN32
    if (sig == SIGINT) {
        // SetConsoleCtrlHandler reports through GetLastError(), not errno.
        const unsigned long win_error = static_cast<unsigned long>(GetLastError());
        std::fprintf(stderr,
                     "[DXAPP] [WARN] could not install the %s handler (error %lu); "
                     "%s will stop the program without a graceful shutdown.\n",
                     name, win_error, name);
        return false;
    }
#endif
    const int error = errno;
    std::fprintf(stderr,
                 "[DXAPP] [WARN] could not install the %s handler (%s); "
                 "%s will stop the program without a graceful shutdown.\n",
                 name, std::strerror(error), name);
    return false;
}

}  // namespace dxapp

#endif  // DXAPP_SIGNAL_ESCALATION_HPP
