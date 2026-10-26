// Drives the _WIN32 branch of signal_escalation.hpp through the graph CLI's
// real handler (graph_cli_interrupt.hpp), with fixtures/win32_stub/windows.h
// standing in for the console. Built with g++ -D_WIN32 by
// tests/scripts/test_win32_console_ctrl.py. Prints "N checks, F failures".
//
// <windows.h> comes only through the header, as in every real TU: it must
// include it with NOMINMAX (the stub's min/max macros would break the
// standard headers below) and WIN32_LEAN_AND_MEAN (no rpcndr.h `small`),
// and leave WIN32_LEAN_AND_MEAN undefined again for the rest of the TU.
#include "multi_model_graph/graph_cli_interrupt.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <limits>

#if defined(min) || defined(max)
#error "signal_escalation.hpp let <windows.h> define min/max (NOMINMAX)"
#endif
#ifdef small
#error "signal_escalation.hpp included the full <windows.h> (WIN32_LEAN_AND_MEAN)"
#endif
#ifdef WIN32_LEAN_AND_MEAN
#error "signal_escalation.hpp leaked WIN32_LEAN_AND_MEAN into the TU"
#endif

namespace {
int g_checks = 0;
int g_failures = 0;

void Check(bool ok, const char* what, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL line %d: %s\n", line, what);
    }
}
#define PROBE_CHECK(cond) Check((cond), #cond, __LINE__)

// One console event at fake time `at` (ms), as the console thread delivers it.
BOOL Press(DWORD type, ULONGLONG at) {
    win32_stub::tick() = at;
    return win32_stub::routine()(type);
}
}  // namespace

void IgnoreSignal(int) {}

int main() {
    namespace cli = dxapp::graph::cli;
    PROBE_CHECK(std::min(1, 2) == 1 && std::numeric_limits<int>::max() > 0);

    // SetConsoleCtrlHandler fails: the warning gives GetLastError(), not
    // errno (which it does not set). The test reads stderr for "(error 5)".
    win32_stub::refuseRegistration() = true;
    errno = ENOENT;
    PROBE_CHECK(!dxapp::installHandlerOrWarn(SIGINT, IgnoreSignal, "SIGINT"));
    PROBE_CHECK(win32_stub::registrations() == 0);
    win32_stub::refuseRegistration() = false;

    win32_stub::tick() = 4242;
    PROBE_CHECK(dxapp::monotonicMs() == 4242);   // the Windows clock is GetTickCount64

    cli::InstallInterruptHandler();               // SIGINT and SIGTERM
    cli::InstallInterruptHandler();               // installing again registers nothing new
    PROBE_CHECK(win32_stub::registrations() == 1);
    if (win32_stub::routine() == 0) {
        std::printf("FAIL: no console control routine registered\n");
        std::printf("%d checks, %d failures\n", g_checks, g_failures + 1);
        return 1;
    }

    // Ctrl-Break and close go to the default routine and never reach the handler.
    PROBE_CHECK(Press(CTRL_BREAK_EVENT, 500) == FALSE);
    PROBE_CHECK(Press(CTRL_CLOSE_EVENT, 600) == FALSE);
    PROBE_CHECK(!cli::Interrupted());

    // Ctrl-C: a graceful request, handled - the process keeps running.
    PROBE_CHECK(Press(CTRL_C_EVENT, 1000) == TRUE);
    PROBE_CHECK(cli::Interrupted());
    // Repeats up to kInterruptCoalesceMs after the first are the same request.
    PROBE_CHECK(Press(CTRL_C_EVENT, 1100) == TRUE);
    PROBE_CHECK(Press(CTRL_C_EVENT, 1000 + dxapp::kInterruptCoalesceMs) == TRUE);
    // One later is a new press: escalate, i.e. hand the event to the default
    // routine, which ends the process (STATUS_CONTROL_C_EXIT).
    PROBE_CHECK(Press(CTRL_C_EVENT, 1000 + dxapp::kInterruptCoalesceMs + 1) == FALSE);

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
