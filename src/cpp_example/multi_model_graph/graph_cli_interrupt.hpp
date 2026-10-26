/**
 * @file graph_cli_interrupt.hpp
 * @brief The multi-model-graph CLI's SIGINT / SIGTERM handling.
 *
 * First request (SIGINT or SIGTERM): the input loop stops reading, the
 * executor finishes the frames already in flight and the report is
 * written - graceful. A later SIGINT (more than kInterruptCoalesceMs after
 * the first request): the process terminates (default SIGINT action), for
 * when that wind-down is itself stuck. SIGTERM never escalates. The same
 * rule as run_dir.hpp's signalHandler. See
 * common/utility/signal_escalation.hpp.
 *
 * Header-only and dependency-free so common_unit_test can exercise the
 * very handler the CLI installs without linking the CLI.
 */
#ifndef DXAPP_MULTI_MODEL_GRAPH_GRAPH_CLI_INTERRUPT_HPP
#define DXAPP_MULTI_MODEL_GRAPH_GRAPH_CLI_INTERRUPT_HPP

#include <atomic>
#include <csignal>

#include "common/utility/signal_escalation.hpp"

namespace dxapp {
namespace graph {
namespace cli {

/// Set by the first SIGINT or SIGTERM (or RequestStop()). Zero-initialised,
/// so no guard runs in the handler.
inline volatile std::sig_atomic_t& InterruptFlag() {
    static volatile std::sig_atomic_t flag = 0;
    return flag;
}

/// When the first SIGINT or SIGTERM (or RequestStop()) arrived (0: none
/// yet). See noteInterruptRequest().
inline std::atomic<long long>& InterruptRequestMs() {
    static std::atomic<long long> first_ms{0};
    return first_ms;
}

/// SIGINT or SIGTERM: set the flag - stop reading, finish the frames in
/// flight, write the report. A SIGINT more than kInterruptCoalesceMs after
/// the first request (SIGINT or SIGTERM) terminates the process; copies
/// sooner than that are the same request. SIGTERM never escalates: `timeout`
/// delivers it twice and escalates with SIGKILL itself. The same rule as
/// run_dir.hpp's signalHandler, so every C++ binary answers alike.
/// Async-signal-safe.
inline void OnInterrupt(int sig) {
    InterruptFlag() = 1;
    const bool later_repeat = noteInterruptRequest(InterruptRequestMs());
    if (sig == SIGINT && later_repeat) terminateBySignal(SIGINT);
}

/// Install OnInterrupt for SIGINT and SIGTERM, persistently (a warning on
/// stderr for each install that fails).
inline void InstallInterruptHandler() {
    installHandlerOrWarn(SIGINT, OnInterrupt, "SIGINT");
    installHandlerOrWarn(SIGTERM, OnInterrupt, "SIGTERM");
}

/// A graceful stop asked for by the program itself (--display's q / ESC).
/// It is a first request like a first signal: its time is recorded the same
/// way, so a Ctrl-C more than kInterruptCoalesceMs later still terminates
/// and one sooner is the same request.
inline void RequestStop() {
    InterruptFlag() = 1;
    noteInterruptRequest(InterruptRequestMs());
}

inline bool Interrupted() { return InterruptFlag() != 0; }

}  // namespace cli
}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_MULTI_MODEL_GRAPH_GRAPH_CLI_INTERRUPT_HPP
