// run_dir.hpp (every runner includes it) compiled with -D_WIN32 against
// fixtures/win32_stub/windows.h, whose far/near/min/max macros stand in for
// the real header's: a name clash after the include fails here as under
// MSVC. -fsyntax-only, by tests/scripts/test_win32_console_ctrl.py.
#include <ctime>

// MSVC's CRT has localtime_s (run_dir.hpp's _WIN32 branch); glibc does not.
inline int localtime_s(std::tm* out, const std::time_t* t) { return localtime_r(t, out) ? 0 : 22; }

#include "common/utility/run_dir.hpp"

int main() { return 0; }
