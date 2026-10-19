// Stand-in for <windows.h>: only what common/utility/signal_escalation.hpp
// uses, with the real signatures and values (BOOL, DWORD, ULONGLONG, WINAPI,
// CTRL_*_EVENT, PHANDLER_ROUTINE), so its _WIN32 branch compiles and runs
// on Linux (tests/scripts/test_win32_console_ctrl.py). The probe drives the
// fake tick and calls the recorded routine as the console would.
//
// It also defines the real header's identifier-like macros, so a clash
// fails here as it would under MSVC: minwindef.h's far and near (always),
// its min and max (unless NOMINMAX), and rpcndr.h's small and hyper (only
// in the full header, i.e. unless WIN32_LEAN_AND_MEAN).
#ifndef DXAPP_TEST_WINDOWS_H_STUB
#define DXAPP_TEST_WINDOWS_H_STUB

#define far
#define near
#define FAR far
#define NEAR near
#ifndef NOMINMAX
#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define small char
#define hyper long long
#endif

typedef int BOOL;
typedef unsigned long DWORD;
typedef unsigned long long ULONGLONG;
#define WINAPI
#define TRUE 1
#define FALSE 0
#define ERROR_ACCESS_DENIED 5L
#define CTRL_C_EVENT 0
#define CTRL_BREAK_EVENT 1
#define CTRL_CLOSE_EVENT 2
#define CTRL_LOGOFF_EVENT 5
#define CTRL_SHUTDOWN_EVENT 6
typedef BOOL(WINAPI* PHANDLER_ROUTINE)(DWORD CtrlType);

namespace win32_stub {
inline PHANDLER_ROUTINE& routine() { static PHANDLER_ROUTINE r = 0; return r; }
inline int& registrations() { static int n = 0; return n; }
inline ULONGLONG& tick() { static ULONGLONG t = 0; return t; }
/// While true, SetConsoleCtrlHandler fails with ERROR_ACCESS_DENIED.
inline bool& refuseRegistration() { static bool refuse = false; return refuse; }
inline DWORD& lastError() { static DWORD e = 0; return e; }
}  // namespace win32_stub

inline BOOL WINAPI SetConsoleCtrlHandler(PHANDLER_ROUTINE routine, BOOL add) {
    if (win32_stub::refuseRegistration()) {
        win32_stub::lastError() = ERROR_ACCESS_DENIED;
        return FALSE;
    }
    if (add) {
        win32_stub::routine() = routine;
        ++win32_stub::registrations();
    } else if (win32_stub::routine() == routine) {
        win32_stub::routine() = 0;
    }
    return TRUE;
}

inline DWORD WINAPI GetLastError() { return win32_stub::lastError(); }

inline ULONGLONG WINAPI GetTickCount64() { return win32_stub::tick(); }

#endif  // DXAPP_TEST_WINDOWS_H_STUB
