"""No C++ identifier in src/ is a name <windows.h> defines as a macro (I1).

signal_escalation.hpp includes <windows.h> on Windows, and it reaches every
C++ example TU (run_dir.hpp, the runners, the graph CLI, common_unit_test);
src/utility/common_util.hpp includes the full <Windows.h> for the
postprocess libraries. minwindef.h always does `#define far` and
`#define near`; the full header also brings rpcndr.h's `small` and `hyper`
and combaseapi.h's `interface`. A local named `far` then compiles on Linux
and breaks the first MSVC build. There is no MSVC here, so the rule is
checked on the source text: comments and string/char literals are removed
first, then no such name may appear as a token.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl"}

# Object-like macros of <windows.h> that read like ordinary identifiers.
WINDOWS_MACRO_NAMES = ("far", "near", "small", "hyper", "interface")

_TOKENS = re.compile(
    r'//[^\n]*'                                   # line comment
    r'|/\*.*?\*/'                                 # block comment
    r'|R"(?P<delim>[^()\\\s]{0,16})\(.*?\)(?P=delim)"'  # raw string
    r'|"(?:\\.|[^"\\\n])*"'                       # string literal
    r"|'(?:\\.|[^'\\\n])*'",                      # char literal
    re.S)
_NAME = re.compile(r"\b(" + "|".join(WINDOWS_MACRO_NAMES) + r")\b")


def strip_comments_and_literals(text: str) -> str:
    """Blank comments and literals, keeping newlines so line numbers hold."""
    return _TOKENS.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def clashes(text: str):
    """(line, name) for every windows.h macro name used as a token."""
    code = strip_comments_and_literals(text)
    return [(code.count("\n", 0, m.start()) + 1, m.group(1)) for m in _NAME.finditer(code)]


def test_the_scanner_ignores_comments_and_literals():
    text = ('// far\n/* small\n near */ const char* s = "hyper far";\n'
            "char c = 'n'; auto r = R\"x(interface)x\";\n"
            "std::thread far(run);\nint small = 1;\n")
    assert clashes(text) == [(5, "far"), (6, "small")]


def test_no_source_uses_a_windows_macro_name_as_an_identifier():
    found = []
    for path in sorted((ROOT / "src").rglob("*")):
        if path.suffix in SUFFIXES and path.is_file():
            rel = path.relative_to(ROOT).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            found += [f"{rel}:{line}: {name}" for line, name in clashes(text)]
    assert not found, "windows.h defines these as macros; rename them:\n" + "\n".join(found)
