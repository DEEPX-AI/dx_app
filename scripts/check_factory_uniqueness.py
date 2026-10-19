#!/usr/bin/env python3
"""Fail when two factory headers declare the same fully qualified class.

A duplicate is invisible while each example is its own binary, but the
generated graph registry includes every factory in one program. Each variant's
factory lives in its own dxapp::v_<variant> namespace
(scripts/generate_cpp_family_layout.py --variant-scope), so the copies of one
family's class (every yolo11 variant declares Yolo11Factory) are distinct
classes; two headers whose QUALIFIED names coincide are the ODR clash.

Every <anything>/factory/*_factory.hpp under the tree is compared, whatever its
depth: a header left outside <task>/<family>/<variant>/factory/ still links.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

CE = Path(__file__).resolve().parents[1] / "src" / "cpp_example"

# Matches "class Name [final] : <base-list> {". Non-greedy up to the first
# '{' so it stops at the class-body opener rather than consuming the whole
# file; re.DOTALL lets the base list span multiple lines.
_CLASS_RE = re.compile(r"class\s+(\w+)(?:\s+final)?\s*:\s*(.+?)\{", re.DOTALL)

# Matches an "I...Factory" interface name, optionally namespace-qualified
# (e.g. "IDetectionFactory" or "dxapp::IDetectionFactory").
_BASE_RE = re.compile(r"(?:\w+\s*::\s*)*\b(I\w+Factory)\b")

# Comments, string and character literals, and preprocessor lines: blanked
# before the braces are counted, so a brace inside one cannot move a class in
# or out of a namespace. A quote after a hex digit is a C++14 digit separator.
_NOT_CODE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|(?<![0-9A-Fa-f])'(?:\\.|[^'\\\n])*'"
    r"|^[ \t]*#(?:[^\n]*\\\n)*[^\n]*",
    re.S | re.M)
_NAMESPACE_OPEN = re.compile(r"\bnamespace\s+(\w+)\s*\{")


class ParseError(Exception):
    """Raised when a *_factory.hpp cannot be parsed for class/base info."""


def _blank(text):
    """text with comments, literals and preprocessor lines turned into spaces:
    same length and same newlines, so an offset in the result is one in text."""
    return _NOT_CODE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def enclosing_namespaces(text, offset):
    """The named namespaces open at offset, outermost first (anonymous ones
    are left out: their contents cannot clash across translation units)."""
    code = _blank(text)
    stack = []
    i = 0
    while i < offset:
        m = _NAMESPACE_OPEN.match(code, i) if code[i] == "n" else None
        if m and (i == 0 or not (code[i - 1].isalnum() or code[i - 1] == "_")) \
                and m.end() <= offset:
            stack.append(m.group(1))
            i = m.end()
            continue
        if code[i] == "{":
            stack.append(None)
        elif code[i] == "}" and stack:
            stack.pop()
        i += 1
    return [name for name in stack if name]


def parse_factory_classes(text):
    """Extract EVERY (qualified_class_name, interface_base) pair from a factory header.

    THE single factory-scanning rule for this repository. Both this script
    and scripts/gen_model_registry.py go through it — the generator imports
    it rather than re-implementing it, because a second implementation is
    free to drift, and a header one of them cannot parse would then silently
    vanish from the generated registry instead of failing loudly.

    The class name is qualified with the namespaces enclosing its
    declaration, from the global scope: `::dxapp::v_yolo11_n_640x640::Yolo11Factory`
    in a scoped variant header, `::dxapp::Yolo11Factory` in one that is not.

    A factory header may declare more than one class (e.g. a private
    postprocessor/visualizer wrapper ahead of the actual factory), so every
    top-level `class ... : ... {` declaration is inspected in order and every
    one whose base list contains an `I...Factory` is returned, IN DECLARATION
    ORDER. A naive "first class in the file" match is wrong on real headers.

    Tolerates the class-declaration variations found in this codebase: an
    optional `final` before the base-class colon, a fully-qualified base
    (`public dxapp::IDetectionFactory`), whitespace/newlines between `class`
    and the base list, and a base list with more than one entry (the first
    base matching `I...Factory` is used, regardless of its position within
    that particular class's base list).

    Raises ParseError with a human-readable reason if no class declaration
    at all is found, or if none of the declared classes has a base matching
    `I...Factory` — this must never be swallowed into a silent skip by the
    caller.
    """
    class_matches = list(_CLASS_RE.finditer(text))
    if not class_matches:
        raise ParseError("no 'class <Name> : <bases> {' declaration found")
    found = []
    for class_match in class_matches:
        class_name, bases_text = class_match.group(1), class_match.group(2)
        for base in bases_text.split(","):
            base_match = _BASE_RE.search(base)
            if base_match:
                scope = enclosing_namespaces(text, class_match.start())
                found.append(("::" + "::".join(scope + [class_name]), base_match.group(1)))
                break
    if not found:
        raise ParseError(
            "no base matching 'I...Factory' in any declared class: {}".format(
                ", ".join(m.group(1) for m in class_matches)
            )
        )
    return found


def parse_factory_class(text):
    """The first (qualified_class_name, interface_base) parse_factory_classes found.

    Uniqueness checking only needs the one class that names the header; the
    generator uses parse_factory_classes() directly because for IT a second
    I...Factory-deriving class in one header is an ambiguity it must refuse
    rather than resolve.
    """
    return parse_factory_classes(text)[0]


def factory_headers(root):
    """Every */factory/*_factory.hpp under root at any depth, common/ aside."""
    root = Path(root)
    return sorted(h for h in root.rglob("*_factory.hpp")
                  if h.parent.name == "factory" and h.relative_to(root).parts[0] != "common")


def collect(root=CE):
    root = Path(root)
    by_class = defaultdict(list)
    errors = []
    for header in factory_headers(root):
        relative = header.relative_to(root).as_posix()
        text = header.read_text(encoding="utf-8", errors="replace")
        try:
            class_name, _base = parse_factory_class(text)
        except ParseError as exc:
            errors.append((relative, str(exc)))
            continue
        by_class[class_name].append(relative)
    return by_class, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cpp-root", default=str(CE),
                        help="factory tree to scan (tests point this elsewhere)")
    args = parser.parse_args(argv)
    by_class, errors = collect(args.cpp_root)
    exit_code = 0

    for path, reason in sorted(errors):
        print("UNPARSEABLE {}: {}".format(path, reason))
        exit_code = 1
    if errors:
        print("{} factory header(s) could not be parsed".format(len(errors)))

    duplicates = {k: v for k, v in by_class.items() if len(v) > 1}
    for name, paths in sorted(duplicates.items()):
        print("DUPLICATE class {}: {}".format(name, ", ".join(paths)))
        exit_code = 1
    if duplicates:
        print("{} duplicate factory class name(s)".format(len(duplicates)))

    if exit_code == 0:
        print("factory class names unique ({} headers)".format(
            sum(len(v) for v in by_class.values())))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
