#!/usr/bin/env python3
"""
validate_app.py — Static analysis and smoke tests for dx_app applications.

Validates a dx_app model directory against coding standards, framework conventions,
and optional runtime smoke tests.

Usage:
    python .deepx/scripts/validate_app.py src/python_example/object_detection/yolov8n/
    python .deepx/scripts/validate_app.py src/python_example/object_detection/yolov8n/ --smoke-test
    python .deepx/scripts/validate_app.py src/python_example/object_detection/yolov8n/ --strict --json
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class CheckResult:
    """Result of a single validation check."""

    def __init__(self, name: str, passed: bool, message: str = "",
                 severity: str = "error", scope: str = "static"):
        self.name = name
        self.passed = passed
        self.message = message
        self.severity = severity  # "error", "warning", "info"
        self.scope = scope        # "static", "smoke"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
            "scope": self.scope,
        }

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


class ValidationReport:
    """Aggregate validation results."""

    def __init__(self, target_dir: str):
        self.target_dir = target_dir
        self.results: List[CheckResult] = []

    def add(self, result: CheckResult):
        self.results.append(result)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "error")

    @property
    def errors(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        errors = len(self.errors)
        warnings = len(self.warnings)
        return (
            f"Validation: {passed}/{total} checks passed, "
            f"{errors} errors, {warnings} warnings"
        )

    def to_dict(self) -> dict:
        return {
            "target_dir": self.target_dir,
            "passed": self.passed,
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }

    def print_report(self, verbose: bool = False):
        for result in self.results:
            if not result.passed or verbose:
                status = "PASS" if result.passed else "FAIL"
                icon = "  " if result.passed else "!!"
                sev = f"[{result.severity.upper()}]" if not result.passed else ""
                print(f"  {icon} [{status}] {result.name} {sev}")
                if result.message and (not result.passed or verbose):
                    print(f"          {result.message}")
        print()
        print(f"  {self.summary()}")


# ---------------------------------------------------------------------------
# Static Checks (11)
# ---------------------------------------------------------------------------

def check_required_files(app_dir: Path) -> CheckResult:
    """Check 1: Required files exist."""
    required = ["config.json", "__init__.py"]
    factory_dir = app_dir / "factory"
    missing = []

    for f in required:
        if not (app_dir / f).exists():
            missing.append(f)

    if not factory_dir.exists():
        missing.append("factory/")
    else:
        if not (factory_dir / "__init__.py").exists():
            missing.append("factory/__init__.py")
        factory_files = list(factory_dir.glob("*_factory.py"))
        if not factory_files:
            missing.append("factory/<model>_factory.py")

    if missing:
        return CheckResult(
            "required_files", False,
            f"Missing: {', '.join(missing)}"
        )
    return CheckResult("required_files", True, "All required files present")


def check_syntax(app_dir: Path) -> CheckResult:
    """Check 2: All .py files have valid syntax."""
    py_files = list(app_dir.rglob("*.py"))
    errors = []

    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=str(py_file))
        except SyntaxError as e:
            rel_path = py_file.relative_to(app_dir)
            errors.append(f"{rel_path}:{e.lineno}: {e.msg}")

    if errors:
        return CheckResult(
            "syntax", False,
            f"{len(errors)} syntax error(s): {'; '.join(errors[:3])}"
        )
    return CheckResult("syntax", True, f"{len(py_files)} files parsed OK")


def check_no_relative_imports(app_dir: Path) -> CheckResult:
    """Check 3: No relative imports (from . or from ..)."""
    py_files = list(app_dir.glob("*.py"))  # Only top-level scripts
    violations = []

    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        # Relative import at top-level script
                        rel = py_file.relative_to(app_dir)
                        violations.append(
                            f"{rel}:{node.lineno}: "
                            f"relative import (level={node.level})"
                        )
        except SyntaxError:
            pass  # Already caught by check_syntax

    # Note: factory/__init__.py is allowed to use relative imports
    if violations:
        return CheckResult(
            "no_relative_imports", False,
            f"{len(violations)} relative import(s) in app scripts: "
            f"{'; '.join(violations[:3])}"
        )
    return CheckResult("no_relative_imports", True, "No relative imports in app scripts")


def check_logging(app_dir: Path) -> CheckResult:
    """Check 4: Uses logging module, not bare print() for diagnostics."""
    py_files = list(app_dir.glob("*.py"))
    print_uses = []

    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                # Skip comments and docstrings
                if stripped.startswith("#"):
                    continue
                # Detect bare print() calls (not in comments)
                if re.match(r'^print\s*\(', stripped):
                    rel = py_file.relative_to(app_dir)
                    print_uses.append(f"{rel}:{i}")
        except (OSError, UnicodeDecodeError):
            pass

    if print_uses:
        return CheckResult(
            "logging", False,
            f"{len(print_uses)} print() call(s) found — use logging: "
            f"{', '.join(print_uses[:5])}",
            severity="warning"
        )
    return CheckResult("logging", True, "No bare print() calls")


def check_no_hardcoded_paths(app_dir: Path) -> CheckResult:
    """Check 5: No hardcoded .dxnn model paths."""
    py_files = list(app_dir.rglob("*.py"))
    violations = []

    path_patterns = [
        r'["\'][^"\']*\.dxnn["\']',          # "path/to/model.dxnn"
        r'["\']/dev/video\d+["\']',           # "/dev/video0"
        r'["\']/home/[^"\']+["\']',           # "/home/user/..."
        r'["\']/tmp/[^"\']+["\']',            # "/tmp/..."
    ]

    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            for i, line in enumerate(source.splitlines(), 1):
                if line.strip().startswith("#"):
                    continue
                for pattern in path_patterns:
                    if re.search(pattern, line):
                        rel = py_file.relative_to(app_dir)
                        violations.append(f"{rel}:{i}")
                        break
        except (OSError, UnicodeDecodeError):
            pass

    if violations:
        return CheckResult(
            "no_hardcoded_paths", False,
            f"{len(violations)} hardcoded path(s): {', '.join(violations[:5])}"
        )
    return CheckResult("no_hardcoded_paths", True, "No hardcoded paths")


def check_ifactory_implementation(app_dir: Path) -> CheckResult:
    """Check 6: Factory implements all 5 required IFactory methods."""
    factory_dir = app_dir / "factory"
    if not factory_dir.exists():
        return CheckResult(
            "ifactory", False, "factory/ directory not found"
        )

    factory_files = list(factory_dir.glob("*_factory.py"))
    if not factory_files:
        return CheckResult(
            "ifactory", False, "No *_factory.py file found"
        )

    required_methods = {
        "create_preprocessor",
        "create_postprocessor",
        "create_visualizer",
        "get_model_name",
        "get_task_type",
    }

    for factory_file in factory_files:
        try:
            with open(factory_file, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = {
                        n.name for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    }
                    missing = required_methods - methods
                    if missing:
                        return CheckResult(
                            "ifactory", False,
                            f"{factory_file.name}: missing methods: "
                            f"{', '.join(sorted(missing))}"
                        )
                    # Check constructor accepts config
                    init_methods = [
                        n for n in node.body
                        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                    ]
                    if init_methods:
                        init = init_methods[0]
                        arg_names = [a.arg for a in init.args.args]
                        if "config" not in arg_names:
                            return CheckResult(
                                "ifactory", False,
                                f"{factory_file.name}: __init__ missing "
                                f"'config' parameter",
                                severity="warning"
                            )
        except SyntaxError:
            pass  # Already caught by check_syntax

    return CheckResult("ifactory", True, "Factory implements all 5 methods")


def check_parse_common_args(app_dir: Path) -> CheckResult:
    """Check 7: App scripts use parse_common_args(), not custom argparse."""
    app_scripts = [
        f for f in app_dir.glob("*.py")
        if f.name != "__init__.py" and not f.name.startswith("_")
    ]
    violations = []

    for script in app_scripts:
        try:
            with open(script, "r", encoding="utf-8") as f:
                source = f.read()

            has_parse_common = "parse_common_args" in source
            has_argparse_import = "import argparse" in source
            has_argumentparser = "ArgumentParser" in source

            if has_argparse_import or has_argumentparser:
                if not has_parse_common:
                    violations.append(
                        f"{script.name}: uses custom argparse instead of "
                        f"parse_common_args()"
                    )
        except (OSError, UnicodeDecodeError):
            pass

    if violations:
        return CheckResult(
            "parse_common_args", False,
            f"{len(violations)} script(s) use custom argparse: "
            f"{'; '.join(violations[:3])}"
        )
    return CheckResult("parse_common_args", True, "All scripts use parse_common_args()")


def check_unused_imports(app_dir: Path) -> CheckResult:
    """Check 8: Detect obviously unused imports."""
    py_files = list(app_dir.glob("*.py"))
    warnings = []

    for py_file in py_files:
        if py_file.name == "__init__.py":
            continue
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)

            imported_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name.split(".")[-1]
                        imported_names.add(name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname or alias.name
                        imported_names.add(name)

            # Check if imported names appear in the rest of the source
            for name in imported_names:
                if name == "*":
                    continue
                # Count occurrences (excluding import lines)
                lines = source.splitlines()
                use_count = 0
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("import ") or stripped.startswith("from "):
                        continue
                    if name in line:
                        use_count += 1

                if use_count == 0:
                    rel = py_file.relative_to(app_dir)
                    warnings.append(f"{rel}: unused import '{name}'")

        except SyntaxError:
            pass

    if warnings:
        return CheckResult(
            "unused_imports", False,
            f"{len(warnings)} potentially unused import(s): "
            f"{'; '.join(warnings[:5])}",
            severity="warning"
        )
    return CheckResult("unused_imports", True, "No obviously unused imports")


def check_readme_quality(app_dir: Path) -> CheckResult:
    """Check 9: README exists and has minimum content (optional)."""
    readme = app_dir / "README.md"
    if not readme.exists():
        return CheckResult(
            "readme", False,
            "No README.md found",
            severity="info"
        )

    try:
        with open(readme, "r", encoding="utf-8") as f:
            content = f.read()

        word_count = len(content.split())
        if word_count < 20:
            return CheckResult(
                "readme", False,
                f"README.md is too short ({word_count} words)",
                severity="info"
            )
    except (OSError, UnicodeDecodeError):
        return CheckResult(
            "readme", False, "Could not read README.md",
            severity="info"
        )

    return CheckResult("readme", True, f"README.md present ({word_count} words)")


def check_app_yaml(app_dir: Path) -> CheckResult:
    """Check 10: app.yaml metadata file (optional)."""
    app_yaml = app_dir / "app.yaml"
    if not app_yaml.exists():
        return CheckResult(
            "app_yaml", False,
            "No app.yaml metadata file",
            severity="info"
        )

    try:
        # Basic YAML validation without requiring pyyaml
        with open(app_yaml, "r", encoding="utf-8") as f:
            content = f.read()
        required_keys = ["name", "task", "model"]
        for key in required_keys:
            if f"{key}:" not in content:
                return CheckResult(
                    "app_yaml", False,
                    f"app.yaml missing required key: {key}",
                    severity="warning"
                )
    except (OSError, UnicodeDecodeError):
        return CheckResult(
            "app_yaml", False, "Could not read app.yaml",
            severity="warning"
        )

    return CheckResult("app_yaml", True, "app.yaml present and valid")


def check_unreachable_code(app_dir: Path) -> CheckResult:
    """Check 11: Detect unreachable code after return/raise/sys.exit."""
    py_files = list(app_dir.rglob("*.py"))
    warnings = []

    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            in_function = False
            prev_was_return = False
            for i, line in enumerate(lines, 1):
                stripped = line.strip()

                if stripped.startswith("def ") or stripped.startswith("class "):
                    in_function = True
                    prev_was_return = False
                    continue

                if prev_was_return and stripped and not stripped.startswith("#"):
                    if not stripped.startswith(("def ", "class ", "except",
                                               "elif", "else:", "finally:",
                                               "@")):
                        indent = len(line) - len(line.lstrip())
                        if indent > 0:  # Only flag indented code
                            rel = py_file.relative_to(app_dir)
                            warnings.append(f"{rel}:{i}")
                            prev_was_return = False
                            continue

                if in_function and stripped.startswith(
                    ("return ", "return\n", "raise ", "sys.exit(")
                ):
                    prev_was_return = True
                else:
                    prev_was_return = False

        except (OSError, UnicodeDecodeError):
            pass

    if warnings:
        return CheckResult(
            "unreachable_code", False,
            f"{len(warnings)} potentially unreachable line(s): "
            f"{', '.join(warnings[:5])}",
            severity="warning"
        )
    return CheckResult("unreachable_code", True, "No unreachable code detected")


# ---------------------------------------------------------------------------
# Smoke Tests (3)
# ---------------------------------------------------------------------------

def smoke_test_help(app_dir: Path) -> CheckResult:
    """Smoke 1: Run --help on each app script."""
    app_scripts = [
        f for f in app_dir.glob("*.py")
        if f.name != "__init__.py"
        and not f.name.startswith("_")
        and ("sync" in f.name or "async" in f.name)
    ]

    if not app_scripts:
        return CheckResult(
            "smoke_help", False,
            "No app scripts found to test",
            scope="smoke"
        )

    failures = []
    for script in app_scripts:
        try:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                capture_output=True, text=True, timeout=10,
                cwd=str(app_dir)
            )
            if result.returncode != 0:
                failures.append(
                    f"{script.name}: exit code {result.returncode}"
                )
        except subprocess.TimeoutExpired:
            failures.append(f"{script.name}: timeout")
        except Exception as e:
            failures.append(f"{script.name}: {e}")

    if failures:
        return CheckResult(
            "smoke_help", False,
            f"{len(failures)} script(s) failed --help: "
            f"{'; '.join(failures[:3])}",
            scope="smoke"
        )
    return CheckResult(
        "smoke_help", True,
        f"{len(app_scripts)} script(s) pass --help",
        scope="smoke"
    )


def smoke_test_import(app_dir: Path) -> CheckResult:
    """Smoke 2: Verify factory can be imported."""
    factory_dir = app_dir / "factory"
    if not factory_dir.exists():
        return CheckResult(
            "smoke_import", False,
            "factory/ directory not found",
            scope="smoke"
        )

    factory_files = list(factory_dir.glob("*_factory.py"))
    if not factory_files:
        return CheckResult(
            "smoke_import", False,
            "No factory file found",
            scope="smoke"
        )

    factory_file = factory_files[0]
    module_name = factory_file.stem

    # Build import test script
    test_code = f"""
import sys
from pathlib import Path
_module_dir = Path("{app_dir}")
_v3_dir = _module_dir.parent.parent
for _p in [str(_v3_dir), str(_module_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from factory import *
print("import OK")
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            stderr = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown"
            return CheckResult(
                "smoke_import", False,
                f"Factory import failed: {stderr}",
                scope="smoke"
            )
    except subprocess.TimeoutExpired:
        return CheckResult(
            "smoke_import", False,
            "Factory import timed out",
            scope="smoke"
        )

    return CheckResult("smoke_import", True, "Factory imports OK", scope="smoke")


def smoke_test_npu(app_dir: Path) -> CheckResult:
    """Smoke 3: Check if NPU is available (informational)."""
    try:
        result = subprocess.run(
            ["dxrt-cli", "-s"],
            capture_output=True, text=True, timeout=5
        )
        if "ready" in result.stdout.lower():
            return CheckResult(
                "smoke_npu", True,
                "NPU device detected and ready",
                scope="smoke"
            )
        else:
            return CheckResult(
                "smoke_npu", False,
                "NPU not ready: " + result.stdout.strip()[:80],
                severity="info",
                scope="smoke"
            )
    except FileNotFoundError:
        return CheckResult(
            "smoke_npu", False,
            "dxrt-cli not found — DX-RT not installed",
            severity="info",
            scope="smoke"
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            "smoke_npu", False,
            "dxrt-cli timed out",
            severity="info",
            scope="smoke"
        )


# ---------------------------------------------------------------------------
# Scope-Aware Analysis
# ---------------------------------------------------------------------------

def detect_app_scope(app_dir: Path) -> Dict[str, bool]:
    """Detect which features the app uses for scope-aware analysis."""
    scope = {
        "has_async": False,
        "has_cpp_postprocess": False,
        "has_config": False,
        "has_readme": False,
        "has_tests": False,
        "is_python": True,
        "is_cpp": False,
    }

    for f in app_dir.glob("*.py"):
        if "async" in f.name:
            scope["has_async"] = True
        if "cpp_postprocess" in f.name:
            scope["has_cpp_postprocess"] = True

    scope["has_config"] = (app_dir / "config.json").exists()
    scope["has_readme"] = (app_dir / "README.md").exists()
    scope["is_cpp"] = bool(list(app_dir.glob("*.cpp")) or
                           list(app_dir.glob("CMakeLists.txt")))

    return scope


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(app_dir: Path, smoke_test: bool = False,
                   strict: bool = False) -> ValidationReport:
    """Run all validation checks on the target directory."""
    report = ValidationReport(str(app_dir))
    scope = detect_app_scope(app_dir)

    # Static checks (always run)
    static_checks = [
        check_required_files,
        check_syntax,
        check_no_relative_imports,
        check_logging,
        check_no_hardcoded_paths,
        check_ifactory_implementation,
        check_parse_common_args,
        check_unused_imports,
        check_readme_quality,
        check_app_yaml,
        check_unreachable_code,
    ]

    for check_fn in static_checks:
        result = check_fn(app_dir)
        # In strict mode, warnings become errors
        if strict and result.severity == "warning" and not result.passed:
            result.severity = "error"
        report.add(result)

    # Smoke tests (optional)
    if smoke_test:
        smoke_checks = [
            smoke_test_help,
            smoke_test_import,
            smoke_test_npu,
        ]
        for check_fn in smoke_checks:
            result = check_fn(app_dir)
            report.add(result)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate a dx_app model directory."
    )
    parser.add_argument(
        "target_dir",
        help="Path to the model directory to validate"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run smoke tests (--help, import, NPU check)"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as errors"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show all check results including passes"
    )

    args = parser.parse_args()
    target = Path(args.target_dir).resolve()

    if not target.exists():
        print(f"Error: Directory not found: {target}", file=sys.stderr)
        sys.exit(2)

    if not target.is_dir():
        print(f"Error: Not a directory: {target}", file=sys.stderr)
        sys.exit(2)

    report = run_validation(
        target,
        smoke_test=args.smoke_test,
        strict=args.strict
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"\nValidating: {target}\n")
        report.print_report(verbose=args.verbose)

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
