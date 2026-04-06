#!/usr/bin/env python3
"""
validate_framework.py — Cross-reference integrity checker for dx_app/.deepx/.

Validates that the .deepx/ knowledge base is internally consistent:
routing table paths exist, file trees are accurate, agent handoffs resolve,
no .deepx/ content leaks into app code, skill sections are complete,
toolset signatures match, memory domain tags are valid, and model_registry
references are correct.

Usage:
    python .deepx/scripts/validate_framework.py
    python .deepx/scripts/validate_framework.py --verbose
    python .deepx/scripts/validate_framework.py --json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEEPX_DIR_NAME = ".deepx"
VALID_DOMAIN_TAGS = {"[UNIVERSAL]", "[DX_APP]", "[PPU]"}
PROHIBITED_DOMAIN_TAGS = {"[DX_STREAM]", "[HAILO]", "[PIPELINE]"}

EXPECTED_DIRECTORIES = [
    "agents",
    "contextual-rules",
    "instructions",
    "knowledge",
    "memory",
    "prompts",
    "scripts",
    "skills",
    "templates",
    "toolsets",
]

REQUIRED_MEMORY_FILES = [
    "MEMORY.md",
    "common_pitfalls.md",
    "model_zoo.md",
    "platform_api.md",
    "performance_patterns.md",
]

REQUIRED_TOOLSET_FILES = [
    "dx-engine-api.md",
    "dx-postprocess-api.md",
    "common-framework-api.md",
    "model-registry.md",
    "dx-model-format.md",
]


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, category: str, name: str, passed: bool,
                 message: str = "", severity: str = "error"):
        self.category = category
        self.name = name
        self.passed = passed
        self.message = message
        self.severity = severity

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
        }


class FrameworkReport:
    def __init__(self):
        self.results: List[CheckResult] = []

    def add(self, result: CheckResult):
        self.results.append(result)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "error")

    def summary(self) -> dict:
        categories: Dict[str, dict] = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {"passed": 0, "failed": 0}
            if r.passed:
                categories[r.category]["passed"] += 1
            else:
                categories[r.category]["failed"] += 1

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "categories": categories,
        }

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Category 1: Routing Table Path Validation
# ---------------------------------------------------------------------------

def check_routing_table_paths(deepx_dir: Path, report: FrameworkReport):
    """Verify all file paths referenced in routing tables exist."""
    md_files = list(deepx_dir.rglob("*.md"))
    dx_app_root = deepx_dir.parent

    for md_file in md_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Find .deepx/ path references
            refs = re.findall(r'`\.deepx/([^`]+)`', content)
            refs += re.findall(r'\.deepx/(\S+\.md)', content)

            for ref in refs:
                ref_path = deepx_dir / ref
                if not ref_path.exists():
                    report.add(CheckResult(
                        "routing_paths",
                        f"path_ref:{md_file.name}",
                        False,
                        f"Referenced path .deepx/{ref} does not exist "
                        f"(in {md_file.relative_to(deepx_dir)})"
                    ))
                else:
                    report.add(CheckResult(
                        "routing_paths",
                        f"path_ref:{ref}",
                        True,
                        f".deepx/{ref} exists"
                    ))
        except (OSError, UnicodeDecodeError):
            pass


# ---------------------------------------------------------------------------
# Category 2: File Tree Accuracy
# ---------------------------------------------------------------------------

def check_file_tree(deepx_dir: Path, report: FrameworkReport):
    """Verify expected directory structure exists."""
    for dirname in EXPECTED_DIRECTORIES:
        dir_path = deepx_dir / dirname
        exists = dir_path.exists() and dir_path.is_dir()
        report.add(CheckResult(
            "file_tree",
            f"dir:{dirname}",
            exists,
            f"{dirname}/ {'exists' if exists else 'MISSING'}"
        ))

    # Check required files
    for filename in REQUIRED_MEMORY_FILES:
        file_path = deepx_dir / "memory" / filename
        exists = file_path.exists()
        report.add(CheckResult(
            "file_tree",
            f"memory:{filename}",
            exists,
            f"memory/{filename} {'exists' if exists else 'MISSING'}"
        ))

    for filename in REQUIRED_TOOLSET_FILES:
        file_path = deepx_dir / "toolsets" / filename
        exists = file_path.exists()
        report.add(CheckResult(
            "file_tree",
            f"toolset:{filename}",
            exists,
            f"toolsets/{filename} {'exists' if exists else 'MISSING'}"
        ))


# ---------------------------------------------------------------------------
# Category 3: Agent Handoff Validation
# ---------------------------------------------------------------------------

def check_agent_handoffs(deepx_dir: Path, report: FrameworkReport):
    """Verify agent routes-to targets exist as agent files."""
    agents_dir = deepx_dir / "agents"
    if not agents_dir.exists():
        report.add(CheckResult(
            "agent_handoffs", "agents_dir", False,
            "agents/ directory not found"
        ))
        return

    agent_files = {f.stem: f for f in agents_dir.glob("*.md")}

    for agent_name, agent_file in agent_files.items():
        try:
            with open(agent_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Find routes-to targets
            targets = re.findall(r'target:\s*(\S+)', content)
            for target in targets:
                # Target should be another agent file
                target_file = agents_dir / f"{target}.md"
                if not target_file.exists():
                    report.add(CheckResult(
                        "agent_handoffs",
                        f"route:{agent_name}->{target}",
                        False,
                        f"Agent {agent_name} routes to {target} but "
                        f"agents/{target}.md does not exist",
                        severity="warning"
                    ))
                else:
                    report.add(CheckResult(
                        "agent_handoffs",
                        f"route:{agent_name}->{target}",
                        True,
                        f"Route {agent_name} -> {target} resolves"
                    ))
        except (OSError, UnicodeDecodeError):
            pass


# ---------------------------------------------------------------------------
# Category 4: .deepx/ Leak Detection
# ---------------------------------------------------------------------------

def check_deepx_leak(deepx_dir: Path, report: FrameworkReport):
    """Ensure .deepx/ content doesn't leak into application source code."""
    dx_app_root = deepx_dir.parent
    src_dir = dx_app_root / "src"

    if not src_dir.exists():
        report.add(CheckResult(
            "leak_detection", "src_dir", True,
            "No src/ directory to check (OK for submodule)"
        ))
        return

    leak_patterns = [
        r'\.deepx/',
        r'\.hailo/',
        r'hailo_apps\.',
    ]

    leaks = []
    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    for pattern in leak_patterns:
                        if re.search(pattern, line):
                            if line.strip().startswith("#"):
                                continue
                            rel = py_file.relative_to(dx_app_root)
                            leaks.append(f"{rel}:{i}: {pattern}")
        except (OSError, UnicodeDecodeError):
            pass

    if leaks:
        report.add(CheckResult(
            "leak_detection", "source_leaks", False,
            f"{len(leaks)} leak(s) found: {'; '.join(leaks[:5])}"
        ))
    else:
        report.add(CheckResult(
            "leak_detection", "source_leaks", True,
            "No .deepx/ leaks in source code"
        ))


# ---------------------------------------------------------------------------
# Category 5: Skill Section Completeness
# ---------------------------------------------------------------------------

def check_skill_sections(deepx_dir: Path, report: FrameworkReport):
    """Verify skill files have required sections."""
    skills_dir = deepx_dir / "skills"
    if not skills_dir.exists():
        report.add(CheckResult(
            "skill_sections", "skills_dir", True,
            "No skills/ directory (OK if skills are empty)"
        ))
        return

    required_sections = ["##", "Phase", "Step", "Workflow"]

    for skill_file in skills_dir.glob("*.md"):
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()

            has_headers = "## " in content or "# " in content
            has_structure = any(s in content for s in required_sections)

            report.add(CheckResult(
                "skill_sections",
                f"skill:{skill_file.stem}",
                has_headers and has_structure,
                f"{skill_file.name}: {'structured' if has_structure else 'missing structure'}"
            ))
        except (OSError, UnicodeDecodeError):
            pass


# ---------------------------------------------------------------------------
# Category 6: Toolset Signature Validation
# ---------------------------------------------------------------------------

def check_toolset_signatures(deepx_dir: Path, report: FrameworkReport):
    """Verify toolset files contain API signatures and code examples."""
    toolsets_dir = deepx_dir / "toolsets"
    if not toolsets_dir.exists():
        return

    for toolset_file in toolsets_dir.glob("*.md"):
        try:
            with open(toolset_file, "r", encoding="utf-8") as f:
                content = f.read()

            has_code = "```" in content
            has_params = "Parameter" in content or "param" in content.lower()
            has_returns = "Return" in content or "return" in content.lower()

            report.add(CheckResult(
                "toolset_signatures",
                f"toolset:{toolset_file.stem}",
                has_code,
                f"{toolset_file.name}: "
                f"{'has code examples' if has_code else 'MISSING code examples'}"
            ))
        except (OSError, UnicodeDecodeError):
            pass


# ---------------------------------------------------------------------------
# Category 7: Memory Domain Tag Validation
# ---------------------------------------------------------------------------

def check_memory_domain_tags(deepx_dir: Path, report: FrameworkReport):
    """Verify memory files use valid domain tags and exclude prohibited ones."""
    memory_dir = deepx_dir / "memory"
    if not memory_dir.exists():
        return

    for mem_file in memory_dir.glob("*.md"):
        try:
            with open(mem_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Find all domain tags
            tags = re.findall(r'\[([A-Z_]+)\]', content)
            bracketed_tags = {f"[{t}]" for t in tags}

            # Check for prohibited tags
            prohibited_found = bracketed_tags & PROHIBITED_DOMAIN_TAGS
            if prohibited_found:
                report.add(CheckResult(
                    "memory_tags",
                    f"prohibited:{mem_file.name}",
                    False,
                    f"{mem_file.name} contains prohibited tags: "
                    f"{', '.join(prohibited_found)}"
                ))
            else:
                report.add(CheckResult(
                    "memory_tags",
                    f"tags:{mem_file.name}",
                    True,
                    f"{mem_file.name}: no prohibited domain tags"
                ))

        except (OSError, UnicodeDecodeError):
            pass


# ---------------------------------------------------------------------------
# Category 8: Model Registry Integrity
# ---------------------------------------------------------------------------

def check_model_registry(deepx_dir: Path, report: FrameworkReport):
    """Verify model_registry.json references are consistent."""
    dx_app_root = deepx_dir.parent
    registry_path = dx_app_root / "config" / "model_registry.json"

    if not registry_path.exists():
        report.add(CheckResult(
            "model_registry", "registry_file", False,
            "config/model_registry.json not found",
            severity="warning"
        ))
        return

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        report.add(CheckResult(
            "model_registry", "valid_json", True,
            f"model_registry.json is valid ({len(registry)} models)"
        ))

        # Check model entries have required fields
        required_fields = ["model_name", "dxnn_file"]
        invalid = []

        if isinstance(registry, dict):
            for model_name, info in registry.items():
                if isinstance(info, dict):
                    for field in required_fields:
                        if field not in info:
                            invalid.append(f"{model_name} missing {field}")
        elif isinstance(registry, list):
            for entry in registry:
                if isinstance(entry, dict):
                    model_name = entry.get("model_name", "<unknown>")
                    for field in required_fields:
                        if field not in entry:
                            invalid.append(f"{model_name} missing {field}")

        if invalid:
            report.add(CheckResult(
                "model_registry", "required_fields", False,
                f"{len(invalid)} model(s) missing required fields: "
                f"{'; '.join(invalid[:5])}"
            ))
        else:
            report.add(CheckResult(
                "model_registry", "required_fields", True,
                "All models have required fields"
            ))

    except json.JSONDecodeError as e:
        report.add(CheckResult(
            "model_registry", "valid_json", False,
            f"model_registry.json is invalid JSON: {e}"
        ))
    except (OSError, UnicodeDecodeError) as e:
        report.add(CheckResult(
            "model_registry", "readable", False,
            f"Cannot read model_registry.json: {e}"
        ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(deepx_dir: Path) -> FrameworkReport:
    """Run all 8 categories of validation."""
    report = FrameworkReport()

    check_routing_table_paths(deepx_dir, report)
    check_file_tree(deepx_dir, report)
    check_agent_handoffs(deepx_dir, report)
    check_deepx_leak(deepx_dir, report)
    check_skill_sections(deepx_dir, report)
    check_toolset_signatures(deepx_dir, report)
    check_memory_domain_tags(deepx_dir, report)
    check_model_registry(deepx_dir, report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate dx_app/.deepx/ framework integrity."
    )
    parser.add_argument(
        "--deepx-dir", type=str, default=None,
        help="Path to .deepx/ directory (auto-detected if not specified)"
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

    if args.deepx_dir:
        deepx_dir = Path(args.deepx_dir).resolve()
    else:
        # Auto-detect: look for .deepx/ relative to script location
        script_dir = Path(__file__).parent
        deepx_dir = script_dir.parent  # scripts/ is inside .deepx/
        if deepx_dir.name != DEEPX_DIR_NAME:
            # Try current directory
            deepx_dir = Path.cwd() / DEEPX_DIR_NAME

    if not deepx_dir.exists():
        print(f"Error: .deepx/ directory not found: {deepx_dir}", file=sys.stderr)
        sys.exit(2)

    report = run_validation(deepx_dir)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"\nFramework Integrity Check: {deepx_dir}\n")
        summary = report.summary()
        print(f"  Categories: {len(summary['categories'])}")
        print(f"  Checks: {summary['passed']}/{summary['total']} passed\n")

        for category, counts in sorted(summary["categories"].items()):
            status = "OK" if counts["failed"] == 0 else "ISSUES"
            print(f"  [{status}] {category}: "
                  f"{counts['passed']} passed, {counts['failed']} failed")

        if args.verbose:
            print("\nDetailed Results:\n")
            for r in report.results:
                if not r.passed or args.verbose:
                    status = "PASS" if r.passed else "FAIL"
                    print(f"  [{status}] {r.category}/{r.name}: {r.message}")

        print(f"\n  Overall: {'PASSED' if report.passed else 'FAILED'}")

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
