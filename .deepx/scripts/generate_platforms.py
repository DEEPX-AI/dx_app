#!/usr/bin/env python3
"""
generate_platforms.py — Sync dx_app/.deepx/ knowledge to platform-specific formats.

Generates platform configurations for GitHub Copilot (.github/), Claude Code (.claude/),
and Cursor (.cursor/) from the canonical .deepx/ knowledge base.

Transformations:
- INTERACTION markers → platform-specific prompts
- Tool references → platform tool mapping
- Path rewriting → relative to platform config location
- Contextual rules → platform-specific format

Usage:
    python .deepx/scripts/generate_platforms.py --generate
    python .deepx/scripts/generate_platforms.py --check
    python .deepx/scripts/generate_platforms.py --diff
    python .deepx/scripts/generate_platforms.py --platform copilot
    python .deepx/scripts/generate_platforms.py --platform claude
    python .deepx/scripts/generate_platforms.py --platform cursor
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLATFORM_CONFIGS = {
    "copilot": {
        "target_dir": ".github",
        "instructions_file": "copilot-instructions.md",
        "rules_dir": None,  # Copilot uses single file
    },
    "claude": {
        "target_dir": ".claude",
        "instructions_file": "CLAUDE.md",
        "rules_dir": "rules",
    },
    "cursor": {
        "target_dir": ".cursor",
        "instructions_file": "rules/dx-app.mdc",
        "rules_dir": "rules",
    },
}

# Tool mapping: .deepx/ canonical names → platform-specific names
TOOL_MAPPING = {
    "copilot": {
        "Read": "read file",
        "Write": "write file",
        "Edit": "edit file",
        "Bash": "run command",
        "Grep": "search",
        "Glob": "find files",
        "TodoWrite": "track progress",
    },
    "claude": {
        "Read": "Read",
        "Write": "Write",
        "Edit": "Edit",
        "Bash": "Bash",
        "Grep": "Grep",
        "Glob": "Glob",
        "TodoWrite": "TodoWrite",
    },
    "cursor": {
        "Read": "read_file",
        "Write": "write_file",
        "Edit": "edit_file",
        "Bash": "run_terminal_cmd",
        "Grep": "grep_search",
        "Glob": "file_search",
        "TodoWrite": "N/A",
    },
}


# ---------------------------------------------------------------------------
# Content Transformers
# ---------------------------------------------------------------------------

def transform_interaction_markers(content: str, platform: str) -> str:
    """Transform <!-- INTERACTION --> markers to platform format."""
    if platform == "copilot":
        # Copilot: remove interaction markers (not supported)
        content = re.sub(
            r'<!-- INTERACTION:.*?-->',
            '',
            content,
            flags=re.DOTALL
        )
    elif platform == "claude":
        # Claude: convert to question tool format
        def replace_interaction(match):
            text = match.group(0)
            question = re.search(r'INTERACTION:\s*(.+?)(?:\n|OPTIONS)', text)
            options = re.search(r'OPTIONS:\s*(.+?)-->', text)
            if question:
                q = question.group(1).strip()
                if options:
                    opts = options.group(1).strip()
                    return f"**Ask the user:** {q}\n**Options:** {opts}\n"
                return f"**Ask the user:** {q}\n"
            return ""

        content = re.sub(
            r'<!-- INTERACTION:.*?-->',
            replace_interaction,
            content,
            flags=re.DOTALL
        )
    elif platform == "cursor":
        # Cursor: convert to inline prompt
        def replace_interaction(match):
            text = match.group(0)
            question = re.search(r'INTERACTION:\s*(.+?)(?:\n|OPTIONS)', text)
            if question:
                return f"> Ask: {question.group(1).strip()}"
            return ""

        content = re.sub(
            r'<!-- INTERACTION:.*?-->',
            replace_interaction,
            content,
            flags=re.DOTALL
        )

    return content


def transform_tool_references(content: str, platform: str) -> str:
    """Replace tool names with platform-specific equivalents."""
    mapping = TOOL_MAPPING.get(platform, {})
    for canonical, platform_name in mapping.items():
        # Replace `ToolName` references
        content = re.sub(
            rf'\b{canonical}\b(?=\s+tool|\s+to\s)',
            platform_name,
            content
        )
    return content


def transform_paths(content: str, platform: str) -> str:
    """Rewrite .deepx/ paths to platform-specific locations."""
    config = PLATFORM_CONFIGS.get(platform, {})
    target_dir = config.get("target_dir", ".deepx")

    if target_dir != ".deepx":
        # Only rewrite if the platform uses a different directory
        content = content.replace(".deepx/", f"{target_dir}/")

    return content


def transform_contextual_rules(rule_content: str, platform: str) -> str:
    """Convert contextual rules format for different platforms."""
    if platform == "cursor":
        # Cursor uses .mdc format with frontmatter
        # Keep YAML frontmatter as-is (Cursor supports it)
        return rule_content

    elif platform == "claude":
        # Claude Code uses markdown rules in .claude/rules/
        # Strip YAML frontmatter, keep content
        content = re.sub(
            r'^---\n.*?\n---\n',
            '',
            rule_content,
            flags=re.DOTALL
        )
        return content.strip()

    elif platform == "copilot":
        # Copilot doesn't support contextual rules
        # Extract the glob and description for documentation
        glob_match = re.search(r'glob:\s*"([^"]+)"', rule_content)
        desc_match = re.search(r'description:\s*(.+)', rule_content)
        if glob_match and desc_match:
            return (
                f"## Rules for `{glob_match.group(1)}`\n\n"
                f"{desc_match.group(1).strip()}\n\n"
                + re.sub(r'^---\n.*?\n---\n', '', rule_content, flags=re.DOTALL)
            )
        return rule_content

    return rule_content


# ---------------------------------------------------------------------------
# Routing Table Generator
# ---------------------------------------------------------------------------

def generate_routing_table(deepx_dir: Path) -> str:
    """Generate context routing table from .deepx/ contents."""
    rows = [
        ("Python app, detection, classification",
         "skills/dx-build-python-app.md, toolsets/common-framework-api.md, "
         "toolsets/model-registry.md"),
        ("C++ app, high performance",
         "skills/dx-build-cpp-app.md, toolsets/dx-engine-api.md, "
         "toolsets/dx-postprocess-api.md"),
        ("Async, performance",
         "skills/dx-build-async-app.md, memory/performance_patterns.md"),
        ("Model, download, setup",
         "skills/dx-model-management.md, toolsets/model-registry.md, "
         "memory/model_zoo.md"),
        ("Postprocess, pybind11",
         "toolsets/dx-postprocess-api.md, contextual-rules/postprocess.md"),
        ("Testing, validation",
         "contextual-rules/tests.md, scripts/validate_app.py"),
        ("**Always load**",
         "memory/common_pitfalls.md"),
    ]

    table = "| If the task mentions... | Read these files |\n"
    table += "|---|---|\n"
    for trigger, files in rows:
        table += f"| {trigger} | {files} |\n"

    return table


# ---------------------------------------------------------------------------
# Platform Generators
# ---------------------------------------------------------------------------

def generate_copilot(deepx_dir: Path, output_dir: Path):
    """Generate .github/copilot-instructions.md."""
    template_path = deepx_dir / "templates" / "copilot-instructions.md"

    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = _default_copilot_template()

    # Replace placeholders
    routing_table = generate_routing_table(deepx_dir)
    content = content.replace("{ROUTING_TABLE}", routing_table)

    hardware_table = (
        "| Architecture | Value | Use case |\n"
        "|---|---|---|\n"
        "| DX-M1 | `dx_m1` | Full performance NPU |\n"
        "| DX-M1A | `dx_m1a` | Extended variant |\n"
    )
    content = content.replace("{HARDWARE_TABLE}", hardware_table)

    skills_table = (
        "| Skill | Description |\n"
        "|---|---|\n"
        "| dx-build-python-app | Build Python inference app |\n"
        "| dx-build-cpp-app | Build C++ inference app |\n"
        "| dx-build-async-app | Build async high-performance app |\n"
        "| dx-model-management | Manage models and registry |\n"
        "| dx-validate | Validate applications |\n"
    )
    content = content.replace("{SKILLS_TABLE}", skills_table)

    # Transform
    content = transform_interaction_markers(content, "copilot")
    content = transform_tool_references(content, "copilot")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "copilot-instructions.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    return output_file


def generate_claude(deepx_dir: Path, output_dir: Path):
    """Generate .claude/CLAUDE.md and .claude/rules/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rules_dir = output_dir / "rules"
    rules_dir.mkdir(exist_ok=True)
    generated = []

    # Generate CLAUDE.md
    routing_table = generate_routing_table(deepx_dir)
    claude_content = _generate_claude_md(deepx_dir, routing_table)
    claude_content = transform_interaction_markers(claude_content, "claude")

    claude_file = output_dir / "CLAUDE.md"
    with open(claude_file, "w", encoding="utf-8") as f:
        f.write(claude_content)
    generated.append(claude_file)

    # Generate contextual rules
    ctx_rules_dir = deepx_dir / "contextual-rules"
    if ctx_rules_dir.exists():
        for rule_file in ctx_rules_dir.glob("*.md"):
            with open(rule_file, "r", encoding="utf-8") as f:
                content = f.read()
            content = transform_contextual_rules(content, "claude")
            output_rule = rules_dir / rule_file.name
            with open(output_rule, "w", encoding="utf-8") as f:
                f.write(content)
            generated.append(output_rule)

    return generated


def generate_cursor(deepx_dir: Path, output_dir: Path):
    """Generate .cursor/rules/."""
    rules_dir = output_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Generate main rules file
    routing_table = generate_routing_table(deepx_dir)
    main_content = _generate_cursor_mdc(deepx_dir, routing_table)
    main_content = transform_interaction_markers(main_content, "cursor")
    main_content = transform_tool_references(main_content, "cursor")

    main_file = rules_dir / "dx-app.mdc"
    with open(main_file, "w", encoding="utf-8") as f:
        f.write(main_content)
    generated.append(main_file)

    # Generate contextual rules as .mdc files
    ctx_rules_dir = deepx_dir / "contextual-rules"
    if ctx_rules_dir.exists():
        for rule_file in ctx_rules_dir.glob("*.md"):
            with open(rule_file, "r", encoding="utf-8") as f:
                content = f.read()
            content = transform_contextual_rules(content, "cursor")
            output_rule = rules_dir / rule_file.with_suffix(".mdc").name
            with open(output_rule, "w", encoding="utf-8") as f:
                f.write(content)
            generated.append(output_rule)

    return generated


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def _default_copilot_template() -> str:
    return """# dx_app — GitHub Copilot Instructions

> Auto-generated from `.deepx/`. Do not edit directly.

## Context Routing Table

{ROUTING_TABLE}

## Skills

{SKILLS_TABLE}

## Hardware

{HARDWARE_TABLE}

## Critical Conventions

1. All Python apps must implement IFactory (5 methods)
2. Use parse_common_args() — never custom argparse
3. Model paths come from CLI args — never hardcode
4. Query config/model_registry.json before creating apps
5. Use logging module — not print()
"""


def _generate_claude_md(deepx_dir: Path, routing_table: str) -> str:
    return f"""# dx_app — Claude Code Entry Point

> Auto-generated from `.deepx/`. Do not edit directly.

## Shared Knowledge

All skills, instructions, toolsets, knowledge bases, and memory live in `.deepx/`.
Read `.deepx/README.md` for the complete master index.

## Context Routing Table

{routing_table}

## Quick Reference

```bash
./build.sh      # Build C++ and pybind11 bindings
./setup.sh      # Download models and test media
```

## Critical Conventions

1. All Python apps must implement IFactory (5 methods)
2. Use parse_common_args() — never custom argparse
3. Model paths come from CLI args — never hardcode
4. Query config/model_registry.json before creating apps
5. Use logging module — not print()
6. Read .deepx/ skill docs first. Do NOT read source code unless skill is insufficient.

## Hardware

| Architecture | Value | Use case |
|---|---|---|
| DX-M1 | `dx_m1` | Full performance NPU |
| DX-M1A | `dx_m1a` | Extended variant |

## Memory

Persistent knowledge in `.deepx/memory/`. Read at task start, update when learning.
"""


def _generate_cursor_mdc(deepx_dir: Path, routing_table: str) -> str:
    return f"""---
description: dx_app standalone inference application rules
globs: ["**/*.py", "**/*.cpp", "**/*.hpp", "**/*.cmake"]
---

# dx_app Rules

## Context Routing

{routing_table}

## Conventions

- IFactory: 5 methods required (create_preprocessor, create_postprocessor,
  create_visualizer, get_model_name, get_task_type)
- CLI: parse_common_args() only — no custom argparse
- Models: query config/model_registry.json before creating apps
- Logging: use logging module, not print()
- C++ Standard: C++14 only
- SIGINT: all loop-based C++ examples must install handler
"""


# ---------------------------------------------------------------------------
# Check / Diff
# ---------------------------------------------------------------------------

def check_sync_status(deepx_dir: Path, dx_app_root: Path) -> Dict[str, str]:
    """Check if platform configs are in sync with .deepx/."""
    results = {}

    for platform, config in PLATFORM_CONFIGS.items():
        target_dir = dx_app_root / config["target_dir"]
        if not target_dir.exists():
            results[platform] = "missing"
        else:
            # Simple hash comparison of key files
            instructions_file = target_dir / config["instructions_file"]
            if instructions_file.exists():
                results[platform] = "present"
            else:
                results[platform] = "incomplete"

    return results


def generate_diff(deepx_dir: Path, dx_app_root: Path,
                  platform: str) -> Optional[str]:
    """Generate diff between current and would-be-generated content."""
    import tempfile

    config = PLATFORM_CONFIGS.get(platform)
    if not config:
        return None

    current_dir = dx_app_root / config["target_dir"]
    if not current_dir.exists():
        return f"[{platform}] No existing {config['target_dir']}/ directory"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / config["target_dir"]

        if platform == "copilot":
            generate_copilot(deepx_dir, tmp_path)
        elif platform == "claude":
            generate_claude(deepx_dir, tmp_path)
        elif platform == "cursor":
            generate_cursor(deepx_dir, tmp_path)

        # Compare
        diffs = []
        for new_file in tmp_path.rglob("*"):
            if new_file.is_dir():
                continue
            rel = new_file.relative_to(tmp_path)
            current_file = current_dir / rel

            if current_file.exists():
                with open(current_file) as f:
                    current_lines = f.readlines()
                with open(new_file) as f:
                    new_lines = f.readlines()

                diff = difflib.unified_diff(
                    current_lines, new_lines,
                    fromfile=f"current/{rel}",
                    tofile=f"generated/{rel}"
                )
                diff_text = "".join(diff)
                if diff_text:
                    diffs.append(diff_text)
            else:
                diffs.append(f"+++ NEW FILE: {rel}")

        return "\n".join(diffs) if diffs else f"[{platform}] In sync"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sync .deepx/ knowledge to platform configs."
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate platform configs from .deepx/"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check if platform configs are in sync"
    )
    parser.add_argument(
        "--diff", action="store_true",
        help="Show diff between current and generated configs"
    )
    parser.add_argument(
        "--platform", type=str, default=None,
        choices=["copilot", "claude", "cursor"],
        help="Target specific platform (default: all)"
    )
    parser.add_argument(
        "--deepx-dir", type=str, default=None,
        help="Path to .deepx/ directory"
    )

    args = parser.parse_args()

    # Locate .deepx/
    if args.deepx_dir:
        deepx_dir = Path(args.deepx_dir).resolve()
    else:
        script_dir = Path(__file__).parent
        deepx_dir = script_dir.parent
        if deepx_dir.name != ".deepx":
            deepx_dir = Path.cwd() / ".deepx"

    if not deepx_dir.exists():
        print(f"Error: .deepx/ not found: {deepx_dir}", file=sys.stderr)
        sys.exit(2)

    dx_app_root = deepx_dir.parent
    platforms = [args.platform] if args.platform else ["copilot", "claude", "cursor"]

    if args.check:
        status = check_sync_status(deepx_dir, dx_app_root)
        for platform in platforms:
            if platform in status:
                print(f"  [{platform}] {status[platform]}")
        sys.exit(0)

    if args.diff:
        for platform in platforms:
            diff = generate_diff(deepx_dir, dx_app_root, platform)
            if diff:
                print(diff)
        sys.exit(0)

    if args.generate:
        for platform in platforms:
            config = PLATFORM_CONFIGS[platform]
            target_dir = dx_app_root / config["target_dir"]
            print(f"Generating {platform} config in {target_dir}/...")

            if platform == "copilot":
                files = [generate_copilot(deepx_dir, target_dir)]
            elif platform == "claude":
                files = generate_claude(deepx_dir, target_dir)
            elif platform == "cursor":
                files = generate_cursor(deepx_dir, target_dir)
            else:
                continue

            for f in files:
                if isinstance(f, Path):
                    rel = f.relative_to(dx_app_root)
                    print(f"  Created: {rel}")

        print("\nDone.")
        sys.exit(0)

    # No action specified
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
