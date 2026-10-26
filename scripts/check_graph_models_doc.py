#!/usr/bin/env python3
"""Guard: fail when docs/graph_models.md is stale.

The configure step writes its copy of the model table into the build tree
and never touches the tracked file (SP1 U-12), so a registry or factory
change that forgets to regenerate the doc is caught here, next to the other
scripts/check_*.py guards. Update: python3 scripts/gen_model_registry.py --docs-only
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    sys.exit(subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "gen_model_registry.py"), "--check-docs"],
        cwd=str(ROOT)))
