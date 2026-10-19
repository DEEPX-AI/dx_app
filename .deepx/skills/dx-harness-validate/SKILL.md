---
name: dx-harness-validate
description: "Internal harness development: validate this repo's .deepx/ integrity, acquiring the suite harness first when running from a standalone checkout"
---

# Internal Validate — `dx-runtime/dx_app` Harness Integrity

Validates the `.deepx/` canonical source of this repo and the platform files
generated from it. Use whenever a task modifies anything under `**/.deepx/**`
(standalone inference apps).

## Quick Start

```bash
# 1. Acquire the suite harness (no-op when a suite is already above this repo)
#    and run the CI-identical verification: drift check + validate_framework.py
bash .deepx/scripts/harness_bootstrap.sh --check

# 2. This repo's own framework checks (also run by step 1)
python .deepx/scripts/validate_framework.py
```

## Why step 1 exists

The `dx-agent-gen` generator and the shared fragments live **only** in
dx-all-suite. In a standalone clone of this repo, `dx-agent-gen generate` and
`dx-agent-gen check` cannot run at all, so `.deepx/` edits would ship
unverified and become drift that only CI catches.

`harness_bootstrap.sh` resolves a suite — explicit `--suite-dir` /
`$DX_SUITE_DIR`, a parent directory, the `.dx-harness/suite` cache, or a
shallow clone into the git-ignored `.dx-harness/` — then delegates to the
suite's `subrepo_check.sh`.

## Exit codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | Clean — generated files match `.deepx/`, validator passed | proceed |
| 1 | Drift (`MISSING:` / `CHANGED:`) or validator failure | regenerate from a suite checkout, then commit the regenerated files here |
| 3 | No suite could be acquired | **STOP** — do not edit `.deepx/`; see the Standalone Checkout HARD GATE in this repo's CLAUDE.md / AGENTS.md |

## Scope

Runs exactly what the CI `subrepo-gate` runs, so local green means CI green.

Suite-wide **conformance** tests are deliberately **not** run here: those checks
are cross-level (they compare all 5 repos against each other) and cannot be
satisfied when the other sub-repos are absent. Run them from a full dx-all-suite
checkout:

```bash
bash .deepx/tools/scripts/harness_gate.sh all
```

## When to use

- After modifying any file under `.deepx/` in this repo
- Before claiming internal harness work is complete
- As the Instruction File Verification Loop when this repo is checked out alone
