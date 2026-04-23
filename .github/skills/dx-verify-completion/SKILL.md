---
name: dx-verify-completion
description: Verify work before claiming done
---

<!-- AUTO-GENERATED from .deepx/ — DO NOT EDIT DIRECTLY -->
<!-- Source: .deepx/skills/dx-verify-completion/SKILL.md -->
<!-- Run: dx-agentic-gen generate -->

# Skill: Verify Before Completion for dx_app

> **Type: RIGID — Iron Law.** No completion claims without fresh evidence.
> "It works" is not evidence. Command output is evidence.

## When to Invoke

- Before saying "the app is complete" or "done"
- Before committing generated code or creating a pull request

## Iron Law

```
No fresh validation output → DO NOT claim completion
Validation shows FAIL      → DO NOT claim completion
Validation shows PASS      → claim with evidence
```

## Gate Function (5 Steps)

### Step 1: File Inventory
```bash
ls -la <app_dir>/ && ls -la <app_dir>/factory/    # Python
ls -la <app_dir>/ && ls -la <app_dir>/include/     # C++
```
Missing files = FAIL. Do not proceed.

### Step 2: Syntax Validation
```bash
for f in $(find <app_dir> -name '*.py' -not -path '*__pycache__*'); do
    python -c "import py_compile; py_compile.compile('$f', doraise=True)" && echo "OK: $f" || echo "FAIL: $f"
done
```

### Step 3: JSON Validation
```bash
for f in $(find <app_dir> -name '*.json'); do
    python -c "import json; json.load(open('$f')); print(f'OK: $f')" || echo "FAIL: $f"
done
```

### Step 4: Factory Compliance
```bash
python -c "
import ast, sys
tree = ast.parse(open('<app_dir>/factory/<model>_factory.py').read())
required = {'create_preprocessor','create_postprocessor','create_visualizer','get_model_name','get_task_type'}
found = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in required}
missing = required - found
if missing: print(f'FAIL: Missing: {missing}'); sys.exit(1)
print('PASS: All 5 IFactory methods present')
"
```

### Step 5: Framework Validator
```bash
python .deepx/scripts/validate_app.py
```
Any FAIL at any step = stop and fix.

### Step 5.5: Cross-Validation with Reference Model (if available)

If a precompiled DXNN exists in `assets/models/` for the same model OR an existing
verified example exists in `src/python_example/`, run the differential diagnosis.
This step is SKIP if no reference assets exist.

```bash
MODEL_NAME="<model>"
DX_APP_ROOT="$(cd ../.. && pwd)"

# Test A: Run generated app with precompiled model
REF_MODEL="${DX_APP_ROOT}/assets/models/${MODEL_NAME}.dxnn"
if [ -f "$REF_MODEL" ]; then
    python <app_dir>/<model>_sync.py --model "$REF_MODEL" \
        --image <TASK_SAMPLE_IMAGE> --no-display --verbose
    echo "Precompiled model exit code: $?"
fi

# Test B: Compare against existing verified example
TASK="<task_type>"
EXISTING="${DX_APP_ROOT}/src/python_example/${TASK}/${MODEL_NAME}/${MODEL_NAME}_sync.py"
if [ -f "$EXISTING" ] && [ -f "$REF_MODEL" ]; then
    python "$EXISTING" --model "$REF_MODEL" \
        --image <TASK_SAMPLE_IMAGE> --no-display --verbose
    echo "Existing app exit code: $?"
fi
```

**Evidence**: Reference model test results, comparison results, diagnosis.
See `dx-validate.md` Level 5.5 for the full Differential Diagnosis Decision Matrix.

## Checklist: Python Apps

- [ ] `factory/<model>_factory.py` — syntax OK, 5 methods present
- [ ] `factory/__init__.py` — syntax OK, has import statement
- [ ] `config.json` — valid JSON, task-appropriate keys
- [ ] `<model>_sync.py` — syntax OK, `parse_common_args()`, `SyncRunner`
- [ ] `<model>_async.py` — syntax OK, `parse_common_args()`, `AsyncRunner`
- [ ] `<model>_sync_cpp_postprocess.py` — syntax OK, imports `dx_postprocess`
- [ ] `<model>_async_cpp_postprocess.py` — syntax OK, imports `dx_postprocess`
- [ ] `__init__.py`, `session.json`, `README.md` — exist and valid
- [ ] `setup.sh` — exists and valid bash (`bash -n setup.sh`)
- [ ] `run.sh` — exists and valid bash (`bash -n run.sh`)
- [ ] `session.log` — exists and non-empty
- [ ] No hardcoded `.dxnn` paths, no relative imports, no bare `print()` in factory

## Checklist: C++ Apps

- [ ] `CMakeLists.txt` — has `project()` and `find_package(dx_engine)`
- [ ] `config.json` — valid JSON
- [ ] `main.cpp` — has `#include` and `main()`
- [ ] Build: `cmake -B build && cmake --build build`

## Completion Report

Present after all checks pass:

```
## Completion Report: <ModelDisplay> <TaskType> App

**Status**: PASS  |  **Output dir**: <path>

| File | Status |  | File | Status |
|------|--------|--|------|--------|
| factory/<model>_factory.py | PASS (5/5) |  | <model>_async_cpp_postprocess.py | PASS |
| factory/__init__.py | PASS |  | session.json | PASS |
| config.json | PASS |  | README.md | PASS |
| <model>_sync.py | PASS |  | setup.sh | PASS |
| <model>_async.py | PASS |  | run.sh | PASS |
| <model>_sync_cpp_postprocess.py | PASS |  | session.log | PASS |
|  |  |  | Framework validator | PASS |

### Framework Validator
<paste actual output from validate_app.py>
```

## Anti-Patterns

- Do NOT claim "files created" without running validation
- Do NOT claim "syntax correct" without py_compile output
- Do NOT skip the framework validator
- Do NOT reuse a previous session's output — run fresh

## Session Log Rules

Save **actual command execution output** to `${WORK_DIR}/session.log`.

**What session.log MUST contain**:
- Every shell command executed (prefixed with `$`)
- The real stdout/stderr output of each command
- Validation output (from py_compile, validate_app.py)
- Any error messages and recovery steps

**What session.log must NOT be**:
- A hand-written summary of "Key Decisions" sections
- A `cat << 'EOF'` block written at the end
- A markdown report with curated snippets
