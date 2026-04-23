---
name: dx-tdd
description: Test-driven development for dx_app
---

# Skill: Test-Driven Development for dx_app

> **Type: RIGID — Iron Law.** No file is written until its validation passes.
> Write → validate → fix or proceed. Never batch.

## When to Invoke

- During any dx_app code generation (Python or C++ apps)
- After each file is created or modified

## Iron Law

```
WRITE file → VALIDATE → PASS? → next file
                       → FAIL? → fix → re-validate
```

## Red-Green-Verify Cycle

1. **Red**: Define what the file must satisfy
2. **Green**: Write the file
3. **Verify**: Run the check immediately — every file, every time

## Validation Order: Python Apps

Each file must pass before creating the next.

| # | File | Check |
|---|------|-------|
| 1 | `factory/<model>_factory.py` | py_compile + 5-method interface check |
| 2 | `factory/__init__.py` | py_compile + import statement present |
| 3 | `config.json` | JSON parse |
| 4 | `<model>_sync.py` | py_compile |
| 5 | `<model>_async.py` | py_compile |
| 6 | `<model>_sync_cpp_postprocess.py` | py_compile |
| 7 | `<model>_async_cpp_postprocess.py` | py_compile |
| 8 | `session.json` | JSON parse |
| 9 | `README.md` | file exists |
| 10 | `setup.sh` | file exists + bash syntax check (`bash -n`) |
| 11 | `run.sh` | file exists + bash syntax check (`bash -n`) |
| 12 | `session.log` | file exists (generated at end of build) |

### Check 1: Factory — syntax + interface
```bash
python -c "import py_compile; py_compile.compile('factory/<model>_factory.py', doraise=True)"
python -c "
import ast, sys
tree = ast.parse(open('factory/<model>_factory.py').read())
required = {'create_preprocessor','create_postprocessor','create_visualizer','get_model_name','get_task_type'}
found = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in required}
missing = required - found
if missing: print(f'FAIL: Missing: {missing}'); sys.exit(1)
print('PASS: All 5 IFactory methods present')
"
```

### Check 2: Factory __init__.py
```bash
python -c "import py_compile; py_compile.compile('factory/__init__.py', doraise=True)"
python -c "assert 'import' in open('factory/__init__.py').read(); print('PASS')"
```

### Check 3 & 8: JSON files
```bash
python -c "import json; json.load(open('config.json')); print('PASS: config.json')"
python -c "import json; json.load(open('session.json')); print('PASS: session.json')"
```

### Checks 4-7: Variant scripts
```bash
for f in <model>_sync.py <model>_async.py <model>_sync_cpp_postprocess.py <model>_async_cpp_postprocess.py; do
    python -c "import py_compile; py_compile.compile('$f', doraise=True)" && echo "OK: $f"
done
```

### Check 9: README
```bash
test -f README.md && echo "PASS" || echo "FAIL: README.md missing"
```

### Checks 10-11: Deployment Scripts (setup.sh, run.sh)
```bash
# setup.sh — must exist and have valid bash syntax
test -f setup.sh && bash -n setup.sh && echo "PASS: setup.sh" || echo "FAIL: setup.sh missing or invalid"

# run.sh — must exist and have valid bash syntax
test -f run.sh && bash -n run.sh && echo "PASS: run.sh" || echo "FAIL: run.sh missing or invalid"
```

### Check 12: session.log
```bash
# session.log — generated at end of build via tee
test -f session.log && echo "PASS: session.log" || echo "FAIL: session.log missing"
```

## Validation Order: C++ Apps

| # | File | Check |
|---|------|-------|
| 1 | `CMakeLists.txt` | exists + `project()` present |
| 2 | `config.json` | JSON parse |
| 3 | `main.cpp` | exists + `#include` present |
| 4 | Build test | `cmake -B build && cmake --build build` |

## Framework-Level Validation

After all files pass: `python .deepx/scripts/validate_app.py`

## Cross-Validation with Reference Model (Post-TDD)

After TDD cycle completes all file-level validations, run the cross-validation
from `dx-validate.md` Level 5.5 if a precompiled reference model or existing
verified example is available. This catches runtime correctness issues that
static validation cannot detect (e.g., wrong postprocessor behavior, config
mismatch, output format differences).

```bash
# Quick check: is a precompiled reference available?
MODEL_NAME="<model_name>"
DX_APP_ROOT="$(cd ../.. && pwd)"
[ -f "${DX_APP_ROOT}/assets/models/${MODEL_NAME}.dxnn" ] && echo "Run Level 5.5 cross-validation" || echo "SKIP: no reference"
```

## Common Failures

| Failure | Fix |
|---------|-----|
| `SyntaxError` | Check the exact line reported |
| Missing IFactory method | Add missing method to factory |
| `JSONDecodeError` | Use double quotes, no trailing commas |
| Empty `__init__.py` | Add `from .<model>_factory import <Class>Factory` |

## Anti-Patterns

- Do NOT create all files then validate at the end
- Do NOT skip py_compile because "it looks right"
- Do NOT skip the 5-method interface check on the factory
- Do NOT validate JSON by eyeballing — always parse programmatically
