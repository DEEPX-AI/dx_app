# Agent Protocols for dx_app

Behavioral protocols that all agents must follow when operating within the dx_app
codebase. These protocols ensure consistency, safety, and quality across all
agent-driven development.

## Protocol 1: Context Before Action

Before writing any code, the agent MUST understand the current state:
1. Check if the model directory already exists under `src/python_example/<task>/`
   or `src/cpp_example/<task>/`
2. Query `config/model_registry.json` for model metadata
3. Identify the correct IFactory interface for the task type
4. Review existing similar models for patterns (e.g., look at another YOLO variant)

**Rationale**: Creating a YOLOv8n app without checking that YOLOv8nPostprocessor
exists in `common/processors/` leads to runtime ImportError.

## Protocol 2: Confirm Before Generating

Present the implementation plan to the user before creating files:
- List all files that will be created
- Show the factory interface that will be implemented
- Confirm input source and model path assumptions

Never generate code without user confirmation for non-trivial requests.

## Protocol 3: Single Source of Truth

- **Factory is the single source of truth** for component creation
- **config.json is the single source of truth** for runtime parameters
- **model_registry.json is the single source of truth** for model metadata
- **parse_common_args() is the single source of truth** for CLI arguments

Never duplicate these concerns. Never define argparse in model scripts.
Never hardcode thresholds in factory code.

## Protocol 4: Validate Incrementally

After each file creation, validate immediately:

```python
# After creating factory
python -c "import py_compile; py_compile.compile('factory/model_factory.py', doraise=True)"

# After creating sync variant
python -c "import py_compile; py_compile.compile('model_sync.py', doraise=True)"

# After creating config.json
python -c "import json; json.load(open('config.json'))"
```

Do not batch validation to the end — catch errors file-by-file.

## Protocol 5: Follow Existing Patterns

When creating a new model application, follow the exact patterns of existing
applications for the same task type:

- Detection: Follow `src/python_example/object_detection/yolov8n/`
- Classification: Follow `src/python_example/classification/` examples
- Segmentation: Follow `src/python_example/semantic_segmentation/bisenetv1/`

Copy structure and conventions, substituting only model-specific details.

## Protocol 6: Error Messages Must Be Actionable

When validation fails, the error message must tell the user exactly what to do:

```
# BAD
"Error: Module not found"

# GOOD
"Error: dx_postprocess module not found. Run './build.sh' in the dx_app root
directory to compile C++ postprocess bindings."
```

## Protocol 7: No Side Effects Outside Scope

An agent building a model application MUST NOT:
- Modify files in `common/` (framework layer)
- Modify `config/model_registry.json` (unless explicitly asked)
- Install packages or modify the Python environment
- Modify CMakeLists.txt at the project root level

Scope is limited to the model directory: `src/python_example/<task>/<model>/`

## Protocol 8: Document What You Create

Every model application must include:
- Module docstring in each Python file
- Usage example in the docstring
- config.json with documented threshold values

## Protocol 9: Test Before Declaring Success

Never claim an application is "complete" without running at minimum:
1. Syntax validation (`py_compile`) on all Python files
2. JSON validation on config.json
3. Import test (if dx_engine is available):
   ```bash
   PYTHONPATH=src/python_example python -c "from factory import <Model>Factory; print('OK')"
   ```

## Protocol 10: Preserve Existing Functionality

When modifying an existing model application:
- Run the original application first to establish baseline behavior
- Make changes incrementally
- Re-run after each change to verify no regression
- If any variant (sync/async/cpp_postprocess) breaks, fix it before proceeding

## Protocol 11: NPU Verification

Before running any inference test, verify the NPU is accessible:

```bash
dxrt-cli -s
```

Expected output shows device info. If this command fails:
- NPU hardware may not be present (skip NPU-dependent tests)
- DX-RT may not be installed (install dx_engine package)
- Driver may not be loaded (check with `lsmod | grep deepx`)

Never attempt inference without confirming NPU availability first. On CI systems
without NPU, ensure all tests are properly decorated with `@requires_npu`.

## Protocol Summary

| # | Protocol | Key Rule |
|---|---|---|
| 1 | Context Before Action | Read before write |
| 2 | Confirm Before Generating | Plan before code |
| 3 | Single Source of Truth | No duplication |
| 4 | Validate Incrementally | Test after each file |
| 5 | Follow Existing Patterns | Copy then customize |
| 6 | Actionable Error Messages | Tell user what to do |
| 7 | No Side Effects | Stay in model dir |
| 8 | Document What You Create | Docstrings + usage |
| 9 | Test Before Declaring Success | Verify before claim |
| 10 | Preserve Existing Functionality | No regressions |
| 11 | NPU Verification | `dxrt-cli -s` first |
| 12 | Output Isolation | dx-agentic-dev/ by default |

## Protocol 12: Output Isolation

All AI-generated applications MUST be created under `dx-agentic-dev/`, not in the
production `src/` directory tree. This is a HARD GATE — no exceptions without
explicit user approval.

### Rules

1. **Default output**: `dx-agentic-dev/<YYYYMMDD-HHMMSS>_<model>_<task>/`
2. **Session metadata**: Every session directory includes `session.json` and `README.md`
3. **Dynamic imports**: Use the root-finding boilerplate (see skill docs) instead of
   the standard `_v3_dir = _module_dir.parent.parent` pattern
4. **Production write**: Only when user EXPLICITLY requests production placement
5. **Validation**: `validate_app.py` works with any path — pass the `dx-agentic-dev/` path

### Why This Matters

Without output isolation, an agent asked to "build a yolo26n app" would overwrite the
existing production `src/python_example/object_detection/yolo26n/` directory. Output
isolation ensures experiments are safely contained.

### Session ID Format

`YYYYMMDD-HHMMSS_<model>_<task>` — e.g., `20250403-143022_yolo26n_object_detection`

Use Python `datetime.now().strftime('%Y%m%d-%H%M%S')` or Bash `$(date +%Y%m%d-%H%M%S)`
to generate the timestamp prefix. These use the **system local timezone** (NOT UTC).
Do NOT use `datetime.utcnow()`, `datetime.now(timezone.utc)`, or `date -u`.
