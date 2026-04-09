# .deepx/ — dx_app Agentic Knowledge

Self-contained knowledge base for building DEEPX standalone inference applications.
This directory works independently — no parent repository access required.

## Directory Structure

```
.deepx/
├── README.md                          # This file — master index
├── agents/                            # AI agent definitions
│   ├── dx-app-builder.md              # Master router agent
│   ├── dx-python-builder.md           # Python app specialist
│   ├── dx-cpp-builder.md              # C++ app specialist
│   ├── dx-benchmark-builder.md        # Benchmark and profiling specialist
│   ├── dx-model-manager.md            # Model download and registry agent
│   └── dx-validator.md                # App and framework validation agent
├── contextual-rules/                  # Scope-aware coding rules
│   ├── python-example.md              # Rules for src/python_example/**
│   ├── cpp-example.md                 # Rules for src/cpp_example/**
│   ├── postprocess.md                 # Rules for src/postprocess/**
│   └── tests.md                       # Rules for tests/**
├── instructions/                      # Development instructions (6 files)
│   ├── agent-protocols.md             # Agent communication protocols
│   ├── architecture.md                # System architecture overview
│   ├── coding-standards.md            # Coding standards and conventions
│   ├── factory-pattern.md             # IFactory pattern reference
│   ├── orchestration.md               # Multi-agent orchestration rules
│   └── testing-patterns.md            # Testing patterns and pytest conventions
├── knowledge/                         # Structured knowledge bases
│   └── knowledge_base.yaml            # Bottleneck patterns, insights, recipes
├── memory/                            # Persistent learned knowledge
│   ├── MEMORY.md                      # Memory index and update protocol
│   ├── common_pitfalls.md             # Domain-tagged pitfalls (ALWAYS read)
│   ├── model_zoo.md                   # 133 models across 15 tasks
│   ├── platform_api.md                # DX-RT platform API and diagnostics
│   └── performance_patterns.md        # FPS optimization techniques
├── prompts/                           # Reusable prompt templates
│   ├── new-python-detection.md        # Create Python detection app
│   ├── new-python-segmentation.md     # Create Python segmentation app
│   ├── new-cpp-app.md                 # Create C++ app
│   └── orchestrated-build.md          # Multi-agent orchestrated build
├── scripts/                           # Validation and generation scripts
│   ├── validate_app.py                # App directory validator (11 checks + 3 smoke)
│   ├── validate_framework.py          # .deepx/ integrity checker (8 categories)
│   └── generate_platforms.py          # Platform config generator
├── skills/                            # Skill definitions (8 files)
│   ├── dx-build-python-app.md         # Build Python inference app
│   ├── dx-build-cpp-app.md            # Build C++ inference app
│   ├── dx-build-async-app.md          # Build async high-performance app
│   ├── dx-model-management.md         # Model download and registry
│   ├── dx-validate.md                 # 5-level validation pyramid
│   ├── dx-brainstorm-and-plan.md      # Process skill — brainstorm before code
│   ├── dx-tdd.md                      # Process skill — test-driven development
│   └── dx-verify-completion.md        # Process skill — verify before claiming done
├── templates/                         # Output templates
│   └── copilot-instructions.md        # GitHub Copilot instructions template
└── toolsets/                          # API reference documentation
    ├── dx-engine-api.md               # InferenceEngine / InferenceOption API
    ├── dx-postprocess-api.md          # 37 pybind11 postprocess bindings
    ├── common-framework-api.md        # SyncRunner, AsyncRunner, IFactory, CLI
    ├── model-registry.md              # model_registry.json schema and queries
    └── dx-model-format.md             # .dxnn model format specification
```

## Agents (6 files)

| Agent | Description | Routes To |
|---|---|---|
| `dx-app-builder` | Master router — classifies requests and routes to specialists | dx-python-builder, dx-cpp-builder, dx-benchmark-builder, dx-model-manager |
| `dx-python-builder` | Builds Python inference apps (4 variants, IFactory pattern) | — |
| `dx-cpp-builder` | Builds C++ inference apps (InferenceEngine API) | — |
| `dx-benchmark-builder` | Benchmarks and profiles inference performance | — |
| `dx-model-manager` | Downloads, registers, and manages .dxnn models | — |
| `dx-validator` | Validates app code and .deepx/ framework files | — (leaf agent) |

## Toolsets (5 files)

| File | Description |
|---|---|
| `dx-engine-api.md` | InferenceEngine constructor, infer(), run_async(), get_input_shape(), error codes |
| `dx-postprocess-api.md` | 37 pybind11 bindings: detection (11), classification (4), segmentation (4), pose (2), face (3), others |
| `common-framework-api.md` | SyncRunner (1104 lines), AsyncRunner, IFactory (11 interfaces), parse_common_args() (11 flags) |
| `model-registry.md` | model_registry.json schema, query patterns, 133 models, naming conventions |
| `dx-model-format.md` | .dxnn format, tensor specs, data types, compilation flow, format versions |

## Memory (5 files + index)

| File | When to Read |
|---|---|
| `MEMORY.md` | Index and update protocol |
| `common_pitfalls.md` | **ALWAYS** — 10 domain-tagged pitfalls |
| `model_zoo.md` | Choosing models, checking compatibility |
| `platform_api.md` | NPU devices, DX-RT runtime, drivers |
| `performance_patterns.md` | FPS optimization, profiling, benchmarking |

## Knowledge (1 file)

| File | Description |
|---|---|
| `knowledge_base.yaml` | Bottleneck patterns (BP-001/003/005), insights (INS-001/002/004/006/007), recipes |

## Prompts (4 files)

| File | Description |
|---|---|
| `new-python-detection.md` | Template: create Python detection app |
| `new-python-segmentation.md` | Template: create Python segmentation app |
| `new-cpp-app.md` | Template: create C++ inference app |
| `orchestrated-build.md` | Meta-template: multi-agent 5-phase build |

## Contextual Rules (4 files)

| File | Glob | Description |
|---|---|---|
| `python-example.md` | `src/python_example/**` | IFactory, parse_common_args, 4-variant naming, imports |
| `cpp-example.md` | `src/cpp_example/**` | CMakeLists.txt, SIGINT, RAII, error checks |
| `postprocess.md` | `src/postprocess/**` | pybind11 pattern, module naming, build.sh |
| `tests.md` | `tests/**` | pytest markers, NPU skip, fixtures, DXAPP_VERIFY |

## Scripts (3 files)

| Script | Description |
|---|---|
| `validate_app.py` | 11 static checks + 3 smoke tests for app directories |
| `validate_framework.py` | 8-category .deepx/ integrity checker |
| `generate_platforms.py` | Sync .deepx/ to .github/, .claude/, .cursor/ |

### validate_app.py

```bash
# Static analysis
python .deepx/scripts/validate_app.py src/python_example/object_detection/yolov8n/

# With smoke tests
python .deepx/scripts/validate_app.py src/python_example/object_detection/yolov8n/ --smoke-test

# Strict mode (warnings → errors) + JSON output
python .deepx/scripts/validate_app.py src/python_example/object_detection/yolov8n/ --strict --json
```

### validate_framework.py

```bash
# Check .deepx/ integrity
python .deepx/scripts/validate_framework.py

# Verbose output
python .deepx/scripts/validate_framework.py --verbose
```

### generate_platforms.py

```bash
# Generate all platform configs
python .deepx/scripts/generate_platforms.py --generate

# Check sync status
python .deepx/scripts/generate_platforms.py --check

# Show diff for a specific platform
python .deepx/scripts/generate_platforms.py --diff --platform copilot
```

## Templates (1 file)

| File | Description |
|---|---|
| `copilot-instructions.md` | GitHub Copilot global instructions with {ROUTING_TABLE}, {SKILLS_TABLE}, {HARDWARE_TABLE} placeholders |

## Context Routing Table

| If the task mentions... | Read these files |
|---|---|
| Python app, detection, classification | `prompts/new-python-detection.md`, `toolsets/common-framework-api.md`, `toolsets/model-registry.md` |
| C++ app, high performance | `prompts/new-cpp-app.md`, `toolsets/dx-engine-api.md`, `toolsets/dx-postprocess-api.md` |
| Async, performance | `toolsets/common-framework-api.md`, `memory/performance_patterns.md` |
| Model, download, setup | `toolsets/model-registry.md`, `memory/model_zoo.md` |
| Postprocess, pybind11 | `toolsets/dx-postprocess-api.md`, `contextual-rules/postprocess.md` |
| Testing, validation | `contextual-rules/tests.md`, `scripts/validate_app.py` |
| **Always load** | `memory/common_pitfalls.md` |

## Platform Integration

| Platform | Target Directory | Generated By |
|---|---|---|
| GitHub Copilot | `.github/copilot-instructions.md` | `generate_platforms.py --platform copilot` |
| Claude Code | `.claude/CLAUDE.md` + `.claude/rules/` | `generate_platforms.py --platform claude` |
| Cursor | `.cursor/rules/*.mdc` | `generate_platforms.py --platform cursor` |

## Developer Workflow

### Building a New App

1. Read `memory/common_pitfalls.md` (always)
2. Choose a prompt template from `prompts/`
3. Query `config/model_registry.json` via patterns in `toolsets/model-registry.md`
4. Follow the prompt template steps
5. Validate with `scripts/validate_app.py`

### Adding Knowledge

1. Read `memory/MEMORY.md` for update protocol
2. Add entries with domain tags: `[UNIVERSAL]`, `[DX_APP]`, or `[PPU]`
3. Include symptom/cause/fix for pitfalls
4. Run `scripts/validate_framework.py` to verify integrity

### Syncing to Platforms

1. Edit canonical content in `.deepx/`
2. Run `python .deepx/scripts/generate_platforms.py --check`
3. Run `python .deepx/scripts/generate_platforms.py --generate`
4. Review diffs and commit

## Key Facts

- **dx_app v3.0.0**: 15 AI tasks, 133 models, 4 Python variants, C++ examples
- **Framework**: SyncRunner (1104 lines), AsyncRunner, 11 IFactory interfaces
- **Postprocess**: 37 pybind11 C++ bindings
- **CLI**: parse_common_args() with 11 flags
- **NPU**: DX-M1 / DX-M1A via DX-RT 3.0.x
- **Model format**: .dxnn (v7+, INT8/UINT8/FP16)
