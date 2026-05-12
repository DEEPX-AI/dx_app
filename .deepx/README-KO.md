# .deepx/ — dx_app Agentic 지식 베이스

DEEPX 독립 실행형 추론 애플리케이션 빌드를 위한 자체 완결형 지식 베이스.
이 디렉터리는 독립적으로 동작합니다 — 상위 저장소 접근이 필요하지 않습니다.

## 디렉터리 구조

```
.deepx/
├── README.md                          # 이 파일 — 마스터 인덱스
├── agents/                            # AI 에이전트 정의
│   ├── dx-app-builder.md              # 마스터 라우터 에이전트
│   ├── dx-python-builder.md           # Python 앱 전문가
│   ├── dx-cpp-builder.md              # C++ 앱 전문가
│   ├── dx-benchmark-builder.md        # 벤치마크 및 프로파일링 전문가
│   ├── dx-model-manager.md            # 모델 다운로드 및 레지스트리 에이전트
│   └── dx-validator.md                # 앱 및 프레임워크 검증 에이전트
├── contextual-rules/                  # 스코프 인식 코딩 규칙
│   ├── python-example.md              # src/python_example/** 규칙
│   ├── cpp-example.md                 # src/cpp_example/** 규칙
│   ├── postprocess.md                 # src/postprocess/** 규칙
│   └── tests.md                       # tests/** 규칙
├── instructions/                      # 개발 지침 (6개 파일)
│   ├── agent-protocols.md             # 에이전트 통신 프로토콜
│   ├── architecture.md                # 시스템 아키텍처 개요
│   ├── coding-standards.md            # 코딩 표준 및 컨벤션
│   ├── factory-pattern.md             # IFactory 패턴 레퍼런스
│   ├── orchestration.md               # 멀티 에이전트 오케스트레이션 규칙
│   └── testing-patterns.md            # 테스트 패턴 및 pytest 컨벤션
├── knowledge/                         # 구조화된 지식 베이스
│   └── knowledge_base.yaml            # 병목 패턴, 인사이트, 레시피
├── memory/                            # 영속적 학습 지식
│   ├── MEMORY.md                      # 메모리 인덱스 및 업데이트 프로토콜
│   ├── common_pitfalls.md             # 도메인 태그된 함정 (항상 읽음)
│   ├── model_zoo.md                   # 15개 task에 걸친 133개 모델
│   ├── platform_api.md                # DX-RT 플랫폼 API 및 진단
│   └── performance_patterns.md        # FPS 최적화 기법
├── prompts/                           # 재사용 가능한 프롬프트 템플릿
│   ├── new-python-detection.md        # Python detection 앱 생성
│   ├── new-python-segmentation.md     # Python segmentation 앱 생성
│   ├── new-cpp-app.md                 # C++ 앱 생성
│   └── orchestrated-build.md          # 멀티 에이전트 오케스트레이션 빌드
├── scripts/                           # 검증 스크립트
│   ├── validate_app.py                # 앱 디렉터리 검증기 (11개 체크 + 3개 smoke)
│   └── validate_framework.py          # .deepx/ 무결성 체크 (8개 카테고리)
├── skills/                            # Skill 정의 (8개 파일)
│   ├── dx-agentic-app-build-python.md         # Python 추론 앱 빌드
│   ├── dx-agentic-app-build-cpp.md            # C++ 추론 앱 빌드
│   ├── dx-agentic-app-build-async.md          # 비동기 고성능 앱 빌드
│   ├── dx-agentic-app-model-management.md         # 모델 다운로드 및 레지스트리
│   ├── dx-agentic-app-validate.md                 # 5단계 검증 피라미드
│   ├── dx-brainstorm-and-plan.md      # 프로세스 skill — 코드 전 브레인스토밍
│   ├── dx-tdd.md                      # 프로세스 skill — 테스트 주도 개발
│   └── dx-verify-completion.md        # 프로세스 skill — 완료 선언 전 검증; 필수 산출물 시행
├── templates/                         # 지침 템플릿 (dx-agentic-gen으로 처리)
│   ├── en/                            # 영문 `.tmpl` 파일
│   └── ko/                            # 한국어 `.tmpl` 파일
└── toolsets/                          # API 레퍼런스 문서
    ├── dx-engine-api.md               # InferenceEngine / InferenceOption API
    ├── dx-postprocess-api.md          # 37개 pybind11 postprocess 바인딩
    ├── common-framework-api.md        # SyncRunner, AsyncRunner, IFactory, CLI
    ├── model-registry.md              # model_registry.json 스키마 및 쿼리
    └── dx-model-format.md             # .dxnn 모델 포맷 사양
```

## Agents (6개 파일)

| 에이전트 | 설명 | 라우팅 대상 |
|---|---|---|
| `dx-app-builder` | 마스터 라우터 — 요청을 분류하고 전문가에게 라우팅 | dx-python-builder, dx-cpp-builder, dx-benchmark-builder, dx-model-manager |
| `dx-python-builder` | Python 추론 앱 빌드 (4개 variant, IFactory 패턴) | — |
| `dx-cpp-builder` | C++ 추론 앱 빌드 (InferenceEngine API) | — |
| `dx-benchmark-builder` | 추론 성능 벤치마크 및 프로파일링 | — |
| `dx-model-manager` | .dxnn 모델 다운로드, 등록, 관리 | — |
| `dx-validator` | 앱 코드 및 .deepx/ 프레임워크 파일 검증 | — (leaf 에이전트) |

## Toolsets (5개 파일)

| 파일 | 설명 |
|---|---|
| `dx-engine-api.md` | InferenceEngine 생성자, infer(), run_async(), get_input_shape(), 에러 코드 |
| `dx-postprocess-api.md` | 37개 pybind11 바인딩: detection (11), classification (4), segmentation (4), pose (2), face (3), 기타 |
| `common-framework-api.md` | SyncRunner (1104 라인), AsyncRunner, IFactory (11개 인터페이스), parse_common_args() (11개 플래그) |
| `model-registry.md` | model_registry.json 스키마, 쿼리 패턴, 133개 모델, 명명 컨벤션 |
| `dx-model-format.md` | .dxnn 포맷, 텐서 스펙, 데이터 타입, 컴파일 흐름, 포맷 버전 |

## Memory (5개 파일 + 인덱스)

| 파일 | 읽는 시점 |
|---|---|
| `MEMORY.md` | 인덱스 및 업데이트 프로토콜 |
| `common_pitfalls.md` | **항상** — 도메인 태그된 10개 함정 |
| `model_zoo.md` | 모델 선택, 호환성 확인 |
| `platform_api.md` | NPU 디바이스, DX-RT 런타임, 드라이버 |
| `performance_patterns.md` | FPS 최적화, 프로파일링, 벤치마킹 |

## Knowledge (1개 파일)

| 파일 | 설명 |
|---|---|
| `knowledge_base.yaml` | 병목 패턴 (BP-001/003/005), 인사이트 (INS-001/002/004/006/007), 레시피 |

## Prompts (4개 파일)

| 파일 | 설명 |
|---|---|
| `new-python-detection.md` | 템플릿: Python detection 앱 생성 |
| `new-python-segmentation.md` | 템플릿: Python segmentation 앱 생성 |
| `new-cpp-app.md` | 템플릿: C++ 추론 앱 생성 |
| `orchestrated-build.md` | 메타 템플릿: 멀티 에이전트 5단계 빌드 |

## Contextual Rules (4개 파일)

| 파일 | Glob | 설명 |
|---|---|---|
| `python-example.md` | `src/python_example/**` | IFactory, parse_common_args, 4-variant 명명, import |
| `cpp-example.md` | `src/cpp_example/**` | CMakeLists.txt, SIGINT, RAII, 에러 체크 |
| `postprocess.md` | `src/postprocess/**` | pybind11 패턴, 모듈 명명, build.sh |
| `tests.md` | `tests/**` | pytest 마커, NPU skip, fixture, DXAPP_VERIFY |

## Scripts (2개 파일)

| 스크립트 | 설명 |
|---|---|
| `validate_app.py` | 앱 디렉터리에 대한 11개 정적 체크 + 3개 smoke 테스트 |
| `validate_framework.py` | 8개 카테고리 .deepx/ 무결성 체크 |

> 플랫폼 파일 생성은 suite 레벨의 **`dx-agentic-gen`** CLI가 처리합니다.
> suite 루트의 [`.deepx/tools/README.md`](../../../.deepx/tools/README.md)를 참조하세요.

### validate_app.py

```bash
# 정적 분석
python .deepx/scripts/validate_app.py src/python_example/object_detection/yolov8n/

# Smoke 테스트 포함
python .deepx/scripts/validate_app.py src/python_example/object_detection/yolov8n/ --smoke-test

# Strict 모드 (warning → error) + JSON 출력
python .deepx/scripts/validate_app.py src/python_example/object_detection/yolov8n/ --strict --json
```

### validate_framework.py

```bash
# .deepx/ 무결성 체크
python .deepx/scripts/validate_framework.py

# Verbose 출력
python .deepx/scripts/validate_framework.py --verbose
```

### 플랫폼 생성 (`dx-agentic-gen`)

```bash
# 모든 플랫폼 설정 생성 (단일 저장소)
dx-agentic-gen generate

# 동기화 상태 체크
dx-agentic-gen check

# EN/KO fragment 패리티 검증
dx-agentic-gen lint

# Suite 전체 (5개 저장소 모두)
bash .deepx/tools/scripts/run_all.sh generate
```

전체 가이드는 suite 루트의 [`.deepx/tools/README.md`](../../../.deepx/tools/README.md) 및
[`.deepx/tools/scripts/README.md`](../../../.deepx/tools/scripts/README.md)를
참조하세요.

## Templates (en/ + ko/)

지침 템플릿은 `templates/en/` 및 `templates/ko/` 아래에 `.tmpl` 파일로 위치합니다.
Fragment placeholder (`{{FRAGMENT:<name>}}`)는 `dx-agentic-gen`에 의해 suite 레벨
fragment 라이브러리를 기준으로 해석됩니다.

## Context 라우팅 표

| 작업이 언급하는 내용... | 읽어야 할 파일 |
|---|---|
| Python 앱, detection, classification | `prompts/new-python-detection.md`, `toolsets/common-framework-api.md`, `toolsets/model-registry.md` |
| C++ 앱, 고성능 | `prompts/new-cpp-app.md`, `toolsets/dx-engine-api.md`, `toolsets/dx-postprocess-api.md` |
| Async, 성능 | `toolsets/common-framework-api.md`, `memory/performance_patterns.md` |
| 모델, 다운로드, 설정 | `toolsets/model-registry.md`, `memory/model_zoo.md` |
| Postprocess, pybind11 | `toolsets/dx-postprocess-api.md`, `contextual-rules/postprocess.md` |
| 테스트, 검증 | `contextual-rules/tests.md`, `scripts/validate_app.py` |
| **항상 로드** | `memory/common_pitfalls.md` |

## 플랫폼 통합

모든 플랫폼 파일은 `.deepx/` canonical source로부터 `dx-agentic-gen`에 의해 생성됩니다:

| 플랫폼 | 대상 파일 |
|---|---|
| GitHub Copilot | `.github/copilot-instructions.md`, `.github/agents/`, `.github/skills/`, `.github/instructions/` |
| Claude Code | `CLAUDE.md`, `.claude/agents/`, `.claude/skills/` |
| Cursor | `.cursor/rules/*.mdc` |
| OpenCode | `AGENTS.md`, `.opencode/agents/`, `opencode.json` |

## 개발자 워크플로우

### 새 앱 빌드

1. `memory/common_pitfalls.md` 읽기 (항상)
2. `prompts/`에서 prompt 템플릿 선택
3. `toolsets/model-registry.md`의 패턴을 통해 `config/model_registry.json` 쿼리
4. Prompt 템플릿 단계 따르기
5. `scripts/validate_app.py`로 검증

### 지식 추가

1. 업데이트 프로토콜을 위해 `memory/MEMORY.md` 읽기
2. 도메인 태그가 붙은 항목 추가: `[UNIVERSAL]`, `[DX_APP]`, 또는 `[PPU]`
3. 함정에는 증상/원인/해결책 포함
4. 무결성 검증을 위해 `scripts/validate_framework.py` 실행

### 플랫폼으로 동기화

1. `.deepx/`의 canonical 콘텐츠 편집
2. `dx-agentic-gen check` 실행 (drift 리포트)
3. `dx-agentic-gen generate` 실행
4. Diff 검토 및 커밋 (pre-commit hook이 안전망으로 check + lint를 재실행)

## 핵심 정보

- **dx_app v3.0.0**: 15개 AI task, 133개 모델, 4개 Python variant, C++ 예제
- **프레임워크**: SyncRunner (1104 라인), AsyncRunner, 11개 IFactory 인터페이스
- **Postprocess**: 37개 pybind11 C++ 바인딩
- **CLI**: 11개 플래그를 갖는 parse_common_args()
- **NPU**: DX-RT 3.0.x를 통한 DX-M1 / DX-M1A (단종)
- **모델 포맷**: .dxnn (v7+, INT8/UINT8/FP16)
