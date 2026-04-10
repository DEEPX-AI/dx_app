# DX-APP 에이전틱 개발 가이드

## 개요

DX-APP은 DEEPX NPU 가속기에서 독립형 추론 애플리케이션을 구축하기 위한
AI 기반 에이전틱 개발을 지원합니다. 보일러플레이트를 수동으로 작성하는 대신
자연어로 원하는 것을 설명하면, 전문 에이전트 네트워크가 프로덕션 수준의
추론 코드를 생성하고 검증하여 결과를 보고합니다.

이 가이드에서는 에이전트 아키텍처, 사용 가능한 스킬, 검증 프레임워크,
dx_app 독립형 추론 개발을 위한 문제 해결을 다룹니다.

---

## 에이전트 아키텍처

6개의 에이전트가 협력하여 dx_app 추론 애플리케이션을 빌드, 검증, 관리합니다.

| 에이전트 | 설명 | 라우팅 대상 |
|---|---|---|
| `dx-app-builder` | 마스터 라우터 — 요청 유형을 분류하여 적절한 전문 에이전트에 디스패치 | `dx-python-builder`, `dx-cpp-builder`, `dx-benchmark-builder`, `dx-model-manager` |
| `dx-python-builder` | 4가지 변형으로 Python 추론 앱 빌드: `sync`, `async`, `sync_cpp_postprocess`, `async_cpp_postprocess` (Sub-agent — dx-app-builder가 호출) | — |
| `dx-cpp-builder` | `InferenceEngine` API를 사용한 C++ 추론 앱 빌드 (Sub-agent — dx-app-builder가 호출) | — |
| `dx-benchmark-builder` | 대상 하드웨어에서의 추론 성능 벤치마크 및 프로파일링 (Sub-agent — dx-app-builder가 호출) | — |
| `dx-model-manager` | `.dxnn` 컴파일 모델 다운로드, 등록, 관리 (Sub-agent — dx-app-builder가 호출) | — |
| `dx-validator` | 생성된 앱 코드와 `.deepx/` 프레임워크 무결성 검증 | — |

### 라우팅 흐름

```
사용자 요청
    │
    ▼
dx-app-builder  (의도 분류)
    │
    ├──► dx-python-builder    (Python 추론 앱)
    ├──► dx-cpp-builder       (C++ 추론 앱)
    ├──► dx-benchmark-builder (성능 프로파일링)
    └──► dx-model-manager     (모델 작업)
            │
            ▼
      dx-validator  (생성 후 자동 호출)
```

---

## 스킬

스킬은 에이전트가 코드 생성 중 호출하는 재사용 가능한 워크플로우를 캡슐화합니다.

| 스킬 | 설명 |
|---|---|
| `dx-build-python-app` | IFactory 패턴을 사용하여 4가지 변형의 Python 추론 앱 빌드 |
| `dx-build-cpp-app` | `InferenceEngine` 런타임 API를 사용한 C++ 추론 앱 빌드 |
| `dx-build-async-app` | 파이프라인화된 전처리/추론/후처리 단계의 비동기 고성능 앱 빌드 |
| `dx-model-management` | 레지스트리에서 `.dxnn` 모델 다운로드 및 모델 경로 설정 |
| `dx-validate` | 생성된 코드에 대해 5단계 검증 피라미드 실행 |
| `dx-brainstorm-and-plan` | 코드 생성 전 브레인스토밍 및 계획 수립 (프로세스 스킬) |
| `dx-tdd` | 테스트 주도 개발 — 파일 생성 직후 즉시 검증 (프로세스 스킬) |
| `dx-verify-completion` | 완료 선언 전 검증 — 증거 기반 확인 (프로세스 스킬) |

---

## 지원 AI 도구

dx_app 에이전틱 개발은 4가지 AI 코딩 도구에서 작동합니다. 각 도구는
자체 설정을 통해 지식 베이스를 자동으로 로드합니다.

| 도구 | 설정 파일 | 사용 가능한 에이전트 |
|---|---|---|
| **Claude Code** | `CLAUDE.md` | 컨텍스트 라우팅을 통해 6개 에이전트 전체 |
| **GitHub Copilot** | `.github/copilot-instructions.md`, `.github/agents/`의 6개 에이전트, `.github/instructions/`의 4개 instruction | `@dx-app-builder`, `@dx-python-builder`, `@dx-cpp-builder`, `@dx-benchmark-builder`, `@dx-model-manager`, `@dx-validator` |
| **Cursor** | `.cursor/rules/dx-app.mdc` (항상), 3개 glob 규칙 (`python-example`, `cpp-example`, `tests`) | 자동 적용 규칙과 함께 자유 형식 대화 |
| **OpenCode** | `AGENTS.md`, `opencode.json`, `.opencode/agents/`의 6개 에이전트, `.opencode/skills/`의 5개 스킬 | `@dx-app-builder` 또는 `/dx-build-python-app` |

### Copilot 파일별 자동 Instruction

다음 글로브 패턴에 매칭되는 파일 편집 시, Copilot이 자동으로
컨텍스트별 instruction을 주입합니다:

| 글로브 패턴 | 주입되는 Instruction | 내용 |
|---|---|---|
| `src/python_example/**` | `python-example.instructions.md` | IFactory 패턴, SyncRunner/AsyncRunner 사용법, 4변형 네이밍 |
| `src/cpp_example/**` | `cpp-example.instructions.md` | C++14 표준, RAII 패턴, InferenceEngine API |
| `src/postprocess/**` | `postprocess.instructions.md` | 후처리 규칙, pybind 바인딩 |
| `tests/**` | `tests.instructions.md` | pytest 패턴, fixture, NPU 마커 |

### OpenCode 스킬 (슬래시 명령)

| 슬래시 명령 | 설명 |
|---|---|
| `/dx-build-python-app` | IFactory를 사용한 단계별 Python 앱 생성 |
| `/dx-build-cpp-app` | InferenceEngine을 사용한 C++ 앱 |
| `/dx-build-async-app` | 비동기 고성능 앱 |
| `/dx-model-management` | 모델 다운로드 및 레지스트리 |
| `/dx-validate` | 5단계 검증 피라미드 실행 |
| `/dx-brainstorm-and-plan` | 코드 생성 전 브레인스토밍 및 계획 수립 |
| `/dx-tdd` | 점진적 검증을 포함한 테스트 주도 개발 |
| `/dx-verify-completion` | 증거 기반 완료 검증 |

---

## 사용자 시나리오

### 시나리오 1: Python 감지 앱 빌드

**프롬프트:**

```
"yolo26n으로 사람 감지하는 Python 앱 만들어줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | 프롬프트를 직접 입력. `CLAUDE.md`가 `dx-build-python-app` 스킬로 라우팅. variant, 작업 유형, 모델에 대해 2-3개 질문 후 `dx-agentic-dev/<session_id>/`에 파일 생성 및 검증 (명시적 요청 시 `src/...`에 직접 생성). |
| **GitHub Copilot** | `@dx-app-builder` 뒤에 프롬프트 입력. `dx-python-builder`로 라우팅, 4가지 variant 생성, `dx-validator` 실행. |
| **Cursor** | 프롬프트를 직접 입력. `dx-app.mdc`(항상 로드)가 컨텍스트 제공. `src/python_example/` 파일 편집 시 `python-example.mdc` 활성화. |
| **OpenCode** | `@dx-app-builder` 뒤에 프롬프트 입력, 또는 `/dx-build-python-app` 스킬 직접 사용. |

### 시나리오 2: C++ 앱 빌드

**프롬프트:**

```
"InferenceEngine으로 yolo26n C++ 추론 앱 만들어줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | 프롬프트를 직접 입력. `dx-build-cpp-app` 스킬로 라우팅. |
| **GitHub Copilot** | `@dx-cpp-builder` 뒤에 프롬프트 입력. |
| **Cursor** | 프롬프트를 직접 입력. `src/cpp_example/` 파일 편집 시 `cpp-example.mdc` 활성화, C++14 및 RAII 규칙 주입. |
| **OpenCode** | `@dx-app-builder` 뒤에 프롬프트 입력, 또는 `/dx-build-cpp-app` 스킬 직접 사용. |

### 시나리오 3: 모델 다운로드 및 등록

**프롬프트:**

```
"DX-M1용 yolo26n 모델 다운로드해줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | `@dx-model-manager` 뒤에 프롬프트 입력. |
| **GitHub Copilot** | `@dx-model-manager` 뒤에 프롬프트 입력. |
| **Cursor** | 프롬프트를 직접 입력. |
| **OpenCode** | `@dx-model-manager` 뒤에 프롬프트 입력, 또는 `/dx-model-management` 스킬 사용. |

### 시나리오 4: 생성된 코드 검증

**프롬프트:**

```
"방금 만든 감지 앱 검증해줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | `@dx-validator` 뒤에 프롬프트 입력. |
| **GitHub Copilot** | `@dx-validator` 뒤에 프롬프트 입력. |
| **Cursor** | 프롬프트를 직접 입력. |
| **OpenCode** | `@dx-validator` 뒤에 프롬프트 입력, 또는 수동 실행: `python .deepx/scripts/validate_app.py src/python_example/object_detection/yolo26n/` |

### 시나리오 5: 포즈 추정 앱 빌드

**프롬프트:**

```
"yolo26n-pose로 포즈 추정 앱 만들어줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | 프롬프트를 직접 입력. `dx-build-python-app` 스킬이 `pose_estimation` 작업 유형으로 라우팅. 키포인트 시각화 및 스켈레톤 그리기 로직 생성. |
| **GitHub Copilot** | `@dx-app-builder` 뒤에 프롬프트 입력. 포즈 전용 후처리와 함께 `dx-python-builder`로 라우팅. |
| **Cursor** | 프롬프트를 직접 입력. `src/python_example/pose_estimation/` 파일 생성 시 `python-example.mdc` 활성화. |
| **OpenCode** | `@dx-app-builder` 뒤에 프롬프트 입력, 또는 `/dx-build-python-app` 스킬 직접 사용. |

### 시나리오 6: 인스턴스 세그멘테이션 앱 빌드

**프롬프트:**

```
"yolo26n-seg로 인스턴스 세그멘테이션 앱 만들어줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | 프롬프트를 직접 입력. `dx-build-python-app` 스킬이 `instance_segmentation` 작업 유형으로 라우팅. 마스크 오버레이 시각화 생성. |
| **GitHub Copilot** | `@dx-app-builder` 뒤에 프롬프트 입력. 세그멘테이션 전용 후처리와 함께 `dx-python-builder`로 라우팅. |
| **Cursor** | 프롬프트를 직접 입력. `src/python_example/instance_segmentation/` 파일 생성 시 `python-example.mdc` 활성화. |
| **OpenCode** | `@dx-app-builder` 뒤에 프롬프트 입력, 또는 `/dx-build-python-app` 스킬 직접 사용. |

### 시나리오 7: 분류 앱 빌드

**프롬프트:**

```
"EfficientNet-B0으로 이미지 분류 앱 만들어줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | 프롬프트를 직접 입력. `dx-build-python-app` 스킬이 `classification` 작업 유형으로 라우팅. Top-K 레이블 예측 로직 생성. |
| **GitHub Copilot** | `@dx-app-builder` 뒤에 프롬프트 입력. 분류 후처리(softmax + Top-K)와 함께 `dx-python-builder`로 라우팅. |
| **Cursor** | 프롬프트를 직접 입력. `src/python_example/classification/` 파일 생성 시 `python-example.mdc` 활성화. |
| **OpenCode** | `@dx-app-builder` 뒤에 프롬프트 입력, 또는 `/dx-build-python-app` 스킬 직접 사용. |

### 시나리오 8: 비동기 고성능 앱 빌드

**프롬프트:**

```
"yolo26n으로 비동기 고성능 감지 앱 만들어줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | 프롬프트를 직접 입력. `dx-build-async-app` 스킬로 라우팅. 큐 기반 병렬 처리로 파이프라인화된 전처리/추론/후처리 단계 생성. |
| **GitHub Copilot** | `@dx-app-builder` 뒤에 프롬프트 입력. 비동기 variant 중심으로 `dx-python-builder`로 라우팅. |
| **Cursor** | 프롬프트를 직접 입력. 비동기 파일 생성 시 `python-example.mdc` 활성화. |
| **OpenCode** | `@dx-app-builder` 뒤에 프롬프트 입력, 또는 `/dx-build-async-app` 스킬 직접 사용. |

---

## 빠른 시작

자연어로 사람 감지 앱을 요청하세요:

```
@dx-app-builder "yolo26n으로 사람 감지하는 Python 앱 만들어줘"
```

에이전트가 수행하는 작업:

1. **명확화 질문** — 변형(`sync` / `async`), 모델 정밀도, 작업 유형(`detection`, `classification`, `segmentation` 등)
2. **빌드 계획 제시** — 생성할 파일 목록, 다운로드할 모델, 작성할 설정
3. **`dx-python-builder`에 라우팅** — 전문 에이전트가 인수인계
4. **파일 생성** — `dx-agentic-dev/<session_id>/`에 생성 (명시적 요청 시 `src/`에 직접 생성)
5. **검증 및 보고** — `dx-validator`가 검사를 실행하고 요약 출력

### 필수 질문 (HARD-GATE)

`@dx-app-builder` 사용 시, 에이전트는 코드 생성 전에 3가지 필수 질문을
반드시 거칩니다:

1. **언어/변형**: Python (sync / async / cpp_postprocess / async_cpp_postprocess) 또는 C++?
2. **AI 작업**: detection, classification, segmentation, pose 등
3. **모델**: 특정 모델명 (예: `yolo26n`) 또는 자동 추천

이 질문은 **생략할 수 없습니다** — 프롬프트에 충분한 정보가 포함되어 있더라도
에이전트가 각 결정을 명시적으로 확인한 후 진행합니다.

---

## 생성 결과물

기본적으로 AI가 생성한 코드는 기존 소스 코드와의 충돌을 방지하기 위해
`dx-agentic-dev/` 격리 디렉토리에 배치됩니다.

### 기본 출력 (dx-agentic-dev/)

```
dx-agentic-dev/<session_id>/
├── README.md              # 세션 메타데이터 및 실행 지침
├── session.json           # 기계 판독 가능한 세션 설정
├── setup.sh               # 환경 설정 스크립트 (필수)
├── run.sh                 # 앱 실행 스크립트 (필수)
├── session.log            # 에이전트 세션 로그 (필수)
└── src/python_example/{task}/{model}/
    ├── __init__.py
    ├── config.json
    ├── {model}_factory.py
    ├── {model}_sync.py
    ├── {model}_async.py
    ├── {model}_sync_cpp_postprocess.py
    └── {model}_async_cpp_postprocess.py
```

세션 ID 형식: `YYYYMMDD-HHMMSS_model_task` (예: `20260403-143022_yolo26n_detection`).

### 프로덕션 출력 (src/)

프로덕션 배치를 명시적으로 요청하면 파일이 표준 소스 트리인
`src/python_example/{task}/{model}/`에 직접 작성됩니다.

### 파일 설명

| 파일 | 용도 |
|---|---|
| `config.json` | 모델 경로, 작업 유형, 입력 차원, 레이블 맵 |
| `{model}_factory.py` | `IFactory` 구현 — 전/후처리를 위한 5개 메서드 인터페이스 |
| `{model}_sync.py` | 동기식 단일 스레드 추론 진입점 |
| `{model}_async.py` | 비동기 파이프라인 추론 진입점 |
| `{model}_sync_cpp_postprocess.py` | pybind를 통한 C++ 후처리 동기 추론 |
| `{model}_async_cpp_postprocess.py` | pybind를 통한 C++ 후처리 비동기 추론 |

> **참고:** `setup.sh`, `run.sh`, `session.log`는 모든 세션 출력 디렉토리에 필수 산출물입니다.

---

## 5단계 검증 피라미드

`dx-validator`는 비용 오름차순으로 검사를 적용합니다. 각 단계가 다음 단계를 게이트합니다.

```
        ▲
       /5\       성능 벤치마크 (FPS 목표)
      /───\
     / 4   \     NPU 통합 테스트 (하드웨어 필요)
    /───────\
   /   3     \   스모크 테스트 (--help, 모듈 임포트)
  /───────────\
 /     2       \ 설정 검증 (모델 경로, 작업 유형)
/───────────────\
       1         정적 검사 (임포트, 네이밍, 구조)
```

| 단계 | 검사 대상 | 하드웨어 필요 |
|---|---|---|
| 1 — 정적 | 절대 임포트, 네이밍 규칙, 파일 구조, IFactory 메서드 | 아니요 |
| 2 — 설정 | `config.json` 스키마, `.dxnn` 모델 경로 해석, 유효한 작업 유형 | 아니요 |
| 3 — 스모크 | `--help` 플래그 에러 없이 실행, 모듈이 정상 임포트 | 아니요 |
| 4 — NPU 통합 | NPU가 있는 상태에서 샘플 이미지로 엔드-투-엔드 추론 | 예 |
| 5 — 성능 | FPS가 모델 및 가속기의 목표 임계값 충족 | 예 |

---

## 검증 명령

```bash
# 정적 검사 (1, 2단계에 걸친 11개 검사)
python .deepx/scripts/validate_app.py src/python_example/{task}/{model}/

# 스모크 테스트 포함 (1~3단계)
python .deepx/scripts/validate_app.py src/python_example/{task}/{model}/ --smoke-test

# 프레임워크 무결성 — .deepx/ 디렉토리 구조 검증
python .deepx/scripts/validate_framework.py
```

---

## 지식 베이스 구조

에이전트 지식은 dx_app 프로젝트 루트의 `.deepx/` 디렉토리에 있습니다.

| 디렉토리 | 수량 | 내용 |
|---|---|---|
| `agents/` | 6 | 에이전트 정의 및 라우팅 규칙 |
| `skills/` | — | dx-runtime 공유 스킬에서 제공 |
| `toolsets/` | 5 | API 참조 (InferenceEngine, IFactory, dxrt-cli, 모델 레지스트리, pybind 헬퍼) |
| `memory/` | 5 | 지속적 지식 (주요 함정, 플랫폼 API 메모, 최적화 패턴, 카메라/디스플레이 메모, 모델 설정 캐시) |
| `contextual-rules/` | 4 | 코딩 표준, 임포트 규칙, 네이밍 규칙, 디렉토리 레이아웃 규칙 |
| `prompts/` | 4 | 각 전문 에이전트의 시스템 프롬프트 |
| `scripts/` | 3 | `validate_app.py`, `validate_framework.py`, `generate_platforms.py` |

에이전트는 작업 시작 시 이 디렉토리들을 읽습니다. 메모리 파일은 개발 중
새로운 패턴이나 수정 사항이 발견될 때 업데이트됩니다.

---

## 세션 센티넬

에이전트는 자동화된 테스트를 위해 각 작업의 시작과 끝에 고정된 마커를 출력합니다:

| 마커 | 출력 시점 |
|---|---|
| `[DX-AGENTIC-DEV: START]` | 에이전트 응답의 첫 번째 줄 |
| `[DX-AGENTIC-DEV: DONE (output-dir: <relative_path>)]` | 모든 작업 완료 후 마지막 줄. `<relative_path>`는 프로젝트 루트 기준 세션 출력 디렉토리의 상대 경로. 파일이 생성되지 않은 경우 `(output-dir: ...)` 부분을 생략. |

핸드오프를 통해 호출된 하위 에이전트는 센티넬을 출력하지 않으며, 최상위 에이전트만 출력합니다.

규칙:
1. **필수** — 첫 번째 응답의 절대적 첫 줄에 `[DX-AGENTIC-DEV: START]`를 출력합니다. 다른 텍스트, tool call, reasoning보다 반드시 먼저 출력해야 합니다. 사용자가 "알아서 진행해"라고 해도 생략 불가 — 자동 테스트가 실패합니다.
2. 모든 작업, 검증, 파일 생성이 완료된 후 마지막 줄에 `[DX-AGENTIC-DEV: DONE (output-dir: <path>)]`를 출력합니다.
3. 핸드오프/라우팅을 통해 호출된 하위 에이전트인 경우 센티넬을 출력하지 않습니다 — 최상위 에이전트만 출력합니다.
4. 사용자가 세션에서 여러 프롬프트를 전송하면 각 프롬프트에 대해 START/DONE을 출력합니다.
5. DONE의 `output-dir`는 프로젝트 루트에서 세션 출력 디렉토리까지의 상대 경로여야 합니다.
6. **기획 산출물(spec, plan, 설계 문서)만 작성한 상태에서는 절대 DONE을 출력하지 마세요.** DONE은 모든 산출물(구현 코드, 스크립트, 설정 파일, 검증 결과)이 생성된 후에만 출력합니다.

---

## 문제 해결

| 문제 | 원인 | 해결 방법 |
|---|---|---|
| 에이전트가 상대 임포트 작성 (`from .factory import ...`) | LLM 기본 동작 | 모든 임포트는 절대적이어야 함: `from dx_app.python_example.detection.yolo26n.yolo26n_factory import ...` |
| Factory 클래스 메서드 누락 | 불완전한 `IFactory` 구현 | 5개 필수 메서드 모두 구현: `create_preprocessor`, `create_postprocessor`, `create_label_map`, `create_input_config`, `create_visualizer` |
| 런타임에 모델 미발견 | `.dxnn` 파일 경로 미등록 | `dx-model-manager`를 통해 `model_registry.json`에서 모델 다운로드 및 등록 |
| NPU 사용 불가 / 디바이스 에러 | 드라이버가 가속기를 감지하지 못함 | `dxrt-cli -s`로 디바이스 상태 확인; DEEPX 커널 모듈 로드 여부 검증 |
| `validate_app.py` 즉시 실패 | Python 경로 또는 venv 미설정 | dx_app 가상 환경 활성화 후 `PYTHONPATH`에 프로젝트 루트 포함 확인 |

---

## 추가 참고 자료

- [DX-APP 프로젝트 개요](09_DX-APP_Project_Overview.md)
- [DX-APP Python 예제 사용 가이드](05_DX-APP_Python_Example_Usage_Guide.md)
- [DX-APP C++ 예제 사용 가이드](03_DX-APP_CPP_Example_Usage_Guide.md)
- [DX-APP 예제 소스 구조](11_DX-APP_Example_Source_Structure.md)
