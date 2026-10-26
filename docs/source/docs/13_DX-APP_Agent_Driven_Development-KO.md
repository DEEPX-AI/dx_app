# DX-APP Agent-Driven 개발 가이드

## 개요

DX-APP은 DEEPX NPU 가속기에서 독립형 추론 애플리케이션을 구축하기 위한
DEEPX agent-driven development (dx-agent-dev)를 지원합니다. boilerplate를 수동으로 작성하는 대신
자연어로 원하는 것을 설명하면, 전문 에이전트 네트워크가 프로덕션 수준의
추론 코드를 생성하고 검증하여 결과를 보고합니다.

이 가이드에서는 에이전트 아키텍처, 사용 가능한 스킬, 검증 프레임워크,
dx_app 독립형 추론 개발을 위한 문제 해결을 다룹니다.

---

## 에이전트 아키텍처

6개의 에이전트가 협력하여 dx_app 추론 애플리케이션을 빌드, 검증, 관리합니다.

| 에이전트 | 설명 | 라우팅 대상 |
|---|---|---|
| `dx-app-builder` | 마스터 라우터 — 요청 유형을 분류하고<br>적절한 전문 에이전트에 dispatch | `dx-python-builder`<br>`dx-cpp-builder`<br>`dx-benchmark-builder`<br>`dx-model-manager` |
| `dx-python-builder` | 4가지 변형으로 Python 추론 앱 빌드:<br>`sync`, `async`, `sync_cpp_postprocess`,<br>`async_cpp_postprocess`<br>(Sub-agent — dx-app-builder가 호출) | — |
| `dx-cpp-builder` | `InferenceEngine` API를 사용한 C++ 추론 앱 빌드 (Sub-agent — dx-app-builder가 호출) | — |
| `dx-benchmark-builder` | 대상 하드웨어에서의 추론 성능 벤치마크 및 profiling (Sub-agent — dx-app-builder가 호출) | `dx-python-builder`<br>(더 빠른 variant로 재빌드) |
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
    ├──► dx-benchmark-builder (성능 profiling)
    └──► dx-model-manager     (모델 작업)
            │
            ▼
      validate_app.py  (생성 후 빌더가 실행)
```

---

## 스킬

스킬은 에이전트가 코드 생성 중 호출하는 재사용 가능한 workflow를 캡슐화합니다.

**범용 SWE 프로세스**  

| 스킬 | 설명 |
|------|------|
| `dx-agent-brainstorm` | 코드 생성 전 brainstorming 및 계획 수립 |
| `dx-agent-tdd` | 테스트 주도 개발 — 파일 생성 직후 즉시 검증 |
| `dx-agent-verify` | 완료 선언 전 검증 — 증거 기반 확인 |
| `dx-swe-writing-plans` | 코드 작성 전 구조화된 구현 계획 작성 |
| `dx-swe-executing-plans` | 리뷰 checkpoint가 포함된 구현 계획 실행 |
| `dx-swe-debugging` | 체계적 디버깅 — 수정 제안 전 진단 수행 |
| `dx-swe-parallel-agents` | 2개 이상의 독립 작업을 병렬 에이전트에 dispatch |
| `dx-swe-subagent-dev` | 독립 서브 에이전트를 활용한 구현 계획 실행 |
| `dx-swe-receiving-review` | 기술적 엄밀성을 갖춘 코드 리뷰 피드백 수신 및 처리 |
| `dx-swe-requesting-review` | 작업이 요구 사항을 충족하는지 검증하기 위한 코드 리뷰 요청 |
| `dx-skill-router` | 작업을 적절한 스킬로 자동 라우팅 |

**DEEPX 빌드**  

| 스킬 | 설명 |
|------|------|
| `dx-agent-app-build-python` | IFactory 패턴을 사용하여 4가지 변형의 Python 추론 앱 빌드 |
| `dx-agent-app-build-cpp` | `InferenceEngine` 런타임 API를 사용한 C++ 추론 앱 빌드 |
| `dx-agent-app-build-async` | 파이프라인화된 전처리/추론/후처리 단계의 비동기 고성능 앱 빌드 |
| `dx-agent-app-model-management` | registry에서 `.dxnn` 모델 다운로드 및 모델 경로 설정 |
| `dx-agent-app-validate` | 생성된 코드에 대해 검증 단계(정적, 설정, 컴포넌트, NPU smoke, 출력 정확도, 교차 검증, 통합) 실행 |

**Harness**  

| 스킬 | 설명 |
|------|------|
| `dx-harness-writing-skills` | 스킬 파일 생성 및 편집 |
| `dx-harness-validate` | 이 저장소의 `.deepx/` 무결성 검증 |

---

## 지원 AI 도구

dx_app agent-driven 개발은 5가지 AI 코딩 도구에서 작동합니다. 각 도구는
자체 설정을 통해 지식 베이스를 자동으로 로드합니다.

| 도구 | 설정 파일 | 사용 가능한 에이전트 |
|---|---|---|
| **Claude Code** | `CLAUDE.md`<br>`.claude/agents/`의 6개 에이전트<br>`.claude/skills/`의 18개 스킬 | `@dx-app-builder`, `@dx-validator`<br>(+ handoff로 호출되는 4개 하위 에이전트) |
| **GitHub Copilot** | `.github/copilot-instructions.md`<br>`.github/agents/`의 6개 에이전트<br>`.github/skills/`의 18개 스킬<br>`.github/instructions/`의 4개 instruction | `@dx-app-builder`, `@dx-validator`<br>(`dx-python-builder`, `dx-cpp-builder`,<br>`dx-benchmark-builder`, `dx-model-manager`는<br>`@dx-app-builder`를 통해 호출되는 하위 에이전트) |
| **Cursor** | `.cursor/rules/dx-app.mdc` (항상)<br>6개 에이전트 규칙 + 18개 스킬 규칙<br>`python-example.mdc`, `cpp-example.mdc`,<br>`tests.mdc` (총 28개) | 자동 적용 규칙과 함께<br>자유 형식 대화 |
| **OpenCode** | `AGENTS.md`, `opencode.json`<br>`.opencode/agents/`의 6개 에이전트<br>`.deepx/skills/`의 18개 스킬 | `@dx-app-builder`<br>또는 `/dx-agent-app-build-python` |
| **Codex CLI** | `AGENTS.md`<br>`.codex/skills/dx-codex-identity/SKILL.md` | 자동 로드된 지침과 함께<br>자유 형식 대화 |

### Copilot 파일별 자동 Instruction

다음 glob 패턴에 매칭되는 파일 편집 시, Copilot이 자동으로
컨텍스트별 instruction을 주입합니다:

| glob 패턴 | 주입되는 Instruction | 내용 |
|---|---|---|
| `src/python_example/**` | `python-example.instructions.md` | IFactory 패턴, `parse_common_args()`, 4변형 naming, 상대 import 금지 |
| `src/cpp_example/**` | `cpp-example.instructions.md` | C++14 표준, RAII 패턴, InferenceEngine API |
| `src/postprocess/**` | `postprocess.instructions.md` | 후처리 규칙, pybind 바인딩 |
| `tests/**` | `tests.instructions.md` | pytest 마커 (`slow`, `smoke`), 검증 참조 문서 |

### OpenCode 스킬 (slash 명령)

**범용 SWE 프로세스**

| slash 명령 | 설명 |
|---|---|
| `/dx-agent-brainstorm` | 코드 생성 전 brainstorming 및 계획 수립 |
| `/dx-agent-tdd` | 점진적 검증을 포함한 테스트 주도 개발 |
| `/dx-agent-verify` | 증거 기반 완료 검증 |
| `/dx-swe-writing-plans` | 구조화된 구현 계획 작성 |
| `/dx-swe-executing-plans` | 리뷰 checkpoint가 포함된 구현 계획 실행 |
| `/dx-swe-debugging` | 체계적 디버깅 — 수정 전 진단 |
| `/dx-swe-parallel-agents` | 독립 작업을 병렬 에이전트에 dispatch |
| `/dx-swe-subagent-dev` | 독립 서브 에이전트를 활용한 계획 실행 |
| `/dx-swe-receiving-review` | 기술적 엄밀성을 갖춘 코드 리뷰 피드백 수신 |
| `/dx-swe-requesting-review` | 작업 검증을 위한 코드 리뷰 요청 |
| `/dx-skill-router` | 적절한 스킬로 자동 라우팅 |

**DEEPX 빌드**

| slash 명령 | 설명 |
|---|---|
| `/dx-agent-app-build-python` | IFactory를 사용한 단계별 Python 앱 생성 |
| `/dx-agent-app-build-cpp` | InferenceEngine을 사용한 C++ 앱 |
| `/dx-agent-app-build-async` | 비동기 고성능 앱 |
| `/dx-agent-app-model-management` | 모델 다운로드 및 registry |
| `/dx-agent-app-validate` | 검증 단계(정적, 설정, 컴포넌트, NPU smoke, 출력 정확도, 교차 검증, 통합) 실행 |

**Harness**

| slash 명령 | 설명 |
|---|---|
| `/dx-harness-writing-skills` | 스킬 파일 생성 및 편집 |
| `/dx-harness-validate` | 이 저장소의 `.deepx/` 무결성 검증 |

### 플랫폼별 파일 참조

각 AI 코딩 에이전트는 dx_app 레벨에서 서로 다른 설정 파일을 자동 로딩합니다.

**자동 로딩 파일**  

| 파일 | 자동 로딩 대상 | 로딩 |
|------|----------------|------|
| `.github/copilot-instructions.md` | Copilot Chat/CLI | 자동 |
| `CLAUDE.md` | Claude Code | 자동 |
| `AGENTS.md` + `opencode.json` | OpenCode | 자동 |
| `AGENTS.md` + `.codex/skills/dx-codex-identity/SKILL.md` | Codex CLI | 자동 |
| `.cursor/rules/dx-app.mdc` | Cursor | 자동 |
| `.cursor/rules/python-example.mdc` | Cursor | 자동 (glob) |
| `.cursor/rules/cpp-example.mdc` | Cursor | 자동 (glob) |
| `.cursor/rules/tests.mdc` | Cursor | 자동 (glob) |

**에이전트 파일 (수동 @mention)**  

| 에이전트 | Copilot (`@mention`) | OpenCode (`@mention`) | Claude Code (`.claude/agents/`) |
|---------|------|---------|---------|
| `dx-app-builder` | `.github/agents/`<br>`dx-app-builder.agent.md` | `.opencode/agents/`<br>`dx-app-builder.md` | `.claude/agents/`<br>`dx-app-builder.md` |
| `dx-python-builder` | `.github/agents/`<br>`dx-python-builder.agent.md` | `.opencode/agents/`<br>`dx-python-builder.md` | `.claude/agents/`<br>`dx-python-builder.md` |
| `dx-cpp-builder` | `.github/agents/`<br>`dx-cpp-builder.agent.md` | `.opencode/agents/`<br>`dx-cpp-builder.md` | `.claude/agents/`<br>`dx-cpp-builder.md` |
| `dx-benchmark-builder` | `.github/agents/`<br>`dx-benchmark-builder.agent.md` | `.opencode/agents/`<br>`dx-benchmark-builder.md` | `.claude/agents/`<br>`dx-benchmark-builder.md` |
| `dx-model-manager` | `.github/agents/`<br>`dx-model-manager.agent.md` | `.opencode/agents/`<br>`dx-model-manager.md` | `.claude/agents/`<br>`dx-model-manager.md` |
| `dx-validator` | `.github/agents/`<br>`dx-validator.agent.md` | `.opencode/agents/`<br>`dx-validator.md` | `.claude/agents/`<br>`dx-validator.md` |

**Copilot 파일별 지침**  

| 파일 | 범위 |
|------|-------|
| `.github/instructions/python-example.instructions.md` | Python 예제 파일 |
| `.github/instructions/cpp-example.instructions.md` | C++ 예제 파일 |
| `.github/instructions/postprocess.instructions.md` | 후처리 파일 |
| `.github/instructions/tests.instructions.md` | 테스트 파일 |

**스킬 파일 (전체 플랫폼 — `/slash-command`)**  

| 스킬 | 파일 |
|-------|------|
| `/dx-agent-brainstorm` | `.deepx/skills/dx-agent-brainstorm/SKILL.md` |
| `/dx-agent-app-build-async` | `.deepx/skills/dx-agent-app-build-async/SKILL.md` |
| `/dx-agent-app-build-cpp` | `.deepx/skills/dx-agent-app-build-cpp/SKILL.md` |
| `/dx-agent-app-build-python` | `.deepx/skills/dx-agent-app-build-python/SKILL.md` |
| `/dx-swe-parallel-agents` | `.deepx/skills/dx-swe-parallel-agents/SKILL.md` |
| `/dx-swe-executing-plans` | `.deepx/skills/dx-swe-executing-plans/SKILL.md` |
| `/dx-agent-app-model-management` | `.deepx/skills/dx-agent-app-model-management/SKILL.md` |
| `/dx-swe-receiving-review` | `.deepx/skills/dx-swe-receiving-review/SKILL.md` |
| `/dx-swe-requesting-review` | `.deepx/skills/dx-swe-requesting-review/SKILL.md` |
| `/dx-skill-router` | `.deepx/skills/dx-skill-router/SKILL.md` |
| `/dx-swe-subagent-dev` | `.deepx/skills/dx-swe-subagent-dev/SKILL.md` |
| `/dx-swe-debugging` | `.deepx/skills/dx-swe-debugging/SKILL.md` |
| `/dx-agent-tdd` | `.deepx/skills/dx-agent-tdd/SKILL.md` |
| `/dx-agent-app-validate` | `.deepx/skills/dx-agent-app-validate/SKILL.md` |
| `/dx-agent-verify` | `.deepx/skills/dx-agent-verify/SKILL.md` |
| `/dx-swe-writing-plans` | `.deepx/skills/dx-swe-writing-plans/SKILL.md` |
| `/dx-harness-writing-skills` | `.deepx/skills/dx-harness-writing-skills/SKILL.md` |
| `/dx-harness-validate` | `.deepx/skills/dx-harness-validate/SKILL.md` |

**공유 지식 베이스 (`.deepx/`)**  

`.deepx/` 디렉토리는 모든 에이전트 지식의 정규 소스(canonical source)입니다.
플랫폼별 파일(`.github/agents|skills|instructions/`, `.github/copilot-instructions.md`, `.claude/`, `.cursor/rules/`, `.opencode/agents/`, `CLAUDE.md`, `AGENTS.md` 및 각 `-KO` 파일)은 dx-all-suite에 포함된 `dx-agent-gen generate`에 의해 `.deepx/`에서 생성됩니다. `.deepx/` 전체가 자동 로딩되지는 않으며(OpenCode의 `opencode.json`은 `.deepx/memory/common_pitfalls.md`와 `.deepx/instructions/coding-standards.md`만 로드), 에이전트와 스킬이 작업 실행 중 필요한 파일을 참조합니다.

| 디렉토리 | 파일 | 설명 |
|-----------|-------|-------------|
| `.deepx/agents/` | 6개 파일 (`dx-app-builder.md`, `dx-python-builder.md`, `dx-cpp-builder.md`, `dx-benchmark-builder.md`, `dx-model-manager.md`, `dx-validator.md`) | 권위 있는 에이전트 정의 |
| `.deepx/skills/` | 18개 디렉토리 | 상세 스킬 workflow (디렉토리당 `SKILL.md` 1개) |
| `.deepx/templates/` | 6개 템플릿 파일 | 플랫폼 파일용 instruction 템플릿 (`dx-agent-gen`이 처리) |
| `.deepx/knowledge/` | 1개 파일 | 도메인별 참조 자료 |
| `.deepx/toolsets/` | 6개 파일 | API 참조 및 schema 문서 |
| `.deepx/instructions/` | 6개 파일 | 코딩 표준 및 workflow 규칙 |
| `.deepx/memory/` | 5개 파일 | 지속적 지식 — 함정 및 세션 메모리 |
| `.deepx/contextual-rules/` | 4개 파일 | 범위별 코딩 규칙 (C++ 예제, Python 예제, 후처리, 테스트) |
| `.deepx/prompts/` | 4개 파일 | 재사용 가능한 프롬프트 템플릿 |
| `.deepx/scripts/` | 3개 파일 | 검증 및 harness 스크립트 |
| `.deepx/docs/` | 2개 파일 | CI sub-repo gate 가이드 (EN/KO) |

---

## 사용자 시나리오

### 시나리오 1: Python 감지 앱 빌드

**프롬프트:**

```
"yolo26n으로 사람 감지하는 Python 앱 만들어줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | 프롬프트를 직접 입력. `CLAUDE.md`가 먼저 `/dx-skill-router`를 실행하고 앱 빌드를 `dx-app-builder` 에이전트(`.claude/agents/`)에 넘김. 이 에이전트가 필수 질문 3개(언어/variant, AI 작업, 모델)를 확인하고 `dx-agent-app-build-python` 스킬을 로드한 뒤 `dx-agent-dev/<session_id>/`에 파일 생성 및 검증 (명시적 요청 시 `src/...`에 직접 생성). |
| **GitHub Copilot** | `@dx-app-builder` 뒤에 프롬프트 입력. `dx-python-builder`로 라우팅, 요청한 variant 생성, `validate_app.py` 실행. |
| **Cursor** | 프롬프트를 직접 입력. `dx-app.mdc`(항상 로드)가 컨텍스트 제공. `src/python_example/` 파일 편집 시 `python-example.mdc` 활성화. |
| **OpenCode** | `@dx-app-builder` 뒤에 프롬프트 입력, 또는 `/dx-agent-app-build-python` 스킬 직접 사용. |

### 시나리오 2: C++ 앱 빌드

**프롬프트:**

```
"InferenceEngine으로 yolo26n C++ 추론 앱 만들어줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | 프롬프트를 직접 입력. `CLAUDE.md`가 빌드를 `dx-app-builder` 에이전트에 넘기고, 이 에이전트가 `dx-agent-app-build-cpp` 스킬을 로드. |
| **GitHub Copilot** | `@dx-app-builder` 뒤에 프롬프트 입력. `dx-cpp-builder`로 라우팅. |
| **Cursor** | 프롬프트를 직접 입력. `src/cpp_example/` 파일 편집 시 `cpp-example.mdc` 활성화, C++14 및 RAII 규칙 주입. |
| **OpenCode** | `@dx-app-builder` 뒤에 프롬프트 입력, 또는 `/dx-agent-app-build-cpp` 스킬 직접 사용. |

### 시나리오 3: 모델 다운로드 및 등록

**프롬프트:**

```
"DX-M1용 yolo26n 모델 다운로드해줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | `@dx-app-builder` 뒤에 프롬프트 입력. `dx-model-manager`로 라우팅. |
| **GitHub Copilot** | `@dx-app-builder` 뒤에 프롬프트 입력. `dx-model-manager`로 라우팅. |
| **Cursor** | 프롬프트를 직접 입력. |
| **OpenCode** | `@dx-app-builder` 뒤에 프롬프트 입력, 또는 `/dx-agent-app-model-management` 스킬 사용. |

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
| **OpenCode** | `@dx-validator` 뒤에 프롬프트 입력, 또는 수동 실행: `python .deepx/scripts/validate_app.py dx-agent-dev/<session_id>/` |

### 시나리오 5: 포즈 추정 앱 빌드

**프롬프트:**

```
"yolo26n-pose로 포즈 추정 앱 만들어줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | 프롬프트를 직접 입력. `CLAUDE.md`가 빌드를 `dx-app-builder` 에이전트에 넘기고, 이 에이전트가 `pose_estimation` 작업 유형으로 `dx-agent-app-build-python` 스킬을 로드. keypoint 시각화 및 skeleton 그리기 로직 생성. |
| **GitHub Copilot** | `@dx-app-builder` 뒤에 프롬프트 입력. 포즈 전용 후처리와 함께 `dx-python-builder`로 라우팅. |
| **Cursor** | 프롬프트를 직접 입력. `dx-app.mdc`(항상 로드)가 컨텍스트 제공. 앱을 `src/python_example/pose_estimation/`에 배치하면 `python-example.mdc`도 적용. |
| **OpenCode** | `@dx-app-builder` 뒤에 프롬프트 입력, 또는 `/dx-agent-app-build-python` 스킬 직접 사용. |

### 시나리오 6: 인스턴스 세그멘테이션 앱 빌드

**프롬프트:**

```
"yolo26n-seg로 인스턴스 세그멘테이션 앱 만들어줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | 프롬프트를 직접 입력. `CLAUDE.md`가 빌드를 `dx-app-builder` 에이전트에 넘기고, 이 에이전트가 `instance_segmentation` 작업 유형으로 `dx-agent-app-build-python` 스킬을 로드. mask overlay 시각화 생성. |
| **GitHub Copilot** | `@dx-app-builder` 뒤에 프롬프트 입력. 세그멘테이션 전용 후처리와 함께 `dx-python-builder`로 라우팅. |
| **Cursor** | 프롬프트를 직접 입력. `dx-app.mdc`(항상 로드)가 컨텍스트 제공. 앱을 `src/python_example/instance_segmentation/`에 배치하면 `python-example.mdc`도 적용. |
| **OpenCode** | `@dx-app-builder` 뒤에 프롬프트 입력, 또는 `/dx-agent-app-build-python` 스킬 직접 사용. |

### 시나리오 7: 분류 앱 빌드

**프롬프트:**

```
"EfficientNet-Lite0으로 이미지 분류 앱 만들어줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | 프롬프트를 직접 입력. `CLAUDE.md`가 빌드를 `dx-app-builder` 에이전트에 넘기고, 이 에이전트가 `image_classification` 작업 유형으로 `dx-agent-app-build-python` 스킬을 로드. Top-K 레이블 예측 로직 생성. |
| **GitHub Copilot** | `@dx-app-builder` 뒤에 프롬프트 입력. 분류 후처리(softmax + Top-K)와 함께 `dx-python-builder`로 라우팅. |
| **Cursor** | 프롬프트를 직접 입력. `dx-app.mdc`(항상 로드)가 컨텍스트 제공. 앱을 `src/python_example/image_classification/`에 배치하면 `python-example.mdc`도 적용. |
| **OpenCode** | `@dx-app-builder` 뒤에 프롬프트 입력, 또는 `/dx-agent-app-build-python` 스킬 직접 사용. |

### 시나리오 8: 비동기 고성능 앱 빌드

**프롬프트:**

```
"yolo26n으로 비동기 고성능 감지 앱 만들어줘"
```

| 도구 | 사용 방법 |
|---|---|
| **Claude Code** | 프롬프트를 직접 입력. `CLAUDE.md`가 빌드를 `dx-app-builder` 에이전트에 넘기고, 이 에이전트가 `dx-agent-app-build-async` 스킬을 로드. 큐 기반 병렬 처리로 파이프라인화된 전처리/추론/후처리 단계 생성. |
| **GitHub Copilot** | `@dx-app-builder` 뒤에 프롬프트 입력. 비동기 variant 중심으로 `dx-python-builder`로 라우팅. |
| **Cursor** | 프롬프트를 직접 입력. `dx-app.mdc`(항상 로드)가 컨텍스트 제공. 앱을 `src/python_example/`에 배치하면 `python-example.mdc`도 적용. |
| **OpenCode** | `@dx-app-builder` 뒤에 프롬프트 입력, 또는 `/dx-agent-app-build-async` 스킬 직접 사용. |

---

## 빠른 시작

자연어로 사람 감지 앱을 요청하세요:

```
@dx-app-builder "yolo26n으로 사람 감지하는 Python 앱 만들어줘"
```

에이전트가 수행하는 작업:

1. **필수 질문 3개** — 언어/variant(Python `sync` / `async` / `cpp_postprocess` 또는 C++), AI 작업(`detection`, `classification`, `segmentation` 등), 모델
2. **빌드 계획 제시** — 생성할 파일 목록, 다운로드할 모델, 작성할 설정
3. **`dx-python-builder`에 라우팅** — 전문 에이전트가 인수인계
4. **파일 생성** — `dx-agent-dev/<session_id>/`에 생성 (명시적 요청 시 `src/`에 직접 생성)
5. **검증 및 보고** — 빌더가 `validate_app.py`를 실행하고, NPU가 있으면 sync 예제를 한 번 실행한 뒤 완료 보고서 출력

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

기본적으로 에이전트가 생성한 코드는 기존 소스 코드와의 충돌을 방지하기 위해
`dx-agent-dev/` 격리 디렉토리에 배치됩니다.

### 기본 출력 (dx-agent-dev/)

```
dx-agent-dev/<session_id>/
├── README.md              # 세션 메타데이터 및 실행 지침
├── session.json           # 기계 판독 가능한 세션 설정
├── setup.sh               # 환경 설정 스크립트 (필수)
├── run.sh                 # 앱 실행 스크립트 (필수)
├── session.log            # 에이전트 세션 로그 (필수)
├── __init__.py
├── config.json
├── factory/
│   ├── __init__.py
│   └── {model}_factory.py
├── {model}_sync.py
├── {model}_async.py                  # 요청 시
├── {model}_sync_cpp_postprocess.py   # 요청 시
└── {model}_async_cpp_postprocess.py  # 요청 시
```

`{model}_sync.py`는 항상 생성되며, 나머지 3가지 variant는 요청한 경우에만 생성됩니다.

세션 ID 형식: `YYYYMMDD-HHMMSS_<agent>_<coding_model>_<target_model>_<task>` — `<agent>`는 `claude`, `codex`, `copilot`, `cursor`, `opencode` 중 하나 (예: `20260403-143022_claude_sonnet46_yolo26n_detection`). timestamp는 시스템 로컬 시간입니다.

### 프로덕션 출력 (src/)

프로덕션 배치를 명시적으로 요청하면 파일이 표준 소스 트리인
`src/python_example/{task}/{family}/{variant}/`에 직접 작성됩니다 (`{variant}`는 `.dxnn` stem).

### 파일 설명

| 파일 | 용도 |
|---|---|
| `config.json` | factory가 읽는 추론 임계값 (예: `score_threshold`, `nms_threshold`, `top_k`). 모델 경로는 `--model` 또는 `model_registry.json`에서 가져옴 |
| `{model}_factory.py` | `IFactory` 구현: `create_preprocessor`, `create_postprocessor`, `create_visualizer`, `get_model_name`, `get_task_type` |
| `{model}_sync.py` | 동기식 단일 스레드 추론 진입점 |
| `{model}_async.py` | 비동기 파이프라인 추론 진입점 |
| `{model}_sync_cpp_postprocess.py` | pybind를 통한 C++ 후처리 동기 추론 |
| `{model}_async_cpp_postprocess.py` | pybind를 통한 C++ 후처리 비동기 추론 |

!!! note "참고"

    `setup.sh`, `run.sh`, `session.log`는 모든 세션 출력 디렉토리에 필수 산출물입니다.

---

## 검증 단계

`dx-agent-app-validate` 스킬이 아래 단계를 정의합니다. 1~3단계는 NPU 없이 실행되며, 4~6단계는 NPU와 모델 파일이 필요합니다.

| 단계 | 검사 대상 | 하드웨어 필요 |
|---|---|---|
| 1 — 정적 | 문법, import, IFactory 메서드, `parse_common_args()`, 하드코딩된 경로 없음 | 아니요 |
| 2 — 설정 | `config.json`이 유효한 JSON이고 작업별 키를 포함, 모델이 `model_registry.json`에 등록됨 | 아니요 |
| 3 — 컴포넌트 | factory가 전처리기, 후처리기, visualizer를 생성, OpenCV에 GUI 백엔드 있음, 앱을 suite 밖으로 복사해도 실행됨 | 아니요 |
| 4 — 스모크 | NPU에서 단일 프레임 추론 | 예 |
| 5 — 출력 정확도 | 검출 수 > 0, 유효한 박스·점수·클래스 ID, 후처리기가 모델 계열과 일치 | 예 |
| 5.5 — 교차 검증 | 사전 컴파일된 참조 모델 또는 검증된 기존 예제와 비교 (둘 다 없으면 생략) | 예 |
| 6 — 통합 | `--save`를 사용한 비디오 전체 파이프라인 | 예 |

---

## 검증 명령

```bash
# 정적 검사 (11개 검사: 필수 파일, 문법, 상대 import, logging, 하드코딩된 경로, IFactory, parse_common_args, 미사용 import, README, app.yaml, 도달 불가 코드)
python .deepx/scripts/validate_app.py dx-agent-dev/<session_id>/

# smoke test 3개 추가 (--help, factory import, NPU 상태) — 세션의 setup.sh 실행 후
python .deepx/scripts/validate_app.py dx-agent-dev/<session_id>/ --smoke-test

# 프레임워크 무결성 — .deepx/ 디렉토리 구조 검증
python .deepx/scripts/validate_framework.py
```

!!! note "참고"

    `validate_app.py`는 세션 출력 디렉토리를 검사합니다. 소스 트리의 variant 디렉토리에는 `__init__.py`가 없어 필수 파일 검사에서 실패합니다. 그 밖의 옵션: `--strict`, `--json`, `--verbose`.

---

## 지식 베이스 구조

에이전트 지식은 dx_app 프로젝트 루트의 `.deepx/` 디렉토리에 있습니다.

| 디렉토리 | 수량 | 내용 |
|---|---|---|
| `agents/` | 6 | 에이전트 정의 및 라우팅 규칙 |
| `skills/` | 18 | 스킬 workflow 디렉토리 (각각 `SKILL.md` 1개) |
| `templates/` | 6 | 플랫폼 파일용 instruction 템플릿 (`dx-agent-gen`이 처리) |
| `knowledge/` | 1 | 도메인별 참조 자료 |
| `toolsets/` | 6 | API 참조 (InferenceEngine, 공통 프레임워크 / IFactory, `.dxnn` 모델 형식, 모델 registry, pybind 후처리, OCR 앱) |
| `memory/` | 5 | 지속적 지식 (index, 주요 함정, model zoo, 성능 패턴, 플랫폼 API) |
| `contextual-rules/` | 4 | C++ 예제, Python 예제, 후처리, 테스트별 범위 규칙 |
| `prompts/` | 4 | 재사용 가능한 프롬프트 템플릿 (새 C++ 앱, Python detection/segmentation, orchestrated build) |
| `scripts/` | 3 | `validate_app.py`, `validate_framework.py`, `harness_bootstrap.sh` |
| `docs/` | 2 | CI sub-repo gate 가이드 (EN/KO) |


!!! note "참고"

    플랫폼 파일(`.github/agents|skills|instructions/`, `.github/copilot-instructions.md`, `.claude/`, `.cursor/rules/`, `.opencode/agents/`, `CLAUDE.md`, `AGENTS.md` 및 각 `-KO` 파일)은 dx-all-suite에 포함된 `dx-agent-gen generate`에 의해 생성됩니다. 
    직접 편집하지 마세요. `.deepx/` 소스 파일을 편집한 후 생성기를 다시 실행하세요. dx_app만 단독으로 checkout한 경우 `.deepx/`를 편집하기 전에 `bash .deepx/scripts/harness_bootstrap.sh --check`를 실행하세요.

에이전트는 작업 시작 시 이 디렉토리들을 읽습니다. 메모리 파일은 개발 중
새로운 패턴이나 수정 사항이 발견될 때 업데이트됩니다.

---

## 세션 sentinel

에이전트는 자동화된 테스트를 위해 각 작업의 시작과 끝에 고정된 마커를 출력합니다:

| 마커 | 출력 시점 |
|---|---|
| `[DX-AGENT-DEV: START]` | 에이전트 응답의 첫 번째 줄 |
| `[DX-AGENT-DEV: DONE (output-dir: <relative_path>)]` | 모든 작업 완료 후 마지막 줄. `<relative_path>`는 프로젝트 루트 기준 세션 출력 디렉토리의 상대 경로. 파일이 생성되지 않은 경우 `(output-dir: ...)` 부분을 생략. |

handoff를 통해 호출된 하위 에이전트는 sentinel을 출력하지 않으며, 최상위 에이전트만 출력합니다.

규칙:  

1. **필수** — 첫 번째 응답의 절대적 첫 줄에 `[DX-AGENT-DEV: START]`를 출력합니다. 다른 텍스트, tool call, reasoning보다 반드시 먼저 출력해야 합니다. 사용자가 "알아서 진행해"라고 해도 생략 불가 — 자동 테스트가 실패합니다.  
2. 모든 작업, 검증, 파일 생성이 완료된 후 마지막 줄에 `[DX-AGENT-DEV: DONE (output-dir: <path>)]`를 출력합니다.  
3. handoff/라우팅을 통해 호출된 하위 에이전트인 경우 sentinel을 출력하지 않습니다 — 최상위 에이전트만 출력합니다.  
4. 사용자가 세션에서 여러 프롬프트를 전송하면 각 프롬프트에 대해 START/DONE을 출력합니다.  
5. DONE의 `output-dir`는 프로젝트 루트에서 세션 출력 디렉토리까지의 상대 경로여야 합니다.  
6. **기획 산출물(spec, plan, 설계 문서)만 작성한 상태에서는 절대 DONE을 출력하지 마세요.** DONE은 모든 산출물(구현 코드, 스크립트, 설정 파일, 검증 결과)이 생성된 후에만 출력합니다.  
7. START 직후와 DONE 직전에 DEEPX 배너를 출력합니다. 배너는 sentinel 줄의 위치를 바꾸지 않습니다.  
8. DONE 전에 세션 디렉토리에 필수 파일이 모두 있는지 확인합니다.  

---

## 문제 해결

- **문제:** 에이전트가 상대 import 작성 (`from .factory import ...`)  
    **원인:** LLM 기본 동작  
    **해결 방법:** 절대 import만 사용해야 합니다.  
    예: `from common.processors import YOLOv8Postprocessor`

- **문제:** Factory 클래스 메서드 누락  
    **원인:** 불완전한 `IFactory` 구현  
    **해결 방법:** 5개 필수 메서드를 모두 구현하세요.  
    `create_preprocessor`, `create_postprocessor`, `create_visualizer`, `get_model_name`, `get_task_type`

- **문제:** 런타임에 모델 미발견  
    **원인:** `.dxnn` 파일 경로 미등록  
    **해결 방법:** `dx-model-manager`를 통해 `model_registry.json`에서 모델을 다운로드하고 등록하세요.

- **문제:** NPU 사용 불가 / 디바이스 에러  
    **원인:** 드라이버가 가속기를 감지하지 못함  
    **해결 방법:** `dxcli -s`로 디바이스 상태를 확인하고 DEEPX 커널 모듈 로드 여부를 검증하세요 (`dxrt-cli -s`도 같은 도구입니다).

- **문제:** `validate_app.py` 즉시 실패  
    **원인:** 대상 디렉토리에 `__init__.py`, `config.json` 또는 `factory/`가 없거나, (`--smoke-test` 사용 시) `common`을 import할 수 없음  
    **해결 방법:** dx_app 루트에서 세션 디렉토리(`dx-agent-dev/<session_id>/`)를 대상으로 실행하세요. 세션의 `setup.sh`가 `./common`을 복사한 후에 실행해야 합니다.

---

## 추가 참고 자료

- [DX-APP 아키텍처 및 사용 레퍼런스](09_DX-APP_Project_Overview.md)
- [DX-APP Python 예제 사용 가이드](05_DX-APP_Python_Example_Usage_Guide.md)
- [DX-APP C++ 예제 사용 가이드](03_DX-APP_CPP_Example_Usage_Guide.md)
- [DX-APP 예제 소스 구조](11_DX-APP_Example_Source_Structure.md)
