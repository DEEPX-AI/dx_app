# dx_app — Claude Code 진입점

> dx_app 독립형 추론 개발을 위한 자체 완결형 진입점.

## 개요

dx_app은 15개 AI 작업에 걸쳐 133개의 컴파일된 `.dxnn` 모델을 제공하며, Python (4개 변형) 및 C++ 예제를 포함합니다.

## 응답 언어

응답 언어는 사용자의 프롬프트 언어에 맞추세요 — 질문하거나 응답할 때
사용자가 사용하는 언어와 동일한 언어를 사용하세요.

**기술 용어 규칙**: 한국어로 응답할 때 영어 기술 용어는 원래 영어 형태를 유지하세요.
영어 용어를 한글 음차로 표기하지 마세요 (한글 음차 표기 금지). 정착된 한국어 외래어
(모델, 서버, 파일, 데이터)는 허용됩니다.


## 권장 모델 (사전 비행 점검)

DX Agentic Development는 강력한 instruction following과 knowledge base 활용 능력을
요구합니다. 작업을 시작하기 전에 권장 모델 중 하나로 실행 중인지 확인하세요:

- **Claude Sonnet 4.6** 이상
- **Claude Opus 4.6** 이상

이 모델이 아닌 경우 (예: GPT-4.1, GPT-4o, Gemini 등), 첫 번째 응답의 시작 부분에
다음 알림을 출력하세요:

```
══════════════════════════════════════════════════════════════
⚠  DX-AGENTIC-DEV: MODEL NOTICE

DX Agentic Development recommends Claude Sonnet 4.6+ or
Opus 4.6+. Your current model may produce lower quality
in instruction following, knowledge base utilization, and
API accuracy. Please switch to a recommended model.
══════════════════════════════════════════════════════════════
```

알림 상자 출력 후 정상적으로 작업을 진행하세요. 이것은 정보성 경고이며,
차단 게이트가 아닙니다.

### 모델 알림 반복 (비권장 모델 전용)

위의 MODEL NOTICE 상자를 출력한 경우 (즉, 권장 모델이 아닌 경우), 사용자가
답변해야 하는 **실제 질문 줄 바로 앞에** 이 짧은 리마인더를 반드시 출력해야
합니다 — brainstorming 흐름의 시작 부분이 아닙니다.

**타이밍**: 모든 파일 읽기, 컨텍스트 분석, 서문 텍스트 이후, `?` (실제 질문)가
포함된 줄 바로 앞에 이 리마인더를 삽입하세요:

```
---
⚠ **Non-recommended model** — output quality may be degraded. Recommended: Claude Sonnet 4.6+ / Opus 4.6+
---
```

**예시 — 잘못됨** (반복이 상자와 함께 스크롤되어 지나감):
```
[DX-AGENTIC-DEV: START]
══ MODEL NOTICE ══
---  ⚠ Non-recommended model ---     ← 너무 이름, 스크롤되어 지나감
... (파일 읽기, 컨텍스트 분석) ...
첫 번째 질문: ...?
```

**예시 — 올바름** (반복이 질문 바로 앞에 나타남):
```
[DX-AGENTIC-DEV: START]
══ MODEL NOTICE ══
... (파일 읽기, 컨텍스트 분석) ...
---  ⚠ Non-recommended model ---     ← 질문 바로 앞
첫 번째 질문: ...?
```

이 리마인더는 한 번만 출력하세요 (첫 번째 질문 앞에), 매 질문마다 출력하지 마세요.


## 공유 지식

모든 스킬, 지시사항, 도구셋, 메모리는 `.deepx/`에 있습니다.
전체 색인은 `.deepx/README.md`를 참조하세요.

## 빠른 참조

```bash
./install.sh && ./build.sh          # Build C++ and pybind11 bindings
./setup.sh                          # Download models and test media
dxrt-cli -s                         # Verify NPU availability
pytest tests/                       # Run unit tests
```

## 스킬

| 명령 | 설명 |
|---------|-------------|
| /dx-build-python-app | Python 추론 앱 빌드 (sync, async, cpp_postprocess, async_cpp_postprocess) |
| /dx-build-cpp-app | InferenceEngine을 사용한 C++ 추론 앱 빌드 |
| /dx-build-async-app | 고성능 비동기 추론 앱 빌드 |
| /dx-model-management | .dxnn 모델 다운로드, 등록, 설정 |
| /dx-validate | 모든 단계 게이트에서 검증 체크 실행 |
| /dx-validate-and-fix | 전체 피드백 루프: 검증, 수집, 승인, 적용, 확인 |
| /dx-brainstorm-and-plan | 브레인스토밍, 2-3가지 접근법 제안, 스펙 자체 검토 후 계획 |
| /dx-tdd | 검증 주도 개발, 선택적 Red-Green-Refactor 단위 테스트 |
| /dx-verify-completion | 프로세스: 완료 선언 전 검증 — 주장보다 증거 우선 |
| /dx-writing-plans | 세분화된 태스크로 구현 계획 작성 |
| /dx-executing-plans | 리뷰 체크포인트와 함께 계획 실행 |
| /dx-subagent-driven-development | 태스크별 신규 서브에이전트로 계획 실행, 2단계 리뷰 |
| /dx-systematic-debugging | 체계적 디버깅 — 수정 제안 전 4단계 근본 원인 조사 |
| /dx-receiving-code-review | 코드 리뷰 피드백을 기술적 엄밀성으로 평가 |
| /dx-requesting-code-review | 기능 완료 후 코드 리뷰 요청 |
| /dx-skill-router | 스킬 탐색 및 호출 — 모든 작업 전 스킬 확인 |
| /dx-writing-skills | 스킬 파일 생성 및 편집 |
| /dx-dispatching-parallel-agents | 독립 태스크를 위한 병렬 서브에이전트 디스패치 |

## 대화형 워크플로우 (반드시 준수)

**빌드하기 전에 항상 사용자와 주요 결정 사항을 검토하세요.** 이것은 HARD GATE입니다.

### 코드 생성 전:
1. **브레인스토밍**: 2-3개의 명확화 질문 — 변형, 작업 유형, 모델. 빌드 계획을 제시하고 승인을 받으세요.
2. **TDD로 빌드**: 생성 직후 각 파일을 검증하세요.
3. **검증**: 주장보다 증거 우선 — 성공 선언 전에 검증 스크립트를 실행하세요.

### 앱 빌드 필수 질문 (HARD-GATE)

<HARD-GATE>
코드 생성 전, 에이전트는 반드시 3가지 모두에 대해 명시적 답변을 질문하고 받아야 합니다:
1. **언어/변형**: Python (sync/async/cpp_postprocess/async_cpp_postprocess) 또는 C++?
2. **AI 작업**: detection, classification, segmentation, pose, face_detection, depth_estimation 등?
3. **모델**: 특정 모델 이름 (예: 'yolo26n') 또는 자동 선택을 위한 'recommend'

사용자의 프롬프트가 충분한 컨텍스트를 제공하는 것처럼 보여도 이 질문은 건너뛸 수 없습니다.
"yolo26n detection 앱 빌드"도 확인이 필요합니다: Python 또는 C++? 어떤 변형?
</HARD-GATE>

### 에이전트 라우팅 (필수)

**모든 앱 빌드 요청은 반드시 `@dx-app-builder`**(마스터 라우터)를 통해야 합니다.
`@dx-python-builder`, `@dx-cpp-builder` 또는 다른 전문 에이전트를 직접 호출하지 마세요.
`@dx-app-builder`는 전문 에이전트가 건너뛰는 필수 brainstorming 질문(Q1: 언어/변형,
Q2: AI 작업, Q3: 모델)을 강제합니다.

### 출력 격리 (HARD GATE)
모든 AI 생성 코드는 기본적으로 `dx-agentic-dev/<session_id>/`에 저장됩니다.
기존 소스 디렉토리 (예: `src/`, `semseg_260323/` 또는 사용자의 기존 코드가 있는
디렉토리)에 생성된 코드를 직접 작성하지 마세요.
사용자가 명시적으로 요청할 때만 `src/`에 작성하세요.

**세션 ID 형식**: `YYYYMMDD-HHMMSS_<model>_<task>` — 타임스탬프는 반드시
**시스템 로컬 타임존**을 사용해야 합니다 (UTC 아님). Bash에서 `$(date +%Y%m%d-%H%M%S)`,
Python에서 `datetime.now().strftime('%Y%m%d-%H%M%S')`를 사용하세요. `date -u`,
`datetime.utcnow()`, `datetime.now(timezone.utc)`는 사용하지 마세요.

### 규칙 충돌 해결 (HARD GATE)
사용자의 요청이 HARD GATE 규칙(IFactory, skeleton-first,
SyncRunner/AsyncRunner, Output Isolation)과 충돌할 때, 에이전트는 반드시:
1. 사용자의 의도를 인정
2. 충돌을 설명하고 구체적 규칙을 인용
3. 프레임워크 내의 올바른 대안을 제안
4. 올바른 접근법으로 진행 — 조용히 따르지 말 것

**사용자가 명시적으로 API 메서드를 지정하더라도** (예: "`InferenceEngine.run()` 사용",
"`run_async()` 사용"), 에이전트는 반드시 해당 호출을 IFactory 패턴 안에 래핑해야 합니다.
상위 프로젝트 지시사항의 일반적인 충돌 패턴을 참조하세요.

## 핵심 규칙

1. **절대 임포트**: `from dx_app.src.python_example.common.xyz import ...`
2. **모델 해석**: `config/model_registry.json` 조회 — .dxnn 경로를 하드코딩하지 말 것
3. **IFactory 패턴**: 모든 앱은 5개 메서드로 IFactory를 구현 (`create_preprocessor`, `create_postprocessor`, `create_visualizer`, `get_model_name`, `get_task_type`)
4. **CLI 인자**: `common/runner/args.py`의 `parse_common_args()` 사용
5. **NPU 확인**: 추론 작업 전 `dxrt-cli -s`
6. **로깅**: `logging.getLogger(__name__)` — 단순 `print()` 금지
7. **스킬 문서로 충분**: 스킬이 불충분하지 않는 한 소스 코드를 읽지 말 것
8. **상대 임포트 금지**: 항상 패키지 루트에서 절대 임포트 사용
9. **하드코딩된 모델 경로 금지**: 모든 모델 경로는 CLI 인자 또는 model_registry.json에서
10. **4개 변형**: Python 앱은 sync, async, sync_cpp_postprocess, async_cpp_postprocess
11. **PPU 모델 자동 감지**: 모델 이름 `_ppu` 접미사, `model_registry.json`의 `csv_task: "PPU"`, 또는 compiler session 컨텍스트로 PPU 모델을 자동 감지. PPU 모델은 `src/python_example/ppu/`에 배치하며 간소화된 후처리(별도 NMS 불필요).
12. **기존 예제 검색**: 코드 생성 전 `src/python_example/<task>/<model>/`에서 기존 예제를 검색. 발견 시 사용자에게 질문: (a) 기존 예제만 설명, 또는 (b) 기존 예제 기반으로 새 예제 생성. 조용히 건너뛰거나 덮어쓰지 말 것.
13. **PPU 예제 생성은 필수**: 컴파일된 .dxnn 모델이 PPU인 경우 에이전트는 반드시 동작하는 예제를 생성해야 함 — PPU 모델에 대해 예제 생성을 건너뛰지 말 것.
14. **참조 모델과 교차 검증**: `assets/models/`에 사전 컴파일된 DXNN이 있거나 `src/python_example/`에 기존 검증된 예제가 있을 때, Level 5.5 감별 진단을 실행하여 앱 코드 vs 컴파일 문제를 분리. `dx-validate.md` Level 5.5 참조.
15. **필수 출력 아티팩트**: 모든 세션은 반드시 13개 아티팩트 모두 생성 (factory, config, 4개 변형, __init__.py, session.json, README.md, setup.sh, run.sh, session.log). 에이전트의 MANDATORY OUTPUT REQUIREMENTS 섹션 참조. 완료 선언 전 자체 검증 체크 실행.
16. **Skeleton 우선 개발** — 코드 작성 전 `.deepx/skills/dx-build-python-app.md` skeleton
    템플릿을 먼저 읽으세요. `src/python_example/<task>/<model>/`에서 가장 유사한 기존 예제를
    복사하고 모델별 부분만 수정(factory, postprocessor). 데모 스크립트를 처음부터 작성하지 마세요.
    프레임워크를 우회하는 독립형 스크립트를 제안하지 마세요 (예: factory/runner 패턴 없이 직접
    `InferenceEngine` 사용).
17. **SyncRunner/AsyncRunner만 사용** — 프레임워크의 SyncRunner (단일 모델) 또는 AsyncRunner
    (다중 모델)를 사용. 대안적 실행 방식을 제안하지 마세요 (독립형 스크립트, 직접 API 호출,
    커스텀 runner, 수동 `run_async()` 루프).
18. **C++14**: C++ 예제는 C++14 표준만 사용
19. **RAII**: C++ 코드는 `std::unique_ptr` 사용, raw `new`/`delete` 금지

## 컨텍스트 라우팅 테이블

| 작업에서 언급하는 내용... | 읽어야 할 파일 |
|---|---|
| **Python app, detection, classification** | `.deepx/skills/dx-build-python-app.md`, `.deepx/toolsets/common-framework-api.md` |
| **C++ app, native** | `.deepx/skills/dx-build-cpp-app.md`, `.deepx/toolsets/dx-engine-api.md` |
| **Async, performance, throughput** | `.deepx/skills/dx-build-async-app.md`, `.deepx/memory/performance_patterns.md` |
| **Model, download, registry** | `.deepx/skills/dx-model-management.md`, `.deepx/toolsets/model-registry.md` |
| **Validation, testing** | `.deepx/skills/dx-validate.md`, `.deepx/instructions/testing-patterns.md` |
| **Validation, feedback, fix** | `.deepx/skills/dx-validate.md`, parent `dx-runtime/.deepx/skills/dx-validate-and-fix.md` |
| **항상 읽기 (모든 작업)** | `.deepx/memory/common_pitfalls.md`, `.deepx/instructions/coding-standards.md` |
| **Brainstorm, plan, design** | `.deepx/skills/dx-brainstorm-and-plan.md` |
| **TDD, validation, incremental** | `.deepx/skills/dx-tdd.md` |
| **Completion, verify, evidence** | `.deepx/skills/dx-verify-completion.md` |

## Python 임포트

```python
from dx_app.src.python_example.common.runner.args import parse_common_args
from dx_app.src.python_example.common.runner.factory_runner import FactoryRunner
from dx_app.src.python_example.common.utils.model_utils import load_model_config
import logging

logger = logging.getLogger(__name__)
```

## 파일 구조

```
src/python_example/{task}/{model}/
├── __init__.py
├── config.json
├── {model}_factory.py
├── {model}_sync.py
├── {model}_async.py
├── {model}_sync_cpp_postprocess.py
└── {model}_async_cpp_postprocess.py

src/cpp_example/{task}/{model}/
├── CMakeLists.txt
├── main.cpp
├── config.json
├── include/
└── src/
```

## 플레이스홀더 코드 금지 (MANDATORY)

스텁/플레이스홀더 코드를 절대 생성하지 마세요. 여기에는 다음이 포함됩니다:
- 주석 처리된 import: `# from dxnn_sdk import InferenceEngine`
- 가짜 결과: `result = np.zeros(...)`
- TODO 마커: `# TODO: implement actual inference`
- 실제 async 구현 없이 "sync 버전과 유사"

모든 생성된 코드는 knowledge base의 실제 API를 사용하여 기능적이어야 합니다.
필요한 SDK/API를 모르는 경우, 먼저 관련 스킬 문서를 읽으세요.


## 실험적 기능 — 금지

실험적이거나 존재하지 않는 기능을 제안, 제공 또는 구현하지 마세요. 여기에는 다음이
포함됩니다:
- "웹 기반 비주얼 컴패니언" (web-based visual companion)
- 로컬 URL 기반 다이어그램 뷰어 또는 대시보드
- 사용자가 시각화를 위해 로컬 URL을 열어야 하는 모든 기능
- 현재 도구 세트에 존재하지 않는 모든 기능

**Superpowers brainstorming skill 오버라이드**: superpowers `brainstorming` 스킬에는
"Visual Companion" 단계 (체크리스트의 2단계)가 포함되어 있습니다. 이 단계는 DEEPX
프로젝트에서 반드시 건너뛰어야 합니다. Visual companion은 우리 환경에 존재하지 않습니다.
brainstorming 체크리스트에서 "Offer visual companion"이라고 하면, 건너뛰고
"Ask clarifying questions" (3단계)로 직접 진행하세요.

기능이 존재하지 않으면, 존재하는 척하지 마세요. 검증된, 문서화된 기능만 사용하세요.

**Autopilot / autonomous mode 오버라이드**: 사용자가 부재 중일 때 (autopilot mode,
auto-response "work autonomously", 또는 `--yolo` 플래그), brainstorming 스킬의
"Ask clarifying questions" 단계는 "knowledge base 규칙에 따라 기본 결정 내리기"로
대체되어야 합니다. `ask_user`를 호출하지 마세요 — knowledge base 기본값을 사용하여
brainstorming spec 생성으로 바로 진행하세요. 이후의 모든 게이트 (spec 리뷰, 계획,
TDD, 필수 산출물, 실행 검증)는 예외 없이 여전히 적용됩니다.


## 브레인스토밍 — 계획 전 Spec (HARD GATE)

superpowers `brainstorming` 스킬 또는 `/dx-brainstorm-and-plan` 사용 시:

1. **Spec 문서는 MANDATORY** — `writing-plans`로 전환하기 전에, spec 문서를
   `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`에 반드시 작성해야 합니다.
   spec을 건너뛰고 바로 계획 작성으로 가는 것은 위반입니다.
2. **사용자 승인 게이트는 MANDATORY** — spec 작성 후, 계획 작성으로 진행하기 전에
   사용자가 반드시 검토하고 승인해야 합니다. 관련 없는 사용자 응답 (예: 다른 질문에
   답변)을 spec 승인으로 처리하지 마세요.
3. **계획 문서는 spec을 참조해야 합니다** — 계획 헤더에는 승인된 spec 문서에 대한
   링크가 포함되어야 합니다.
4. **`/dx-brainstorm-and-plan` 선호** — 일반 superpowers `brainstorming` 스킬 대신
   프로젝트 레벨의 brainstorming 스킬을 사용하세요. 프로젝트 레벨 스킬에는
   도메인별 질문과 사전 점검이 포함되어 있습니다.
5. **규칙 충돌 확인은 MANDATORY** — brainstorming 중, agent는 사용자 요구사항이
   HARD GATE 규칙 (IFactory 패턴, skeleton-first, Output Isolation,
   SyncRunner/AsyncRunner)과 충돌하는지 반드시 확인해야 합니다. 충돌이 감지되면,
   agent는 brainstorming 중에 이를 해결해야 합니다 — 설계 spec에서 위반 요청을
   조용히 따르지 마세요. 위의 "규칙 충돌 해결"을 참조하세요.


## 자율 모드 보호 (MANDATORY)

사용자가 부재 중일 때 — autopilot mode, `--yolo` 플래그, 또는 시스템 auto-response
"The user is not available to respond" — 다음 규칙이 적용됩니다:

1. **"Work autonomously"는 "묻지 않고 모든 규칙을 따르라"는 의미이지, "규칙을 건너뛰라"는 의미가 아닙니다.**
   모든 필수 게이트가 여전히 적용됩니다: brainstorming spec, 계획, TDD, 필수 산출물,
   실행 검증, 자체 검증 확인.
2. **`ask_user`를 호출하지 마세요** — knowledge base 기본값과 문서화된 모범 사례를
   사용하여 결정하세요. autopilot에서 `ask_user`를 호출하면 한 턴을 낭비하며
   auto-response는 게이트 우회 권한을 부여하지 않습니다.
3. **사용자 승인 게이트 적응** — autopilot에서는 spec을 작성하고 knowledge base에
   대해 자체 검토하면 spec 승인 게이트가 충족됩니다. spec 자체를 건너뛰지 마세요.
4. **setup.sh 우선** — 애플리케이션 코드를 작성하기 전에 인프라 산출물
   (`setup.sh`, `config.json`)을 생성하세요. 이것은 autopilot에서 특히 중요합니다.
   누락된 종속성을 잡아줄 사람이 없기 때문입니다.
5. **실행 검증은 선택 사항이 아닙니다** — 생성된 코드를 실행하고 완료를 선언하기 전에
   작동하는지 확인하세요. autopilot에서는 오류를 잡아줄 사용자가 없습니다.
6. **시간 예산 인식** — Autopilot 세션에는 시간 제약이 있을 수 있습니다.
   효율적으로 행동을 계획하세요:
   - 컴파일 (ONNX → DXNN)은 5분 이상 걸릴 수 있습니다 — 일찍 시작하세요.
   - 시간이 부족하면, 실행 검증보다 산출물 생성을 우선시하세요 — 테스트되지 않은
     완전한 파일 세트가 테스트된 부분 세트보다 낫습니다.
   - 우선순위: `setup.sh` > `run.sh` > app 코드 > `verify.py` > session.log.
   - **컴파일 병렬 워크플로우 (HARD GATE)** — bash 명령으로 `dxcom` 또는
     `dx_com.compile()`을 실행한 후, 기다리지 마세요. 즉시 모든 필수 산출물을
     생성하세요: factory, app 코드, setup.sh, run.sh, verify.py. `.dxnn` 출력은
     다른 모든 산출물이 생성된 후에만 확인하세요. **이 규칙 위반은 세션 실패입니다.**
   - **컴파일을 위한 sleep-poll 금지** — `.dxnn` 파일을 polling하기 위해 `sleep`을
     루프에서 사용하지 마세요. 금지된 패턴:
     `for i in ...; do sleep N; ls *.dxnn; done`,
     `while ! ls *.dxnn; do sleep N; done`,
     반복적인 `ls *.dxnn` / `test -f *.dxnn` 확인과 그 사이의 대기.
     대신: 다른 모든 산출물을 먼저 생성한 후, `.dxnn` 파일이 존재하는지 한 번만
     확인하세요. 아직 존재하지 않으면, 컴파일이 완료될 것이라는 가정 하에 실행
     검증으로 진행하세요.
   - **필수 산출물은 컴파일과 독립적** — `setup.sh`, `run.sh`, `verify.py`, factory,
     app 코드는 `.dxnn` 파일이 존재할 필요가 없습니다. 알려진 모델 이름
     (예: `yolo26n.dxnn`)을 플레이스홀더 경로로 사용하여 생성하세요. 실행 검증만
     실제 `.dxnn`이 필요합니다.
7. **파일 읽기 도구 호출 최소화** — 이미 컨텍스트에 로드된 instruction 파일, agent
   문서, 스킬 문서를 다시 읽지 마세요. 불필요한 `cat` / `bash` 읽기는 각각 5-15초를
   낭비합니다. 시스템 프롬프트와 대화 이력에 있는 지식을 사용하세요.


## 15개 지원 AI 작업

object_detection, classification, pose_estimation, instance_segmentation,
semantic_segmentation, face_detection, depth_estimation, image_denoising,
image_enhancement, super_resolution, embedding, obb_detection, hand_landmark, ppu

## 하드웨어

| 아키텍처 | 값 |
|---|---|
| DX-M1 | `dx_m1` |

## 메모리

영속적 지식은 `.deepx/memory/`에 있습니다. 작업 시작 시 읽고, 학습 시 업데이트하세요.

## Git 작업 — 사용자 관리

작업 종료 시 git 브랜치 작업 (merge, PR, push, cleanup)에 대해 묻지 마세요.
사용자가 모든 git 작업을 직접 처리합니다. "merge to main", "create PR",
또는 "delete branch" 같은 옵션을 제시하지 마세요 — 작업만 완료하세요.


## Git 안전 — Superpowers 산출물

**`docs/superpowers/` 하위 파일을 절대 `git add`하거나 `git commit`하지 마세요.**
이들은 superpowers 스킬 시스템에서 생성된 임시 계획 산출물 (spec, plan)입니다.
`.gitignore`에 포함되어 있지만, 일부 도구는 `git add -f`로 `.gitignore`를 우회할 수
있습니다. 파일 생성은 괜찮습니다 — 커밋은 금지입니다.


## 세션 센티넬 (자동화 테스트용 MANDATORY)

사용자 프롬프트를 처리할 때, 테스트 하네스의 자동화된 세션 경계 감지를 위해
이 정확한 마커를 출력하세요:

- **응답의 첫 번째 줄**: `[DX-AGENTIC-DEV: START]`
- **모든 작업 완료 후 마지막 줄**: `[DX-AGENTIC-DEV: DONE (output-dir: <relative_path>)]`
  여기서 `<relative_path>`는 세션 출력 디렉토리입니다 (예: `dx-agentic-dev/20260409-143022_yolo26n_detection/`)

규칙:
1. **중요 — `[DX-AGENTIC-DEV: START]`를 첫 번째 응답의 절대적인 첫 줄로 출력하세요.**
   이것은 다른 어떤 텍스트, 도구 호출, 추론보다 먼저 나타나야 합니다.
   사용자가 "그냥 진행하라" 또는 "자체 판단을 사용하라"고 지시해도,
   START 센티넬은 협상 불가입니다 — 자동화 테스트는 이것 없이 실패합니다.
2. 모든 작업, 검증, 파일 생성이 완료된 후 맨 마지막 줄에 `[DX-AGENTIC-DEV: DONE (output-dir: <path>)]`를
   출력하세요
3. 상위 레벨 agent에 의해 handoff/routing으로 호출된 **서브 agent**인 경우,
   이 센티넬을 출력하지 마세요 — 최상위 agent만 출력합니다
4. 사용자가 세션에서 여러 프롬프트를 보내면, 각 프롬프트에 대해 START/DONE을 출력하세요
5. DONE의 `output-dir`은 프로젝트 루트에서 세션 출력 디렉토리까지의 상대 경로여야 합니다.
   파일이 생성되지 않았다면, `(output-dir: ...)` 부분을 생략하세요.
6. **계획 산출물만 생성한 후에는 절대 DONE을 출력하지 마세요** (spec, plan, 설계
   문서). DONE은 모든 산출물이 생성되었음을 의미합니다 — 구현 코드, 스크립트,
   설정, 검증 결과. brainstorming 또는 계획 단계를 완료했지만 실제 코드를 아직
   구현하지 않았다면, DONE을 출력하지 마세요. 대신, 구현으로 진행하거나
   사용자에게 진행 방법을 물어보세요.
7. **DONE 전 필수 산출물 확인**: DONE을 출력하기 전에, 아래의 자체 검증 확인을
   실행하세요. 필수 파일이 누락된 경우, DONE을 출력하기 전에 생성하세요.
   **이 단계를 절대 건너뛰지 마세요.**
   ```bash
   WORK_DIR="<session_output_directory>"
   echo "=== Mandatory Deliverable Check ==="
   for f in setup.sh run.sh verify.py session.log README.md config.json; do
       [ -f "${WORK_DIR}/$f" ] && echo "  ✓ $f" || echo "  ✗ MISSING: $f"
   done
   ls "${WORK_DIR}"/*.dxnn 2>/dev/null && echo "  ✓ .dxnn model" || echo "  ✗ MISSING: .dxnn model"
   ```
   산출물 중 MISSING이 있으면, 돌아가서 생성하세요. 누락된 산출물이 있는 상태에서
   최종 보고서를 제시하거나 DONE을 출력하지 마세요.
8. **세션 내보내기 안내**: DONE 센티넬 줄 바로 앞에, CLI 플랫폼에 맞는 세션 저장
   안내를 출력하세요:

   | 플랫폼 | 명령 | 형식 |
   |--------|------|------|
   | **Copilot CLI** | `/share html` | HTML 트랜스크립트 |
   | **Cursor CLI** (`agent`) | 내장 내보내기 없음 — 테스트 하네스가 `--output-format stream-json`으로 자동 저장 | JSON stream |
   | **OpenCode** | `/export` | JSON |

   Copilot CLI의 경우: `To save this session as HTML, type: /share html`
   OpenCode의 경우: `To save this session as JSON, type: /export`
   Cursor CLI의 경우: 사용자 작업이 필요 없습니다 — 테스트 하네스가 출력을 자동 캡처합니다.

   테스트 하네스 (`test.sh`)는 내보낸 아티팩트를 자동으로 감지하고
   세션 출력 디렉토리에 복사합니다.


## 계획 출력 (MANDATORY)

계획 문서를 생성할 때 (예: writing-plans 또는 brainstorming 스킬을 통해),
파일을 저장한 직후 **대화 출력에 전체 계획 내용을 항상 인쇄하세요**. 파일 경로만
언급하지 마세요 — 사용자가 별도의 파일을 열지 않고 프롬프트에서 직접 계획을
검토할 수 있어야 합니다.


## Instruction File Verification Loop (HARD GATE) — 내부 개발 전용

canonical source 수정 시 — `**/.deepx/**/*.md` 파일(agents, skills, templates,
fragments 포함) — 작업 완료 선언 전에 다음 루프를 **반드시** 수행해야 합니다:

1. **Generator 실행** — `.deepx/` 변경을 모든 플랫폼으로 전파:
   ```bash
   dx-agentic-gen generate
   # Suite 전체: bash tools/dx-agentic-dev-gen/scripts/run_all.sh generate
   ```
2. **Drift 검증** — 생성물과 commit 상태 일치 확인:
   ```bash
   dx-agentic-gen check
   ```
   drift 발견 시 1단계로 복귀.
3. **자동화 테스트 루프** — 테스트는 generator 출력이 정책을 만족하는지 검증:
   ```bash
   python -m pytest tests/test_agentic_scenarios/ -v --tb=short
   ```
   실패 처리:
   - generator 버그 → generator 수정 → 1단계
   - `.deepx/` 콘텐츠 누락 → `.deepx/` 수정 → 1단계
   - 테스트 규칙 부족 → 테스트 강화 → 1단계
4. **수동 감사** — 테스트 결과에 의존하지 않고, 생성된 파일들을 독립적으로 읽어
   크로스 플랫폼 sync(CLAUDE vs AGENTS vs copilot)와 레벨 간 sync(suite → 하위)를
   검증.
5. **갭 분석** — 수동 감사에서 발견한 이슈:
   - generator가 놓친 경우 → **generator 규칙 수정** 후 1단계
   - 테스트가 놓친 경우 → **테스트 강화** 후 1단계
6. **반복** — 3~5단계 모두 통과할 때까지.

### Pre-flight Classification (MANDATORY)

저장소 내의 `.md` 또는 `.mdc` 파일을 수정하기 전에, 반드시 다음 3가지 카테고리
중 하나로 분류하세요. **이 단계를 절대 건너뛰지 마세요** — generator 관리 파일을
직접 수정하면 다음 generate에서 조용히 덮어써지는 사일런트 손상이 됩니다.

1. **Canonical source** (`**/.deepx/**/*.md`) — 직접 수정 후 위의 Verification
   Loop을 실행합니다.
2. **Generator output** — 알려진 출력 경로의 파일:
   `CLAUDE.md`, `CLAUDE-KO.md`, `AGENTS.md`, `AGENTS-KO.md`,
   `copilot-instructions.md`, `.github/agents/`, `.github/skills/`,
   `.claude/agents/`, `.claude/skills/`, `.opencode/agents/`, `.cursor/rules/`
   → **직접 수정 금지.** `.deepx/` source(template, fragment, 또는
   agent/skill)를 찾아 수정한 후 `dx-agentic-gen generate`를 실행하세요.
3. **독립 소스** — 위 두 카테고리에 해당하지 않는 모든 파일 (`docs/source/`,
   `source/docs/`, `tests/`, 서브 프로젝트의 `README.md` 등)
   → 직접 수정 가능. 수정 후 `dx-agentic-gen check`를 한 번 실행하여 예상치
   못한 drift가 없는지 확인하세요.

**Anti-pattern**: 분류 없이 바로 파일을 수정하는 것. 해당 파일이 generator
output인지 확실하지 않으면, 수정 전후에 `dx-agentic-gen check`를 실행하세요
— check가 수정 내용을 덮어쓰면 해당 파일은 generator가 관리하는 파일이므로
`.deepx/` source를 통해 수정해야 합니다.

Pre-commit hook이 generator output 무결성을 강제합니다: 생성된 파일이
최신이 아니면 `git commit`이 실패합니다. Hook 설치:
```bash
tools/dx-agentic-dev-gen/scripts/install-hooks.sh
```

이 게이트는 `.deepx/` 파일이 작업의 *주요 산출물*인 경우(규칙 추가, 플랫폼 sync,
KO 번역 생성, agents/skills 수정)에 적용됩니다. 기능 구현 중 `.deepx/`에 단순
한 줄 수정이 발생하는 경우에는 적용되지 않습니다.

