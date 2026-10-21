# 이 repository의 CI drift gate (`subrepo-gate`)

이 repository는 dx-agent-dev harness의 한 level이다: `.deepx/` 디렉터리가 canonical
source이고, 여기서 `CLAUDE.md`, `AGENTS.md`, `.claude/`, `.github/agents|skills`,
`.cursor/rules`, `.opencode/`가 생성된다. generator와 공유 fragment는
**dx-all-suite**에만 존재하므로 이 repo는 스스로 재생성할 수 없다 — 아래 CI gate는
이 repo의 commit이 생성 파일을 stale 상태로 남기지 않았는지 검사한다.

## 파일

| 파일 | 실행 환경 | Runner | Token |
|---|---|---|---|
| `.github/workflows/dx-agent-dev-subrepo-gate-ghes.yml` | 폐쇄망 GHES (`gh.deepx.ai/deepx/dx_app`) | `self-hosted` | `GH_DCI_TOKEN` |
| `.github/workflows/dx-agent-dev-subrepo-gate-cloud.yml` | github.com — private 미러 `deepx-dhyang/dx_app` 및 public `DEEPX-AI/dx_app` | GitHub 제공 `ubuntu-latest` | `GC_DCI_TOKEN` (미러) / 없음 (public) |

Job 이름 (required status check로 이 이름을 사용): **`subrepo-gate`**. 각 job에는
`github.server_url` 기반 host guard가 있어 host마다 둘 중 정확히 하나만 실행된다.
두 파일은 2026-09-08에 단일 파일 `dx-agent-dev-subrepo-gate.yml`을 대체했다.
`.github/workflows/`는 release-excluded이므로 cloud 파일은 공개 repo에 수동으로
commit한다 (suite 가이드 §5.3 참고).

## Gate가 실행하는 것

1. 이 repo를 `self/`로 checkout한다 (`fetch-depth: 1`, `persist-credentials: false`).
2. dx-all-suite(`https://<host>/<owner>/dx-all-suite.git`)를 shallow clone한다 —
   같은 이름 branch가 있으면 그것 — sync 도구가 sub-repo를 먼저, dx-all-suite를 나중에
   push하므로 브랜치가 나타날 때까지 `SUITE_BRANCH_RETRIES`회 재시도
   (2회 x `SUITE_BRANCH_RETRY_DELAY` 15초 = 최대 30초) — 없으면 suite의 default branch.
   token은 `git -c http.extraheader=…`로 전달하며 URL에 넣지 않는다. fallback한 suite의
   tools/scripts에 `subrepo_check.sh`가 없으면(너무 오래됨) 명확한 에러로 거부한다 —
   dx-all-suite를 먼저 sync한 뒤 job을 re-run.
3. `python3`가 3.8 이상이고 `jinja2`와 `yaml`을 import할 수 있는지 확인한다 (ghes: 시스템 python
   또는 job-local `.gate-venv`; cloud: `actions/setup-python` + `pip`).
4. suite tree 안의 이 repo의 canonical 위치에서 drift check를 실행한다:

```bash
bash suite/.deepx/tools/scripts/subrepo_check.sh --subrepo-path dx-runtime/dx_app --repo-dir "$GITHUB_WORKSPACE/self" --suite-dir "$GITHUB_WORKSPACE/suite"
```

5. suite clone을 제거한다 (`if: always()`).

## 로컬 재현

**같은 branch**의 dx-all-suite checkout에서, 이 repo가 `dx-runtime/dx_app`에 init된 상태로:

```bash
# this repo's working tree (including uncommitted changes) against the current suite
bash .deepx/tools/scripts/subrepo_check.sh --subrepo-path dx-runtime/dx_app --repo-dir dx-runtime/dx_app --suite-dir . --mode worktree

# the committed HEAD only (what CI sees)
bash .deepx/tools/scripts/subrepo_check.sh --subrepo-path dx-runtime/dx_app --repo-dir dx-runtime/dx_app --suite-dir .
```

이 repo의 독립 clone에서(suite checkout이 없을 때)는 스크립트가 suite를 직접
clone한다 — 아무 dx-all-suite checkout에 들어 있는 복사본을 실행하면 된다:
`bash /path/to/dx-all-suite/.deepx/tools/scripts/subrepo_check.sh --subrepo-path dx-runtime/dx_app --repo-dir .`

Exit code: 0 clean · 1 drift (`MISSING:` / `CHANGED:`) · 2 setup/도구 오류
(suite 접근 불가, 잘못된 경로, python3 없음 또는 3.8 미만, 또는 generator crash —
`Traceback`은 drift가 아니므로 재생성할 것이 없음).

## Red일 때

```bash
# in the suite checkout, same branch, this repo at dx-runtime/dx_app
bash .deepx/tools/scripts/run_all.sh generate
bash .deepx/tools/scripts/run_all.sh check      # must print "All generated files are up-to-date." for every level
```

그 다음 재생성된 파일을 **이 repo에서** commit한다. 생성 파일을 직접 수정하지 말고
`.deepx/` source를 수정한 뒤 재생성한다. 같은 이름의 suite branch가 아직 없으면
gate는 2회(약 30초) 재시도한 뒤 suite의 default branch와 비교하므로, 변경이 새 fragment에
의존한다면 suite branch를 먼저 push한다. 오래된 run의 `bash: suite/.deepx/tools/scripts/
subrepo_check.sh: No such file`는 바로 그 race — suite branch가 생긴 뒤 job을 re-run.

## 참고

- Suite 수준 가이드 (전체 환경, token, `harness-gate`, onboarding, 공개 채널 복사 절차): dx-all-suite의 `.deepx` 디렉터리 안 `docs/ci-gates-KO.md`
- `subrepo_check.sh` 레퍼런스: dx-all-suite의 `.deepx` 디렉터리 안 `tools/scripts/README-KO.md` §3
