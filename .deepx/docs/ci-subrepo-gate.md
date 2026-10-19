# CI drift gate for this repository (`subrepo-gate`)

This repository is one level of the dx-agent-dev harness: its `.deepx/` directory
is the canonical source from which `CLAUDE.md`, `AGENTS.md`, `.claude/`,
`.github/agents|skills`, `.cursor/rules` and `.opencode/` are generated. The
generator and the shared fragments live only in **dx-all-suite**, so this repo
cannot regenerate itself — the CI gate below checks that a commit here did not
leave generated files stale.

## Files

| File | Runs on | Runner | Token |
|---|---|---|---|
| `.github/workflows/dx-agent-dev-subrepo-gate-ghes.yml` | closed-network GHES (`gh.deepx.ai/deepx/dx_app`) | `self-hosted` | `GH_DCI_TOKEN` |
| `.github/workflows/dx-agent-dev-subrepo-gate-cloud.yml` | github.com — private mirror `deepx-dhyang/dx_app` and public `DEEPX-AI/dx_app` | GitHub-hosted `ubuntu-latest` | `GC_DCI_TOKEN` (mirror) / none (public) |

Job name (use this as the required status check): **`subrepo-gate`**. Each job
carries a host guard on `github.server_url`, so exactly one of the two runs per
host. Both replaced the single `dx-agent-dev-subrepo-gate.yml` on 2026-09-08.
`.github/workflows/` is release-excluded, so the cloud file is committed to the
public repo by hand (see the suite guide, §5.3).

## What the gate executes

1. Checks out this repo into `self/` (`fetch-depth: 1`, `persist-credentials: false`).
2. Clones dx-all-suite (`https://<host>/<owner>/dx-all-suite.git`), shallow,
   same-name branch if it exists — waiting up to `SUITE_BRANCH_WAIT` (180 s) for it to
   appear, because the sync tool pushes the sub-repos before dx-all-suite — else the suite's
   default branch; with the token passed as `git -c http.extraheader=…`, never inside the URL.
   A fallback suite that has no `subrepo_check.sh` under its tools/scripts (too old) is refused
   with an explicit error: sync dx-all-suite first, then re-run the job.
3. Ensures `python3` is >= 3.8 and can import `jinja2` and `yaml` (ghes: system python or a
   job-local `.gate-venv`; cloud: `actions/setup-python` + `pip`).
4. Runs the drift check at this repo's canonical position inside the suite tree:

```bash
bash suite/.deepx/tools/scripts/subrepo_check.sh --subrepo-path dx-runtime/dx_app --repo-dir "$GITHUB_WORKSPACE/self" --suite-dir "$GITHUB_WORKSPACE/suite"
```

5. Removes the suite clone (`if: always()`).

## Reproduce locally

From a dx-all-suite checkout on the **same branch**, with this repo initialised
at `dx-runtime/dx_app`:

```bash
# this repo's working tree (including uncommitted changes) against the current suite
bash .deepx/tools/scripts/subrepo_check.sh --subrepo-path dx-runtime/dx_app --repo-dir dx-runtime/dx_app --suite-dir . --mode worktree

# the committed HEAD only (what CI sees)
bash .deepx/tools/scripts/subrepo_check.sh --subrepo-path dx-runtime/dx_app --repo-dir dx-runtime/dx_app --suite-dir .
```

From a standalone clone of this repo (no suite checkout at hand) the script
clones the suite itself — run the copy that ships with any dx-all-suite checkout:
`bash /path/to/dx-all-suite/.deepx/tools/scripts/subrepo_check.sh --subrepo-path dx-runtime/dx_app --repo-dir .`

Exit codes: 0 clean · 1 drift (`MISSING:` / `CHANGED:`) · 2 setup/tool error
(unreachable suite, bad path, python3 missing or < 3.8, or the generator crashed —
a `Traceback` is NOT drift: nothing to regenerate).

## When it is red

```bash
# in the suite checkout, same branch, this repo at dx-runtime/dx_app
bash .deepx/tools/scripts/run_all.sh generate
bash .deepx/tools/scripts/run_all.sh check      # must print "All generated files are up-to-date." for every level
```

Then commit the regenerated files **in this repo**. Never hand-edit generated
files — edit the `.deepx/` source and regenerate. If a suite branch with the same
name does not exist yet, the gate waits up to 180 s for it and then compares against
the suite's default branch; push the suite branch first when your change depends on
new fragments. `bash: suite/.deepx/tools/scripts/subrepo_check.sh: No such file` on an
old run means exactly that race — re-run the job once the suite branch exists.

## See also

- Suite-level guide (all environments, tokens, `harness-gate`, onboarding, public-channel copy procedure): `docs/ci-gates.md` inside the dx-all-suite `.deepx` directory
- `subrepo_check.sh` reference: `tools/scripts/README.md` §3 inside the dx-all-suite `.deepx` directory
