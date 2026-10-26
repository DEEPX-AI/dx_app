"""The dxapp-checks workflow: the checks job runs the single entry point; the
NPU job is opt-in and never runs fork code (U-16). GitHub's own schema is not
available here - these pin the structure the design depends on."""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "dxapp-checks.yml"


@pytest.fixture(scope="module")
def doc():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _triggers(doc):
    return doc.get("on", doc.get(True))  # YAML 1.1 reads a bare `on` key as True


def _runs(job):
    return [step.get("run", "") for step in job["steps"]]


def test_triggers(doc):
    triggers = _triggers(doc)
    for event in ("push", "pull_request", "workflow_dispatch"):
        assert event in triggers, event


def test_every_run_step_uses_bash_with_pipefail(doc):
    # shell: bash is `bash -eo pipefail` on GitHub: `binary | tee log` fails with the binary.
    assert doc["defaults"]["run"]["shell"] == "bash"


def test_permissions_are_read_only(doc):
    assert doc["permissions"] == {"contents": "read"}


def test_the_checks_job_runs_the_single_entry_point(doc):
    job = doc["jobs"]["checks"]
    runs = [r.strip() for r in _runs(job)]
    assert "bash scripts/ci_checks.sh" in runs
    # `pip install pytest` is fine; running a check directly is not.
    assert not any("-m pytest" in r or "check_" in r or "gen_model_registry" in r for r in runs), runs
    assert "ubuntu-24.04" in str(job["runs-on"])
    assert "DXAPP_CHECKS_RUNNER" in str(job["runs-on"])
    assert "needs" not in job


def test_fork_pull_requests_always_run_on_the_hosted_runner(doc):
    # DXAPP_CHECKS_RUNNER may name a self-hosted runner; a fork's code must
    # never reach it. The fork test comes first, so it wins over the variable.
    runs_on = " ".join(str(doc["jobs"]["checks"]["runs-on"]).split())
    assert runs_on.startswith(
        "${{ github.event.pull_request.head.repo.fork && 'ubuntu-24.04' || "), runs_on
    assert runs_on.index("head.repo.fork") < runs_on.index("vars.DXAPP_CHECKS_RUNNER")


def test_no_checkout_leaves_the_token_in_git_config(doc):
    # Neither job pushes; on a persistent self-hosted runner the job token
    # must not stay in .git/config for every later step.
    for name, job in doc["jobs"].items():
        checkouts = [s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout@")]
        assert checkouts, name
        for step in checkouts:
            assert (step.get("with") or {}).get("persist-credentials") is False, (name, step)


def test_the_checks_job_lets_git_trust_the_checkout(doc):
    # tests/scripts' hermetic guard lists the tracked files with git; a
    # "dubious ownership" refusal in a container would leave it nothing to guard.
    # Added only once: on a self-hosted runner ~/.gitconfig persists across runs.
    runs = [r.strip() for r in _runs(doc["jobs"]["checks"])]
    trust = ('git config --global --get-all safe.directory | grep -qxF "$GITHUB_WORKSPACE" '
             '|| git config --global --add safe.directory "$GITHUB_WORKSPACE"')
    assert trust in runs
    assert runs.index(trust) < runs.index("bash scripts/ci_checks.sh")


def test_the_npu_job_is_opt_in_and_never_runs_fork_code(doc):
    job = doc["jobs"]["npu"]
    cond = " ".join(str(job["if"]).split())
    assert "vars.DXAPP_NPU_CI == 'true'" in cond
    assert "github.event_name == 'workflow_dispatch'" in cond
    assert "refs/heads/main" in cond and "refs/heads/staging" in cond
    assert "pull_request" not in cond
    assert job["needs"] == "checks"
    assert "self-hosted" in job["runs-on"] and "dxapp-npu" in job["runs-on"]
    assert job["concurrency"]["group"] == "dxapp-npu"


def test_the_npu_job_checks_the_npu_first_then_runs_the_graph_tests(doc):
    runs = "\n".join(_runs(doc["jobs"]["npu"]))
    order = [runs.index(needle) for needle in (
        "sanity_check.sh", "ninja -C build_x86_64", "graph_engine_test", "test_graph_engine.py")]
    assert order == sorted(order)
    for needle in ("Sanity check PASSED!", "common_unit_test", "test_graph_cli.py",
                   "test_graph_python.py", " 0 failures, 0 skipped$", "--require-dxrt"):
        assert needle in runs, needle


def test_the_sanity_step_is_judged_by_its_text(doc):
    # sanity_check.sh's exit code is not trusted: the step ignores it
    # (`|| true`), then needs the PASSED line and rejects any [ERROR] line.
    sanity = [r for r in _runs(doc["jobs"]["npu"]) if "sanity_check.sh" in r]
    assert len(sanity) == 1, sanity
    run = sanity[0]
    for needle in ("|| true", "grep -q 'Sanity check PASSED!'", "\\[ERROR\\]", "exit 1"):
        assert needle in run, needle


def test_every_npu_run_is_under_an_external_timeout(doc):
    # A hung hardware case (e.g. a lost job id) must fail its step rather than
    # hold the shared NPU runner until the job's timeout-minutes.
    lines = [line.strip() for run in _runs(doc["jobs"]["npu"]) for line in run.splitlines()]
    for needle in ("sanity_check.sh", "/graph_engine_test |", "/common_unit_test |", "-m pytest"):
        hits = [line for line in lines if needle in line]
        assert hits, needle
        for line in hits:
            assert "timeout " in line.split(needle)[0], line
    assert "--kill-after=" in "\n".join(lines)


def test_the_strict_claim_names_the_real_workflow():
    cmake = (ROOT / "src" / "cpp_example" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "catches in CI." not in cmake
    assert ".github/workflows/dxapp-checks.yml" in cmake and "scripts/ci_checks.sh" in cmake
    gen = " ".join((ROOT / "scripts" / "gen_model_registry.py").read_text(encoding="utf-8").split())
    assert "--strict (CI)" not in gen and 'drift (CI)"' not in gen
    assert gen.count("scripts/ci_checks.sh: codegen-strict") == 2  # docstring (re-wrapped) and --help
