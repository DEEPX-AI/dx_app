#!/usr/bin/env bash
# Every dx_app repository check that needs no NPU, in one place (U-16).
# Developers and any CI job run exactly this:
#
#   bash scripts/ci_checks.sh                 # every check
#   bash scripts/ci_checks.sh --list          # the check names, in order
#   bash scripts/ci_checks.sh --only a,b      # a subset
#   bash scripts/ci_checks.sh --require-dxrt  # a check skipped for want of
#                                             # the dxrt headers FAILS instead
#
# A check needing what this machine lacks is SKIPPED with the reason
# (header-odr, cxx14-headers and cross-compile need the dxrt headers; the
# GitHub runner has none). The run fails only on a FAILED check.
#
# Environment:
#   PYTHON            the Python for the checks and pytest (default: the
#                     active virtualenv's, else python3)
#   DXRT_INCLUDE_DIR  where dxrt/dxrt_api.h is looked for (default
#                     /usr/local/include; point it at a missing directory to
#                     see what the CI runner sees)
#   DXAPP_CROSS_CXX   the aarch64 cross compiler (default aarch64-linux-gnu-g++)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$ROOT" || exit 2

CHECKS=(guard-graph-boundary guard-factory-uniqueness guard-model-registry
        guard-variant-scope codegen-strict codegen-check-docs header-odr
        cxx14-headers cross-compile python-compile workflow-yaml tests-scripts)

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    default_python="${VIRTUAL_ENV}/bin/python"
else
    default_python=python3
fi
PYTHON="${PYTHON:-$default_python}"
DXRT_INCLUDE_DIR="${DXRT_INCLUDE_DIR:-/usr/local/include}"
DXAPP_CROSS_CXX="${DXAPP_CROSS_CXX:-aarch64-linux-gnu-g++}"
export DXRT_INCLUDE_DIR DXAPP_CROSS_CXX

only=""
require_dxrt=0
while [ $# -gt 0 ]; do
    case "$1" in
        --list) printf '%s\n' "${CHECKS[@]}"; exit 0 ;;
        --only)
            [ $# -ge 2 ] || { echo "ci_checks: --only needs a comma-separated list" >&2; exit 2; }
            only="$2"; shift 2 ;;
        --require-dxrt) require_dxrt=1; shift ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "ci_checks: unknown argument: $1" >&2; exit 2 ;;
    esac
done

selected=("${CHECKS[@]}")
if [ -n "$only" ]; then
    IFS=',' read -r -a selected <<< "$only"
    for name in "${selected[@]}"; do
        case " ${CHECKS[*]} " in
            *" ${name} "*) ;;
            *) echo "ci_checks: unknown check: ${name} (see --list)" >&2; exit 2 ;;
        esac
    done
fi

W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT
skip_reason=""

has_dxrt() { [ -f "${DXRT_INCLUDE_DIR}/dxrt/dxrt_api.h" ]; }

skip_for_dxrt() {
    skip_reason="no dxrt headers at ${DXRT_INCLUDE_DIR}/dxrt"
    if [ "$require_dxrt" = 1 ]; then
        echo "ci_checks: --require-dxrt: ${skip_reason}" >&2
        return 1
    fi
    return 77
}

check_workflow_yaml() {
    local py files
    shopt -s nullglob
    files=(.github/workflows/*.yml)
    shopt -u nullglob
    [ ${#files[@]} -gt 0 ] || { echo "ci_checks: no .github/workflows/*.yml" >&2; return 1; }
    for py in "$PYTHON" python3 /usr/bin/python3; do
        if "$py" -c "import yaml" >/dev/null 2>&1; then
            "$py" - "${files[@]}" <<'PYEOF'
import sys

import yaml

bad = 0
for path in sys.argv[1:]:
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), dict):
            raise ValueError("no jobs: mapping")
        print("ok   " + path)
    except Exception as exc:  # noqa: BLE001 - report every file
        bad += 1
        print("BAD  {}: {}".format(path, exc))
sys.exit(1 if bad else 0)
PYEOF
            return
        fi
    done
    echo "ci_checks: no Python with PyYAML (tried ${PYTHON}, python3, /usr/bin/python3): pip install pyyaml" >&2
    return 1
}

run_check() {
    case "$1" in
        guard-graph-boundary) "$PYTHON" scripts/check_graph_boundary.py ;;
        guard-factory-uniqueness) "$PYTHON" scripts/check_factory_uniqueness.py ;;
        guard-model-registry) "$PYTHON" scripts/check_model_registry.py ;;
        guard-variant-scope)
            "$PYTHON" scripts/generate_cpp_family_layout.py --variant-scope src/cpp_example --check ;;
        codegen-strict)
            "$PYTHON" scripts/gen_model_registry.py --strict --out-dir "$W/generated" \
                --docs "$W/graph_models.md" --manifest scripts/modelzoo_manifest.json > /dev/null ;;
        codegen-check-docs) "$PYTHON" scripts/check_graph_models_doc.py ;;
        header-odr) has_dxrt || { skip_for_dxrt; return; }; bash scripts/check_header_odr.sh ;;
        cxx14-headers) has_dxrt || { skip_for_dxrt; return; }; bash scripts/check_cxx14.sh ;;
        cross-compile)
            if ! command -v "$DXAPP_CROSS_CXX" >/dev/null 2>&1; then
                skip_reason="no ${DXAPP_CROSS_CXX}"
                return 77
            fi
            has_dxrt || { skip_for_dxrt; return; }
            bash scripts/check_cross_compile.sh ;;
        python-compile)
            PYTHONPYCACHEPREFIX="$W/pycache" "$PYTHON" -m compileall -q \
                src/python_example src/bindings/python scripts tests ;;
        workflow-yaml) check_workflow_yaml ;;
        tests-scripts)
            if [ -n "${DXAPP_CI_CHECKS_ACTIVE:-}" ]; then
                echo "ci_checks: tests-scripts refused: already running inside ci_checks.sh" \
                     "(a tests/scripts test must not start the suite that runs it)" >&2
                return 1
            fi
            # TARGET's known failures run as strict xfails: green while they
            # fail, red the day one passes (tests/scripts/known_failures.py).
            DXAPP_CI_CHECKS_ACTIVE=1 "$PYTHON" -m pytest tests/scripts -q -rfExX -p no:cacheprovider \
                --known-failures tests/scripts/known_target_failures.txt ;;
    esac
}

summary=()
failed=0
for name in "${selected[@]}"; do
    echo "== ${name}"
    skip_reason=""
    start=$SECONDS
    run_check "$name"
    status=$?
    secs=$((SECONDS - start))
    if [ "$status" -eq 0 ]; then
        summary+=("PASS  ${name} (${secs}s)")
    elif [ "$status" -eq 77 ] && [ -n "$skip_reason" ]; then
        # 77 is a SKIP only when this script gave the reason; a check command
        # that itself exits 77 is a FAIL.
        summary+=("SKIP  ${name}: ${skip_reason}")
    else
        summary+=("FAIL  ${name} (exit ${status})")
        failed=1
    fi
done
echo
echo "== ci_checks summary"
printf '%s\n' "${summary[@]}"
exit "$failed"
