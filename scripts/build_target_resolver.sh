#!/bin/bash

DXAPP_RESOLVER_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
DXAPP_CPP_EXAMPLE_DIR="${DXAPP_RESOLVER_DIR}/src/cpp_example"
DXAPP_RUN_DEMO="${DXAPP_RESOLVER_DIR}/run_demo.sh"
DXAPP_CATEGORY_EXCLUDES=(common build sample __pycache__ utils)

dxapp_is_excluded_category() {
    local category="$1"
    local excluded
    for excluded in "${DXAPP_CATEGORY_EXCLUDES[@]}"; do
        if [ "${category}" = "${excluded}" ]; then
            return 0
        fi
    done
    return 1
}

dxapp_list_categories() {
    local category_path category
    for category_path in "${DXAPP_CPP_EXAMPLE_DIR}"/*; do
        [ -d "${category_path}" ] || continue
        category=$(basename "${category_path}")
        dxapp_is_excluded_category "${category}" && continue
        printf '%s\n' "${category}"
    done | sort
}

dxapp_resolve_minimal_targets() {
    awk '
        /^DEMO_CPP_BASE=\(/ { in_array=1; next }
        in_array && /^\)/ { in_array=0; next }
        in_array {
            gsub(/#.*/, "")
            for (i = 1; i <= NF; i++) {
                if ($i != "") {
                    print $i "_sync"
                    print $i "_async"
                }
            }
        }
    ' < "${DXAPP_RUN_DEMO}" | sort -u
}

dxapp_resolve_category_targets() {
    local category="$1"
    local category_dir="${DXAPP_CPP_EXAMPLE_DIR}/${category}"
    local model_dir model

    if [ -z "${category}" ]; then
        echo "[DXAPP] [ERROR] --category requires a category name or 'list'." >&2
        return 1
    fi

    if [ ! -d "${category_dir}" ] || dxapp_is_excluded_category "${category}"; then
        echo "[DXAPP] [ERROR] Unknown category: ${category}" >&2
        return 1
    fi

    for model_dir in "${category_dir}"/*; do
        [ -d "${model_dir}" ] || continue
        model=$(basename "${model_dir}")
        [ -f "${model_dir}/${model}_sync.cpp" ] && printf '%s_sync\n' "${model}"
        [ -f "${model_dir}/${model}_async.cpp" ] && printf '%s_async\n' "${model}"
    done | sort -u
}
