# =============================================================================
# dx_tool_completion.bash - Bash completion for dx_tool.sh (CLI mode)
# =============================================================================
# This file provides tab completion for CLI arguments (e.g., dx_tool.sh search <TAB>).
# Interactive mode (menu) has built-in tab completion via readline.
#
# Install:
#   source scripts/dx_tool_completion.bash
#
# Permanent install:
#   cp scripts/dx_tool_completion.bash ~/.local/share/bash-completion/completions/dx_tool.sh
#   or add to ~/.bashrc:
#   source /path/to/dx_app/scripts/dx_tool_completion.bash
# =============================================================================

_dx_tool_completions() {
    local cur prev commands
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Subcommand list
    commands="add extract list search info delete new-task validate run bench help"

    # Estimate project root from dx_tool.sh location
    local script_path="${COMP_WORDS[0]}"
    local project_root=""
    if [[ -f "$script_path" ]]; then
        project_root="$(cd "$(dirname "$script_path")/.." 2>/dev/null && pwd)"
    elif [[ -f "scripts/dx_tool.sh" ]]; then
        project_root="$(pwd)"
    elif [[ -f "../scripts/dx_tool.sh" ]]; then
        project_root="$(cd .. && pwd)"
    fi

    local cpp_dir="$project_root/src/cpp_example"
    local py_dir="$project_root/src/python_example"

    # Category list
    local categories="classification object_detection semantic_segmentation instance_segmentation depth_estimation ppu embedding face_alignment face_detection pose_estimation obb_detection super_resolution image_denoising image_enhancement"

    case "$COMP_CWORD" in
        1)
            # First argument: subcommand
            COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
            ;;
        *)
            # Nth argument: run subcommand multi-arg completion
            if [[ "${COMP_WORDS[1]}" == "run" ]]; then
                case "$prev" in
                    --lang)
                        COMPREPLY=( $(compgen -W "cpp py both" -- "$cur") )
                        return 0
                        ;;
                    --filter)
                        # Suggest model names
                        if [[ -d "$cpp_dir" ]]; then
                            local models
                            models=$(find "$cpp_dir" -mindepth 2 -maxdepth 2 -type d \
                                ! -path "*/common/*" ! -path "*/build/*" ! -path "*/sample/*" \
                                -exec basename {} \; 2>/dev/null | sort -u)
                            COMPREPLY=( $(compgen -W "$models" -- "$cur") )
                        fi
                        return 0
                        ;;
                    *)
                        COMPREPLY=( $(compgen -W "--lang --filter --no-video --video-only --sync-only --async-only --display --help" -- "$cur") )
                        return 0
                        ;;
                esac
            fi
            # bench subcommand multi-arg completion
            if [[ "${COMP_WORDS[1]}" == "bench" ]]; then
                case "$prev" in
                    --lang)
                        COMPREPLY=( $(compgen -W "cpp py both" -- "$cur") )
                        return 0
                        ;;
                    --filter|-f)
                        if [[ -d "$cpp_dir" ]]; then
                            local models
                            models=$(find "$cpp_dir" -mindepth 2 -maxdepth 2 -type d \
                                ! -path "*/common/*" ! -path "*/build/*" ! -path "*/sample/*" \
                                -exec basename {} \; 2>/dev/null | sort -u)
                            COMPREPLY=( $(compgen -W "$models" -- "$cur") )
                        fi
                        return 0
                        ;;
                    --loops|-l|--warmup|-w)
                        COMPREPLY=( $(compgen -W "1 3 5 10 20 50" -- "$cur") )
                        return 0
                        ;;
                    --output-dir)
                        COMPREPLY=( $(compgen -d -- "$cur") )
                        return 0
                        ;;
                    --compare)
                        COMPREPLY=( $(compgen -f -X '!*.csv' -- "$cur") )
                        return 0
                        ;;
                    *)
                        COMPREPLY=( $(compgen -W "--lang --loops --warmup --filter --video --output-dir --csv-only --compare --help" -- "$cur") )
                        return 0
                        ;;
                esac
            fi
            # Second argument: per-subcommand auto-completion
            case "$prev" in
                run|bench)
                    # run/bench options
                    COMPREPLY=( $(compgen -W "--lang --filter --help" -- "$cur") )
                    ;;
                list)
                    # Category filter
                    COMPREPLY=( $(compgen -W "$categories" -- "$cur") )
                    ;;
                search)
                    # Keyword - suggest model names
                    if [[ -d "$cpp_dir" ]]; then
                        local models
                        models=$(find "$cpp_dir" -mindepth 2 -maxdepth 2 -type d \
                            ! -path "*/common/*" ! -path "*/build/*" ! -path "*/sample/*" \
                            -exec basename {} \; 2>/dev/null | sort -u)
                        COMPREPLY=( $(compgen -W "$models" -- "$cur") )
                    fi
                    ;;
                info|delete|rm)
                    # Model name or category/model path
                    if [[ -d "$cpp_dir" ]]; then
                        local completions=""
                        # Model names only
                        completions=$(find "$cpp_dir" -mindepth 2 -maxdepth 2 -type d \
                            ! -path "*/common/*" ! -path "*/build/*" ! -path "*/sample/*" \
                            -exec basename {} \; 2>/dev/null | sort -u)
                        # Category/model path
                        for cat in $categories; do
                            if [[ -d "$cpp_dir/$cat" ]]; then
                                for model_dir in "$cpp_dir/$cat"/*/; do
                                    [[ -d "$model_dir" ]] && completions="$completions $cat/$(basename "$model_dir")"
                                done
                            fi
                        done
                        COMPREPLY=( $(compgen -W "$completions" -- "$cur") )
                    fi
                    ;;
            esac
            ;;
    esac

    return 0
}

# Support both dx_tool.sh and ./dx_tool.sh
complete -F _dx_tool_completions dx_tool.sh
complete -F _dx_tool_completions ./dx_tool.sh
complete -F _dx_tool_completions scripts/dx_tool.sh
complete -F _dx_tool_completions ./scripts/dx_tool.sh
