#!/usr/bin/env python3
"""Generate build_internal.bat from CMakeSettings.json to match a selected configuration."""
import argparse
import json
import shutil
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CPP_EXAMPLE_DIR = PROJECT_ROOT / "src" / "cpp_example"
RUN_DEMO = PROJECT_ROOT / "run_demo.sh"
CATEGORY_EXCLUDES = {"common", "build", "sample", "__pycache__", "utils"}


def list_categories() -> List[str]:
    return sorted(
        path.name
        for path in CPP_EXAMPLE_DIR.iterdir()
        if path.is_dir() and path.name not in CATEGORY_EXCLUDES
    )


def resolve_minimal_targets() -> List[str]:
    text = RUN_DEMO.read_text(encoding="utf-8")
    in_array = False
    bases: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("DEMO_CPP_BASE=("):
            in_array = True
            continue
        if in_array and stripped.startswith(")"):
            break
        if in_array:
            stripped = stripped.split("#", 1)[0]
            bases.extend(part for part in stripped.split() if part)
    return sorted({target for base in bases for target in (f"{base}_sync", f"{base}_async")})


def resolve_category_targets(category: str) -> List[str]:
    if category == "list":
        return []
    category_dir = CPP_EXAMPLE_DIR / category
    if category in CATEGORY_EXCLUDES or not category_dir.is_dir():
        sys.exit(f"[DXAPP] [ERROR] Unknown category: {category}")
    targets: List[str] = []
    for model_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
        model = model_dir.name
        if (model_dir / f"{model}_sync.cpp").is_file():
            targets.append(f"{model}_sync")
        if (model_dir / f"{model}_async.cpp").is_file():
            targets.append(f"{model}_async")
    if not targets:
        sys.exit("[DXAPP] [ERROR] No build targets resolved.")
    return targets


def load_settings(path: pathlib.Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        sys.exit(f"CMakeSettings.json not found: {path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"Failed to parse {path}: {exc}")


def find_configuration(data: Dict, name: str) -> Dict:
    for cfg in data.get("configurations", []):
        if cfg.get("name") == name:
            return cfg
    available = [cfg.get("name", "<unnamed>") for cfg in data.get("configurations", [])]
    sys.exit(f"Configuration '{name}' not found. Available: {', '.join(available)}")


def render_value(raw: str, project_var: str, cfg_name: str, normalize_path: bool = False, project_var_path: str = None) -> str:
    # Translate CMakeSettings placeholders into batch-friendly forms.
    substitute_project = project_var_path if (normalize_path and project_var_path) else project_var
    value = raw.replace("${projectDir}", substitute_project)
    value = value.replace("${name}", cfg_name)
    # ${env.VAR} -> %VAR%
    idx = 0
    while "${env." in value[idx:]:
        start = value.find("${env.", idx)
        end = value.find("}", start)
        if start == -1 or end == -1:
            break
        env_key = value[start + len("${env."): end]
        value = value[:start] + f"%{env_key}%" + value[end + 1:]
        idx = start + len(env_key) + 2
    if normalize_path:
        # Normalize to forward slashes to avoid CMake escape issues.
        # Replace backslashes with forward slashes
        value = value.replace("\\", "/")
    return value


def to_cmake_path(val: str) -> str:
    """Convert Windows path to CMake-friendly form (forward slashes)."""
    return val.replace("\\", "/")


def is_multi_config(generator: str) -> bool:
    gen = generator.lower()
    return "visual studio" in gen or "xcode" in gen or "multi-config" in gen


def sanitize_batch_var_name(target: str) -> str:
    """배치 변수명에 사용할 수 있도록 타겟명을 sanitize합니다.
    
    영숫자와 언더스코어만 허용하고, 나머지는 언더스코어로 치환합니다.
    """
    import re
    return re.sub(r'[^A-Za-z0-9_]', '_', target)


def build_configure_lines(cfg: Dict, project_var: str, targets: Optional[List[str]] = None) -> List[str]:
    # Force Visual Studio generator to avoid missing Ninja on user machines.
    generator = "Visual Studio 17 2022"
    generator_args = "-A x64 -T v143"
    cfg_type = cfg.get("configurationType", "Release")
    variables = cfg.get("variables", [])

    # Automatically mirror DEEPX_SDK_DIR into DXRT_INSTALLED_DIR if not explicitly set.
    variables_with_derived = list(variables)
    has_dxrt_installed = any(v.get("name") == "DXRT_INSTALLED_DIR" for v in variables)
    for var in variables:
        if var.get("name") == "DEEPX_SDK_DIR" and not has_dxrt_installed:
            derived = dict(var)
            derived["name"] = "DXRT_INSTALLED_DIR"
            # Treat as PATH to normalize separators for CMake.
            derived["type"] = derived.get("type", "PATH")
            variables_with_derived.append(derived)
            break

    build_root_raw = cfg.get("buildRoot", f"{project_var}/build/{cfg_type}")
    install_root_raw = cfg.get("installRoot", f"{project_var}/install/{cfg_type}")
    build_root = render_value(build_root_raw, project_var, cfg.get("name", cfg_type))
    install_root = render_value(install_root_raw, project_var, cfg.get("name", cfg_type))

    lines: List[str] = []
    lines.append("@echo off")
    # Use EnableDelayedExpansion when targets are specified
    if targets is not None:
        lines.append("setlocal EnableDelayedExpansion")
    else:
        lines.append("setlocal")
    lines.append("REM Auto-generated by scripts/generate_build_bat.py")
    lines.append("set \"PROJECT_DIR=%~dp0\"")
    lines.append("set \"PROJECT_DIR=%PROJECT_DIR:~0,-1%\"")
    lines.append(f"set \"BUILD_DIR={build_root}\"")
    lines.append(f"set \"INSTALL_DIR={install_root}\"")
    lines.append("set \"PROJECT_DIR_FWD=%PROJECT_DIR:\\=/%\"")
    lines.append(f"set \"GENERATOR={generator}\"")
    lines.append(f"set \"GENERATOR_ARGS={generator_args}\"")

    lines.append("")
    lines.append("REM Clean stale CMake cache to avoid generator mismatch")
    lines.append("if exist \"%BUILD_DIR%/CMakeCache.txt\" (")
    lines.append("  del /f /q \"%BUILD_DIR%/CMakeCache.txt\"")
    lines.append(")")
    lines.append("if exist \"%BUILD_DIR%/CMakeFiles\" (")
    lines.append("  rmdir /s /q \"%BUILD_DIR%/CMakeFiles\"")
    lines.append(")")

    for var in variables_with_derived:
        name = var.get("name")
        raw_val = var.get("value", "")
        normalize = var.get("type", "").upper() == "PATH"
        if not name:
            continue
        value = render_value(
            raw_val,
            "%PROJECT_DIR%",
            cfg.get("name", cfg_type),
            normalize_path=normalize,
            project_var_path="%PROJECT_DIR_FWD%",
        )
        lines.append(f"set \"{name}={value}\"")

    lines.append("")
    lines.append(f"echo Configuring with generator: {generator} (%GENERATOR_ARGS%)")

    cfg_lines: List[str] = []
    cfg_lines.append("cmake -S \"%PROJECT_DIR%\" -B \"%BUILD_DIR%\" ^")
    cfg_lines.append("  -G \"%GENERATOR%\" ^")
    if generator_args:
        cfg_lines.append("  %GENERATOR_ARGS% ^")
    if not is_multi_config(generator):
        cfg_lines.append(f"  -DCMAKE_BUILD_TYPE={cfg_type} ^")
    for var in variables_with_derived:
        name = var.get("name")
        raw_val = var.get("value", "")
        normalize = var.get("type", "").upper() == "PATH"
        if not name:
            continue
        rendered = render_value(
            raw_val,
            "%PROJECT_DIR%",
            cfg.get("name", cfg_type),
            normalize_path=normalize,
            project_var_path="%PROJECT_DIR_FWD%",
        )
        cmake_value = to_cmake_path(rendered) if normalize else rendered
        cfg_lines.append(f"  -D{name}=\"{cmake_value}\" ^")
    # Pass Python executable from build.bat environment if available
    cfg_lines.append("  -DPython_EXECUTABLE=\"%PYTHON_EXE%\" ^")
    cfg_lines.append("  -DPython3_EXECUTABLE=\"%PYTHON_EXE%\"")

    lines.extend(cfg_lines)
    lines.append("IF %ERRORLEVEL% NEQ 0 goto :err")
    lines.append("")
    
    if targets is not None:
        # Target-specific build: build only specified targets and copy binaries
        target_list = " ".join(targets)
        lines.append(f"cmake --build \"%BUILD_DIR%\" --config {cfg_type} --parallel --target {target_list}")
        lines.append("IF %ERRORLEVEL% NEQ 0 goto :err")
        lines.append("")
        lines.append("REM Copy built executables to bin directory")
        lines.append("mkdir \"%PROJECT_DIR%\\bin\" 2>nul")
        for target in targets:
            var_name = sanitize_batch_var_name(target)
            lines.append(f"set \"FOUND_{var_name}=0\"")
            lines.append(f"for /R \"%BUILD_DIR%\" %%F in ({target}.exe) do (")
            lines.append("  if exist \"%%F\" (")
            lines.append(f"    copy /Y \"%%F\" \"%PROJECT_DIR%\\bin\\\" >nul")
            lines.append(f"    set \"FOUND_{var_name}=1\"")
            lines.append(f"    echo [DXAPP] [INFO] Binary copied to bin\\{target}.exe")
            lines.append("  )")
            lines.append(")")
            lines.append(f"if \"!FOUND_{var_name}!\"==\"0\" echo [DXAPP] [WARN] Built executable not found: {target}.exe")
        lines.append("")
        lines.append("echo Target build completed successfully.")
    else:
        # Full build: build all and install
        lines.append(f"cmake --build \"%BUILD_DIR%\" --config {cfg_type} --parallel")
        lines.append("IF %ERRORLEVEL% NEQ 0 goto :err")
        lines.append("")
        lines.append(f"cmake --install \"%BUILD_DIR%\" --config {cfg_type} --prefix \"%INSTALL_DIR%\"")
        lines.append("IF %ERRORLEVEL% NEQ 0 goto :err")
        lines.append("")
        lines.append("echo Build and install completed successfully.")
    
    lines.append("pause")
    lines.append("goto :eof")
    lines.append("")
    lines.append(":err")
    lines.append("echo Failed with exit code %ERRORLEVEL%.")
    lines.append("pause")
    lines.append("exit /b %ERRORLEVEL%")

    return lines


def write_bat(lines: List[str], dest: pathlib.Path) -> None:
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_env_bat(dxrt_dir: Optional[str], dest: pathlib.Path) -> None:
    lines: List[str] = ["@echo off"]
    lines.append("if not defined PROJECT_ROOT for %%A in (\"%~dp0.\") do set \"PROJECT_ROOT=%%~fA\\\"")
    lines.append("set \"VENV_SCRIPTS=%PROJECT_ROOT%venv\\Scripts\"")
    lines.append("set \"VENV_SITE=%PROJECT_ROOT%venv\\Lib\\site-packages\"")

    dxrt_dir = to_cmake_path(dxrt_dir)
    if dxrt_dir:
        lines.append(f"set \"DEEPX_SDK_DIR={dxrt_dir}\"")
        lines.append(f"set \"DXRT_INSTALLED_DIR={dxrt_dir}\"")
        lines.append("set \"CMAKE_PREFIX_PATH=%DEEPX_SDK_DIR%;%CMAKE_PREFIX_PATH%\"")
        lines.append("set \"DXRT_DLL_DIR=%DXRT_INSTALLED_DIR%/bin\"")
        lines.append("set \"DXRT_LIB_DIR=%DXRT_INSTALLED_DIR%/lib/x64\"")
        lines.append("set \"DXRT_INCLUDE_DIR=%DXRT_INSTALLED_DIR%/include\"")
    else:
        lines.append("REM DEEPX_SDK_DIR not defined in CMakeSettings; nothing to export")

    lines.append("")
    lines.append("REM Prefer venv Python and ensure DXRT DLLs are on PATH")
    lines.append("if exist \"%VENV_SCRIPTS%\" set \"PATH=%VENV_SCRIPTS%;%PATH%\"")
    lines.append("if exist \"%DXRT_INSTALLED_DIR%\\bin\\dxrt.dll\" if exist \"%VENV_SITE%\" copy /Y \"%DXRT_INSTALLED_DIR%\\bin\\dxrt.dll\" \"%VENV_SITE%\" >nul")

    lines.append("echo DXRT build environment configured.")
    write_bat(lines, dest)


def run_bat(path: pathlib.Path) -> int:
    # Runs the generated batch in-place so CMake sees the same cwd.
    try:
        completed = subprocess.run(
            f'"{path}"', shell=True, check=False
        )
        return completed.returncode
    except FileNotFoundError:
        sys.exit(f"Cannot run missing file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate build_internal.bat from CMakeSettings.json")
    parser.add_argument("--config", default="x64-Release", help="Configuration name from CMakeSettings.json")
    parser.add_argument("--cmake-settings", default="CMakeSettings.json", help="Path to CMakeSettings.json")
    parser.add_argument("--output", default="build_internal.bat", help="Output batch file path")
    parser.add_argument("--run", action="store_true", help="Run the generated build_internal.bat after creating it")
    parser.add_argument("--targets", nargs="+", help="Build only the specified CMake target(s)")
    parser.add_argument("--minimal", action="store_true", help="Build run_demo C++ targets only")
    parser.add_argument("--category", help="Build C++ targets under a category, or use 'list'")
    args = parser.parse_args()

    # Validate only one selection mode
    selected_modes = sum(bool(value) for value in [args.targets, args.minimal, args.category])
    if selected_modes > 1:
        sys.exit("[DXAPP] [ERROR] Use only one of --targets, --minimal, or --category.")

    # Handle category list early exit
    if args.category == "list":
        for category in list_categories():
            print(category)
        return

    settings_path = pathlib.Path(args.cmake_settings)
    data = load_settings(settings_path)
    cfg = find_configuration(data, args.config)
    project_root = pathlib.Path(__file__).resolve().parent.parent
    cfg_type = cfg.get("configurationType", "Release")
    build_root_raw = cfg.get("buildRoot", f"%PROJECT_DIR%/build/{cfg_type}")

    # Render DEEPX_SDK_DIR if present so we can export it for downstream steps (e.g., pybind build).
    dxrt_dir_val: Optional[str] = None
    for var in cfg.get("variables", []):
        if var.get("name") == "DEEPX_SDK_DIR":
            raw_val = var.get("value", "")
            normalize = var.get("type", "").upper() == "PATH"
            dxrt_dir_val = render_value(
                raw_val,
                "%PROJECT_DIR%",
                cfg.get("name", cfg.get("configurationType", "")),
                normalize_path=normalize,
                project_var_path="%PROJECT_DIR_FWD%",
            )
            break

    # Resolve targets
    targets: Optional[List[str]] = None
    if args.targets:
        targets = args.targets
    elif args.minimal:
        targets = resolve_minimal_targets()
    elif args.category:
        targets = resolve_category_targets(args.category)

    if targets is not None and not targets:
        sys.exit("[DXAPP] [ERROR] No build targets resolved.")

    lines = build_configure_lines(cfg, "%PROJECT_DIR%", targets=targets)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_bat(lines, out_path)
    print(f"Wrote {out_path}")

    env_out = out_path.with_name("build_env.bat")
    write_env_bat(dxrt_dir_val, env_out)
    print(f"Wrote {env_out}")

    if args.run:
        # Clean the build directory to avoid generator or toolset conflicts.
        cleanup_build = render_value(
            build_root_raw,
            str(project_root),
            cfg.get("name", cfg_type),
            normalize_path=True,
            project_var_path=str(project_root).replace("\\", "/"),
        )
        cleanup_path = pathlib.Path(cleanup_build)
        if cleanup_path.exists():
            shutil.rmtree(cleanup_path, ignore_errors=True)
            print(f"Removed existing build directory: {cleanup_path}")
        print(f"Running {out_path}...")
        rc = run_bat(out_path)
        if rc != 0:
            sys.exit(rc)
        print("build_internal.bat finished successfully.")


if __name__ == "__main__":
    main()