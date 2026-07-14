from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD_BAT = ROOT / "build.bat"


def test_build_bat_has_scope_menu_and_env_controls():
    text = BUILD_BAT.read_text(encoding="utf-8")

    assert "BUILD_SCOPE" in text
    assert "BUILD_CATEGORY" in text
    assert "Select build scope:" in text
    assert "--minimal" in text
    assert "--category" in text


def test_build_bat_skips_install_copy_for_target_only_scope():
    text = BUILD_BAT.read_text(encoding="utf-8")

    assert "if /I \"%BUILD_SCOPE%\"==\"all\"" in text
    assert "Skipping install bin copy for target-only build scope" in text
    assert "Skipping dx_postprocess installation for build scope" in text


def test_build_bat_installs_dx_postprocess_for_minimal_scope():
    text = BUILD_BAT.read_text(encoding="utf-8")

    assert 'set "INSTALL_DX_POSTPROCESS=0"' in text
    assert 'if /I "%BUILD_SCOPE%"=="all" set "INSTALL_DX_POSTPROCESS=1"' in text
    assert 'if /I "%BUILD_SCOPE%"=="minimal" set "INSTALL_DX_POSTPROCESS=1"' in text
    assert 'if "%INSTALL_DX_POSTPROCESS%"=="1"' in text


def test_build_bat_declares_invalid_scope_and_missing_category_errors():
    text = BUILD_BAT.read_text(encoding="utf-8")

    assert "[DXAPP] [ERROR] Invalid BUILD_SCOPE:" in text
    assert "[DXAPP] [ERROR] BUILD_CATEGORY is required when BUILD_SCOPE=category." in text


def test_build_bat_rejects_empty_or_invalid_interactive_scope_choice():
    text = BUILD_BAT.read_text(encoding="utf-8")

    assert "Select build scope [1-3]:" in text
    assert "[DXAPP] [ERROR] Build scope selection is required." in text
    assert "[DXAPP] [ERROR] Invalid build scope selection:" in text
    assert 'if "!BUILD_SCOPE_CHOICE!"==""' in text
    assert "Falling back to: all" not in text
    assert 'if not defined BUILD_SCOPE set "BUILD_SCOPE=all"' not in text


def test_build_bat_uses_delayed_expansion_inside_all_build_blocks():
    text = BUILD_BAT.read_text(encoding="utf-8")

    assert "setlocal EnableDelayedExpansion" in text
    assert "echo Will copy installed binaries from: !INSTALL_DIR!\\bin" in text
    assert "echo Checking module directory: !MODULE_DIR!" in text


def test_build_bat_checks_empty_category_with_delayed_expansion():
    """Issue 1: Empty BUILD_CATEGORY check must use delayed expansion empty-string check."""
    text = BUILD_BAT.read_text(encoding="utf-8")
    
    # Must check for empty string using delayed expansion syntax
    assert 'if "!BUILD_CATEGORY!"==""' in text


def test_build_bat_uses_valid_scope_whitelist_flag():
    """Issue 2: Scope validation should use a VALID_SCOPE whitelist flag instead of chained negations."""
    text = BUILD_BAT.read_text(encoding="utf-8")
    
    # Must set and use VALID_SCOPE flag
    assert "VALID_SCOPE" in text
    assert 'set "VALID_SCOPE=0"' in text or 'set "VALID_SCOPE=1"' in text


def test_build_bat_passes_category_with_quoted_delayed_expansion():
    """Issue 3: Category argument must use quoted delayed expansion to avoid expansion issues."""
    text = BUILD_BAT.read_text(encoding="utf-8")
    
    # Must pass category with quoted delayed expansion
    assert '--category "!BUILD_CATEGORY!"' in text


def test_build_bat_does_not_use_generator_args_variable():
    """Category must be passed directly, not stored in GENERATOR_ARGS variable."""
    text = BUILD_BAT.read_text(encoding="utf-8")
    
    # Must NOT use GENERATOR_ARGS with quoted category
    assert 'set "GENERATOR_ARGS=--run --category' not in text


def test_build_bat_invokes_generator_directly_per_scope():
    """Generator must be invoked directly for each scope with category as quoted delayed expansion."""
    text = BUILD_BAT.read_text(encoding="utf-8")
    
    # Must have direct invocation with quoted delayed expansion for category
    assert '.\\scripts\\generate_build_bat.py --run --category "!BUILD_CATEGORY!"' in text or \
           '.\\\\scripts\\\\generate_build_bat.py --run --category "!BUILD_CATEGORY!"' in text
