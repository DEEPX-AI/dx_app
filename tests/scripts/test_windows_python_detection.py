from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WINDOWS_PYTHON_BATS = [
    ROOT / "build.bat",
    ROOT / "run_demo.bat",
    ROOT / "setup.bat",
]


def read_batch(path):
    return path.read_text(encoding="utf-8")


def test_python_path_is_first_python_candidate_for_windows_scripts():
    for path in WINDOWS_PYTHON_BATS:
        text = read_batch(path)
        python_path_index = text.index("PYTHON_PATH")
        venv_index = text.index("venv\\Scripts\\python.exe")

        assert python_path_index < venv_index, path.name
        assert 'set "PYTHON_EXE=%PYTHON_PATH:"=%"' in text


def test_windows_scripts_do_not_auto_install_python():
    for path in WINDOWS_PYTHON_BATS:
        text = read_batch(path)

        assert "winget install --id Python.Python" not in text, path.name
        assert "Attempting automatic install" not in text, path.name
        assert ":python_install" not in text, path.name


def test_windows_scripts_report_missing_python_and_exit():
    for path in WINDOWS_PYTHON_BATS:
        text = read_batch(path)

        assert "[ERR] Python not found." in text, path.name
        assert "Please install Python:" in text, path.name
        assert "https://www.python.org/downloads/" in text, path.name
        assert "exit /b 1" in text or "goto :err" in text, path.name


def test_setup_bat_searches_for_alternative_python_when_auto_detected_python_fails():
    text = read_batch(ROOT / "setup.bat")
    verify_block = text[text.index("REM Verify Python works") :]

    assert "Python found at %PYTHON_EXE% but failed to execute. Searching further..." in verify_block
    assert '"%%~P" --version >nul 2>nul' in verify_block
    assert "if !ERRORLEVEL! EQU 0 set \"PYTHON_EXE=%%~P\"" in verify_block
    assert "python --version >nul 2>nul" in verify_block
    assert "py --version >nul 2>nul" in verify_block


def test_run_demo_bat_searches_py_and_python_when_auto_detected_python_fails():
    text = read_batch(ROOT / "run_demo.bat")
    verify_block = text[text.index("REM Make sure the found Python actually executes") :]

    assert "Python313\\python.exe" in verify_block
    assert "Python39\\python.exe" in verify_block
    assert "%ProgramFiles%\\Python312\\python.exe" in verify_block
    assert "C:\\Python312\\python.exe" in verify_block
    assert "if not defined PYTHON_EXE if exist %%~P (" in verify_block
    assert "python --version >nul 2>nul" in verify_block
    assert "py --version >nul 2>nul" in verify_block
