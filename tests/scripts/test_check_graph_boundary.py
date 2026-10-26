"""scripts/check_graph_boundary.py's CMake drift tripwire (U-52). Hermetic:
the module is imported and pointed at temporary CMake text."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "scripts" / "check_graph_boundary.py"
TOKEN = "${DXAPP_POSTPROCESS_INCLUDES}"


def load_guard():
    spec = importlib.util.spec_from_file_location("check_graph_boundary", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_real_cmake_lists_pass():
    assert load_guard().check_include_roots_drift()


def test_a_graph_target_without_the_postprocess_variable_is_fatal(tmp_path, monkeypatch, capsys):
    guard = load_guard()
    text = guard.CMAKE_LISTS_PATH.read_text(encoding="utf-8")
    block = guard._extract_named_block(guard._TARGET_INCLUDE_BLOCK, text, "graph_engine_test")
    assert block is not None and TOKEN in block
    broken = tmp_path / "CMakeLists.txt"
    broken.write_text(text.replace(block, block.replace(TOKEN, "")), encoding="utf-8")
    monkeypatch.setattr(guard, "CMAKE_LISTS_PATH", broken)
    assert not guard.check_include_roots_drift()
    assert "graph_engine_test" in capsys.readouterr().out


def test_the_postprocess_variable_left_in_a_comment_is_fatal(tmp_path, monkeypatch, capsys):
    """A "# ${DXAPP_POSTPROCESS_INCLUDES}" comment in the block lists nothing."""
    guard = load_guard()
    text = guard.CMAKE_LISTS_PATH.read_text(encoding="utf-8")
    block = guard._extract_named_block(guard._TARGET_INCLUDE_BLOCK, text, "graph_engine_test")
    assert block is not None and TOKEN in block
    broken = tmp_path / "CMakeLists.txt"
    broken.write_text(text.replace(block, block.replace(TOKEN, "# " + TOKEN)), encoding="utf-8")
    monkeypatch.setattr(guard, "CMAKE_LISTS_PATH", broken)
    assert not guard.check_include_roots_drift()
    assert "graph_engine_test" in capsys.readouterr().out


def test_the_vitpose_root_is_one_the_guard_resolves_against():
    assert ROOT / "src" / "postprocess" / "vitpose" in load_guard().INCLUDE_ROOTS


def test_the_shared_roi_crop_header_is_the_only_utility_header_the_engine_may_include():
    """I17: the graph engine includes common/utility/roi_crop.hpp;
    the rest of common/utility/ stays outside the engine's include surface."""
    guard = load_guard()
    engine_file = ROOT / "src" / "cpp_example" / "common" / "graph" / "roi_router.cpp"
    assert guard._include_violation(engine_file, "common/utility/roi_crop.hpp") is None
    assert guard._include_violation(engine_file, "common/utility/common_util.hpp") is not None


def test_the_real_engine_tree_is_clean():
    """The whole guard, as CI runs it: exit 0 on this checkout."""
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(GUARD)], cwd=str(ROOT),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_logic_ops_header_is_an_engine_header():
    """Logic nodes live in common/graph/logic_ops.{hpp,cpp}, included by the
    parser, the executors and the visualizer."""
    guard = load_guard()
    engine_file = ROOT / "src" / "cpp_example" / "common" / "graph" / "graph_config.cpp"
    assert guard._include_violation(engine_file, "common/graph/logic_ops.hpp") is None


def test_the_user_logic_header_is_an_engine_header():
    """A "user" node runs a class registered in common/graph/user_logic.hpp; logic_ops.cpp includes it."""
    guard = load_guard()
    engine_file = ROOT / "src" / "cpp_example" / "common" / "graph" / "logic_ops.cpp"
    assert guard._include_violation(engine_file, "common/graph/user_logic.hpp") is None
