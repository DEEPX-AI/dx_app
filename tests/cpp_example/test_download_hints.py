"""Recovery hints name scripts/download_models.py, never ./setup.sh --models.

setup.sh also re-fetches the sample videos and, under its default --force,
rm -rf's an assets/videos symlink (scripts/get_resource.sh); the graph CLI
already prints download_models.py (graph_consumer.cpp DownloadCommand).
"""
import pytest

from conftest import PROJECT_ROOT

COMMON_UTIL = (PROJECT_ROOT / "src" / "cpp_example" / "common" / "utility"
               / "common_util.hpp")


@pytest.mark.contract
def test_common_util_hints_name_download_models():
    text = COMMON_UTIL.read_text(encoding="utf-8")
    assert "./setup.sh --models" not in text
    assert text.count("python3 scripts/download_models.py --models") == 5
