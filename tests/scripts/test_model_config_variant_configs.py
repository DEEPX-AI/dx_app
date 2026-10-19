"""ModelConfig reads what every variant config.json says, not factory defaults.

A variant's config.json keeps the factory parameters under a nested
``"config"`` object (``"score_threshold"``, ``"nms_threshold"``, ...), next
to top-level metadata and other nested objects (``"registry_config"``,
``"cli"``, ``"preprocessor"``, ...). A reader that skips every nested object
silently hands the factories their compiled-in defaults instead.

These tests compile a small probe against ``common/config/model_config.hpp``
in ``tmp_path`` and compare what it reads, key by key and type-dispatched on
the JSON type, with Python's view of the same file:

* the effective view is the top-level scalars and arrays, with the members of
  a top-level ``"config"`` object on top (nested wins);
* ``"registry_config"`` and every other object are not part of it;
* bool -> ``get<bool>``, integral number -> ``get<int>``, other number ->
  ``get<double>``, string -> ``get<std::string>``, null -> its raw token via
  ``get<std::string>``, array of strings -> ``get_string_list``, any other
  array -> its raw text in ``rawArrays()``, parsed back as JSON.

The key counts are compared too, so a value that leaks in from an object
that is not read shows up. Nothing outside ``tmp_path`` is written.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = ROOT / "src" / "cpp_example"
VARIANT_CONFIGS = sorted(CPP_ROOT.glob("*/*/*/config.json"))
# 499 on 2026-10-01. A lower count means the glob no longer finds them all.
MIN_VARIANT_CONFIGS = 499

POSE_CONFIG = (
    CPP_ROOT / "pose_estimation" / "yolov8_pose" / "yolov8-s-pose_640x640" / "config.json"
)

PROBE_SOURCE = r"""
#include "common/config/model_config.hpp"

#include <climits>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string Hex(const std::string& s) {
    static const char* kDigits = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) {
        out += kDigits[c >> 4];
        out += kDigits[c & 0x0F];
    }
    return out;
}

std::string Unhex(const std::string& s) {
    std::string out;
    for (size_t i = 0; i + 1 < s.size(); i += 2) {
        out += static_cast<char>(std::stoi(s.substr(i, 2), nullptr, 16));
    }
    return out;
}

std::vector<std::string> SplitTabs(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream in(line);
    while (std::getline(in, field, '\t')) fields.push_back(field);
    return fields;
}

// Restores std::cout's buffer on scope exit.
class CoutCapture {
public:
    explicit CoutCapture(std::ostream& sink) : old_(std::cout.rdbuf(sink.rdbuf())) {}
    ~CoutCapture() { std::cout.rdbuf(old_); }
    CoutCapture(const CoutCapture&) = delete;
    CoutCapture& operator=(const CoutCapture&) = delete;

private:
    std::streambuf* old_;
};

std::string Number(double v, int digits) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", digits, v);
    return buf;
}

// "V\t<payload>" or "ABSENT" for one key, read the way `type` says.
std::string Read(const dxapp::ModelConfig& cfg, char type, const std::string& key,
                 const std::string& default_text) {
    const std::string kAbsent = "ABSENT";
    switch (type) {
        case 'b': {
            const bool a = cfg.get<bool>(key, false);
            const bool b = cfg.get<bool>(key, true);
            if (a != b) return kAbsent;
            return std::string("V\t") + (a ? "true" : "false");
        }
        case 'i': {
            const int a = cfg.get<int>(key, INT_MIN);
            const int b = cfg.get<int>(key, INT_MAX);
            if (a != b) return kAbsent;
            return "V\t" + std::to_string(a);
        }
        case 'd': {
            const double v = cfg.get<double>(key, std::numeric_limits<double>::quiet_NaN());
            if (std::isnan(v)) return kAbsent;
            return "V\t" + Number(v, 17);
        }
        case 'f': {  // the factory's own call: get<float>(key, factory default)
            const float v = cfg.get<float>(key, std::stof(default_text));
            return "V\t" + Number(static_cast<double>(v), 9);
        }
        case 's':
        case 'n': {
            const std::string a = cfg.get<std::string>(key, std::string("\x01" "A"));
            const std::string b = cfg.get<std::string>(key, std::string("\x01" "B"));
            if (a != b) return kAbsent;
            return "V\t" + Hex(a);
        }
        case 'l': {
            if (cfg.rawArrays().count(key) == 0) return kAbsent;
            std::string joined;
            for (const std::string& item : cfg.get_string_list(key)) {
                if (!joined.empty()) joined += ',';
                joined += "x" + Hex(item);
            }
            return "V\t" + joined;
        }
        case 'r': {
            const auto it = cfg.rawArrays().find(key);
            if (it == cfg.rawArrays().end()) return kAbsent;
            return "V\t" + Hex(it->second);
        }
        default:
            return "BADTYPE";
    }
}

}  // namespace

// argv[1]: request lines "F\t<path>" and "K\t<type>\t<hex key>[\t<default>]";
// argv[2]: one answer line per request line.
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream requests(argv[1]);
    std::ofstream answers(argv[2]);
    std::unique_ptr<dxapp::ModelConfig> cfg;
    std::string line;
    while (std::getline(requests, line)) {
        const std::vector<std::string> f = SplitTabs(line);
        if (f.size() == 2 && f[0] == "F") {
            std::ostringstream info;
            {
                CoutCapture capture(info);
                cfg = std::make_unique<dxapp::ModelConfig>(f[1]);
            }
            // values_.size() is visible only in the load message "(N keys)".
            const std::string text = info.str();
            const size_t open = text.rfind(" (");
            const std::string keys =
                open == std::string::npos ? "?" : text.substr(open + 2, text.find(' ', open + 2) - open - 2);
            answers << "F\t" << (cfg->isLoaded() ? 1 : 0) << '\t' << keys << '\t'
                    << cfg->rawArrays().size() << '\n';
        } else if (f.size() >= 3 && f[0] == "K" && cfg) {
            answers << Read(*cfg, f[1][0], Unhex(f[2]), f.size() > 3 ? f[3] : "") << '\n';
        } else {
            answers << "BADREQUEST\n";
        }
    }
    return 0;
}
"""


def effective_view(doc: dict) -> dict:
    """What ModelConfig should hold: top level, then a top-level "config" on top."""
    view = {k: v for k, v in doc.items() if not isinstance(v, dict)}
    nested = doc.get("config")
    if isinstance(nested, dict):
        view.update({k: v for k, v in nested.items() if not isinstance(v, dict)})
    return view


def json_type(value) -> str:
    if isinstance(value, bool):
        return "b"
    if isinstance(value, int):
        return "i"
    if isinstance(value, float):
        return "d"
    if isinstance(value, str):
        return "s"
    if value is None:
        return "n"
    if isinstance(value, list):
        return "l" if all(isinstance(x, str) for x in value) else "r"
    raise AssertionError(f"unexpected JSON value {value!r}")


@pytest.fixture(scope="module")
def probe(tmp_path_factory) -> Path:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is not available")
    work = tmp_path_factory.mktemp("model_config_probe")
    source = work / "probe.cpp"
    source.write_text(PROBE_SOURCE)
    binary = work / "probe"
    build = subprocess.run(
        [compiler, "-std=c++14", "-Wall", "-Wextra", "-Werror",
         f"-I{CPP_ROOT}", str(source), "-o", str(binary)],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stderr
    return binary


def run_probe(probe: Path, tmp_path: Path, requests: list[str]) -> list[str]:
    request_file = tmp_path / "requests.txt"
    answer_file = tmp_path / "answers.txt"
    request_file.write_text("".join(line + "\n" for line in requests))
    run = subprocess.run([str(probe), str(request_file), str(answer_file)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    answers = answer_file.read_text().splitlines()
    assert len(answers) == len(requests)
    return answers


def mismatches_for(probe: Path, tmp_path: Path, configs: list[Path]) -> tuple[int, list[str]]:
    """Compare every key of every config; return (keys checked, mismatch lines)."""
    requests: list[str] = []
    expected: list[tuple[str, str | None, object]] = []
    for path in configs:
        view = effective_view(json.loads(path.read_text(encoding="utf-8")))
        requests.append(f"F\t{path}")
        expected.append((str(path), None, view))
        for key, value in view.items():
            requests.append(f"K\t{json_type(value)}\t{key.encode('utf-8').hex()}")
            expected.append((str(path), key, value))

    answers = run_probe(probe, tmp_path, requests)
    problems: list[str] = []
    checked = 0
    for (path, key, value), answer in zip(expected, answers):
        where = Path(path).parent.name if Path(path).name == "config.json" else Path(path).name
        if key is None:
            arrays = sum(isinstance(v, list) for v in value.values())
            want = f"F\t1\t{len(value) - arrays}\t{arrays}"
            if answer != want:
                problems.append(f"{where}: key counts (loaded, values, arrays): "
                                f"want {want.split(chr(9))[1:]}, got {answer.split(chr(9))[1:]}")
            continue
        checked += 1
        kind = json_type(value)
        if not answer.startswith("V\t"):
            problems.append(f"{where}: {key} ({kind}) want {value!r}, got {answer}")
            continue
        payload = answer[2:]
        if kind == "b":
            got = payload == "true"
        elif kind == "i":
            got = int(payload)
        elif kind == "d":
            got = float(payload)
        elif kind == "s":
            got = bytes.fromhex(payload).decode("utf-8")
        elif kind == "n":
            got = None if bytes.fromhex(payload) == b"null" else bytes.fromhex(payload)
        elif kind == "l":
            got = [bytes.fromhex(x[1:]).decode("utf-8") for x in payload.split(",")] if payload else []
        else:
            got = json.loads(bytes.fromhex(payload).decode("utf-8"))
        if got != value or type(got) is not type(value):
            problems.append(f"{where}: {key} ({kind}) want {value!r}, got {got!r}")
    return checked, problems


def test_every_variant_config_reads_as_python_reads_it(probe, tmp_path):
    assert len(VARIANT_CONFIGS) >= MIN_VARIANT_CONFIGS, len(VARIANT_CONFIGS)
    checked, problems = mismatches_for(probe, tmp_path, VARIANT_CONFIGS)
    assert checked > 0
    assert not problems, (
        f"{len(problems)} mismatches over {len(VARIANT_CONFIGS)} configs, {checked} keys; first 40:\n"
        + "\n".join(problems[:40])
    )


def test_yolov8_s_pose_reads_its_thresholds_not_the_factory_defaults(probe, tmp_path):
    # The factory's own calls: get<float>("score_threshold", 0.25f) and
    # get<float>("nms_threshold", 0.65f); config.json says 0.3 and 0.45.
    answers = run_probe(probe, tmp_path, [
        f"F\t{POSE_CONFIG}",
        f"K\tf\t{b'score_threshold'.hex()}\t0.25",
        f"K\tf\t{b'nms_threshold'.hex()}\t0.65",
    ])
    score = float(answers[1].split("\t")[1])
    nms = float(answers[2].split("\t")[1])
    assert score == pytest.approx(0.3, abs=1e-6)
    assert nms == pytest.approx(0.45, abs=1e-6)


def test_overlay_rules_on_small_files(probe, tmp_path):
    cases = {
        # nested wins, whether "config" comes before or after the top-level key
        "after.json": {"score_threshold": 0.1, "config": {"score_threshold": 0.2}},
        "before.json": {"config": {"score_threshold": 0.2}, "score_threshold": 0.1},
        # a scalar at the top level replaced by a list in "config", and back
        "kinds.json": {"a": 1, "b": ["x"], "config": {"a": ["y", "z"], "b": 2}},
        # other objects, and objects inside "config", stay unread
        "others.json": {"registry_config": {"obj_threshold": 0.9}, "cli": {"x": True},
                        "config": {"inner": {"deep": 1}, "top_k": 3}},
        # a flat legacy file, and a "config" that is not an object
        "flat.json": {"score_threshold": 0.3, "class_names": ["a", "b]"], "config": "path.json"},
    }
    paths = []
    for name, doc in cases.items():
        path = tmp_path / name
        # json.dumps keeps key order, so "before.json" really has "config" first.
        path.write_text(json.dumps(doc, indent=2))
        paths.append(path)
    # A repeated key: the last one wins, so a later non-object "config"
    # cancels the overlay (Python's json reads it the same way).
    repeated = tmp_path / "repeated.json"
    repeated.write_text('{"config": {"score_threshold": 0.2}, "score_threshold": 0.1, "config": "x"}')
    paths.append(repeated)
    checked, problems = mismatches_for(probe, tmp_path, paths)
    assert checked == 10
    assert not problems, "\n".join(problems)
