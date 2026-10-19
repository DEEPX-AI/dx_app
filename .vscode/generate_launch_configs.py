#!/usr/bin/env python3
"""
Generate launch.json configurations for all examples, and the tasks.json build tasks
they run first

Usage:
    Generate launch.json file:
       python3 generate_launch_configs.py > launch.json
    Generate tasks.json file:
       python3 generate_launch_configs.py --tasks > tasks.json

    Note: This will overwrite your existing launch.json / tasks.json file.
          Make sure to backup them if needed.
"""

import json
import sys

# Example configurations: (name, binary, model, video, image, rtsp_stream)
# Example configurations are DERIVED from config/model_registry.json rather than
# hard-coded. The hard-coded list had drifted badly: 252 of its 276 .dxnn references
# named files from an older naming era that are no longer on disk. Under the
# per-variant layout each variant is its own C++ target, <variant>_sync and
# <variant>_async, and the sample media comes from each variant's own config, so all
# of it is derivable.
#
# One representative variant per family keeps the config count manageable; every other
# variant has its own targets, which the build-target task can build.
import json as _json
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
_REG = _json.loads((_ROOT / "config" / "model_registry.json").read_text(encoding="utf-8"))


def _has_cpp_targets(entry):
    d = _ROOT / "src" / "cpp_example" / entry["task"] / entry["family"] / entry["variant"]
    return all((d / f"{entry['variant']}_{mode}.cpp").is_file() for mode in ("sync", "async"))


def _variant_media(entry):
    cfg_path = (_ROOT / "src" / "python_example" / entry["task"] / entry["family"]
                / entry["variant"] / "config.json")
    if not cfg_path.is_file():
        return None, None
    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
    img = cfg.get("default_image") or "sample/img/sample_street.jpg"
    vid = cfg.get("default_video") or "assets/videos/snowboard.mp4"
    return img, _Path(vid).name


_seen = {}
for _e in sorted(_REG, key=lambda x: (x["task"], x["family"], x["variant"])):
    if _e["family"] in _seen or _e.get("alias_of") is not None or not _has_cpp_targets(_e):
        continue
    _img, _vid = _variant_media(_e)
    if _img is None:
        continue
    _seen[_e["family"]] = (
        _e["variant"], _e["variant"], _e["dxnn_file"], _vid, _img, "stream6")

# (name, binary, model, video, image, rtsp_stream) -- the shape create_config expects.
examples = list(_seen.values())

# Multi-model example configurations: (name, binary, model1, model2, video, image, rtsp)
# Empty: the yolov7_x_deeplabv3 entry that used to live here names a binary that exists
# in neither example tree (CMake's *_x_* multi-model glob matches nothing), and its two
# .dxnn names are from the pre-rename era. It generated 12 dead launch configs.
multi_model_examples = [
]

def create_config(name, binary, mode, input_type, model, video, image, rtsp_stream):
    """Create a single launch configuration"""
    
    # Determine arguments based on input type
    if input_type == "Image":
        args = ["-m", f"assets/models/{model}", "-i", image, "-l", "30"]
    elif input_type == "Image Dir":
        args = ["-m", f"assets/models/{model}", "-i", "sample/img", "-l", "30"]
    elif input_type == "Video":
        args = ["-m", f"assets/models/{model}", "-v", f"assets/videos/{video}"]
    elif input_type == "Video + Save":
        args = ["-m", f"assets/models/{model}", "-v", f"assets/videos/{video}", "-s"]
    elif input_type == "RTSP":
        args = ["-m", f"assets/models/{model}", "-r", f"rtsp://192.168.30.100:8554/{rtsp_stream}"]
    elif input_type == "Camera":
        args = ["-m", f"assets/models/{model}", "-c", "0"]
    else:
        args = []
    
    config = {
        "type": "cppdbg",
        "request": "launch",
        "name": f"Demo: {name} {mode} ({input_type})",
        "program": "${workspaceFolder}/bin/" + binary + "_" + mode.lower(),
        "args": args,
        "cwd": "${workspaceFolder}",
        "stopAtEntry": False,
        "environment": [],
        "externalConsole": False,
        "MIMode": "gdb",
        "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
                "ignoreFailures": True
            }
        ],
        "preLaunchTask": f"build: {binary}_{mode.lower()}"
    }
    
    return config

def create_multi_model_config(name, binary, mode, input_type, model1, model2, video, image, rtsp_stream):
    """Create a single launch configuration for multi-model examples"""
    
    # Determine arguments based on input type
    if input_type == "Image":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-i", image, "-l", "30"]
    elif input_type == "Image Dir":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-i", "sample/img", "-l", "30"]
    elif input_type == "Video":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-v", f"assets/videos/{video}"]
    elif input_type == "Video + Save":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-v", f"assets/videos/{video}", "-s"]
    elif input_type == "RTSP":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-r", f"rtsp://192.168.30.100:8554/{rtsp_stream}"]
    elif input_type == "Camera":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-c", "0"]
    else:
        args = []
    
    config = {
        "type": "cppdbg",
        "request": "launch",
        "name": f"Demo: {name} {mode} ({input_type})",
        "program": "${workspaceFolder}/bin/" + binary + "_" + mode.lower(),
        "args": args,
        "cwd": "${workspaceFolder}",
        "stopAtEntry": False,
        "environment": [],
        "externalConsole": False,
        "MIMode": "gdb",
        "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
                "ignoreFailures": True
            }
        ],
        "preLaunchTask": f"build: {binary}_{mode.lower()}"
    }
    
    return config

# Generate all configurations
all_configs = []

for name, binary, model, video, image, rtsp_stream in examples:
    for mode in ["Sync", "Async"]:
        for input_type in ["Image", "Image Dir", "Video", "Video + Save", "RTSP", "Camera"]:
            config = create_config(name, binary, mode, input_type, model, video, image, rtsp_stream)
            all_configs.append(config)

# Generate multi-model configurations
for name, binary, model1, model2, video, image, rtsp_stream in multi_model_examples:
    for mode in ["Sync", "Async"]:
        for input_type in ["Image", "Image Dir", "Video", "Video + Save", "RTSP", "Camera"]:
            config = create_multi_model_config(name, binary, mode, input_type, model1, model2, video, image, rtsp_stream)
            all_configs.append(config)

# tasks.json: three shared tasks, then one "build: <target>" task per launch
# configuration's preLaunchTask, and the same targets offered by build-target.
_COMMON_TASKS = [
    {
        "label": "check-and-setup-venv",
        "type": "shell",
        "linux": {
            "command": "bash",
            "args": [
                "-c",
                "if [ -z \"$VIRTUAL_ENV\" ]; then echo \"No venv active, setting up...\"; if [ ! -d \"venv\" ]; then python3 -m venv venv; fi; source venv/bin/activate && pip install --upgrade pip; else echo \"Virtual environment already active\"; fi"
            ]
        },
        "windows": {
            "command": "powershell",
            "args": [
                "-Command",
                "if (-not $env:VIRTUAL_ENV) { Write-Host 'No venv active, setting up...'; if (-not (Test-Path 'venv')) { python -m venv venv }; .\\venv\\Scripts\\Activate.ps1; pip install --upgrade pip } else { Write-Host 'Virtual environment already active' }"
            ]
        },
        "group": "build",
        "presentation": {"echo": True, "reveal": "always", "focus": False, "panel": "shared"},
        "problemMatcher": []
    },
    {
        "label": "build",
        "type": "shell",
        "dependsOn": "check-and-setup-venv",
        "linux": {
            "command": "bash",
            "args": [
                "-c",
                "source venv/bin/activate 2>/dev/null || true && echo \"Building project...\" && ./build.sh --type debug"
            ]
        },
        "windows": {
            "command": "powershell",
            "args": [
                "-Command",
                "try { .\\venv\\Scripts\\Activate.ps1 } catch {}; Write-Host 'Building project...'; build.bat"
            ]
        },
        "group": {"kind": "build", "isDefault": True},
        "presentation": {"echo": True, "reveal": "always", "focus": False, "panel": "shared"},
        "problemMatcher": ["$gcc"]
    },
    {
        "label": "build-target",
        "type": "shell",
        "dependsOn": "check-and-setup-venv",
        "linux": {
            "command": "bash",
            "args": [
                "-c",
                "source venv/bin/activate 2>/dev/null || true && echo \"Building target: ${input:buildTarget}\" && ./build.sh --type debug --target ${input:buildTarget}"
            ]
        },
        "group": "build",
        "presentation": {"echo": True, "reveal": "always", "focus": False, "panel": "shared"},
        "problemMatcher": ["$gcc"]
    },
]


def create_build_task(target):
    """The task a launch configuration's preLaunchTask names"""
    return {
        "label": f"build: {target}",
        "type": "shell",
        "linux": {
            "command": "bash",
            "args": [
                "-c",
                f"source venv/bin/activate 2>/dev/null || true && ./build.sh --type debug --target {target}"
            ]
        },
        "group": "build",
        "problemMatcher": ["$gcc"]
    }


def create_tasks_json():
    targets = [f"{binary}_{mode}" for _name, binary, *_rest in examples
               for mode in ("sync", "async")]
    default = "yolov7_640x640_sync" if "yolov7_640x640_sync" in targets else targets[0]
    return {
        "version": "2.0.0",
        "tasks": _COMMON_TASKS + [create_build_task(t) for t in targets],
        "inputs": [
            {
                "id": "buildTarget",
                "type": "pickString",
                "description": "Select build target",
                "options": targets,
                "default": default
            }
        ]
    }


if "--tasks" in sys.argv[1:]:
    print(json.dumps(create_tasks_json(), indent=2))
else:
    # Output JSON in launch.json format (pretty-printed)
    launch_json = {
        "version": "0.2.0",
        "configurations": all_configs
    }
    print(json.dumps(launch_json, indent=4))
