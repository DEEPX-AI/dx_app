---
name: dx-build-async-app
description: Build async high-performance inference app
---

# Skill: Build Async High-Performance App for dx_app

> **This skill doc is sufficient.** Do NOT read `async_runner.py` source code
> unless this document is insufficient for your task.

## Overview

Build high-throughput asynchronous inference applications that maximize NPU
utilization by overlapping preprocessing, inference, and postprocessing across
multiple frames.

## Output Isolation (MUST FOLLOW)

All AI-generated applications MUST be created under `dx-agentic-dev/`, NOT in the
production `src/` directory. This prevents accidental modification of existing code.

### Session Directory

```
dx-agentic-dev/<YYYYMMDD-HHMMSS>_<model>_<task>/
├── session.json          # Build metadata
├── README.md             # How to run this app
├── factory/
│   ├── __init__.py
│   └── <model>_factory.py
├── <model>_async.py
├── <model>_async_cpp_postprocess.py
└── config.json
```

### session.json Template

```json
{
  "session_id": "<YYYYMMDD-HHMMSS>_<model>_<task>",
  "created_at": "<ISO 8601 timestamp>",
  "model": "<model_name>",
  "task": "<task_type>",
  "variants": ["async", "async_cpp_postprocess"],
  "status": "complete",
  "notes": "<any relevant notes>"
}
```

### Import Boilerplate for dx-agentic-dev/

Since apps in `dx-agentic-dev/` are at a different directory depth than production apps,
use this dynamic root-finding pattern instead of the standard `_v3_dir` pattern:

```python
import sys
from pathlib import Path

# Find dx_app root dynamically
_current = Path(__file__).resolve().parent
while _current != _current.parent:
    if (_current / 'src' / 'python_example' / 'common').exists():
        break
    _current = _current.parent
_v3_dir = _current / 'src' / 'python_example'
_module_dir = Path(__file__).parent

for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
```

### When to Use Production Path

Only create files in `src/python_example/<task>/<model>/` when the user EXPLICITLY says:
- "Add this to the production codebase"
- "Create this in src/"
- "Make this a permanent addition"

Default behavior: ALWAYS use `dx-agentic-dev/`.

## When to Use Async

| Scenario | Use Async? | Why |
|---|---|---|
| Single image | No | No overlap opportunity |
| Image directory | Maybe | Mild benefit from pipelining |
| Video file | Yes | Full pipeline overlap |
| USB camera | Yes | Real-time requirement |
| RTSP stream | Yes | Continuous feed, latency matters |
| Benchmarking | Yes | Measures true throughput |

## AsyncRunner Architecture

### 5-Worker Pipeline

```
Thread 1: read_worker
    Reads frames from input source (video/camera/RTSP)
    Measures: sum_read
    Output: (frame, timestamp) -> read_queue

Thread 2: preprocess_worker
    Runs preprocessor.process(frame)
    Calls ie.run_async([input_tensor])
    Measures: sum_preprocess
    Output: (frame, input_tensor, req_id, ctx) -> reqid_queue

Thread 3: wait_worker
    Calls ie.wait(req_id) to get inference results
    Measures: sum_inference (via inflight tracking)
    Output: (frame, input_tensor, outputs, ctx) -> output_queue

Thread 4: postprocess_worker
    Runs postprocessor.process(outputs, ctx)
    Measures: sum_postprocess
    Output: (frame, results) -> render_queue

Thread 5: render_worker
    Runs visualizer.visualize(frame, results)
    Saves output if --save enabled
    Measures: sum_render, sum_save
    Output: display_img -> display_queue

Main Thread: display_loop
    cv2.imshow() (must be on main thread for GUI)
    Measures: sum_display
```

### Pipeline Overlap (The Key Insight)

Without async (SyncRunner):
```
Frame 0: [Pre][Infer][Post][Viz]
Frame 1:                        [Pre][Infer][Post][Viz]
Frame 2:                                                [Pre][Infer][Post][Viz]
Total: 3 * (pre + infer + post + viz)
```

With async (AsyncRunner):
```
Frame 0: [Pre][Infer       ]
Frame 1:      [Pre][Infer       ]
Frame 2:           [Pre][Infer       ]
                   [Post 0][Viz 0]
                          [Post 1][Viz 1]
                                 [Post 2][Viz 2]
```

The CPU work (preprocess N+1) overlaps with NPU work (inference N).
For inference-dominated models, this yields ~2x throughput.

### Thread-Safe Communication

All inter-thread communication uses `SafeQueue` with configurable maxsize (default 4):

```python
# SafeQueue API
q.put(item, timeout=0.1)    # Returns True if enqueued, False if timeout
q.get(timeout=0.5)           # Returns item or None if timeout
q.try_get()                  # Non-blocking get, returns item or None
```

### Graceful Shutdown

The pipeline uses a SENTINEL chain for clean termination:

```
stop_event.set() -> drain all queues -> push SENTINEL to each queue
Each worker: if item is SENTINEL -> push SENTINEL to next queue -> exit
```

Error propagation: If any worker throws, it stores the exception in
`_worker_error` and triggers `_set_stop()`. The main thread re-raises
after all workers join.

## Building an Async App

### Step 1: Create Factory (Same as Sync)

The factory is shared between sync and async variants.

### Step 2: Create Async Script

```python
#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
<ModelDisplay> Asynchronous Inference Example - DX-APP v3.0.0

Usage:
    python <model>_async.py --model model.dxnn --video input.mp4
    python <model>_async.py --model model.dxnn --camera 0
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import <ModelClass>Factory
from common.runner import AsyncRunner, parse_common_args


def parse_args():
    return parse_common_args("<ModelDisplay> Async Inference")


def main():
    args = parse_args()
    factory = <ModelClass>Factory()
    runner = AsyncRunner(factory)
    runner.run(args)


if __name__ == "__main__":
    main()
```

### Step 3: Async + C++ Postprocess (Maximum Throughput)

```python
#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
<ModelDisplay> Async Inference (C++ Postprocess) - DX-APP v3.0.0
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import <CppPostProcess>
from dx_engine import InferenceOption
from common.utility import convert_cpp_detections
from factory import <ModelClass>Factory
from common.runner import AsyncRunner, parse_common_args


def parse_args():
    return parse_common_args("<ModelDisplay> Async Inference")


def main():
    args = parse_args()
    factory = <ModelClass>Factory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        use_ort = InferenceOption().get_use_ort()
        runner._cpp_postprocessor = <CppPostProcess>(
            input_w, input_h, 0.3, 0.45, use_ort
        )
        runner._cpp_convert_fn = convert_cpp_detections

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)


if __name__ == "__main__":
    main()
```

## Performance Comparison Template

Run all 4 variants and compare:

```bash
# 1. Sync + Python postprocess (baseline)
python <model>_sync.py \
    --model model.dxnn --video test.mp4 --no-display --loop 3 --verbose

# 2. Sync + C++ postprocess
python <model>_sync_cpp_postprocess.py \
    --model model.dxnn --video test.mp4 --no-display --loop 3 --verbose

# 3. Async + Python postprocess
python <model>_async.py \
    --model model.dxnn --video test.mp4 --no-display --loop 3 --verbose

# 4. Async + C++ postprocess (fastest)
python <model>_async_cpp_postprocess.py \
    --model model.dxnn --video test.mp4 --no-display --loop 3 --verbose
```

Expected results (typical detection model, 640x640):

```
Variant                        FPS     Inference   Postprocess
sync + python                  12.3    45 ms       18 ms
sync + cpp                     18.7    45 ms        2 ms
async + python                 22.1    45 ms       18 ms (overlapped)
async + cpp                    32.5    45 ms        2 ms (overlapped)
```

## Async Metrics (Extended)

AsyncRunner tracks additional metrics beyond the 7 sync fields:

| Metric | Description |
|---|---|
| `infer_completed` | Total inference requests completed |
| `render_completed` | Total frames rendered |
| `save_completed` | Total frames saved |
| `display_completed` | Total frames displayed |
| `infer_first_ts` | Timestamp of first inference submit |
| `infer_last_ts` | Timestamp of last inference complete |
| `infer_time_window` | Duration from first to last inference |
| `inflight_current` | Current inflight requests |
| `inflight_max` | Peak inflight requests |
| `inflight_time_sum` | Weighted inflight-time product |

## Common Issues

### Queue Full / Slow Consumer
If postprocess is slower than inference, queues fill up and the pipeline stalls.
Fix: Use C++ postprocess to speed up the bottleneck stage.

### Display Overhead
`cv2.imshow()` must run on the main thread and can be slow (10-30ms per frame).
For benchmarks, always use `--no-display`.

### Graceful Shutdown
Ctrl+C triggers the SENTINEL chain. If workers hang, they time out after 5 seconds
via `thread.join(timeout=5.0)`.

### Image Input
Async works with images too (single-element iterator), but provides no speedup.
Use sync for single images.
