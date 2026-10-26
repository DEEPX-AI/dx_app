# DX-APP Multi-Model Graph

The multi-model graph runs **several models over one input** in a single application: a detector, a segmenter and a pose model side by side, a detector that feeds a re-identification model with each person it finds, or a super-resolution model whose output image is handed to a detector. Decision logic such as zones, line crossing, counting and alerts can be added around the models.  

The whole pipeline is described in a **node-graph JSON file**: which models run, how they are connected, which regions are cropped and handed on. Changing a model or a connection is a text edit. There is no C++ to edit and nothing to rebuild.  

```bash
./bin/multi_model_graph_async \
    --graph src/cpp_example/multi_model_graph/fanout_od_seg_pose.json \
    --output result.png
```

This command runs a detector, a semantic segmenter and a pose model over one image and draws all three results on one picture.  

---

## Overview

### Components

| Component | Description |
|-----------|-------------|
| `bin/multi_model_graph_sync` | Runs the graph one stage at a time, in topological order. The reference: slower, and the first thing to try when a result looks wrong |
| `bin/multi_model_graph_async` | Submits every node whose inputs are ready before waiting on any of them, and keeps several frames in flight. Use it for throughput |
| `dx_graph` (Python module) | Runs the same graphs from Python, with the same engine, reports and pictures as the CLI |
| `src/cpp_example/multi_model_graph/*.json` | 22 sample graphs (see *Sample Graphs*) |
| `run_multi_model_demo.sh` | Validates every sample graph and runs the demo scenarios |
| `docs/graph_models.md` | The models that can run in a graph, with what each one produces and consumes |

Both binaries give the same result for the same input: their `--report` files are byte-identical (without `--report-timing`, which adds measured times).  

### Supported Models and Platforms

- **Models**: 492 of the 497 registered models can run in a graph. A model is named by its variant, the `.dxnn` file stem (for example `yolov8-n_640x640`). `--list-models` and `docs/graph_models.md` list them.  
- **Inputs**: images, video files, cameras (`camera:<N>`) and RTSP streams (`rtsp://...`), one or several at once.  
- **Platforms**: Linux x86_64 is built and tested. aarch64 is compile-checked only. Windows is not built or tested in this release (see *Graph engine on Windows* in [DX-APP Installation and Build](02_DX-APP_Installation_and_Build.md)).  
- **DX-RT**: 3.5.0 or later for the DX Model Zoo 2_5_0 files (`.dxnn` container v9). The graph checks the container before it loads a model, and on an older runtime it stops with a `MODEL_LOAD` error (`--check` exits 1 on it too) that says  
  `<file>: .dxnn container v9 needs DX-RT >= 3.5.0, but this runtime is 3.4.1. Use the v8 file (dxnn/2_4_0) or upgrade DX-RT.`  

---

## Build and Run

### Build

The graph binaries are built by the **full** build, or by `./build.sh --target multi_model_graph_sync multi_model_graph_async`. `dx_graph` is built by the full build only. The default `./build.sh` (`--minimal`) builds none of them.  

```bash
# multi_model_graph_sync / multi_model_graph_async into bin/
./build.sh --all

# Also build dx_graph for a specific Python interpreter
/path/to/python -m pip install pybind11 numpy
./build.sh --all --python_exec /path/to/python
```

`dx_graph` is installed as `bin/python/dx_graph`. It needs pybind11, numpy and that Python's development headers; without them the build skips it and prints `dx_graph Python module skipped: <reason>`.  

### Run Your First Graph

**Step 1. Check the graph without the NPU**  

`--check` validates a graph and lists each node, its model and whether the `.dxnn` file is present. It does not open the NPU.  

```bash
./bin/multi_model_graph_sync --check src/cpp_example/multi_model_graph/fanout_od_seg_pose.json
```

```text
OK: graph "od-seg-pose" is valid (4 nodes, 3 edges, 3 models)

node      model                         produces   consumes   artifact
cam       (source)                      frame      -          sample/img/sample_people.jpg
od        yolov8-n_640x640              boxes      full_frame yolov8-n_640x640.dxnn  [present, v9]
seg       bisenetv2_1024x2048           labelmap   full_frame bisenetv2_1024x2048.dxnn  [present, v9]
pose      yolov8-s-pose_640x640         keypoints  full_frame yolov8-s-pose_640x640.dxnn  [present, v9]
```

**Step 2. Download missing models**  

A model that is not downloaded yet is listed as `[MISSING]`, followed by a `download_models.py` command for the missing files. That command names the files by their Model Zoo names and adds `--output <model-dir>`; the variant names work as well:  

```bash
python3 scripts/download_models.py --models yolov8-n_640x640 bisenetv2_1024x2048 yolov8-s-pose_640x640
```

**Step 3. Run**  

```bash
# One image, result saved as an image
./bin/multi_model_graph_async --graph src/cpp_example/multi_model_graph/fanout_od_seg_pose.json \
    --output result.png

# A video instead of the graph's own input, result saved as a video
./bin/multi_model_graph_async --graph src/cpp_example/multi_model_graph/fanout_od_seg_pose.json \
    --input assets/videos/dance-group.mov --output result.mkv

# A camera, shown in a window (q or ESC stops)
./bin/multi_model_graph_async --graph src/cpp_example/multi_model_graph/fanout_od_seg_pose.json \
    --input camera:0 --display
```

Each frame prints one line with what every node found, and the run ends with a summary. The first command prints:  

```text
graph:  od-seg-pose (4 nodes, 3 edges)
input:  Image file: sample/img/sample_people.jpg
frame 0: od boxes=4 seg labelmap pose keypoints=2
output: result.png
1 frame, 0 failed
```

### Demo Script

`run_multi_model_demo.sh` checks every sample graph and runs the demo scenarios.  

```bash
bash run_multi_model_demo.sh            # interactive menu
bash run_multi_model_demo.sh --check    # check every sample graph, no NPU
bash run_multi_model_demo.sh --all      # --check, then each scenario
bash run_multi_model_demo.sh --scenario hand_cascade
```

| Option | Description |
|--------|-------------|
| `--scenario <id or number>` | Run one scenario: `hand_cascade` (0), `logistics_volume` (1) or `dms` (2) |
| `--async` | Use `multi_model_graph_async` instead of `multi_model_graph_sync` |
| `--check` | Validate every `src/cpp_example/multi_model_graph/*.json`. Does not use the NPU |
| `--all` | `--check`, then each scenario with `multi_model_graph_sync` |
| `--list` | Print the scenarios and the sample graphs, then exit |
| `--output-dir <dir>` | Reports and saved frames (default: `reports/dev/multi_model_demo`) |
| `--models-dir <dir>` | `.dxnn` directory (default: `assets/models`) |

---

## Sample Graphs

The sample graphs are in `src/cpp_example/multi_model_graph/`. Each one is a starting point for your own graph.  

**Model pipelines**  

| File | What it shows | Models |
|------|---------------|--------|
| `fanout_od_seg_pose.json` | One frame, three models in parallel | `yolov8-n_640x640` + `bisenetv2_1024x2048` + `yolov8-s-pose_640x640` |
| `fanout_od_seg_depth.json` | The same, with depth instead of pose | `yolov8-n_640x640` + `bisenetv2_1024x2048` + `fastdepth_224x224` |
| `cascade_od_reid_track.json` | Detector, then a crop per person, then an embedding, with tracking | `yolov8-n_640x640` → `casvit-t_224x224` |
| `cascade_od_attr.json` | Detector, then a crop per person, then attributes | `yolov8-n_640x640` → `deepmar_resnet50_224x224` |
| `cascade_obb_cls.json` | Oriented detector, then an un-rotated crop, then a classifier | `yolo26-l-obb_1024x1024` → `resnet50_224x224` |
| `handoff_denoise_od.json` | Denoise, then detect on the denoised image | `dncnn-color_512x512` → `yolov5-n_640x640` |
| `handoff_sr_od_cls.json` | Super-resolve ×2, detect on the large image, classify each person (*) | `realesrgan-x2_192x192` → `yolov8-n_640x640` → `resnet50_224x224` |
| `chain_sr_od_pose.json` | Super-resolve ×2, detect, then top-down pose per person (*) | `realesrgan-x2_192x192` → `yolov8-n_640x640` → `vitpose-s_256x192` |
| `chain_zerodce_od_pose_emb.json` | Low-light enhancement, detect, then pose and an embedding per person | `zerodce_400x600` → `yolov8-n_640x640` → `vitpose-s_256x192` + `casvit-t_224x224` |
| `hand_cascade.json` | Palm detector, then a crop per palm, then 21 hand landmarks | `mediapipe-hand-detector_192x192` → `mediapipe-hands-lite_224x224` |
| `dms_headpose.json` | Face detector, then head pose (pitch, yaw, roll) on a cpu node | `scrfd-500m_640x640` → `headpose` |
| `logistics_volume.json` | Detection, instance mask and depth in parallel, then a relative package-volume estimate on a cpu node | `yolo26-n_640x640` + `yolo26-n-seg_640x640` + `yolo26-depth-n_768x768` → `volume` |
| `multistream_od_two_sources.json` | Two sources, each its own stream, through one shared tracked detector and one re-id model | `yolov8-n_640x640` → `casvit-t_224x224` |

(*) Not part of the DX-M1 sample runs in this release: with `realesrgan-x2_192x192`, these graphs are at the limit of DX-M1's device memory. If a run stops with `MODEL_LOAD`, see *Device Memory*.  

**Decision graphs (logic nodes)**  

| File | What it decides |
|------|-----------------|
| `zone_intrusion.json` | People in a restricted zone raise an `intrusion` event (cooldown 10 s) |
| `line_counting.json` | Tracked people crossing a line are counted per direction, with a `cross` event each |
| `ppe_zone_check.json` | People in a zone are cropped for attributes; one without a hat raises `no_hat` |
| `loitering_dwell.json` | How long each tracked person stays in a zone; a `dwell` event after 5 s |
| `zone_people_count.json` | People counted in two zones |
| `gallery_known_person.json` | Each person's re-id embedding is named after the nearest gallery entry, or `unknown` |
| `driver_attention.json` | The driver's head pose raises `looking_away` (yaw beyond ±30°) or `head_down` (pitch below −20°) |
| `multistream_people_count.json` | People counted per stream; a `crowded` event in a stream with 5 or more |
| `logistics_volume_alert.json` | The packages' summed volume estimate raises `load_high` above 100 (relative: tune per camera) |

The decision graphs ship with a sample image. Their scenes are videos: run them with, for example, `--input assets/videos/person-pair-hallway.mp4 --loop --display`. `multistream_people_count.json` has two sources: give it one `--input <source_id>=<uri>` per source (see *Several Sources*).  

**Changing a model is a text edit.** To run a different detector in `cascade_od_reid_track.json`, change one string and download the new model. The same binary runs it:  

```bash
# in cascade_od_reid_track.json: "model": "yolov8-n_640x640"  ->  "model": "yolo26-s_640x640"
python3 scripts/download_models.py --models yolo26-s_640x640
./bin/multi_model_graph_async --graph src/cpp_example/multi_model_graph/cascade_od_reid_track.json \
    --output swapped.png
```

Any model whose `produces` and `consumes` fit the slot can be used. `--check` tells you before the run whether it fits.  

---

## Writing a Graph

### Graph File

A graph is a JSON file with a `version`, a `name`, a list of `nodes` and a list of `edges`. This is `cascade_od_reid_track.json`: a detector finds people, each person is cropped, and a re-identification model computes an embedding per crop.  

```json
{
  "version": 1,
  "name": "person-reid",
  "nodes": [
    { "id": "cam",  "type": "source", "uri": "sample/img/sample_people.jpg" },
    { "id": "od",   "model": "yolov8-n_640x640",
      "params": { "score_threshold": 0.35 },
      "track":  { "algo": "iou", "iou": 0.3, "max_age": 30 } },
    { "id": "reid", "model": "casvit-t_224x224" }
  ],
  "edges": [
    { "from": "cam", "to": "od" },
    { "from": "od",  "to": "reid",
      "roi": { "classes": ["person"], "pad": 0.05, "max": 16 } }
  ]
}
```

| Key | Meaning |
|-----|---------|
| `version` | Always `1` in this release |
| `name` | Free text, shown in messages and in `--report` |
| `nodes[].id` | The node's name, used by edges and in every message |
| `nodes[].params` | Overrides the model's `config.json` for this node only (see *Parameters*) |
| `nodes[].track` | Assigns a track id to each box the node outputs (see *Tracking*) |
| `edges[].roi` | Makes the edge a crop edge: the consumer runs once per box (see *ROI Options*) |

The files are strict JSON: comments are not accepted.  

### Nodes

| Kind | Written as | Meaning |
|------|------------|---------|
| source | `{ "id": "cam", "type": "source", "uri": "..." }` | Where frames come from: an image, a video file, `camera:0` or an `rtsp://` URL. A relative `uri` is read from the current directory when it exists there, else from the repository root. `--input` replaces it |
| model | `{ "id": "od", "model": "yolov8-n_640x640" }` | One registered model, named by its variant. A model node has no `"type"` |
| cpu | `{ "id": "hp", "type": "cpu", "op": "headpose" }` | A fixed computation on the CPU after its inputs: `headpose` (pitch, yaw, roll from five face landmarks) or `volume` (a relative package-volume estimate from boxes, instance masks and depth) |
| logic | `{ "id": "zone", "type": "logic", "op": "zone", "params": { ... } }` | A decision around the models: `filter`, `zone`, `line`, `count`, `dwell`, `condition`, `gallery` or `user` (see *Decision Logic*) |

### Edges

| Kind | Written as | The consumer receives |
|------|------------|-----------------------|
| frame | `{ "from": "cam", "to": "od" }` | The whole frame: the source frame, or the image the producer made |
| ROI | `{ "from": "od", "to": "reid", "roi": { } }` | One cropped region per box the producer found. The consumer runs once per crop |
| result | `{ "from": "face", "to": "hp", "carry": "result" }` | The producer's result itself. The consumer must be a cpu or a logic node |

Every model `produces` one result shape (`boxes`, `obboxes`, `instances`, `keypoints`, `labelmap`, `densemap`, `image`, `scores`, `vector`) and `consumes` a `frame`, a `roi` (crop) or `either`. An edge is accepted only when both ends fit:  

- A frame edge needs a consumer that consumes `frame` or `either`. From a model node it is allowed only when the producer's output is an `image` (denoising, low-light enhancement, super-resolution): the consumer then runs on that image (*image hand-off*). Its results carry `origin.inv_align`, which maps them back to the source frame; the drawing applies it.  
- An ROI edge needs a producer of `boxes`, `obboxes` or `instances`, and a consumer that consumes `roi` or `either`.  
- A node cannot receive both a full frame and ROI crops. Split it into two nodes.  
- Two edges cannot share the same `from` and `to`, and the graph must have no cycle.  
- Two sources feeding one node are two streams, not two inputs (see *Several Sources*).  

A graph that breaks a rule is rejected by `--check` with the node or edge and the fix (see *Checking and Troubleshooting*).  

### ROI Options

All options are optional; `"roi": {}` crops every box unchanged.  

| Key | Meaning | Default |
|-----|---------|---------|
| `classes` | Keep only these class names, for example `["person"]` | all classes |
| `min_score` | Confidence floor, 0 to 1 (`0.35`, not `35`) | the producer's threshold |
| `min_area` | Pixel-area floor: checked on the detected box, and again on the crop after `pad` and clipping to the frame | `0` |
| `pad` | Grow each box by this fraction of its size on every side, 0 to 1 (`0.05` is 5 %) | `0` |
| `max` | Keep at most this many boxes, highest score first | no limit |
| `align` | `"face5"`: align the crop from five facial landmarks (face detectors only) | none |

The options are applied in this order: class, score and area filter, then the `max` cut, then tracking, then `pad` and clipping, then alignment (or un-rotation of an oriented box), then the crop. Every second-stage result is drawn in source-frame coordinates, including the inverse of an `align` or of an OBB un-rotation. In `--report`, a result keeps the coordinates of the crop or image it ran on, and its `origin.inv_align` (a 2×3 affine) and `origin.src_box` map it back to the source frame. Logic nodes always work in source-frame pixels.  

### Tracking

`"track"` on a detector node gives every box it outputs a `track_id` that follows the object from frame to frame. It is declared on the producer, so every consumer of its boxes sees the same ids.  

| Key | Meaning | Default |
|-----|---------|---------|
| `algo` | The tracker. `"iou"` is the only one | `"iou"` |
| `iou` | Overlap a box needs to continue a track, greater than 0 and at most 1 | `0.3` |
| `max_age` | Frames an unmatched track survives (0 or more) | `30` |

`"track": {}` uses the defaults. Tracking is applied in frame order, so in `multi_model_graph_async` a tracked node adds some latency; throughput is unaffected.  

### Parameters

A node's `"params"` overrides the model's `config.json` for that node only, so one model can appear twice in a graph with two thresholds:  

```text
factory defaults  <  <task>/<family>/<variant>/config.json  <  node "params"
```

Values are numbers (`score_threshold`, `nms_threshold`, ...), or strings and string arrays for the keys a model reads as text, such as `class_names`. `class_names` renames a model's classes for one node, which keeps an ROI filter readable:  

```json
{ "id": "od", "model": "yolov8-n_640x640", "params": { "class_names": ["pedestrian"] } }
```

### Device Memory

Every model node opens its own engine on the NPU, even two nodes that name the same model. A graph whose engines do not fit in the NPU's memory fails while loading, with `MODEL_LOAD` on the first node that does not fit. Two settings under `"engine"` lower what a graph needs:  

```json
{ "id": "od",  "model": "yolov8-n_640x640", "engine": { "share": true, "buffers": 2 } },
{ "id": "od2", "model": "yolov8-n_640x640", "engine": { "share": true, "buffers": 2 } }
```

- `"share": true`: every node with it and the same model uses one engine. The nodes must have the same `params` and `buffers`. Results do not change.  
- `"buffers": N` (1 to 100): how many inputs DX-RT keeps in flight for the model (DX-RT's default is 6). Most of a model's device memory is this buffer pool, so 2 buffers need about a third. Fewer buffers can lower async throughput; results do not change.  

---

## Decision Logic

Logic nodes are the decisions around the models: which boxes matter, where they are, how many, for how long, and when to raise an event. They run on the CPU, once per frame, and read their inputs on `"carry": "result"` edges.  

| Op | Reads | Main params | Outputs | Events |
|----|-------|-------------|---------|--------|
| `filter` | boxes | `classes`, `min_score`, `min_area` | the boxes it keeps | - |
| `zone` | boxes | `polygon` (required), `mode` (`inside`/`outside`), `anchor` (`bottom`/`center`) | the boxes in the selected region | `enter`, `exit` (tracked boxes only) |
| `line` | tracked boxes | `line` (required), `direction` (`both`/`in`/`out`) | counts `in`, `out`, `total` | `cross` |
| `count` | 1 to 8 box inputs | `classes` | a total and a count per class | - |
| `dwell` | tracked boxes | `threshold_s` (default 10), `grace_s` (default 1) | seconds per track | `dwell` |
| `condition` | boxes, records, or per-crop scores or vectors | `value`, `comparator`, `threshold`, `reduce`, `cooldown_s`, `event` | `value`, `active` | its `event` |
| `gallery` | embeddings | `gallery` (required), `threshold` (default 0.5), `events` (`known`/`all`/`none`, default `known`) | a name per crop, or `unknown` | `match`; `unknown` too with `"events": "all"` |
| `user` | any one input | `name`, `input`, `output` | its input, unchanged | - |

- **Coordinates** are fractions of the frame, 0 to 1, so one graph fits any resolution: `"polygon": [x0, y0, x1, y1, ...]` (at least 3 points), `"line": [x0, y0, x1, y1]`.  
- **Track ids** come from a detector's `"track"`. `line` and `dwell` need them, and `zone` raises `enter`/`exit` only for tracked boxes; `--check` names the detector to add `"track"` to.  
- **`condition`** compares one value (a box count, or a number from its input) with `threshold`, and raises its event when the comparison becomes true. With `cooldown_s` greater than 0 it raises it again at most every `cooldown_s` seconds while the comparison holds.  
- **`gallery`** compares embeddings with a gallery file built once from a folder of images: `python3 scripts/build_gallery_database.py --variant <model> --images <folder> --out <file>.bin`.  
- **`user`** passes its input through in a run. In an exported project (see *Standalone Project*) it becomes a C++ function you write.  
- `--list-ops` lists every logic and cpu op with its inputs, its output and whether it needs track ids; `--list-ops --json` adds each op's events and parameters (types, defaults, ranges).  

`zone_intrusion.json` raises an `intrusion` event when a person stands in the lower-right part of the frame:  

```json
{
  "version": 1,
  "name": "zone-intrusion",
  "nodes": [
    { "id": "cam",    "type": "source", "uri": "sample/img/sample_people.jpg" },
    { "id": "od",     "model": "yolov8-n_640x640",
      "params": { "score_threshold": 0.35 },
      "track":  { "algo": "iou", "iou": 0.3, "max_age": 30 } },
    { "id": "person", "type": "logic", "op": "filter", "params": { "classes": ["person"] } },
    { "id": "zone",   "type": "logic", "op": "zone",
      "params": { "polygon": [0.45, 0.35, 1.0, 0.35, 1.0, 1.0, 0.45, 1.0] } },
    { "id": "alarm",  "type": "logic", "op": "condition",
      "params": { "value": "count", "comparator": ">", "threshold": 0,
                  "cooldown_s": 10, "event": "intrusion" } }
  ],
  "edges": [
    { "from": "cam",    "to": "od" },
    { "from": "od",     "to": "person", "carry": "result" },
    { "from": "person", "to": "zone",   "carry": "result" },
    { "from": "zone",   "to": "alarm",  "carry": "result" }
  ]
}
```

**Events** are printed on stdout after the frame's line, written one JSON object per line to `--events <file>`, and included in `--report`:  

```text
frame 0: od boxes=3 person boxes=2 zone boxes=1 alarm records=1 value=1 active=1
[EVENT] {"frame":0,"time":0,"stream":"cam","node":"zone","event":"enter","track_id":1,"data":{"count":1}}
[EVENT] {"frame":0,"time":0,"stream":"cam","node":"alarm","event":"intrusion","track_id":1,"data":{"value":1,"threshold":0,"track_ids":"1"}}
```

On the picture, zones are tinted and outlined, lines have an arrow toward their `in` side, counts and conditions appear in a panel at the top left, and the last five events are listed in a corner: bottom left, or another corner when that one would cover labels.  

!!! note "NOTE"  

    A line has no hysteresis: a box whose anchor moves back and forth across the line counts a crossing each time. Place a line where people walk through, not where they stand.  

---

## Several Sources

A graph may have several source nodes. **Each source is its own stream**: the models are shared (one engine each, loaded once), but every stream has its own frames, tracks, logic state and output.  

```bash
./bin/multi_model_graph_async \
    --graph src/cpp_example/multi_model_graph/multistream_od_two_sources.json \
    --input cam1=camera:0 --input cam2=assets/videos/dance-group.mov \
    --output run.mkv --frames 300
```

- `--input <source_id>=<uri>` replaces one source's `uri`. A graph with several sources needs the `<source_id>=` form.  
- `--output run.mkv` writes one video per stream: `run_cam1.mkv`, `run_cam2.mkv`.  
- `--display` shows one window with a grid of the streams, each labelled with its source id.  
- `--frames N` stops after N frames of each stream.  
- With several sources, every frame in `--report` and every event names its `stream`.  

---

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--graph <file>` | - | The graph to run |
| `--input [<source>=]<uri>` | each source's `uri` | An image, a video, `camera:<N>` or `rtsp://...`. Repeat once per source |
| `--output <file>` | none | The rendered result. `.mp4`, `.avi` or `.mkv`: a video. Other extensions: images (`<stem>_<index><ext>` for each frame of a video). A camera or RTSP source with image output needs `--frames N` |
| `--display` | off | Show each frame in a window; `q` or ESC stops the run. Needs a desktop session |
| `--draw <id,...>` | every node | Draw only these nodes; every node still runs and reports |
| `--report <file>` | none | Every frame's results as JSON |
| `--report-timing` | off | With `--report`: add the time per node to each frame |
| `--events <file>` | none | Every event of the logic nodes, one JSON object per line |
| `--output-layers <dir>` | none | Per frame, the clean frame and each node's drawing as separate transparent PNGs, with `layers.json`. A video, camera or RTSP source needs `--frames N` |
| `--model-dir <dir>` | `assets/models` | Where the `.dxnn` files are |
| `--frames <N>` | all | Stop after N frames (of each stream) |
| `--loop` | off | Restart a video at its end, until Ctrl-C, `q`/ESC or `--frames N` |
| `--max-inflight <N>` | `16` | Frames in flight at once (`multi_model_graph_async` only) |
| `--max-jobs-per-stage <N>` | `0` (no limit) | Jobs one stage may have outstanding across frames (`multi_model_graph_async` only) |
| `--stall-timeout-ms <N>` | `0` (wait forever) | Fail when work is outstanding and nothing completes for N ms (`multi_model_graph_async` only) |
| `--check [<file>]` | - | Validate the graph and list its models without opening the NPU, then exit |
| `--list-models` | - | List the models; filter with `--consumes <roi, frame or either>` and `--produces <shape>` |
| `--list-ops` | - | List the logic and cpu ops |
| `--json` | off | With `--check`, `--list-models` or `--list-ops`: print JSON instead of a table |
| `--help` | - | Show usage |

**Exit codes**: `0` success, `1` the graph or its models cannot run, or a frame failed, `2` a usage error.  

Ctrl-C (and SIGTERM) stops gracefully: the frames in flight finish and `--report` and `--output` are finalized. A second Ctrl-C more than 200 ms later ends the process at once. If a run may have to be killed, write `.mkv` or `.avi` rather than `.mp4`, which cannot be played without its closing index.  

`DXAPP_PROGRESS=1` prints a `[PROGRESS] frames=N fps=F avg_fps=A node_ms=<id>:<ms>,...` line about once a second.  

---

## Checking and Troubleshooting

**Validate before running.** `--check` parses and validates the graph, lists each node with its shapes and its `.dxnn` file, and names any file that is missing, without opening the NPU. A missing file does not make `--check` fail (a graph can be checked before its models are downloaded), but a file that cannot be loaded does: a file that is not a `.dxnn`, or a v9 file on DX-RT older than 3.5.0. Whether all the models fit in the NPU's memory shows only when they load.  

**Find a model for a slot.** `--list-models` answers which model can go where:  

```bash
./bin/multi_model_graph_async --list-models --consumes roi --produces scores
```

```text
model                             task                  produces   consumes   input      published file     ready
--------------------------------------------------------------------------------------------------------------
faceattr_resnetv1-18_218x178      face_attribute        scores     either     178x218    yes       v9       yes
alexnet_224x224                   image_classification  scores     either     224x224    yes       v9       yes
...
```

(Columns narrowed to fit the page.) `file` shows whether the model is downloaded (`v8`, `v9`, `missing`), and `ready` says whether the graph engine can run it.  

**Error messages** name where the problem is, what it is, and how to fix it:  

```text
ERROR [CODE] <where>: <what>
  -> <how>
```

| Code | When |
|------|------|
| `GRAPH_SCHEMA` | Malformed JSON, a wrong type, a missing or unknown key, or a value out of range |
| `GRAPH_EDGE` | The shapes of an edge do not fit (for example `"seg" produces labelmap, not boxes` on an ROI edge), or a logic node's inputs are wrong |
| `GRAPH_ALIGN` | `"align": "face5"` from a producer without facial landmarks |
| `GRAPH_CYCLE`, `GRAPH_ORPHAN` | The graph has a cycle, or a node no source reaches |
| `GRAPH_VERSION`, `GRAPH_RESERVED` | An unsupported `version`, or something reserved for a later release (the `"prompt"` key, `"type": "fuse"`) |
| `MODEL_UNKNOWN` | The model name is not registered; the message suggests the closest name |
| `MODEL_NOT_READY` | The model is registered but cannot run in a graph (the anomaly-detection models in this release) |
| `MODEL_MISSING` | The `.dxnn` file is not in `--model-dir`; the message gives the `download_models.py` command |
| `MODEL_LOAD` | The file is present but cannot be loaded: device memory (see *Device Memory*), device state, or a v9 file on DX-RT older than 3.5.0 |

**A run that finds nothing says why.** When an ROI edge produces no crops although its producer found boxes (for example `"classes": ["persons"]` when the detector emits `person`), the run prints a warning with the filter and the classes the producer actually emitted.  

---

## Python: dx_graph

`dx_graph` runs the same graph files from Python. It links the same engine as the CLI, so the reports, the rendered pictures and the error messages are the CLI's own.  

```python
import dx_graph

with dx_graph.Graph("src/cpp_example/multi_model_graph/cascade_od_reid_track.json") as g:
    for report in g.stream(source="my_video.mp4"):   # like --input; pipelined like the async CLI
        d = report.to_dict()                         # this frame's --report entry
        canvas = g.render(report)                    # this frame's --output image (numpy, BGR)
```

```bash
PYTHONPATH=bin/python python3 my_script.py
```

- A graph with several sources: `g.stream(sources={"cam1": a, "cam2": b})`; each report's `.stream` names its source.  
- A model the device cannot load raises `dx_graph.GraphError` with `.code == "MODEL_LOAD"`.  
- The executor options are the constructor's `max_frames_in_flight`, `max_jobs_per_stage` and `stall_timeout_ms`.  
- Graphs are written in JSON, as for the CLI. Python cannot build or edit a graph, or add its own stages.  

The full API is described in `src/bindings/python/dx_graph/README.md`.  

---

## Standalone Project

`scripts/export_graph_project.py` writes a graph as a CMake project that builds and runs wherever the DEEPX runtime is installed, with only the code of the graph's models. It gives the same events and the same picture as `multi_model_graph_async`.  

```bash
python3 scripts/export_graph_project.py --graph my_graph.json --out ~/my_graph
cd ~/my_graph && ./setup.sh && ./run.sh      # results/result.jpg, results/events.jsonl
```

| Option | Description |
|--------|-------------|
| `--name <name>` | Project and program name (default: from the graph's `name`) |
| `--no-models` | Leave the `.dxnn` files out; the project's `setup.sh` downloads them |
| `--model-dir <dir>` | Where the `.dxnn` files are (default: `assets/models`) |
| `--dry-run` | Write nothing; with `--json`, print what would be exported and its size |

A `user` logic node becomes a C++ class in `user_logic/<name>.cpp` with a `Run()` stub that you edit and rebuild (`cmake --build build`).  

---

## Limitations in This Release

| Item | Status |
|------|--------|
| Anomaly detection | The four anomaly-detection models are registered but cannot run in a graph (`MODEL_NOT_READY`) |
| 3D / LiDAR models | Rejected in every graph: every source is a camera frame |
| Text-conditioned models (`"prompt"`) and model fusion (`"type": "fuse"`) | Reserved, rejected with `GRAPH_RESERVED` |
| CLIP prompt scores | Not a graph node: a CLIP image model outputs its image embedding |
| SuperPoint tracks | Not in a graph; the `descriptors` output is available |
| DnCNN image hand-off | The denoised image stays at 512×512, the model's resolution |
| `handoff_sr_od_cls.json`, `chain_sr_od_pose.json` | Not part of the DX-M1 sample runs (device-memory limit) |
| Windows, aarch64 | Not built or tested on Windows; aarch64 is compile-checked only |
