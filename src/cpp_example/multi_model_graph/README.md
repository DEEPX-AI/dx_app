# Multi-model graph example

Run several models over one video stream — detection plus segmentation plus
pose, or a detector feeding a re-identification model on each person it finds
— by writing a **node-graph JSON file**. Which models run, how they are
wired, which regions are cut out and handed on: all of it is data. There is
no C++ to edit and nothing to rebuild.

```bash
./bin/multi_model_graph_async \
    --graph src/cpp_example/multi_model_graph/fanout_od_seg_pose.json \
    --output result.png
```

That command runs a detector, a semantic segmenter and a pose model over one
image and composites all three onto one picture. To make it run a different
detector, change one string:

```diff
-    { "id": "od",   "model": "yolov8-n_640x640" },
+    { "id": "od",   "model": "yolo26-s_640x640" },
```

```bash
python3 scripts/download_models.py --models yolo26-s_640x640   # fetch the .dxnn, once (see "Model files" below)
./bin/multi_model_graph_async --graph .../fanout_od_seg_pose.json --output result.png
```

No `cmake`, no `ninja`, no recompilation — the same binary, reading a
different word out of a JSON file. The same edit works for the segmenter,
the pose model, and every second-stage model in the cascade samples: any of
the 492 models that [`docs/graph_models.md`](../../../docs/graph_models.md)
counts as running in a graph, as long as its `produces` and `consumes`
columns fit the slot.
`--check` tells you before you run whether they fit.

**Model names are variants**: the `.dxnn` file stem, as in
`yolov8-n_640x640`. A graph may still name a model by its old registry name
(`yolov8n`) or an `alias_of` name; `--check` resolves it and says so on one
line per node (`note: node "od": "yolov8n" is the old name of
"yolov8-n_640x640"`), a run (and `dx_graph.Graph`) prints the same line on
stderr, and the table at the end of `docs/graph_models.md` lists every old
name. The shipped samples use variant names.

**Model files**: `python3 scripts/download_models.py --models <name...>` fetches
models into `assets/models` (`--output <dir>` elsewhere), and `--all` fetches
all 497 registered models: 493
from the model zoo's current release, `q-lite-dxnn/2_5_0`, and 4 that 2_5_0
does not publish from `2_4_0` (`beit-l-p16_384x384`, `levit-128s_224x224`,
`vit-b-p16_384x384`, `vit-t-p16_224x224`).
`SCRFD500M_PPU` and `YOLOV5Pose_PPU` are not registered. The
2_5_0 files are `.dxnn` container **v9**; this release needs DX-RT 3.5.0 or
later, the first runtime that loads them. Every graph path checks the container before it
creates an engine, so on an older DX-RT it stops with

```text
<file>: .dxnn container v9 needs DX-RT >= 3.5.0, but this runtime is 3.4.1. Use the v8 file (dxnn/2_4_0) or upgrade DX-RT.
```

There `--check` prints that file as `[present, v9 (needs DX-RT >= 3.5.0)]`,
adds an `ERROR [MODEL_LOAD]` line and exits 1; a file this runtime loads is
`[present, v8]` or `[present, v9]`. `--list-models` shows the same container
version in its `file` column. The graphs in this release were validated on
DX-RT 3.4.1 with v8 files and on DX-RT 3.5.0 (driver 2.7.0) with both the v8
and the v9 files: the golden reports of the 9 sample graphs that have one are
byte-identical between the two binaries on either store, and the v8 ones
equal the DX-RT 3.4.1 results.

Two binaries are built from one engine and differ in exactly one value:

| Binary | Executor |
|---|---|
| `multi_model_graph_sync` | one stage at a time, topological order. The reference: slower, and the first thing to try when a result looks wrong |
| `multi_model_graph_async` | dispatch-on-ready. Every node whose inputs are ready is submitted before any of them is waited on |

Both produce the same `FrameReport` for the same input — `--report` writes
it as JSON, and `tests/cpp_example/test_graph_cli.py` compares the two files
byte for byte on real hardware for **all twenty-two shipped sample graphs**
(a graph whose model file is missing is skipped), covering
every payload shape they produce — `frame`, `boxes`, `obboxes`,
`keypoints`, `labelmap`, `densemap`, `image`, `scores`, `vector` and
`records`. The C++
`FrameReport::operator==`, used by the unit suite, compares every shape
exactly too, `obboxes` included.

Shipped samples, next to the CLI sources in `src/cpp_example/multi_model_graph/`:

| File | Shape | Models |
|---|---|---|
| `fanout_od_seg_pose.json` | one frame, three models in parallel | `yolov8-n_640x640` + `bisenetv2_1024x2048` + `yolov8-s-pose_640x640` |
| `fanout_od_seg_depth.json` | same, with depth instead of pose | `yolov8-n_640x640` + `bisenetv2_1024x2048` + `fastdepth_224x224` |
| `cascade_od_reid_track.json` | detector → per-person crop → embedding, with tracking | `yolov8-n_640x640` → `casvit-t_224x224` |
| `cascade_od_attr.json` | detector → per-person crop → attributes | `yolov8-n_640x640` → `deepmar_resnet50_224x224` |
| `cascade_obb_cls.json` | oriented detector → un-rotated crop → classifier | `yolo26-l-obb_1024x1024` → `resnet50_224x224` |
| `handoff_denoise_od.json` | denoise, then detect on the denoised image (image hand-off on a plain edge) | `dncnn-color_512x512` → `yolov5-n_640x640` |
| `handoff_sr_od_cls.json` | super-resolve ×2, then detect on the large image → per-person crop → classifier. Not run on DX-M1: the three models exceed NPU weight memory | `realesrgan-x2_192x192` → `yolov8-n_640x640` → `resnet50_224x224` |
| `multistream_od_two_sources.json` | two sources, each its own stream, through one shared tracked detector and one reid | `yolov8-n_640x640` → `casvit-t_224x224` |
| `chain_sr_od_pose.json` | super-resolve ×2, then detect on the large image → per-person crop → top-down pose. Not run on DX-M1: the three models exceed NPU weight memory | `realesrgan-x2_192x192` → `yolov8-n_640x640` → `vitpose-s_256x192` |
| `chain_zerodce_od_pose_emb.json` | low-light enhancement, then detect on the enhanced image → per-person crop → top-down pose, and an embedding per person | `zerodce_400x600` → `yolov8-n_640x640` → `vitpose-s_256x192` + `casvit-t_224x224` |
| `hand_cascade.json` | palm detector → per-palm crop → 21 hand landmarks | `mediapipe-hand-detector_192x192` → `mediapipe-hands-lite_224x224` |
| `dms_headpose.json` | face detector → head pose (pitch, yaw, roll) on a cpu node | `scrfd-500m_640x640` → `headpose` |
| `logistics_volume.json` | detector, instance mask and depth in parallel → package volume proxy on a cpu node (mask area × median depth, in source pixels; the depth map is normalized per frame, so the proxy is relative) | `yolo26-n_640x640` + `yolo26-n-seg_640x640` + `yolo26-depth-n_768x768` → `volume` |
| `zone_intrusion.json` | people in a restricted zone raise an `intrusion` event (filter → zone → condition, cooldown 10 s) | `yolov8-n_640x640` → logic |
| `line_counting.json` | tracked people crossing a line, counted per direction, a `cross` event each | `yolov8-n_640x640` → logic |
| `ppe_zone_check.json` | people in a zone are cropped for attributes; one without a hat raises `no_hat` (zone → ROI → classifier → condition) | `yolov8-n_640x640` → `deepmar_resnet18_224x224` → logic |
| `loitering_dwell.json` | how long each tracked person stays in a zone; a `dwell` event after 5 s | `yolov8-n_640x640` → logic |
| `zone_people_count.json` | people counted in two zones | `yolov8-n_640x640` → logic |
| `gallery_known_person.json` | each person's ReID embedding named after the nearest gallery entry, or `unknown` | `yolov8-n_640x640` → `repvgg-a0-reid_256x128` → logic |
| `driver_attention.json` | the driver's head pose decides: `looking_away` when the yaw passes ±30°, `head_down` when the pitch drops below −20° (conditions on a cpu node's numbers, cooldown 3 s; with several faces, any face's) | `scrfd-500m_640x640` → `headpose` → logic |
| `multistream_people_count.json` | two sources through one detector; people counted per stream, a `crowded` event in a stream with 5 or more | `yolov8-n_640x640` → logic |
| `logistics_volume_alert.json` | the packages' summed `volume_proxy` raises `load_high` above 100 (cooldown 10 s); relative, so tune the threshold per camera | `yolo26-n_640x640` + `yolo26-n-seg_640x640` + `yolo26-depth-n_768x768` → `volume` → logic |

`hand_cascade.json`, `dms_headpose.json` and `logistics_volume.json` are
shipped scenarios. The hand cascade is a crop chain. Head pose and volume
are cpu nodes.

The hand cascade's ROI edge has `"pad": 0.15`. The palm detector's box is
the palm grown 2× toward the fingers, which can cut off the thumb tip;
MediaPipe crops 2.6× the palm for its landmark model, and `pad` 0.15 per
side makes the 2× box 2.6×. Without it, the thumb tip on `sample_hand.jpg`
lands about 230 px from where the landmark model puts it on the whole
frame; with it, every point is within 15 px. The crop is not rotated:
MediaPipe also turns the crop so that the hand points up, and the graph has
no ROI option for that, so a hand held far from upright gets less accurate
landmarks. CLIP prompt scores are not a graph node (see *Not in this
release*).

The last nine are decision graphs (see *Logic nodes*). On their sample
images they show their decision (`driver_attention.json`'s driver looks
ahead, so it raises no event); their scenes are videos, for example
`--input assets/videos/person-pair-hallway.mp4` (two people walking a
hallway) with `--loop` for a Live run.

An ROI crop uses one rule, `PaddedCropRect` in `common/utility/roi_crop.hpp`:
pad the box around its centre, clamp it to the frame in float, then truncate
the corner and the size once.

---

## The whole vocabulary: four node kinds, three edge kinds

**Nodes**

| Kind | Written as | Meaning |
|---|---|---|
| source | `{ "id": "cam", "type": "source", "uri": "..." }` | where frames come from: an image, a video file, `camera:0`, or an `rtsp://` URL. A relative `uri` is resolved against the repository root, so the samples run from any directory. A graph may have several source nodes; each one is its own stream (see *Several sources: one stream each*) |
| model | `{ "id": "od", "model": "yolov8-n_640x640" }` | one model from the registry — note there is **no `"type"`** on a model node. `--list-models` and [`docs/graph_models.md`](../../../docs/graph_models.md) say which names exist. Optional `"engine": {"buffers": N, "share": true}`: NPU memory (see *Device memory*) |
| cpu | `{ "id": "headpose", "type": "cpu", "op": "headpose" }` | one named reduction, run once per frame after its parents finish. `"op"` is `headpose` or `volume`. No `"model"`, no `"track"`. `headpose` takes no params; it writes one record per face with five landmarks (`pitch`, `yaw`, `roll` in degrees, `face_index` = the face's index in its boxes input, and `track_id` when the face is tracked) and draws each face's axes from the nose (x red, y green, z blue: where the face points) and its yaw, pitch and roll under the face box, as a layer of its own. A condition reads several faces through its `reduce`: `max` (the default) for a `>` test fires when any face passes, `min` for a `<` test. `volume` accepts `scale_factor` (number, default 1), `classes` and `class_ids` (string arrays; omit them for the defaults, pass `[]` to disable that match) |
| logic | `{ "id": "zone", "type": "logic", "op": "zone", "params": { "polygon": [0.5, 0, 1, 0, 1, 1, 0.5, 1] } }` | a decision around the models, run once per frame on the CPU after its inputs: `filter`, `zone`, `line`, `count`, `dwell`, `condition`, `gallery` or `user` (see *Logic nodes*). No `"model"`, no `"track"`, no `"engine"` |

**Edges**

| Kind | Written as | Meaning |
|---|---|---|
| frame | `{ "from": "cam", "to": "od" }` | the consumer gets the whole frame: the source frame, or the image the producer made (see below) |
| ROI | `{ "from": "od", "to": "reid", "roi": { ... } }` | the consumer gets one cropped region per box the producer found, and runs once per crop |
| result | `{ "from": "face", "to": "headpose", "carry": "result" }` | the consumer, which must be a cpu or a logic node, receives the producer payload. Not a crop and not a frame. `headpose` needs one `boxes` result. `volume` needs one `boxes`, one `instances` and one `densemap`; it works in source-frame pixels: boxes are mapped back through their origin, and the instance mask and the depth map may differ in size (each covers the frame, as the overlay draws them; the depth is read at each mask pixel's place, never resized). A node fed by an ROI edge hands a logic node every crop's result (see *Logic nodes*) |
| `port` | `{ "from": "drive", "to": "x", "port": "lane" }` | reads one named output of the producer; without it, the primary output (this example is refused today: no shipped port has an edge-carryable shape; see *Output ports*). A result edge cannot also name a port |

A model node accepts a frame edge only if it `consumes` `frame` or `either`,
and a ROI edge only if it `consumes` `roi` or `either`. A ROI edge's producer
must emit `boxes`, `obboxes` or `instances`. Both columns are in
`docs/graph_models.md`.

The `consumes` column is each model's **input contract**. It comes from the
model's factory interface by default, and a model family may override it
(see *A model's graph contract*). `vitpose-s_256x192` does: it
`consumes` `either`, so `yolov8-n_640x640 → roi person → vitpose-s_256x192` is a
top-down pose cascade, one pose per person crop, and it still runs on a
whole frame as its single-model example does.

Structural rules that will reject an otherwise sensible-looking graph:

* **A plain edge carries the source frame or the producer's image.** From a
  source it carries the source frame. From a model node that produces
  `image` — denoise, low-light enhancement, super-resolution
  (`--list-models --produces image`) — it hands that model's output image
  to the consumer, which runs on the image as produced (it is never resized
  back to the source size). "Denoise, then detect" is `cam → denoise → od`
  (`handoff_denoise_od.json`); "super-resolve, then detect and classify" is
  `cam → sr → od`, then an ROI edge `od → cls` (`handoff_sr_od_cls.json`).
  Everything else is **rejected** at `--check`, not quietly run on the raw
  frame:
  * a plain edge from a producer of anything but `image` —
    `"seg" produces labelmap, and only an image can be handed off on a plain edge`.
    A `boxes`/`obboxes`/`instances` producer keeps its own message
    (`... "od"'s output would be discarded`) and is told to add `"roi": {}`;
  * a plain edge out of a node that runs per crop (it has an ROI in-edge) —
    `"sr" runs once per crop, so a plain edge would hand off one image per crop`;
  * more than one full-frame image into one node in one stream —
    `receives full-frame input from more than one image: "cam", "denoise"`
    (with several sources the message names the stream:
    `... more than one image in stream "cam1": "sr", "cam1"`).
* **Two sources feeding one node are two streams**, not two inputs:
  `cam1 → od` and `cam2 → od` run `od` once per frame of each;
  `cam1 → sr → od` plus `cam2 → od` runs `od` on `sr`'s image in stream
  `cam1` and on `cam2`'s frame in stream `cam2`. A node is full-frame or
  per-crop in every stream: the frame-and-crops rule below still applies to
  the node as a whole.
* **A node may not receive both a full frame and ROI crops.** Split it into
  two nodes — one for each input. A hand-off edge counts as a full frame.
* **Two edges may not share the same `(from, to)` pair.** To feed one
  producer's output into a consumer twice, add a second consumer node.

### Coordinates after a hand-off

A node's results are in the coordinates of the image it ran on. After a
hand-off that is the producer's image, not the source frame, so every result
carries `origin.inv_align` (emitted as `inv_align` in the `--report` JSON):
the 2×3 affine that maps its coordinates back to the source frame. After ×2
super-resolution a detector reports `inv_align` = `[0.5, 0, 0, 0, 0.5, 0]`;
after `dncnn-color_512x512` (512×512 out) on a 640×640 frame, `[1.25, 0, 0, 0,
1.25, 0]`. Chains compose. The renderer applies it, so overlays land on the
source frame.

The size of the handed-off image depends on the model family, because each
family's postprocessor decides it (the image is used as produced):

| Family | Models | Handed-off image |
|---|---|---|
| Super-resolution (Real-ESRGAN) | `realesrgan-x2_192x192`, `realesrgan-x4_192x192`, `realesrgan-x8_192x192` | the input size × the factor: 640×480 → 1280×960 at ×2. Aspect ratio kept. |
| Super-resolution (ESPCN) | `espcn-x2_17x17`, `espcn-x3_17x17`, `espcn-x4_17x17` | the input size × the factor (275×150 → 550×300 at ×2): tiled exactly like `espcn-x2_17x17_sync` (17×17 tiles, halo 4; `"params": {"sr_tile_halo": N}` overrides it). Aspect ratio kept. |
| Denoising (DnCNN) | `dncnn-*_512x512` | always 512×512, the model resolution, whatever the input size. Aspect ratio not kept: a 640×480 frame gives `[1.25, 0, 0, 0, 0.9375, 0]`. |
| Low-light enhancement (Zero-DCE) | `zerodce_400x600`, `zerodce-pp_400x600` | resized back to the input size, so the scale is 1 and `inv_align` is the identity. |

ESPCN in a graph uses the runner's own tiling code, so its `sr` image equals
`espcn-x2_17x17_sync`'s `_output_only` image pixel for pixel. The tile is the
model's input (17×17). The halo is taken from the first of these that is
set: the node's `"sr_tile_halo"`, the model's `config.json` `sr_tile_halo`,
the environment variable `DXAPP_SR_TILE_HALO`, then 4. A bad halo fails
when the model loads, as `MODEL_LOAD`: `"sr_tile_halo" must be a whole
number, got 2.5`, or `SR tile halo 9 exceeds the maximum useful overlap`.
DnCNN is not tiled and still hands off 512×512 (see *Not in this release*).

"The input size" is the size of the image the model itself ran on - the
source frame, or the previous image in a chain. A consumer's detectors run
on the handed-off image at that size, so their boxes, and `roi.min_area`
below, are in that image's pixels. Coordinates still map back to the
source frame exactly, through `inv_align`, whatever the aspect ratio.

ROI crops after a hand-off are cut from the **handed-off image**, so the
detail super-resolution added reaches the crop. Each crop's `src_box` and
`inv_align` still map to the source frame. Two `roi` options read
differently there:

* `min_area` is measured in the pixels of the image the producer ran on.
  After ×2 super-resolution a threshold of 100 px² equals 25 source px².
* `pad` is a ratio of the box size, so the scale does not change it.

How a crop's *dense* results (mask, label map, depth, an `image` result, an
instance's mask) land on the source canvas depends on the crop:

* **Plain and scaled crops** - including every crop cut from a
  super-resolved or denoised image - are placed as exact rectangles: the
  crop's own rectangle mapped through `inv_align`.
* **Rotated or aligned crops** - OBB (`obboxes`) and `align: "face5"`,
  including after a hand-off - are warped through `inv_align` at source
  scale, and colour lands only under the warped footprint of the crop, never
  on the bounding rectangle around it.

Boxes and keypoints always go through `inv_align`. An `instances` result is
drawn with its box and label on top of its mask, as a `boxes` result is.

### Output ports

Most models have one output, their `produces` shape. Twelve also have named
extra outputs, called **ports**. `docs/graph_models.md` shows them after a
`+` in the `produces` column (`boxes + drivable, lane (labelmap)`), and
`--list-models` ends their rows with `  ports: <name>=<shape> ...`:

| Model | Port | Shape | What it holds |
|---|---|---|---|
| `yolopv2_384x640` | `drivable` | `labelmap` | the drivable-area mask: 0/1 per pixel, a binary mask, at the size of the image the model ran on |
| `yolopv2_384x640` | `lane` | `labelmap` | the lane-line mask, likewise 0/1 and binary |
| `superpoint_480x640` | `descriptors` | `densemap` | N×256 float32, row i = the descriptor of keypoint i across the frame's items, in order. Not spatial: no pixel meaning |
| `3ddfa-v2_mobilenetv1_120x120`, `3ddfa-v2_mobilenet-0.5_120x120` | `pose` | `vector` | yaw, pitch and roll in degrees, 3 floats per face, concatenated in item order |
| `mediapipe-hands-lite_224x224` | `handedness` | `scores` | one item per hand: `class_name` `Left`/`Right`/`Unknown`, `class_id` 0/1/-1, `confidence` = the hand-presence score |
| `ppmatting-hrnet-w48-composition_512x512`, `ppmatting-hrnet-w48-distinctions_512x512` | `alpha` | `densemap` | the soft alpha matte, 0 to 1 per pixel |
| `clip-img_resnet50_224x224_openai`, `repvgg-a0-reid_256x128`, `eigenplaces-resnet18_512x512`, `eigenplaces-resnet50_512x512`, `pp-shituv2-feature-extraction_224x224` | `matches` | `scores` | the ranking against the model's gallery file, best first: one item per match, `class_id` = its rank, `confidence` = its cosine score, `class_name` = its gallery label (its image path when unlabelled). A model has this port when its `config.json` names a gallery |

The primary output keeps its own name: `yolopv2_384x640`'s is `boxes`,
`superpoint_480x640`'s `keypoints`. It is unchanged by the ports.

**Selecting a port on an edge.** An edge takes the primary output unless it
has `"port": "<name>"`. The port is a separate key, not part of `"from"`,
because node ids may contain dots, so `"from": "drive.lane"` would be
ambiguous. Naming the primary (`"port": "boxes"` on `yolopv2_384x640`) is the same
as leaving `"port"` out. Validation:

* `"port"` must be a non-empty string (`GRAPH_SCHEMA`: `"port" must be a
  non-empty string, got 5`).
* A source has no ports: `"cam" is a source; "port" selects one of a
  model's outputs` / `-> remove "port" from this edge`.
* An unknown name lists the real ones: `"drive" has no output port "wings"`
  / `-> its outputs: boxes, drivable, lane`.
* Every shape rule applies to the selected port's shape. The message names
  the port: `"drive" port "drivable" produces labelmap, not boxes` on a ROI
  edge, and `"drive" port "lane" produces labelmap, and only an image can
  be handed off on a plain edge` on a plain one. Without `"port"`, the
  message names no port.
* `"align": "face5"` on a non-primary port is `GRAPH_ALIGN`: the port
  carries no landmarks, so crop from the primary boxes.
* One edge per `(from, to)` pair still holds, so a consumer reads one output
  of a producer.

No shipped port has a shape an edge can carry today. A ROI edge needs
`boxes`, `obboxes` or `instances` and a plain edge needs `image`, so the
example `{ "from": "drive", "to": "x", "port": "lane" }` above is refused
with the plain-edge message. Ports are read from `--report`, the render and
`dx_graph`. An edge that selects a port of a shape it can carry is routed
from that port by both executors. A stage that then leaves that port out of
its result, or fills it with another shape (a plugin that breaks its ports
contract), gives the consumer an empty result and the frame's report an
error, in both executors: `node "cls": "drive" port "people" handed off no
boxes` on a ROI edge, `node "x": "drive" handed off no image` on a plain one.

**Tracking stays on the primary boxes.** `"track"` tracks a node's primary
`boxes`. The tracked result keeps its ports, and crops cut from a
non-primary box port would be untracked (`track_id` -1).

**In `--report`**, a result with ports has a `"ports"` object next to its
`"payload"`, one entry per port in the payload format of its shape. A
result without ports has no `"ports"` key and is byte for byte what it was
before. From `yolopv2_384x640` on a 768×576 image:

```json
"ports": {
  "drivable": {"labels": {"cols": 768, "digest": "02aca4279247673d", "nonzero": 19962, "rows": 576, "type": 0}, "shape": "labelmap"},
  "lane": {"labels": {"cols": 768, "digest": "27ed4ab36bdbc998", "nonzero": 4511, "rows": 576, "type": 0}, "shape": "labelmap"}
}
```

**In the render**, each port is drawn in the pass of its shape, right after
its result's primary payload, with the same `origin`. The `yolopv2_384x640` masks
blend only their non-zero pixels at alpha 0.5, `drivable` in green and
`lane` in red, the colours of the YOLOPv2 runner. A `vector` port is not
drawn: a 3DDFA `pose` rides on the face's 68 landmarks, and a `vec[3]` label
would only cover the face box's own label (a node whose primary output is a
vector, an embedding, still prints `vec[N]`). `handedness` prints `Left: 97%`
at the hand crop. SuperPoint's descriptors are not spatial and are never
drawn, so its render is unchanged.

**Memory.** A port lives in its frame's report, so every frame in flight
(up to `--max-inflight`, default 16) holds its ports until its report is
handed out. A `dx_graph` `Report` holds them for as long as you keep it.
Per frame:

* `yolopv2_384x640`: 2 × W × H bytes for its two masks, at the size of the image it
  ran on (884,736 bytes for 768×576, about 4 MB for 1920×1080).
* `superpoint_480x640`: N × 256 × 4 bytes. With its default `top_k` of 500
  keypoints that is 512,000 bytes, about 0.5 MB.

**SuperPoint tracks are not a port.** `superpoint_480x640_sync` also matches
keypoints from frame to frame into tracks. A track needs a step applied to
every frame in frame order. Graph stages decode in completion order, and the
only frame-ordered state the engine keeps is the IoU tracker on box streams,
so tracks would need a new engine concept. The `descriptors` port gives a
`dx_graph` user what tracking needs.

---

## A complete graph file, annotated

```jsonc
{
  "version": 1,                       // 1 is the only version this build runs
  "name": "person-reid",              // free text (a string); in messages and --report

  "nodes": [
    { "id": "cam",  "type": "source",
      "uri": "sample/img/sample_people.jpg" },   // --input overrides this

    { "id": "od",   "model": "yolov8-n_640x640",
      "params": { "score_threshold": 0.35 },     // overrides the model's config.json
      "track":  { "algo": "iou",                 // one id space for every consumer
                  "iou": 0.3,                    // (0, 1]
                  "max_age": 30 } },             // a whole number, 0 or more

    { "id": "reid", "model": "casvit-t_224x224" }
  ],

  "edges": [
    { "from": "cam", "to": "od" },               // frame edge

    { "from": "od",  "to": "reid",               // ROI edge: one run per crop
      "roi": { "classes": ["person"],
               "pad": 0.05,                      // 0 to 1: 0.05 is 5 % per side
               "max": 16 } }                     // a whole number, 1 or more
  ]
}
```

(The samples are strict JSON — the `//` comments above are for this document
only. The parser does not accept comments.)

**`track` is a node attribute, not an edge option.** Two edges leaving the
same detector with independent trackers would hand the same boxes two
different id spaces, so `reid`'s track 3 and `attr`'s track 3 would be
different people. Tracking is a property of the detection stream, so it is
declared once, on the producer.

All three `track` keys are optional; `"track": {}` means the defaults.

| Key | Meaning | Default |
|---|---|---|
| `algo` | the tracker. `"iou"` is the only one; any other value is `GRAPH_SCHEMA` (`unknown tracker "sort"`) | `"iou"` |
| `iou` | the overlap a box needs to continue a track: greater than 0, at most 1 | `0.3` |
| `max_age` | how many frames an unmatched track survives: a whole number, 0 or more. `0` drops a track the first frame it goes unmatched | `30` |

A frame on which a tracked node does not run - its stage failed, or its
hand-off input was unusable, so the frame's report carries an error - does
not reach its tracker: the tracks neither match nor age on that frame. Both
executors behave the same way.

---

## `roi` options

All six are optional; `"roi": {}` means "crop every box, unchanged".

| Key | Meaning | Default |
|---|---|---|
| `classes` | keep only these class names | all classes |
| `min_score` | confidence floor, 0 to 1, as the producer reports it (`0.35`, not `35`) | the producer's own threshold |
| `min_area` | pixel-area floor, 0 or more, applied after clipping (in the pixels of the producer's input image — see *Coordinates after a hand-off*) | `0` |
| `pad` | grow each box by this fraction of its size on every side before cropping, 0 to 1 (see below) | `0` |
| `max` | keep at most this many boxes, highest score first: a whole number, 1 or more. Leave it out for no limit | no limit |
| `align` | `"face5"` — similarity-transform alignment from five facial landmarks | none |

A value outside these ranges, or a fraction where a whole number is
required, is a `GRAPH_SCHEMA` error at `--check` (see *Error messages*).
`1.0` counts as a whole number. The bound on `pad` is 1 because `pad` is
added on **each** side: `pad: 1` already makes the crop three times the box
in each direction, so the object is a ninth of the crop. Beyond that the
crop is mostly background, and a value such as `5` is almost always a
percentage typed as a whole number (5 % is `0.05`).

`align: "face5"` is rejected at validation time when the producer does not
emit landmarks, so a graph that asks a body detector for an aligned face
crop fails before the NPU is opened rather than producing quietly wrong
crops.

### Operator order (normative)

This order is part of the specification, not an implementation detail:
applying `pad` after `align` produces different crops. Copied verbatim from
`common/graph/roi_router.hpp`:

```
producer boxes
  -> classes / min_score / min_area filter
  -> max cut (descending score)
  -> [track]      when declared on the node; frame-ordered
  -> pad + clip to frame
  -> [align:face5 | un-rotate(OBB)]   affine; inverse recorded in RoiRef
  -> crop
  -> second-stage model input
```

A crop that crosses the frame boundary is clipped. If what remains falls
below `min_area` the ROI is dropped and counted in the frame report's
`skipped_out_of_bounds`.

Every second-stage result is restored to **source-frame coordinates** before
it is reported or drawn, including the inverse of an `align` or OBB
un-rotation, so overlays land where the object actually is.

---

## Parameter precedence

```
factory constructor defaults  <  <task>/<family>/<variant>/config.json  <  node "params"
```

A node's `"params"` overrides the model's `config.json` for that node only,
so the same model can appear twice in one graph with two thresholds. Each
value is a number, a string or an array of strings:

* **A number** works for any key, as before.
* **A string or an array of strings** works only for a key the model reads
  as text. Today that is `class_names`, an array, read by 187 models (169
  detectors, 17 instance segmenters and 1 zero-shot instance segmenter);
  `layout`, a string, read by 20 detectors and instance segmenters; and
  `gallery` and `title`, strings, read by the 5 retrieval, re-identification
  and place-recognition models. The kinds come
  from a scan of each factory header at build time: `get<std::string>("k")`
  is a string key, `get_string_list("k")` a list key. `--list-models` does
  not show them; the error message for a wrong value names them. The kind
  must match, too: `"class_names": "person"` is refused (`"class_names"
  must be an array of strings for model "yolov8-n_640x640", got "person"`).
* **Anything else** (`true`, `null`, an object) is refused: `"use_x" must be
  a number, a string or an array of strings, got true` / `-> write a switch
  as 1 or 0`.

`class_names` renames a model's classes for one node. It names class ids in
order, so it can make an ROI filter read naturally:

```jsonc
{ "id": "od", "model": "yolov8-n_640x640", "params": { "class_names": ["pedestrian"] } },
...
{ "from": "od", "to": "reid", "roi": { "classes": ["pedestrian"] } }
```

Class 0 (COCO's `person`) is now `pedestrian`. A class id past the end of
the list is reported as `class_<id>` (a dog is `class_16`).

A string or list entry may hold any character: the params reach the model's
config parser as JSON, escapes included, so `"class_names": ["car \"sedan\"", "a[1]"]`
arrives exactly as written. The same parser reads `config.json`.

A node that sets any params keeps every list in the model's `config.json`
that it does not name itself: overriding `score_threshold` does not wipe a
`class_names` list from the file.

---

## Logic nodes: zones, lines, counts, conditions, events

A logic node is the decision around the models: which boxes matter, where
they are, how many, for how long, and when to say so. It runs once per
frame on the CPU after the nodes it reads, and reads them only on
`"carry": "result"` edges.

| Op | Reads | Params | Hands on | Events |
|---|---|---|---|---|
| `filter` | boxes, obboxes or instances | `classes` (strings), `min_score` (0 to 1), `min_area` (pixels) | the boxes it keeps | - |
| `zone` | boxes, obboxes or instances | `polygon` (required), `mode` `inside`/`outside`, `anchor` `bottom`/`center` | the boxes whose anchor is in the selected region | `enter`, `exit` (boxes with a `track_id`) |
| `line` | tracked boxes | `line` (required), `anchor`, `direction` `both`/`in`/`out` | records: `in`, `out`, `total` | `cross` (`direction`, `in`, `out`) |
| `count` | 1 to 8 box inputs | `classes` | records, one per input: `total` and one number per class, text `node` | - |
| `dwell` | tracked boxes (usually a zone's) | `threshold_s` (> 0, default 10), `grace_s` (default 1) | records per track: `track_id`, `seconds` | `dwell` (`seconds`), once per stay |
| `condition` | boxes, records, or a per-crop node's scores or vectors | `value`, `comparator`, `threshold`, `reduce` `max`/`min`/`sum`, `cooldown_s`, `event` | records: `value`, `active` | its `event` (default: the node id), with `value`, `threshold`, `track_ids` |
| `gallery` | vectors, per crop or full frame | `gallery` (required), `threshold` (0 to 1, default 0.5), `events` `known`/`all`/`none` | records per crop: `track_id`, `score`, `parent_index`, text `name` | `match` (`name`, `score`), `unknown` |
| `user` | any one full-frame input | `name` (a C identifier, not a C++ keyword; nodes sharing a name take one kind), `input`, `output` (shape names, equal) | its input, unchanged | - |

* **Coordinates are fractions 0 to 1 of the source frame**, flat arrays:
  `"polygon": [x0, y0, x1, y1, ...]` (at least 3 points), `"line": [x0, y0,
  x1, y1]` from A to B. One graph fits any resolution; pixels are refused.
  A line's `in` is a crossing from the left of A->B to its right as seen on
  the image (y down): with A at the bottom and B at the top, walking right
  is `in`. A box is placed by its `anchor`: its bottom centre (where a
  person stands) or its centre.
* **Track ids** come from a detector's `"track"`; `filter`, `zone` and
  `user` keep them. `line` and `dwell` need them, and `--check` names the
  detector to add `"track"` to. A track not seen for 5 seconds is forgotten
  (a zone then raises `exit` for it).
* **Boxes in, boxes out**: `filter`, `zone` and `user` hand on boxes, so a
  logic node can feed another one, or an ROI edge to a model: `zone -> roi ->
  classifier` crops only the people in the zone (`ppe_zone_check.json`).
  Logic nodes read and write source-frame pixels (a detector run on a
  handed-off image is mapped back first).
* **Per crop**: a node fed by an ROI edge hands a logic node every crop's
  result, with the crop's `track_id`. `condition` reads them as `"count"`,
  `"with:<label>"` or `"without:<label>"` (the crops whose scores do or do
  not name the label); `gallery` names each crop's vector.
* **`condition`** compares one value: `"count"` for boxes or crops, a
  number its records input writes (`--check` lists them), reduced over
  several records with `reduce`. With `cooldown_s` 0 it raises its event
  once, when the condition becomes true; with more, at most every
  `cooldown_s` seconds while it holds. Names: letters, digits, `_ . : -`.
* **`gallery`** is a DXGAL1 `.bin`, built once from an image folder by
  `python3 scripts/build_gallery_database.py --variant <model> --images
  <folder> --out <file>.bin` (`--labels folder` names entries by
  sub-folder). `--check` refuses a gallery built by another model than the
  one feeding the node. A crop is named after its nearest entry (label,
  else image file stem) at or above `threshold`, else `unknown`; a track
  announces its name once, and again when it changes.
* **`user`** passes its input through in a run; an exported project
  (Studio, Phase 3) turns it into a function stub named `name`.
* **Time and state**: logic nodes count seconds per stream: a file's frame
  index over its frame rate, a camera's or RTSP stream's seconds since its
  first frame (`FrameReport::time_s`). Each logic node keeps its state
  (tracks, counts, cooldowns) per stream, and both binaries run a stream's
  frames through it in order, so their reports are byte-identical.
* **`--loop`**: when a video starts again, its stream's trackers and logic
  nodes start afresh too (new track ids from 0; no zone members, line
  crossings or counts, dwell stays, condition edge or cooldown), so every
  pass raises the events the first one did. Frame indices and stream time
  keep counting.
* **On the picture** (`--output`, `--display`): zones tinted and outlined,
  lines with an arrow to their `in` side, the boxes a zone keeps outlined,
  counts and conditions in a panel at the top left, dwell timers under each
  box, gallery names under each crop; red when a zone holds someone, a line
  was just crossed, a stay passed its threshold or a condition is true. The
  last five events of the stream are listed at the bottom left. A zone's
  label sits inside it at its top-left corner and a line's beside its
  middle, on the side away from the arrow; both, and the panel, move off
  the detector's box labels.

Behaviour worth knowing before a graph goes live:

* **A line has no hysteresis.** Each change of side across the segment is a
  crossing, so a box whose anchor jitters on the line counts an `in` and an
  `out` for every back-and-forth. Place a line where people walk through,
  not where they stand.
* **A zone in `"mode": "outside"`** selects the boxes outside the polygon,
  so a new track first seen outside raises `enter` on that first frame, and
  `exit` when it steps into the polygon.
* **An untracked `gallery` input** (no `"track"` on the detector) has no
  track to remember a name by: every crop announces its name (`match`, or
  `unknown` with `"events": "all"`) on every frame.
* **`condition` compares with a slack of 1e-6**, as stream times are: a
  value within 1e-6 of `threshold` counts as equal to it (a dwell of
  0.15 s that is 0.15 minus a rounding error is `>= 0.15`). Seconds in
  `data` and records (`seconds`, names ending in `_s`) are written to
  0.001 s, like `time`.

### Events

Every event goes, as its frame finishes:

* to **stdout**, always, right after the frame's `frame N:` line:

  ```text
  frame 0: od boxes=3 person boxes=2 zone boxes=1 alarm records=1 value=1 active=1
  [EVENT] {"frame":0,"time":0,"stream":"cam","node":"zone","event":"enter","track_id":1,"data":{"count":1}}
  [EVENT] {"frame":0,"time":0,"stream":"cam","node":"alarm","event":"intrusion","track_id":1,"data":{"value":1,"threshold":0,"track_ids":"1"}}
  ```

* to **`--events <file>`**, the same objects one per line (JSONL), each
  line one write, so a reader tailing the file never sees half a line;
* to **`--report`**, as `"events"` on the frame (a frame without events has
  no such key).

`time` is the stream time in seconds (0.001 s), `track_id` is `-1` when an
event is about no one track, `data` holds the event's numbers and text.
Events come in frame order (with several sources, in the order frames were
read) and, within a frame, in the graph's topological node order.

### `--list-ops --json`

The catalog of every op a graph can use, for a front end's palette and
inspector:

```json
[
  {"op": "zone", "type": "logic", "summary": "...",
   "inputs": {"min": 1, "max": 1, "accepts": [{"shape": "boxes", "per_crop": false}, "..."]},
   "output": "same", "tracking": false, "events": ["enter", "exit"], "numbers": [],
   "params": [{"name": "polygon", "type": "polygon", "required": true, "min_points": 3, "help": "..."},
              {"name": "mode", "type": "choice", "required": false, "default": "inside",
               "choices": ["inside", "outside"], "help": "..."}]}
]
```

The eight logic ops come first, then `headpose` and `volume` (`"type":
"cpu"`). `output` is `"same"` (the input's shape) or `"records"`. Number
params give `default`, `min` and `max` when bounded, `min_exclusive` when
the minimum itself is refused; choices give `choices`; polygons
`min_points`, lines `points`. `"<event>"` in `events` means the `event`
param names it.

Hints for an inspector, each only on the params it applies to: `step` (a
sensible increment for a number); for lists, `items` (`"string"`, or
`"integer"` for ids written as strings, `"24"`) and no `default`, because
leaving a list out is not `[]` (`when_omitted` lists what it means, for
`volume` the built-in package classes; for `filter` and `count` it means
every class); `format: "file"` with `extensions` for a path (`gallery`);
`equals` names the param a value must repeat (`user`'s `output` equals its
`input`); and `values_by_input` (condition's `value`) lists, per accepted
input, the values allowed: `["count"]` for boxes,
`["count", "with:<label>", "without:<label>"]` for per-crop scores, or
`"values_from": "numbers"` for records (a number the input op lists in its
`numbers`).

---

## Take a graph away: a standalone project

`scripts/export_graph_project.py` writes a graph as a CMake project that
builds and runs anywhere the DEEPX runtime is installed. It is this CLI with
only the factories of the graph's models, so it gives the same events and
the same overlay as `multi_model_graph_async`:

```bash
python3 scripts/export_graph_project.py --graph my_graph.json --out ~/my_graph
cd ~/my_graph && ./setup.sh && ./run.sh      # results/result.jpg, results/events.jsonl
```

| Option | Meaning |
|---|---|
| `--name <name>` | the project and program name (default: from the graph's `name`) |
| `--no-models` | leave the `.dxnn` files out; `setup.sh` downloads them (`models/models.tsv`) or copies them (`--models-from <dir>`) |
| `--model-dir <dir>` | where the `.dxnn` files are (default `assets/models`) |
| `--dry-run` | write nothing; with `--json`, print what would be exported and its size. A model file that is not installed has `"bytes": null` (the Model Zoo manifest gives addresses, not sizes); `model_bytes_unknown` counts them |
| `--json` | one JSON summary on stdout (`ok`, `models`, `user_logic`, `inputs`, `bytes`; on failure `error`, `error_key`) |

The project holds `graph.json` (its inputs and galleries copied into
`inputs/` and `data/`), `main.cpp`, `CMakeLists.txt`, the engine and the
used factories under `src/` at their dx_app paths, `models/`, `setup.sh`,
`run.sh` and a README. The program is this CLI: `./run.sh --help`,
`./run.sh --check`, `./run.sh --sync`, `--input`, `--display`, `--loop`.

A source's input is copied into the project unless it is live
(`camera:<N>` or `rtsp://...`, run with `--display`) or a video at an
`http://` or `https://` address, which the program opens as a video file
(OpenCV), by its extension. An `rtsps://` source, and an http(s) address
that is not a video file, cannot be opened by this CLI and are refused.

**`user` nodes.** In a run of this CLI a `user` node passes its input
through. In an export it becomes a class `user_logic/<name>.{hpp,cpp}` with
`Data Run(const Data& input, UserContext& ctx)`; the stub returns `input`
and its comment lists the input's fields. Edit it and rebuild
(`cmake --build build`): `ctx.Raise("event", track_id, data)` raises an
event (stdout `[EVENT]`, `--events`, `--report`), `ctx.Label("text")`
writes in the overlay's top-left panel, members keep state per stream. A
class that throws, returns another kind or raises an invalid event name
records a frame error and its input passes through.

**Where a program finds its files.** Repository-relative paths
(`src/cpp_example/<task>/<family>/<variant>/config.json`, `sample/...`,
galleries) are read against `$DXAPP_ROOT` when it is set, else against the
repository the program was built from, else, for an exported project, the
nearest folder holding `.dxapp-root` (from the program's directory up, then
from the working directory up). `run.sh` sets `DXAPP_ROOT` for the program
it starts only; a `DXAPP_ROOT` exported in a shell would point every dx_app
program there.

---

## Error messages

Every failure names **where** (the node or edge), **what** (the fact) and
**how** (the command or edit that fixes it):

```
ERROR [CODE] <where>: <what>
  -> <how>
```

There are twelve codes. `GRAPH_EDGE` has two more rows, for edge ports and
logic inputs, and `GRAPH_SCHEMA` a second row, for node params. Each example
below is real output from `--check`, except the three marked otherwise.

| Code | When | Example |
|---|---|---|
| `GRAPH_VERSION` | `"version"` is a whole number other than 1 | `ERROR [GRAPH_VERSION] g.json: graph version 2 is not supported by this build`<br>`  -> supports: 1` |
| `GRAPH_SCHEMA` | malformed JSON, a wrong type, a missing or duplicate key, an unreadable file, a value out of range or not a whole number | `ERROR [GRAPH_SCHEMA] edge "od"->"cls": duplicate edge "od" -> "cls"`<br>`  -> declare the connection once; add a second consumer node instead of a second edge`<br>`ERROR [GRAPH_SCHEMA] g.json: "version" must be a whole number, got 1.5`<br>`  -> add "version": 1` |
| `GRAPH_EDGE` | shapes or input contracts do not meet across an edge | `ERROR [GRAPH_EDGE] edge "seg"->"cls": "seg" produces labelmap, not boxes`<br>`  -> remove "roi" from this edge, or use a detector as the source` |
| `GRAPH_EDGE` | an edge's `"port"` names no output of the producer, or is on an edge from a source (see *Output ports*) | `ERROR [GRAPH_EDGE] edge "drive"->"x": "drive" has no output port "wings"`<br>`  -> its outputs: boxes, drivable, lane`<br>`ERROR [GRAPH_EDGE] edge "cam"->"drive": "cam" is a source; "port" selects one of a model's outputs`<br>`  -> remove "port" from this edge` |
| `GRAPH_EDGE` | a logic node's inputs: too few or too many, a kind its op does not read, or boxes without track ids for `line` or `dwell` (see *Logic nodes*) | `ERROR [GRAPH_EDGE] node "crossing": line needs track ids, and the boxes from "od" have none`<br>`  -> add "track": {"algo": "iou"} to node "od"` |
| `GRAPH_SCHEMA` | a `params` value the model cannot read: text for a numeric key, or the wrong kind for a text key (see *Parameter precedence*) | `ERROR [GRAPH_SCHEMA] node "od" "params": "score_threshold" must be a number, got "0.5"`<br>`  -> model "yolov8-n_640x640" reads text only for: class_names`<br>`ERROR [GRAPH_SCHEMA] node "od" "params": "class_names" must be an array of strings for model "yolov8-n_640x640", got "person"`<br>`ERROR [GRAPH_SCHEMA] node "c" "params": "label" must be a number, got "x"`<br>`  -> model "resnet50_224x224" reads only numbers` |
| `GRAPH_ALIGN` | `align: "face5"` from a producer without landmarks | `ERROR [GRAPH_ALIGN] edge "od"->"emb": "align": "face5" needs five facial landmarks, and model "yolov8-n_640x640" produces no landmarks`<br>`  -> remove "align", or use a face detector as the source` |
| `GRAPH_CYCLE` | the nodes do not form a DAG | `ERROR [GRAPH_CYCLE] graph: cycle detected among: a, b`<br>`  -> a graph must be acyclic` |
| `GRAPH_ORPHAN` | a node no source can reach, or a graph with no source | `ERROR [GRAPH_ORPHAN] node "lost": unreachable from any source`<br>`  -> connect it with an edge, or remove it` |
| `GRAPH_RESERVED` | a key held back for a later release | `ERROR [GRAPH_RESERVED] g.json node[1]: "prompt" is reserved for text-conditioned models and is not supported in this release`<br>`  -> remove "prompt"; see README.md for the models this build runs` |
| `MODEL_UNKNOWN` | the name is not in the registry | `ERROR [MODEL_UNKNOWN] node "od": unknown model "YoloV8N" - not in config/model_registry.json`<br>`  -> did you mean "yolov8-n_640x640"? run --list-models, or see docs/graph_models.md` |
| `MODEL_NO_TASK` | registered without a task *(no model this build ships is in this state; not real output)* | `ERROR [MODEL_NO_TASK] node "od": model "X" has no task registered`<br>`  -> ./scripts/add_model.sh --model X --task <task>` |
| `MODEL_NOT_READY` | registered, but the graph engine cannot run it: no factory, no postprocessor, or an interface with no graph stage (the four anomaly models in this release) | `ERROR [MODEL_NOT_READY] node "ad": model "patchcore_224x224" is registered but cannot run: anomaly detection is not graph-ready in this release`<br>`  -> see docs/graph_models.md for models usable in graphs` |
| `MODEL_MISSING` | the `.dxnn` is not in `--model-dir` | *(from a run: `--check` lists the same files and the same command without failing)*<br>`ERROR [MODEL_MISSING] graph "person-reid": 2 model files are not in ./no_models`<br>`  node "od"  model yolov8-n_640x640  ->  ./no_models/yolov8-n_640x640.dxnn`<br>`  node "reid"  model casvit-t_224x224  ->  ./no_models/casvit-t_224x224.dxnn`<br>`  -> python3 scripts/download_models.py --models YoloV8N casvit_t --output ./no_models` |
| `MODEL_LOAD` | the `.dxnn` is present but the runtime cannot load it (device memory, device state, an incompatible file). `--check` reports the cases it can see without the NPU, a file that is not a `.dxnn` and a v9 file on DX-RT older than 3.5.0 (see *Model files*), and exits 1; whether the models fit in device memory shows only when they load | *(real output on DX-M1, from a run, not from `--check`, of a graph that names its models by their old names)*<br>`ERROR [MODEL_LOAD] node "sr2": model "realesrgan_x2" (realesrgan-x2_192x192.dxnn) could not be loaded: [dxrt-exception] Invalid model exception {"failed to register task":lib/task.cpp:145:Task}`<br>`  -> the NPU's device memory may be full: this graph loaded 3 model(s) before this one, and every node loads its own copy, even of the same model, unless the nodes set "engine": {"share": true} (another process on the NPU uses memory too); share an engine, lower a node's "engine": {"buffers": N} (DX-RT's default is 6), remove a node or use a smaller model - DX-M1 cannot hold a second realesrgan-x2_192x192 next to yolov8-n_640x640 and resnet50_224x224` |

`MODEL_LOAD` names the node and keeps the runtime's own reason whole. Its
hint depends on how far loading got: after other models loaded, it is the
device-memory hint above (see *Device memory*); for the first model of the
graph it is `check the NPU with dxrt-cli -s, and that <file> was compiled
for this DX-RT version; to download it again: python3 scripts/download_models.py --models <name> --output <model-dir>`.
For a `.dxnn` container v9 on DX-RT older than 3.5.0 (see *Model files*) it
is `use the v8 file (dxnn/2_4_0) or upgrade DX-RT to >= 3.5.0` instead:
downloading again would fetch the same v9 file.

Note the download command in `MODEL_MISSING`: the name it prints is the
**model zoo's** spelling, which is almost never the variant (`yolov8-n_640x640`
is `YoloV8N` there, `yolo26-l-obb_1024x1024` is `yolo26l-obb`): 488 of the
497 registered models differ this way. `download_models.py --models` accepts the zoo's
name, the `.dxnn` file name with or without `.dxnn` (which is the variant),
or the old registry name, in any letter case, but the zoo's name is the
manifest's own spelling, so that is the one the CLI prints. It is compiled into
the binary from `scripts/modelzoo_manifest.json` at build time, so a binary
deployed without a source tree prints the same name. Every registered model is
published, so a missing file is always offered that command. A future row with
`published` false is left out of the command and gets its own line:
`is not published by the model zoo yet; download_models.py cannot download it`.

The command is `scripts/download_models.py`, never `./setup.sh --models`:
`setup.sh` also fetches the sample videos, and under its default `--force`
`scripts/get_resource.sh` deletes an `assets/videos` symlink (`rm -rf`) to
recreate it. `download_models.py` touches only the files it is asked for and
overwrites them by default, so the same line repairs a file that is there but
broken. `--output` is the `--model-dir` the graph was checked against.

**Every bad value in one object is reported at once**, as unknown keys are
(below): one `roi`, `track` or `params` object with three bad values gives
one message with all three, and one recovery line that covers each:

```console
$ multi_model_graph_sync --check g.json
ERROR [GRAPH_SCHEMA] g.json edge[1] "od"->"reid": 3 invalid values: "min_score" must be between 0 and 1, got 35; "pad" must be between 0 and 1, got 5; "max" must be a whole number of at least 1, got 0
  -> "min_score" is a confidence as the producer reports it (0.35, not 35); "pad" grows each box by this fraction of its size on every side (0.05 = 5 %); "max" keeps at most this many boxes; leave it out for no limit
```

A bad `"version"` is reported alone, because the version decides how the
rest of the file is read, and unknown keys are reported before bad values,
so a mistyped key is answered as a typo.

---

## Finding out what a graph needs: `--check`

`--check` parses and validates a graph, lists every node with its shape and
its artifact, and names any `.dxnn` that is absent — **without opening the
NPU**. It is the fast loop while writing a graph, and it matters more here
than it did for a single-model example, because one graph now names three to
five models.

```console
$ ./bin/multi_model_graph_sync --check src/cpp_example/multi_model_graph/cascade_od_reid_track.json
OK: graph "person-reid" is valid (3 nodes, 2 edges, 2 models)

node      model                         produces   consumes   artifact
cam       (source)                      frame      -          sample/img/sample_people.jpg
od        yolov8-n_640x640              boxes      full_frame yolov8-n_640x640.dxnn  [present, v9]
reid      casvit-t_224x224              vector     either     casvit-t_224x224.dxnn  [present, v9]
```

A graph with several sources gets a `streams:` block after the node table,
naming the nodes each stream runs (a one-source graph prints none):

```console
$ ./bin/multi_model_graph_sync --check src/cpp_example/multi_model_graph/multistream_od_two_sources.json
OK: graph "two-cameras-reid" is valid (4 nodes, 3 edges, 2 models)

node      model                         produces   consumes   artifact
cam1      (source)                      frame      -          sample/img/sample_people.jpg
cam2      (source)                      frame      -          sample/img/sample_person_a1.jpg
od        yolov8-n_640x640              boxes      full_frame yolov8-n_640x640.dxnn  [present, v9]
reid      casvit-t_224x224              vector     either     casvit-t_224x224.dxnn  [present, v9]

streams: 2 - one per source node, read in turn, one frame from each
  cam1      -> od, reid
  cam2      -> od, reid
```

Before the table, `--check` prints a `note:` line for each node that names
a model by an old name; after it, what a node needs besides its `.dxnn`.
None of these lines changes the exit code:

```console
$ ./bin/multi_model_graph_sync --check vpr.json
...
resource: node "v": gallery sample/gallery/vpr_eigenplaces-resnet18_512x512.bin [present, DXGAL1]
note: node "c": the CLIP prompt bank is Python-only; in a graph this model outputs its image embedding, not zero-shot scores
```

* `resource: ... gallery <path>`: the gallery file a retrieval,
  re-identification or place-recognition model compares against: the
  node's `"gallery"` param when it has one, else the model's `config.json`
  entry, an absolute path as given and a relative one against the
  repository: `[present, DXGAL1]`, `[present, but not a DXGAL1 gallery]` or
  `[MISSING]`. A run resolves the same `gallery` the same way, so it opens
  the file `--check` found from any working directory.
* `note: ...`: a fact about the model in a graph, such as the CLIP note
  above, or, before the table, `note: node "od": "yolov8n" is the old name
  of "yolov8-n_640x640"`.

An anomaly model is refused before any of this (`MODEL_NOT_READY`, exit 1);
its message names the companion engines it would need.

The exit code answers *is this graph valid, and can this runtime load what
is here?* — a question a fresh checkout with nothing downloaded must still
be able to ask — so a missing `.dxnn` is reported but does not make
`--check` fail. A real run does fail on it, with the same list and the same
recovery command. A `.dxnn` that is present but cannot load here does make `--check` fail:
a file that is not a `.dxnn` at all (empty, cut short, no `DXNN` header;
the artifact column says `[present, invalid]`, and the error says
`download it again: python3 scripts/download_models.py --models <name> --output <model-dir>`),
or one this DX-RT cannot
load: the artifact column says
`[present, v9 (needs DX-RT >= 3.5.0)]`, one line per such node follows on
stderr, and the exit code is 1 (output on DX-RT 3.4.1):

```console
$ ./bin/multi_model_graph_sync --check sr.json --model-dir v9_models
OK: graph "sr" is valid (2 nodes, 1 edges, 1 models)

node      model                         produces   consumes   artifact
cam       (source)                      frame      -          sample/img/sample_people.jpg
sr        espcn-x2_17x17                image      either     espcn-x2_17x17.dxnn  [present, v9 (needs DX-RT >= 3.5.0)]
ERROR [MODEL_LOAD] node "sr": model "espcn-x2_17x17" (espcn-x2_17x17.dxnn) cannot be loaded here: v9_models/espcn-x2_17x17.dxnn: .dxnn container v9 needs DX-RT >= 3.5.0, but this runtime is 3.4.1. Use the v8 file (dxnn/2_4_0) or upgrade DX-RT.
```

Because it never opens the NPU, `--check` cannot tell whether the device
can hold every model of the graph at once: that is found out when the
models load (`MODEL_LOAD`, see *Device memory*).

### `--check --json`

One JSON object on stdout, nothing on stderr; exit 0 when `errors` is
empty, else 1. A usage error (`--check --json` with no path, `--list-models`
together with `--check`, a bad option value) is exit 2 with the message and
the usage text on stderr and nothing on stdout. Real output, for a graph
`two-od` whose `od` (old name `yolov8n`) and `od2` share one engine and a
0-byte `yolov8-n_640x640.dxnn`, feeding a `cls` (`resnet50`, not downloaded)
with a crop edge from `od` and a plain edge from `od2`; `what` is shortened:

```console
$ ./bin/multi_model_graph_sync --check two_od.json --json --model-dir models
```

```json
{
  "ok": false,
  "graph": "two-od",
  "errors": [
    {"code": "GRAPH_EDGE", "where": "edge \"od2\"->\"cls\"", "what": "\"od2\" produces boxes, and a plain edge carries only ...",
     "how": "add \"roi\": {} to crop \"od2\"'s boxes, or feed \"cls\" from a source node",
     "node": null, "edge": {"from": "od2", "to": "cls"}},
    {"code": "MODEL_LOAD", "where": "node \"od\"", "what": "model \"yolov8n\" (yolov8-n_640x640.dxnn) is not a loadable .dxnn: ...",
     "how": "download it again: python3 scripts/download_models.py --models YoloV8N --output models",
     "node": "od", "edge": null},
    {"code": "MODEL_LOAD", "where": "node \"od2\"", "what": "model \"yolov8-n_640x640\" (yolov8-n_640x640.dxnn) is not a loadable .dxnn: ...",
     "how": "download it again: python3 scripts/download_models.py --models YoloV8N --output models",
     "node": "od2", "edge": null}
  ],
  "models": [
    {"node": "od", "model": "yolov8-n_640x640", "file": "yolov8-n_640x640.dxnn",
     "state": "invalid", "container": null,
     "engine": {"buffers": 2, "share": true, "shared_with": null}},
    {"node": "od2", "model": "yolov8-n_640x640", "file": "yolov8-n_640x640.dxnn",
     "state": "invalid", "container": null,
     "engine": {"buffers": 2, "share": true, "shared_with": "od"}},
    {"node": "cls", "model": "resnet50_224x224", "file": "resnet50_224x224.dxnn",
     "state": "missing", "container": null,
     "engine": {"buffers": 0, "share": false, "shared_with": null}}
  ],
  "notes": ["note: node \"od\": \"yolov8n\" is the old name of \"yolov8-n_640x640\"",
            "note: node \"cls\": \"resnet50\" is the old name of \"resnet50_224x224\""]
}
```

- `graph`: the graph's `"name"`, else the path given.
- `errors`: at most one parse/validation error (`ValidateGraph` stops at the first), then one `MODEL_LOAD` per model node whose present file cannot load. `node`/`edge`: validation errors are matched against the parsed graph (`node "<id>"`, `edge "<from>"->"<to>"` in `where`); parse errors by the index in `where` (`node[i]`, `edge[i]`) looked up in the raw JSON file (`from`/`to` are `null` when that key is missing). Both `null` when the error is about the whole graph (`GRAPH_CYCLE`, `GRAPH_VERSION`, no source, unreadable file).
- `models`: one entry per model node, in declaration order, when the file parsed (also when validation failed): `state` is `present`, `missing`, `invalid` (not a `.dxnn`), `unloadable` (container this DX-RT cannot load) or `unknown` (model not registered; then `file` and `container` are `null`); `container` is `"v8"`/`"v9"` or `null`; `engine.buffers` 0 means DX-RT's default; `engine.shared_with` is the owner's id for a sharing node that is not the owner, else `null`. No device-memory figure (see *Device memory*).
- `notes`: the `note:` lines `--check` prints for old names, one string each.

Object keys come out in alphabetical order; the order above is for reading.

## Finding a model: `--list-models`

```console
$ ./bin/multi_model_graph_async --list-models --consumes roi --produces scores
```

```text
model                                              task                            produces   consumes   input      published file     ready
--------------------------------------------------------------------------------------------------------------------------------------------
faceattr_resnetv1-18_218x178                       face_attribute                  scores     either     178x218    yes       missing  yes
mobilenetv2_224x224                                image_classification            scores     either     224x224    yes       v8       yes
...
```

`--consumes` takes `roi`, `frame` or `either`; `--produces` takes a shape
name (`boxes`, `obboxes`, `instances`, `keypoints`, `labelmap`, `densemap`,
`image`, `scores`, `vector`, `boxes3d`). Together they answer the question a
task name cannot: *which model can I put in the second stage?*

One row per registered model (497), named by its variant, then one
`alias of <variant>` row per `alias_of` name (the old registry names are not
listed; `docs/graph_models.md` has them). `published` is `no` for a model the
model zoo does not publish yet. `file` is the container version of the
model's `.dxnn` in `--model-dir`: `v8`, `v9`, `missing`, or `invalid` for a
file without the `DXNN` header; when this DX-RT cannot load the file it
says why, as in `v9 (needs DX-RT >= 3.5.0)`. `ready` is `no`, with the
reason, for a model the graph engine cannot run (the anomaly models in this
release). A model with extra outputs ends its row with
`  ports: <name>=<shape> ...`.

### `--list-models --json`

One JSON array on stdout and nothing else on stdout or stderr; exit 0. A bad
filter is exit 2 with the text error on stderr and nothing on stdout:

```json
[
  {"model": "yolov8-n_640x640", "task": "object_detection", "produces": "boxes",
   "consumes": "full_frame", "input": {"width": 640, "height": 640}, "published": true,
   "file": "yolov8-n_640x640.dxnn", "container": "v9", "ready": true, "reason": "",
   "ports": [], "alias_of": null},
  {"model": "yolopv2_384x640", "...": "...",
   "ports": [{"name": "drivable", "shape": "labelmap"}, {"name": "lane", "shape": "labelmap"}]},
  {"model": "deit_base384_distilled", "alias_of": "deit-b_384x384_distilled", "...": "every other field copied from the variant"}
]
```

`container` is exactly the table's `file` column (`v8`, `v9`, `v9 (needs DX-RT >= 3.5.0)`, `missing`, `invalid`). Rows come in table order: models, then `alias_of` rows. `--consumes`/`--produces`/`--model-dir` apply as in the table.

Three sources, in order of authority:

1. **`--check <graph.json>`** — what this graph needs, right now.
2. **`--list-models`** — what this binary can run. Authoritative after you
   add your own model.
3. **[`docs/graph_models.md`](../../../docs/graph_models.md)** — the same
   table, generated from the same registry. The build writes its own copy
   (`build_x86_64/generated/graph_models.md`) and never rewrites the tracked
   file; after a registry change, update it with
   `python3 scripts/gen_model_registry.py --docs-only`. The guard
   `python3 scripts/check_graph_models_doc.py` fails, naming that command,
   when the tracked file is stale.

---

## Reserved keys

The design reserves grammar for three features this release does not
implement. Two of them have a spelling in the schema and are rejected by
name with `GRAPH_RESERVED`, so a graph written against a future release
fails loudly instead of being silently misread:

| Reserved | Why it is rejected rather than ignored |
|---|---|
| `"prompt"` on a node | Text-conditioned models need a tokenizer, which this build does not have. Ignoring the key would run the model unconditioned and quietly return the wrong thing |
| `"type": "fuse"` on a node | Multi-model fusion formulas are model-specific and must be designed against a real model spec. A `fuse` node that silently became a pass-through would change results without saying so |

The third — a condition/rule expression grammar on edges — has **no key in
v1**: an expression language is a small language, permanently bound by
backward compatibility once shipped, so none was chosen.

## Unknown keys are an error

Every other key the schema does not define is rejected too, with the key
named, its location named, and — when something close exists — the key you
probably meant:

```console
$ multi_model_graph_sync --check typo.json
ERROR [GRAPH_SCHEMA] typo.json edge[1] "od"->"reid" "roi": 2 unknown keys: "classez", "paddng"
  -> did you mean "classes" for "classez", "pad" for "paddng"? accepted keys: classes, min_score, min_area, pad, max, align
```

This holds at every level — top level, node, `track`, `engine`, edge and `roi` — and
it matters more here than in most configuration formats: a dropped key does
not stop the graph, it runs it with the option missing, and nothing prompts
you to look. Three details worth knowing:

* **Every unknown key in one object is reported at once**, so fixing a file
  with several typos takes one round trip, not one per typo.
* **A mistyped *required* key is answered as a typo**, not as a missing
  key: `{"form": "cam"}` gives `unknown key "form" -> did you mean "from"?`.
* **The accepted keys are always listed**, even when nothing is close
  enough to guess — a wrong suggestion is worse than none.

A key that is genuinely absent rather than mistyped gets the location and a
recovery line of its own:

```console
ERROR [GRAPH_SCHEMA] g.json node[1] (model "yolov8-n_640x640"): missing required key "id"
  -> add "id": a name unique within this graph
```

Node `"type"` values are checked the same way. `"source"` is the only type;
a model node carries a `"model"` name and no `"type"` at all. In
particular `{"type": "model", "model": "yolov8-n_640x640"}` — a natural thing to
write — is **rejected**.

Two deliberate exceptions to the key rule:

* **`params` keys are open; values are numbers unless the model reads that
  key as text.** Its keys are model parameter names, not schema keys, so
  any key with a number is accepted and passed through to the model (see
  *Parameter precedence* for text).
* **`$schema` is accepted and ignored** at the top level, so an editor can
  attach a JSON-schema reference for completion and inline validation.
  Nothing else in this schema is within two edits of it, so tolerating it
  cannot hide a typo for a real key. The tolerance stops there — a
  `"_comment"` key is rejected; `"name"` already carries intent, and every
  extra tolerated key is somewhere a typo can hide.

And one ordering rule: **a version this build does not support wins
first.** A graph marked `"version": 2` is answered with `GRAPH_VERSION`,
not with a list of unknown keys, because a later release's keys are unknown
to this build by definition.

---

## When a run produces nothing, it says so

Some mistakes only exist at run time. A class name that matches nothing —
`"classes": ["persons"]` when the detector emits `person` — passes
`--check`, runs, exits 0, and hands the second stage zero crops. Nothing
about that is distinguishable from a frame in which the detector genuinely
found no people, so the run says which it was:

```console
warning: edge "od"->"reid" produced no crops, although "od" found 3 boxes
  the "roi" filter on this edge sets: "classes": ["persons"] "max"
  "od" emitted these classes: dog, person
  -> check the class names against that list, or relax "min_score"/"min_area"/"max"
```

The typo is printed next to the truth. The warning goes to stderr, fires
at most once per edge per run, and never changes the exit code — a
detector that legitimately finds nothing stays silent, because the warning
is raised only when the producer *did* emit boxes and the filter kept none
of them.

---

## Platforms

- **Linux x86_64** is built and tested (the unit binaries, the graph tests,
  the CLI).
- **aarch64** is compile-checked only: `scripts/check_cross_compile.sh`
  compiles the engine and the CLI with the cross compiler
  (`-fsyntax-only`); nothing is linked or run on aarch64. Concurrency and
  `--max-inflight` were validated with x86_64 dxrt.
- **Windows** is not built or tested on Windows in this release. The graph targets get MSVC
  flags (`/bigobj`, C5038 and C4062 as errors), and Ctrl-C goes through
  `SetConsoleCtrlHandler`: repeats within 200 ms are one request, a later
  press terminates, and Ctrl-Break terminates at once. `tests\windows\run_tests.bat
  graph` runs the CLI smoke tests. The Windows support is verified on Linux only; see
  *Graph engine on Windows* in `docs/source/docs/02_DX-APP_Installation_and_Build.md`.

## Not in this release

| Wanted | Status |
|---|---|
| **Multi-model fusion** (`"type": "fuse"`) | Grammar reserved, rejected with `GRAPH_RESERVED` |
| **Text-conditioned models** (`"prompt"`) | Grammar reserved, rejected with `GRAPH_RESERVED` |
| **Condition / rule expressions on edges** | Rules are logic nodes with a fixed form (*Logic nodes*), not an expression language, which would be permanently binding once shipped |
| **3D / LiDAR models** (`boxes3d`) | Rejected in every graph (`GRAPH_EDGE`: `consumes LiDAR point clouds, not camera frames`): every source is a camera frame |
| **Anomaly detection** | The four anomaly models are registered but not graph-ready (`MODEL_NOT_READY`); EfficientAD's companion engines have no graph stage |
| **CPU nodes besides `headpose` and `volume`** (CLIP prompt scores) | Not graph nodes |
| **SuperPoint tracks in a graph** | Needs an ordered per-stream step; the descriptors port is available |
| **DnCNN resized back to the input size** | Hand-off hands off as produced: DnCNN's image stays 512×512 |

---

## Several sources: one stream each

A graph may declare more than one source node
(`multistream_od_two_sources.json` has two, `cam1` and `cam2`). **Each
source is its own stream**: the source and every node reachable from it.
`multistream_people_count.json` counts people per camera: `count` counts
each stream's own frame and `condition` keeps its state (the cooldown) per
stream, so each camera has its own total and a `crowded` event names the
stream it happened in.

**Models are shared, frames are not.** A model node has one engine on the
NPU, loaded once, and that engine carries the jobs of every stream that
reaches the node: four cameras through three models are three engines, not
twelve. Everything about frames is per stream. Each stream has its own
frame sequence and its own reports, and a node that declares `track` keeps
one tracker per stream, so track ids never cross streams: track 3 in `cam1`
and track 3 in `cam2` are unrelated objects. A frame runs only the nodes of
its own stream, on that stream's frame or hand-off, and never waits for a
node that only another stream reaches. A frame that fails, fails in its own
stream; the other streams run on. (The async executor's
`node "x" never completed`, a scheduler defect that a validated graph never
produces, likewise only ever names a node of the frame's own stream.)

**Reports.** With several sources every frame in `--report` carries
`"stream": "<source_id>"`, and its `"index"` counts the frames of that
stream (0, 1, ... in each). A one-source report has no `"stream"` key.

**Order.** The streams are read in turn: one frame from each stream still
reading, in source declaration order, round after round. Both executors
emit reports in the order frames were read, so `multi_model_graph_sync` and
`multi_model_graph_async` write the same bytes, the two-source sample
included. Two videos of 8 frames (`cam1`) and 5 frames (`cam2`) give this
`(stream, index)` sequence:

```
cam1 0, cam2 0, cam1 1, cam2 1, cam1 2, cam2 2, cam1 3, cam2 3,
cam1 4, cam2 4, cam1 5, cam1 6, cam1 7
```

**Ends.** Each stream ends on its own, when its source gives no more
frames; the run ends when every stream has ended. `--frames N` is per
stream: N frames of each, and a stream that has given N is never read
again (a camera does not grab an N+1-th frame). Live and file sources mix:
`--input cam1=camera:0 --input cam2=clip.mp4 --frames 100` reads 100 camera
frames even when the clip is shorter. Because the streams are read in turn,
a file stream next to a camera is read at the camera's pace.

**Back-pressure.** `--max-inflight` is one window shared by every stream,
not a window per stream: it bounds the frames in flight across all of them.
Because frames are admitted in turn, while k streams are reading, any M
consecutive admissions hold at most ⌈M/k⌉ frames of one stream: a fast
stream cannot take the slots of a slow one. The cost is head-of-line
blocking: reports leave in admission order, so a slow stream - a slow
camera, or frames that take long to finish - holds back the others, never
the reverse.

**Inputs.** `--input <source_id>=<uri>` replaces one source's `uri`; repeat
it once per source, and a source you do not name reads its own `uri`. The
named form works on a one-source graph too (`--input cam=clip.mp4`). A bare
`--input <uri>` names the only source of a one-source graph; with several
sources it is a usage error, exit 2, before any model loads:

```console
$ ./bin/multi_model_graph_async --graph src/cpp_example/multi_model_graph/multistream_od_two_sources.json --input clip.mp4
--input "clip.mp4" does not say which source it replaces: graph "two-cameras-reid" has 2 source nodes (cam1, cam2); write --input <source_id>=<uri>, e.g. --input cam1=clip.mp4
```

The other `--input` usage errors, each exit 2 before any model loads:

* the same source twice (`--input cam1=a.mp4 --input cam1=b.mp4`, or two
  bare values on a one-source graph): `--input names source "cam1" twice`;
* `<source_id>=` with no uri after it: `--input cam2= has no uri after "="`;
* an empty value (`--input ""`): `--input needs a value`;
* a misspelled id on a graph with several sources:
  `--input "camX=a.mp4" does not say which source it replaces: "camX" is not a source node of graph "two-cameras-reid"; its source nodes: cam1, cam2`.

`<id>=` is read as a source name only when `<id>` is the id of a source
node, so a uri that contains `=` stays a uri:
`--input 'rtsp://host/stream?user=a'` replaces the source of a one-source
graph. On a graph with several sources, a prefix that has a `/`, `\` or
`:` in it, and so could begin a path or a URL, gets the "does not say which
source" message above rather than the misspelled-id one.

**Outputs.** `--output` writes one set of files per stream:

* a video extension (`.mp4`, `.avi`, `.mkv`): one video per stream,
  `<stem>_<source_id><ext>` (`--output run.mkv` writes `run_cam1.mkv` and
  `run_cam2.mkv`), each at its own stream's frame rate and frame size;
* any other extension: `<stem>_<source_id>_<index><ext>` for every frame,
  `<index>` in six digits (`run_cam1_000000.png`), even from a source that is
  a single image - with several sources images are always numbered. No
  extension means `.png`, as with one source.

The live-source refusal (image output from `camera:` or `rtsp://` without
`--frames`) is checked for each stream's own uri, and the missing-directory
check names the first file a stream would write:
`could not write out/run_cam1.mkv: directory out does not exist` (exit 1).
`--display` opens one window, `multi_model_graph`, holding every stream: a
grid with one cell per source (2 side by side, 3-4 in 2x2, more in
ceil(sqrt(N)) columns), each cell labelled with its source id at the top
left and showing that stream's latest frame (see *Results for front ends*);
`q` or ESC stops the whole run, as Ctrl-C does. A stop request stops
reading every stream, finishes the frames in flight in every stream, and
finalizes `--report` and every output file.

**Messages.** Every frame prints one stdout line, in node order, with no
window: `frame 3: palm boxes=1 landmark keypoints=1`. A records node adds
its numbers (`headpose records=1 pitch=-5.98 yaw=13.74 roll=2.09`); more
than three items keeps the first three and a `+|N` tail. With several
sources the line is `stream cam1 frame 3: ...`. A failed frame is also
reported on stderr as `stream cam1 frame 3: <error>` (or `frame 3: <error>`
with one source), and the summary gives one `stream <source_id>: N frames, F failed` line per
stream before the total (lines from the runtime left out):

```console
$ ./bin/multi_model_graph_sync --graph src/cpp_example/multi_model_graph/multistream_od_two_sources.json --input cam1=clip.avi --report report.json
graph:  two-cameras-reid (4 nodes, 3 edges)
input:  cam1: Video file: clip.avi
input:  cam2: Image file: sample/img/sample_person_a1.jpg
report: report.json
stream cam1: 8 frames, 0 failed
stream cam2: 1 frame, 0 failed
9 frames, 0 failed
```

`--check` lists the streams, and the nodes each one runs, after the node table
(see *Finding out what a graph needs: `--check`*). A stream whose source gives
no frame at all does not stop the others: they run to their end, then the run
prints `ERROR [GRAPH_SCHEMA] source "<uri>": produced no frames` for each
empty stream and exits 1. `--report` and `--output` hold every frame the other
streams gave, and the `output:`, `report:` and per-stream summary lines are
printed as in any finished run (the empty stream's own is `0 frames`). A stop
request that arrives while the models load prints `interrupted before the
first frame` and blames no stream (exit 0), as with one source.

---

## Tracking costs latency

An IoU tracker is stateful and strictly frame-ordered: feeding it frame 3
before frame 2 breaks the ids. The design keeps detection
inference parallel and makes only the tracking step wait for frame order.
**Every asynchronous graph pays reorder latency** — a frame that finishes
early waits until every earlier frame has been handed out — **and a graph that
declares `track` additionally serializes its tracking step at a gate per node and stream**,
so one slow frame holds back the tracking of every later one of its stream. Throughput is
unaffected: inference for later frames keeps running.

This build does that. The asynchronous executor keeps up to `--max-inflight`
frames in flight (default 16). A tracked node's results are applied in frame
order through a gate per node and stream: frame N+1 of a stream waits at it
until frame N of the same stream has run, frames of other streams never touch
it, and inference for later frames keeps running on the NPU. Finished
reports pass through a reorder buffer and come out in the order the frames
were read (frame order, with one source), each equal to what the synchronous
executor produces for that frame. A finished frame
still waiting in that buffer counts toward `--max-inflight`, so a frame that is
slow to finish stops the sources from being read further ahead rather than
letting the buffer grow without bound.

---

## Device memory

Every model node opens its own engine on the NPU, even two nodes that name
the same model, and each engine holds its own device memory; other
processes using the NPU draw on the same memory. A graph whose engines do
not fit fails while it loads, with `MODEL_LOAD` on the first node that does
not fit, and every engine opened before it is closed again before the error
is reported. On DX-M1, a second `realesrgan-x2_192x192` next to
`yolov8-n_640x640`, `resnet50_224x224` and a first `realesrgan-x2_192x192`
cannot load (the example under *Error messages*).

Two node settings lower what a graph needs, both under `"engine"`:

```json
{ "id": "od",  "model": "yolov8-n_640x640", "engine": { "share": true, "buffers": 2 } },
{ "id": "od2", "model": "yolov8-n_640x640", "engine": { "share": true, "buffers": 2 } }
```

* `"buffers": N` (1 to 100) is how many inputs DX-RT keeps in flight for
  the model (`InferenceOption::bufferCount`; DX-RT's default is 6). Most of
  a model's device memory is that buffer pool: `parse_model -m <file>`
  prints it under *Model Memory Usage* (`yolov8-n_640x640`: 73.87 MB in
  total, 70.31 MB of it six buffers), so 2 buffers need about a third. Fewer
  buffers can lower async throughput; results do not change.
* `"share": true` lets every node with it and the same model variant use ONE
  engine. They must have the same `params` and the same `buffers`
  (`GRAPH_SCHEMA` on `node "<id>" "engine"` otherwise). Results are the same
  as with an engine each. The executors call each stage from one thread and
  route every result by the job that asked for it, so sharing is safe in
  both binaries; `--max-jobs-per-stage` counts per node, so a shared engine
  may hold that many jobs per sharing node.

Either setting on the two `realesrgan-x2_192x192` nodes of the example
above (`"share": true` on both, or `"buffers": 2` on each) lets that graph
load on DX-M1.

`--check` cannot measure device memory: it never opens the NPU, and DX-RT
has no API for a model's memory before it loads. `--check --json` lists
each model node's `engine` settings and which node's engine it shares, so a
front end can add up `parse_model` figures itself.

---

## Options

| Option | Default | Notes |
|---|---|---|
| `--graph <file>` | — | the graph to run |
| `--input [<source>=]<uri>` | each source node's `uri` | image, video, `camera:<N>`, `rtsp://...`. Repeatable, once per source: `<source>=` names the source node it replaces, and a graph with several sources needs it (see *Several sources: one stream each*) |
| `--output <file>` | none | the rendered result. `.mp4`, `.avi` or `.mkv` (any letter case): one video (codec `mp4v` for `.mp4`/`.mkv`, `MJPG` for `.avi`; the source's fps when it is between 1 and 240, else 30). Any other extension, or none: images, exactly `<file>` for a single image, `<stem>_<index><ext>` for each frame of a multi-frame source. Images from a live source (`camera:`, `rtsp://`, from `--input` or the graph's `uri`) never end, so they need `--frames N`; without it the run is refused (exit 2) before any model loads. A directory that does not exist fails before any model loads too (`could not write <file>: directory <dir> does not exist`, exit 1). With several sources, one file set per stream: one video `<stem>_<source_id><ext>` each, or images `<stem>_<source_id>_<index><ext>`, always numbered; both checks apply to each stream |
| `--display` | off | show each rendered frame in a window. `q` or ESC stops the run gracefully, as Ctrl-C does; closing the window does not stop it (the next frame reopens it). Needs a window system: with none of `DISPLAY`, `WAYLAND_DISPLAY`, `QT_QPA_PLATFORM` set it is refused (exit 2). One that is set but does not answer (`DISPLAY=:99` with no server there, `QT_QPA_PLATFORM=xcb` without `DISPLAY`) is refused the same way, before any model loads: a window is opened once in a child process first, because Qt aborts the process that cannot connect. With several sources, one window, `multi_model_graph`, a grid of the streams with each cell labelled with its source id (see *Results for front ends*); `q` or ESC stops the run |
| `--draw <id[,id...]>` | every node | draw only these nodes, in the window, `--output` and `--output-layers`; every node still runs, prints and reports. A logic node's id draws its overlay (a zone's polygon, a line's arrow, its labels): `--draw zone` draws the zone but not its detector's boxes, `--draw od,zone` both. The events panel lists only the drawn nodes' events, so it shows when a drawn node raises one. Repeatable; ids are comma-separated. An id that is not a node of the graph, or is a source, is a usage error (exit 2) naming the graph's other nodes: `--draw: "nope" is not a node of this graph that draws; its nodes: od, seg, depth`. With `--check` or a listing it is a usage error too |
| `--output-layers <dir>` | none | per frame, the clean source frame and each drawn node's drawing alone on a transparent RGBA PNG the size of the frame, plus `<dir>/layers.json` listing them in stacking order (see *Results for front ends*). `<dir>` is made when it does not exist (its parent must; otherwise exit 1, before any model loads). A video, camera or RTSP source needs `--frames N` (exit 2 otherwise): it writes every frame. With `--check` or a listing it is a usage error (exit 2) |
| `--report <file>` | none | every `FrameReport` as JSON, ordered by `(node, parent_index, roi_index)`. Written frame by frame and finalized on every exit short of a kill (see *`--report`* below) |
| `--report-timing` | off | with `--report`: each frame also gets `"node_ms": {"<node>": ms}`, the time per node (see *`--report`*). Without `--report` it is a usage error (exit 2) |
| `--events <file>` | none | every event of the graph's logic nodes, one JSON object per line (see *Events*). Written as each frame finishes; each line is one write. With `--check`, `--list-models` or `--list-ops` it is a usage error (exit 2) |
| `--model-dir <dir>` | `assets/models` | where the `.dxnn` files are |
| `--max-inflight <N>` | `16` | frames in flight at once; at least 1. Async only: `multi_model_graph_sync` refuses it (exit 2) |
| `--max-jobs-per-stage <N>` | `0` | jobs one stage may have outstanding across frames; 0 = no limit. Async only: `multi_model_graph_sync` refuses it (exit 2) |
| `--stall-timeout-ms <N>` | `0` | fail with `ERROR: AsyncExecutor stalled: …` (exit 1) when work is outstanding and nothing completes for N ms; 0 = wait forever. Async only: `multi_model_graph_sync` refuses it (exit 2) |
| `--frames <N>` | all | stop after N frames; with several sources, after N frames of each stream |
| `--loop` | off | a video file that ends starts again from its first frame, so the run goes on until Ctrl-C, `q`/ESC in `--display` or `--frames N` (per stream). Frame indices, stream time, `[PROGRESS]` and the event panel keep counting; each restart starts the stream's trackers (new ids from 0) and its logic nodes afresh (zone members, line crossings and counts, dwell stays, a condition's edge and cooldown), so every pass raises the events the first one did. Images, cameras and RTSP are unaffected. With per-frame image `--output` and a video source it needs `--frames N` (exit 2 otherwise); with `--check`, `--list-models` or `--list-ops` it is a usage error (exit 2) |
| `--check [<file>]` | — | validate and report without opening the NPU, then exit: 0 for a valid graph (missing `.dxnn` files are listed, not an error), 1 for an invalid graph or a present `.dxnn` that cannot load here (not a `.dxnn`, or a container this DX-RT cannot load) |
| `--list-models` | — | every model, then the `alias of` rows; filter with `--consumes` / `--produces`; `--model-dir` sets the directory the `file` column reads |
| `--list-ops` | — | every logic and cpu op: inputs, output, tracking, accepted kinds; with `--json` the full catalog, params included (see *`--list-ops --json`*). Takes no graph |
| `--json` | off | with `--check`, `--list-models` or `--list-ops`: one JSON document on stdout instead of the tables (see *`--list-models --json`* and *`--check --json`*). With anything else it is a usage error (exit 2) |
| `--help` | — | |

`--max-jobs-per-stage` and `--stall-timeout-ms` are `dx_graph`'s
`max_jobs_per_stage` and `stall_timeout_ms`: non-negative whole numbers, so
`-1`, `1.5` or a missing value is a usage error naming the option. Every
numeric option (`--frames` too) takes at most 2147483647; a larger value is
a usage error as well. The sync
binary runs one frame at a time and would ignore all three async-only
options, so it refuses them instead:

```console
$ ./bin/multi_model_graph_sync --graph g.json --max-inflight 4
--max-inflight applies to multi_model_graph_async only: this binary runs one frame at a time
```

The live-source and `--display` refusals, word for word:

```console
$ ./bin/multi_model_graph_async --graph g.json --input camera:0 --output out.png
--output out.png writes one image per frame, and "camera:0" is a live source that never ends: write a video instead (--output <stem>.mkv, .avi or .mp4) or stop after N frames (--frames N)
$ ./bin/multi_model_graph_async --graph g.json --display      # no window system
--display needs a window system, and neither DISPLAY nor WAYLAND_DISPLAY is set: run it in a desktop session, or save the result with --output
$ DISPLAY=:99 ./bin/multi_model_graph_async --graph g.json --display   # nothing on :99
--display could not open a window with DISPLAY=:99: no window system answered there; run it in a desktop session, or save the result with --output
```

Exit codes: `0` success, `1` a `GraphError` (the graph or its models cannot
run) or a failed frame, `2` a usage error.

Ctrl-C **and SIGTERM** are graceful: the input loop stops reading, the
frames already in flight finish (in every stream) and `--report` / `--output`
are finalized;
the exit code is the run's own (`0` when every frame succeeded). A stop
request that arrives while the models load ends the run before the first
frame: `interrupted before the first frame`, `0 frames, 0 failed`, exit 0.
A later
Ctrl-C, more than 200 ms after the first request, terminates at once, for a
wind-down that is itself stuck; repeats within 200 ms (a wrapper such as
`timeout` forwarding the same keypress) count as the same request. SIGTERM
never escalates (`timeout` may deliver it twice). To force an end, follow it
with SIGKILL: `timeout` does that only when given `-k`, as in
`timeout -k 10 60 ./bin/...`. `q` or ESC in the `--display` window is the
same graceful stop, so a Ctrl-C more than 200 ms after it terminates too.

A run that is terminated that way, or killed, finalizes nothing. `--report`
then holds every finished frame and needs its closing lines (below). A
video is left without its closing index: an `.mp4` has no `moov` atom and
OpenCV and most players cannot open it at all, while an `.mkv` or `.avi`
still plays up to about the last frames written. So after the first stop
request, let the run finish; if you may have to kill it, write `.mkv` or
`.avi`.

### Results for front ends

Three options turn one run into what a page needs to show it.

- **`--draw od,zone`** keeps the picture to the nodes chosen before a Live
  run starts. The other nodes still run, print their `frame N:` lines and
  fill `--report`; they only draw nothing.
- **`--output-layers <dir>`** splits each frame's picture into layers a page
  stacks and toggles itself, for one image or for `--frames N` of a video
  (a paused video: run that frame again as an image). For each frame:

  | File | What it holds |
  |---|---|
  | `<stem>[_<source>]_<frame>_base.png` | the clean source frame, as read (3 channels) |
  | `<stem>[_<source>]_<frame>_<node>.png` | one node's drawing alone, RGBA, transparent where it drew nothing; a node that drew nothing on the frame has no file |
  | `<stem>[_<source>]_<frame>_events.png` | the recent-events panel, RGBA, when it has lines |
  | `layers.json` | every frame written so far, rewritten after each frame |

  `<stem>` is `--output`'s file name without its extension, else the graph
  file's; `_<source>` appears with several sources only; `<frame>` is the
  frame index in 6 digits. `layers.json` lists each frame's layers bottom
  first, a model node with its `model`, a logic or cpu node with its `op`:

  ```json
  {
    "frames": [
      {
        "stream": "cam",
        "frame": 0,
        "base": "result_000000_base.png",
        "layers": [
          { "node": "seg", "model": "bisenetv2_1024x2048", "file": "result_000000_seg.png" },
          { "node": "depth", "model": "fastdepth_224x224", "file": "result_000000_depth.png" },
          { "node": "od", "model": "yolov8-n_640x640", "file": "result_000000_od.png" },
          { "node": "zone", "op": "zone", "file": "result_000000_zone.png" },
          { "panel": "events", "file": "result_000000_events.png" }
        ]
      }
    ]
  }
  ```

  The order is the drawing order of `--output`: each node at the first pass
  it draws in (images, label maps, dense maps, instance masks, boxes,
  oriented boxes, keypoints, scores and vectors), logic nodes after them in
  id order, the events panel last. So the base with every layer blended
  over it in that order (`out = layer * a + out * (1 - a)`) is the
  `--output` image, within a few levels of rounding; the one case it is not
  exact is a node that draws in two passes (an instance segmenter's masks,
  then its boxes) over a pixel another node drew in between. Labels are
  laid out as in the merged image, so toggling a layer off never moves the
  others. The run prints `layers: <dir>/layers.json` at its end.
- **`--display` with several sources** shows one window, a grid of the
  streams (2 side by side, 3-4 in 2x2, more in ceil(sqrt(N)) columns), each
  cell labelled with its source id and showing that stream's latest frame,
  letterboxed when the streams' sizes differ. The cells are the size of the
  first frame rendered, the whole grid scaled down to fit 1920x1080; so one
  capture of the window (Studio's Live on Xvfb) shows every stream. With
  one source the window is the frame itself, as before.

### `--report`

- **Written as each frame finishes**, not held until the input ends, so
  memory stays bounded on a camera or RTSP source that runs for hours. The
  file is opened when the source opens, before the first frame: a path that
  cannot be written fails at once (`could not write <path>`, exit 1).
- **The bytes are those of a whole-file dump**: `{"frames": [...],
  "graph": "<name>"}`, indented by two spaces, with a trailing newline. The
  sync and async binaries write byte-identical files.
- **`--report-timing`** adds `"node_ms": {"<node>": ms}` to every frame:
  milliseconds per non-source node, rounded to 0.001. In
  `multi_model_graph_sync` it is the time inside the node's stage (all
  crops of a ROI node, summed) or its cpu reduction; in
  `multi_model_graph_async` it runs from the node's dispatch to its last
  job's completion, so it includes waiting behind other frames' jobs. A ROI
  node with no crops is `0`. Timings differ from run to run, so without the
  flag a report holds none and the two binaries' files stay byte-identical.
- **Events**: a frame whose logic nodes raised events has `"events": [{"node",
  "event", "track_id", "time", "data"}]` (see *Events*); other frames, and
  every graph without logic nodes, have no such key.
- **Every exit short of a kill finalizes it**: the normal end; Ctrl-C,
  SIGTERM, `q`/ESC; a frame whose `--output` cannot be written; an error that
  stops the run, such as `--stall-timeout-ms`; a source that produced no
  frames (exit 1; `"frames": []` with one source, every frame of the other
  streams with several); and a stop request before the first frame
  (`"frames": []`, exit 0).
- **After a SIGKILL** (or a terminating second Ctrl-C) the file holds every
  frame finished so far and ends with that frame's closing `    }`, so a JSON
  parser rejects it. Each frame goes out in one write, so appending the
  closing lines repairs it (the graph's `"name"`, `""` when it has none):

  ```bash
  printf '\n  ],\n  "graph": "%s"\n}\n' person-reid >> report.json
  ```

  If no frame had finished, the file ends with `[`: then the format is
  `'],\n  "graph": "%s"\n}\n'` instead. In the rare case the kill lands
  inside a frame's write, that frame is cut short: delete back to the
  previous frame's closing `    }` first.
- **NaN and infinities** are written as the strings `"NaN"`, `"Infinity"`
  and `"-Infinity"`. JSON has no such numbers, and strict parsers
  (JavaScript's `JSON.parse`, `jq`, nlohmann's own `json::parse`) reject
  bare `NaN`/`Infinity` tokens; a string is valid everywhere, and Python's
  `float()` and JavaScript's `Number()` turn it back into the number. Finite
  values are unchanged, and no report contains `null`. Two report files with
  NaN in the same place compare equal as text, while the C++
  `FrameReport::operator==` treats NaN as never equal, so an in-memory
  comparison of a NaN-bearing report fails by design.

### Progress lines (`DXAPP_PROGRESS=1`)

With `DXAPP_PROGRESS=1` both binaries print, at most once a second and once
at the end, after the per-frame lines:

```text
[PROGRESS] frames=24 fps=31.8 avg_fps=30.9 node_ms=od:12.41,reid:6.02
```

`frames` counts finished frames of every stream; `fps` is the rate since
the previous line and `avg_fps` since the first frame. `node_ms` is the
per-node time (as in `--report-timing`) averaged over the frames since the
previous line that ran the node; the final line carries run averages. Node
ids are percent-encoded outside `[A-Za-z0-9_.-]` (`cam 1` is `cam%201`).
The first line, before any rate can be measured, carries only `frames=`.

---

## A model's graph contract

What a model `consumes` and which ports it has come from its factory
interface (`IPoseFactory`, `IDetectionFactory`, ...), through the table in
`scripts/gen_model_registry.py`. A model family overrides either one in
that script's `GRAPH_TRAITS` table (the per-variant factory headers are
generated, so the overrides live beside the generator): `vitpose` and
`dark_hrnet` consume `either`, `superpoint` has the `descriptors` port and
`ppmatting` the `alpha` port. A gallery embedding model gets its `matches`
port from its `config.json`. A factory header may still declare either one
itself, with a single line:

```cpp
static constexpr GraphInput graphInput() { return GraphInput::kEither; }   // kFullFrame, kRoi or kEither
static constexpr const char* graphPorts() { return "descriptors"; }       // comma-separated port names
```

The build's code generator parses those lines out of the header, so no
factory is instantiated to learn its contract. A header declaration must
agree with `GRAPH_TRAITS` for its family, or generation fails with
`TRAIT MISMATCH`. Every
generated registry row also carries a `static_assert` that the factory's own
declaration equals what the generator parsed. A declaration the generator
misread or missed, such as one inherited from another factory, then fails
the build (`graphInput() and gen_model_registry.py disagree`) instead of
shipping the interface default.

A factory header that mentions `graphInput` or `graphPorts` anywhere, in a
comment too, must hold exactly one declaration in the form above, or
generation fails with `UNPARSEABLE TRAIT <header>`. So a comment that names
a trait in a header that declares none, or that quotes the declaration a
second time, stops the build. A port name must also be one the generator
knows for the factory's result type (`UNKNOWN PORT`).

`vitpose-s_256x192` decodes its output as the 17 VitPose heatmaps, as the
Python example does; its C++ and Python keypoints agree to within 2.1e-5 px,
and it reads neither `score_threshold` nor `nms_threshold`.
`dark-hrnet-w32_256x192` uses the same decoder, and its C++ and Python
keypoints agree to the same 2.1e-5 px on its v9 `.dxnn`.

## Your own registry (`IModelRegistry`)

The engine reads models only through `IModelRegistry`
(`common/graph/i_registry.hpp`). A registry of your own, such as a plugin,
implements that interface. This release added fields to the structs it
passes, so code built against an earlier `i_registry.hpp` (or the
`shape.hpp` it includes) must be rebuilt:

* `StageResult::ports` (in `shape.hpp`): a stage's extra outputs, by name.
  The PORTS CONTRACT in `i_registry.hpp` says what a stage must put there.
* `ModelInfo::ports`, a list of `PortInfo {name, shape}`: the extra outputs
  a model declares.
* `ModelInfo::text_params`, a list of `ParamInfo {name, kind}` with `kind`
  `kText` or `kTextList`: the params keys the model reads as a string or a
  list of strings. Every other key is a number.
* `StageParams::text` and `StageParams::lists`: a node's string and
  string-array params, next to the existing `numeric`.

Leaving them empty gives the old behaviour: no ports, and numbers only.

---

## From Python

The same graphs run from Python through the `dx_graph` module, which is built
by the same CMake build and links this engine — not a copy of it:

```python
import dx_graph

with dx_graph.Graph("src/cpp_example/multi_model_graph/cascade_od_reid_track.json") as g:
    for report in g.stream(source="my_video.mp4"):   # --input; pipelined as the async CLI
        d = report.to_dict()                         # this frame's --report entry
        canvas = g.render(report)                    # this frame's --output image
```

Reports, renderings and error messages are the CLI's own, checked on hardware
by `tests/cpp_example/test_graph_python.py`. Every `run()` and `stream()`
starts with the graph's trackers reset, as the CLI starts on its input, so a
reused Graph gives what a fresh one gives; track ids follow objects across
the frames of one stream. A graph with several sources reads them all with
`g.stream()` or `g.stream(sources={"cam1": a, "cam2": b})` (the CLI's
`--input cam1=a --input cam2=b`); each report's `.stream` names its source.
`to_dict()` holds the same `"NaN"` / `"Infinity"`
/ `"-Infinity"` strings as the `--report` file, a model the device cannot
load raises `dx_graph.GraphError` with `.code == "MODEL_LOAD"`, and the
executor options are the constructor's `max_frames_in_flight`,
`max_jobs_per_stage` and `stall_timeout_ms`. One difference: the CLI stops
gracefully on SIGTERM, while a Python program keeps Python's default SIGTERM
action (the process ends at once); `dx_graph` installs no signal handler.
`./build.sh --all --python_exec <your python>` builds the module for that
interpreter and ships it as `bin/python/dx_graph` (the default `./build.sh`,
`--minimal`, does not). Build, API, threading and limits:
[`src/bindings/python/dx_graph/README.md`](../../bindings/python/dx_graph/README.md).

---

## Retargeting a sample without rebuilding

This is the claim the example exists to make, so it is worth trying:

```bash
# edit cascade_od_reid_track.json: "model": "yolov8-n_640x640" -> "yolo26-s_640x640"
python3 scripts/download_models.py --models yolo26-s_640x640   # a v9 file: needs DX-RT >= 3.5.0 (see "Model files")
./bin/multi_model_graph_async \
    --graph src/cpp_example/multi_model_graph/cascade_od_reid_track.json \
    --output /tmp/swapped.png
```

No rebuild, no recompilation, no code change. The same holds for swapping
`bisenetv2_1024x2048` for any other model whose `produces` column says `labelmap`, or
`casvit-t_224x224` for anything that `consumes` `roi` or `either`.
