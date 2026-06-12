# Building PaddleOCR / RapidDoc Apps on the DeepX NPU (app reference)

> How to BUILD runtime apps on the PaddlePaddle OCR/document ecosystem on the DX-M1 NPU
> using DEEPX's integrated forks. This is the **app-building** companion to the
> compile/integration reference at `dx-compiler/.deepx/toolsets/paddlepaddle-deepx.md`
> (read that too for the model/NPU-engine side). Read this BEFORE building an OCR
> inference app or a PDF→Markdown app.

## Key architectural note (READ FIRST)

These apps do **NOT** use the dx_app `IFactory` / `SyncRunner` / `AsyncRunner` pattern.
PaddleOCR-deepx and RapidDoc ship their **own NPU pipelines** (the models run on the
DX-M1 via the fork's runtime, not via `dx_engine.InferenceEngine` directly). So build a
**standalone app that drives the fork's API** — do not wrap it in IFactory. (This is the
documented exception to the "always IFactory" rule, because the inference is owned by the
PaddleOCR/RapidDoc pipeline, not by a single `.dxnn` you call.)

| App | Built on | DEEPX source (branch) | Pattern |
|---|---|---|---|
| OCR inference (video/webcam) | PaddleOCR-deepx (PP-OCRv5 det+rec) | `DEEPX-AI/PaddleOCR-deepx` @ **`deepx`** | OpenCV capture loop → `PaddleOCR.predict(frame)` on NPU |
| PDF → Markdown | RapidDoc (PP-StructureV3 pipeline) | `DEEPX-AI/RapidDoc` @ **`rapid_doc_deepx`** | `demo/demo_offline.py` 7-stage NPU pipeline |

## Setup (both apps; standard DEEPX-fork bring-up)

```bash
# Clone into an ISOLATED working dir (e.g. the session dir) — NEVER reuse/modify/delete a
# pre-existing user repo found elsewhere on disk.
git clone -b deepx https://github.com/DEEPX-AI/PaddleOCR-deepx.git           # OCR app
git clone -b rapid_doc_deepx https://github.com/DEEPX-AI/RapidDoc.git         # PDF→md app
pip install -r requirements.deepx.txt && pip install -e .
./setup.sh                                             # download prebuilt onnx+dxnn models (foreground; NOT a dxcom compile)
source <fork>/deepx_scripts/set_env.sh 1 2 1 3 2 4    # DX-RT env (RapidDoc); PaddleOCR-deepx: see its deepx-branch setup
export DXNN_DEVICES=0                                  # NPU device(s)
```
- **Models come from `./setup.sh`** (RapidDoc: `setup_assets()` → `setup_sample_models.sh`
  downloads `onnx_models/` + `dxnn_models/`). Do NOT hand-compile any `.dxnn`, and do NOT
  run the download as a background task in a headless build — both have deadlocked the build.
- Run the suite sanity check first (`dx-runtime/scripts/sanity_check.sh --dx_rt`) — NPU must PASS.
- Generated app + scripts go to `dx-agent-dev/<session_id>/` (output isolation).

## A. OCR inference app — video file + webcam

```python
import cv2
from paddleocr import PaddleOCR                       # DEEPX fork → DX-M1 NPU models
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False)

def open_source(src):                                  # --source video.mp4  OR  --source 0 (webcam)
    return cv2.VideoCapture(int(src) if str(src).isdigit() else src)

cap = open_source(args.source); writer = None
while True:
    ok, frame = cap.read()
    if not ok: break
    res = ocr.predict(frame)                           # per-frame NPU OCR: boxes + text + score
    vis = draw_ocr(frame, res)                         # overlay boxes + recognized strings
    if writer is None and args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'),
                                 cap.get(cv2.CAP_PROP_FPS) or 15, (vis.shape[1], vis.shape[0]))
    if writer: writer.write(vis)
    if args.show: cv2.imshow('ocr', vis); cv2.waitKey(1)
# save one annotated frame as sample_detect.jpg; report per-frame latency / FPS
```
- `--source <path.mp4>` and `--source <int>` (webcam) MUST both work (single code path via `open_source`).
- Skip frames if webcam FPS exceeds NPU throughput; report measured per-frame latency + FPS in the README.

## B. PDF → Markdown app

```bash
# from the RapidDoc rapid_doc_deepx checkout (after setup above)
python demo/demo_offline.py <input.pdf|dir> --finegrained          # 7-stage NPU pipeline
python demo/demo_offline.py scanned.pdf  --parse-method ocr        # force OCR (scanned)
python demo/demo_offline.py digital.pdf  --parse-method txt        # text layer only (fast)
python run_with_npu_monitor.py python demo/demo_offline.py docs/ --finegrained   # + NPU utilization
```
- Output: Markdown + JSON under `demo/output-offline-<mode>/` (preserves headings/tables).
- Wrap this in the app's `run.sh`; save a sample input PDF + its rendered Markdown (`sample_output.md`).
- Report per-stage NPU timings in the README.

## Mandatory deliverables (per app)

`setup.sh` (clone fork + env + deps), `run.sh` (one-command launcher with env activation +
`DXNN_DEVICES`), `README.md` (run steps + measured NPU latency/FPS or stage timings), and a
visual sample (`sample_detect.jpg` for OCR; `sample_output.md` for PDF→md). No placeholder code.

## Anti-patterns (STOP)

- Wrapping PaddleOCR/RapidDoc in dx_app `IFactory` — inference is owned by the fork's pipeline.
- Using upstream PaddleOCR/RapidDoc `main` (no DeepX NPU backend) — use the `deepx` /
  `rapid_doc_deepx` branches.
- Hand-compiling models with `dxcom` instead of `./setup.sh` — the fork ships prebuilt
  onnx+dxnn; manual compile is the known build-deadlock cause.
- Reusing/deleting a pre-existing user repo found on disk — always clone fresh into the
  session dir; deleting a user's repo is a destructive action.
- Running without sourcing the DX-RT env (`deepx_scripts/set_env.sh`) → device init errors.
- Webcam OCR that ignores NPU throughput (no frame-skip) → growing latency.
