# Building PaddleOCR / RapidDoc Apps on the DeepX NPU (app reference)

> How to BUILD runtime apps on the PaddlePaddle OCR/document ecosystem on the DX-M1 NPU
> using DEEPX's integrated forks. This is the **app-building** companion to the
> compile/integration reference at `dx-compiler/.deepx/toolsets/paddlepaddle-deepx.md`
> (read that too for the model/NPU-engine side). Read this BEFORE building an OCR
> inference app or a PDF→Markdown app.

## Key architectural note (READ FIRST)

These apps do **NOT** use the dx_app `IFactory` / `SyncRunner` / `AsyncRunner` pattern.
PaddleOCR-deepx and RapidDoc ship their **own NPU pipelines** (the models run on the
DX-M1 via the fork's runtime, not via `dx_engine.InferenceEngine` directly). This is the
documented exception to the "always IFactory" rule.

**BUT you MUST still GENERATE A STANDALONE APP — you must NOT just run the fork's example
script.** The deliverable is **your own entry program** that *imports the fork's pipeline
API as a library* and drives it. Two hard consequences:

1. **Write your own entry** (`pdf_to_markdown.py` / `ocr_video.py`) that calls the fork's
   Python API directly. **NEVER** make `run.sh` shell out to the fork's `demo/demo_offline.py`
   (or any `demo/*` / example script) — wrapping the example is **NOT** an app and FAILS the
   showcase gate. Model your entry's logic on the demo, but it is *your* code.
2. **Make it self-contained (vendoring).** The DEEPX RapidDoc fork is **not on PyPI**, so
   **vendor its importable Python package** (`rapid_doc/`, ~3.7 MB pure-Python, no binaries)
   **into the app dir** and import from the vendored copy. `setup.sh` installs the pip deps
   and downloads the NPU models; the app then runs **without a runtime clone of the fork**.
   (PaddleOCR-deepx's `paddleocr` is pip-installable, so the OCR app installs it as a normal
   dependency instead of vendoring.)

| App | Built on | DEEPX source (branch) | Pattern (what YOU generate) |
|---|---|---|---|
| OCR inference (video/webcam) | PaddleOCR-deepx (PP-OCRv5 det+rec) | `DEEPX-AI/PaddleOCR-deepx` @ **`deepx`** | own `ocr_video.py`: OpenCV capture loop → `PaddleOCR.predict(frame)` on NPU (`paddleocr` as pip dep) |
| PDF → Markdown | RapidDoc (PP-StructureV3 pipeline) | `DEEPX-AI/RapidDoc` @ **`rapid_doc_deepx`** | own `pdf_to_markdown.py` importing the **vendored `rapid_doc` package** API (NOT `demo/demo_offline.py`) |

## Setup — clone to OBTAIN the package, then VENDOR it (RapidDoc)

```bash
# 1) Clone the fork into a TEMP/ISOLATED dir only to obtain its source — NEVER reuse,
#    modify, or delete a pre-existing user repo found elsewhere on disk.
git clone -b rapid_doc_deepx https://github.com/DEEPX-AI/RapidDoc.git /tmp/_rapiddoc_src

# 2) VENDOR the importable package + the small helper scripts INTO the app dir (no models):
APP=dx-agent-dev/<session_id>
cp -r /tmp/_rapiddoc_src/rapid_doc        "$APP"/rapid_doc            # ~3.7MB pure-Python pipeline
cp -r /tmp/_rapiddoc_src/deepx_scripts    "$APP"/deepx_scripts        # set_env.sh etc.
cp    /tmp/_rapiddoc_src/setup_sample_models.sh "$APP"/               # model downloader
cp    /tmp/_rapiddoc_src/requirements.deepx.txt "$APP"/ ; cp /tmp/_rapiddoc_src/LICENSE "$APP"/ 2>/dev/null

# 3) setup.sh (generated) then: venv → pip install -r requirements.deepx.txt → dx_engine bridge
#    → ./setup_sample_models.sh (downloads onnx_models/ + dxnn_models/, foreground).
# 4) run.sh (generated) runs YOUR entry, with the vendored package importable:
#    source deepx_scripts/set_env.sh 1 2 1 3 2 4 ; export DXNN_DEVICES=0 ; python pdf_to_markdown.py ...
```
- The app imports the **vendored** `rapid_doc` (e.g. `PYTHONPATH=. python pdf_to_markdown.py`
  or a `sys.path` insert in the entry) — **no runtime clone of the fork.**
- **Models come from `./setup_sample_models.sh`** (downloads `onnx_models/` + `dxnn_models/`).
  Do NOT hand-compile any `.dxnn`, and do NOT run the download as a background task in a
  headless build — both have deadlocked the build. Models are NOT committed to the showcase.
- **Model downloads MUST be resilient (retry + resume).** The `sdk.deepx.ai` CDN
  intermittently resets large transfers (PP-OCRv5 server ≈ 302 MB, RapidDoc onnx_models
  ≈ 930 MB), so a plain `curl -fsSL` fails the whole pull on a single reset. Use:
  ```bash
  curl -fSL --retry 15 --retry-all-errors --retry-delay 4 -C - "$URL" -o "$DEST"
  ```
  (`-C -` resumes the partial file). This applies to the paddleocr `setup.sh` curl AND
  the RapidDoc fork's `deepx_scripts/get_resource.sh` curl (patch it after vendoring).
- **Skip the download ONLY when ALL required model dirs are present.** RapidDoc needs BOTH
  `dxnn_models/` (NPU) AND `onnx_models/` (incl. the formula model
  `onnx_models/pp_formulanet_plus_m.onnx`). A skip guard keyed on `dxnn_models/` alone
  leaves `onnx_models/` empty after a partial download and `run.sh` then fails with a
  missing-formula-model `FileNotFoundError`. Gate on every required dir (and re-run the
  downloader with `--force` when re-downloading).
- For the **OCR app**, `paddleocr` (DEEPX `deepx` branch) is pip-installable → install it in
  `setup.sh` (no vendoring); the entry imports `from paddleocr import PaddleOCR`.
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

## B. PDF → Markdown app — your own entry over the vendored API

Write `pdf_to_markdown.py` (YOUR code) that imports the **vendored** `rapid_doc` package
and drives the pipeline — do NOT call `demo/demo_offline.py`. The public API the demos use:

```python
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # make the vendored ./rapid_doc importable
from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from rapid_doc.data.data_reader_writer import FileBasedDataWriter
from rapid_doc.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, read_fn
from rapid_doc.utils.enum_class import MakeMode
# ... build args (--input, --parse-method auto|txt|ocr, --output-dir), read PDF bytes,
#     run doc_analyze on the NPU, then write Markdown (+ JSON) via FileBasedDataWriter.
# Model the orchestration on the fork's demo_offline.py, but this is YOUR standalone app.
```
- `run.sh` runs **`python pdf_to_markdown.py`** (after sourcing `deepx_scripts/set_env.sh`
  and exporting `DXNN_DEVICES`) — never the fork's demo script.
- Support `--parse-method auto|txt|ocr`. Output Markdown + JSON (preserves headings/tables/
  formulas); copy the rendered Markdown to `sample_output.md`.
- Save a sample input PDF + its rendered Markdown; report per-stage NPU timings in the README.

## Mandatory deliverables (per app)

- **Your own entry program** — `pdf_to_markdown.py` (RapidDoc) or `ocr_video.py` (PaddleOCR).
  This is the core deliverable; a `run.sh` that only calls the fork's demo is NOT acceptable.
- **Vendored package** (RapidDoc): `rapid_doc/` + `deepx_scripts/` + `setup_sample_models.sh`
  copied into the app dir (so it runs without a runtime clone). OCR: `paddleocr` as a pip dep.
- `setup.sh` (venv + deps + model download via `setup_sample_models.sh`; NO fork clone at run time),
  `run.sh` (sources `set_env.sh` + `DXNN_DEVICES`, then runs YOUR entry),
  `README.md` (run steps + measured NPU latency/FPS or stage timings),
  and a visual sample (`sample_detect.jpg` for OCR; `sample_output.md` for PDF→md). No placeholder code.

## Anti-patterns (STOP)

- **`run.sh` that shells out to the fork's `demo/demo_offline.py` (or any `demo/*`/example)** —
  that wraps the example instead of generating an app. **FAILS the showcase gate.** Write your
  own entry that imports the (vendored) pipeline API.
- **Depending on a runtime clone of the whole fork** — vendor the `rapid_doc` package into the
  app so the showcase is self-contained (only the NPU models are downloaded, never committed).
- Wrapping PaddleOCR/RapidDoc in dx_app `IFactory` — inference is owned by the fork's pipeline.
- Using upstream PaddleOCR/RapidDoc `main` (no DeepX NPU backend) — use the `deepx` /
  `rapid_doc_deepx` branches.
- Hand-compiling models with `dxcom` instead of `setup_sample_models.sh` — the fork ships
  prebuilt onnx+dxnn; manual compile is the known build-deadlock cause.
- Reusing/deleting a pre-existing user repo found on disk — always clone fresh into a temp/
  session dir; deleting a user's repo is a destructive action.
- Running without sourcing the DX-RT env (`deepx_scripts/set_env.sh`) → device init errors.
- Webcam OCR that ignores NPU throughput (no frame-skip) → growing latency.
