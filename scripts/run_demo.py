"""DX-APP Interactive Demo - Cross-platform Python launcher."""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Enable ANSI colors on Windows 10+
if sys.platform == "win32":
    os.system("")

# ─── Colors ───
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[90m"

DX_APP_PATH = Path(__file__).resolve().parent.parent


def resolve_model_path(name: str) -> Path:
    """assets/models first, then a parent checkout's workspace/res/models."""
    local = DX_APP_PATH / "assets" / "models" / name
    if local.is_file():
        return local
    directory = DX_APP_PATH
    while directory != directory.parent:
        candidate = directory / "workspace" / "res" / "models" / name
        if candidate.is_file():
            return candidate
        directory = directory.parent
    return local


# ═══════════════════════════════════════════════════════════════
# Demo Registry
# ═══════════════════════════════════════════════════════════════
DEMOS = [
    # (label, group, cpp_base, py_dir, py_base, model, video, image, py_async, image_only)
    ("Object Detection         (YOLOv7)", "Detection", "yolov7", "object_detection/yolov7", "yolov7", "yolov7_640x640.dxnn", "assets/videos/snowboard.mp4", "sample/img/sample_street.jpg", True, False),
    ("Object Detection         (YOLOv11N)", "Detection", "yolo11", "object_detection/yolo11", "yolo11", "yolo11-n_640x640.dxnn", "assets/videos/boat.mp4", "sample/img/sample_street.jpg", True, False),
    ("Face Detection           (SCRFD500M)", "Detection", "scrfd", "face_detection/scrfd", "scrfd", "scrfd-500m_640x640.dxnn", "assets/videos/dance-group.mov", "sample/img/sample_face.jpg", True, False),
    ("OBB Detection            (YOLO26N-OBB)", "Detection", "yolo26_obb", "oriented_object_detection/yolo26_obb", "yolo26_obb", "yolo26-n-obb_1024x1024.dxnn", "assets/videos/obb.mp4", "sample/img/sample_airport_satellite_view.png", True, False),
    ("Pose Estimation          (YOLOv8s-Pose)", "Pose & Landmark", "yolov8_pose", "pose_estimation/yolov8_pose", "yolov8_pose", "yolov8-s-pose_640x640.dxnn", "assets/videos/dance-solo.mov", "sample/img/sample_people.jpg", True, False),
    ("Hand Landmark            (HandLandmarkLite)", "Pose & Landmark", "mediapipe_hands_lite", "hand_landmark/mediapipe_hands_lite", "mediapipe_hands_lite", "mediapipe-hands-lite_224x224.dxnn", "assets/videos/hand.mp4", "sample/img/sample_hand.jpg", True, False),
    ("Face Alignment           (3DDFA-V2-MobileNetV1)", "Pose & Landmark", "3ddfa_v2", "face_landmark/3ddfa_v2", "3ddfa_v2", "3ddfa-v2_mobilenetv1_120x120.dxnn", "assets/videos/face-alignment-closeup.mp4", "sample/img/sample_face_a1.jpg", True, False),
    ("Instance Segmentation    (YOLOv8N-Seg)", "Segmentation", "yolov8_seg", "instance_segmentation/yolov8_seg", "yolov8_seg", "yolov8-n-seg_640x640.dxnn", "assets/videos/dogs.mp4", "sample/img/sample_street.jpg", True, False),
    ("Semantic Segmentation    (DeepLabV3+MobileNet)", "Segmentation", "deeplabv3", "semantic_segmentation/deeplabv3", "deeplabv3", "deeplabv3plus_mobilenetv1_512x512.dxnn", "assets/videos/blackbox-city-road.mp4", "sample/img/sample_parking.jpg", True, False),
    ("Classification           (ResNet50)", "Classification", "resnet", "image_classification/resnet", "resnet", "resnet50_224x224.dxnn", "assets/videos/dogs.mp4", "sample/img/sample_dog.jpg", True, False),
    ("Depth Estimation         (YOLO26-Depth-S)", "Depth Estimation", "yolo26_depth", "depth_estimation/yolo26_depth", "yolo26_depth", "yolo26-depth-s_768x768.dxnn", "assets/videos/blackbox-city-road.mp4", "sample/img/sample_parking.jpg", True, False),
    ("Image Denoising          (DnCNN-50)", "Image Restoration", "dncnn", "image_denoising/dncnn", "dncnn", "dncnn-50_512x512.dxnn", "assets/videos/noisy_hand.mp4", "sample/img/sample_denoising.jpg", True, False),
    ("Super Resolution         (ESPCN-X4)", "Image Restoration", "espcn", "super_resolution/espcn", "espcn", "espcn-x4_17x17.dxnn", "assets/videos/lowres-drone-city-road.mp4", "sample/img/sample_lowres275x150.png", True, False),
    ("Image Enhancement        (Zero-DCE)", "Image Restoration", "zerodce", "low_light_enhancement/zerodce", "zerodce", "zerodce_400x600.dxnn", "assets/videos/lowlight.mp4", "sample/img/sample_lowlight.jpg", True, False),
    ("Embedding                (ArcFace)", "Recognition", "arcface", "face_recognition/arcface", "arcface", "arcface_mobilefacenet_112x112.dxnn", "assets/videos/face-pair-sofa.mp4", "sample/img/face_pair", True, True),
    ("Attribute Recognition    (DeepMAR-ResNet50)", "Recognition", "deepmar", "person_attribute/deepmar", "deepmar", "deepmar_resnet50_224x224.dxnn", "assets/videos/person-pair-hallway.mp4", "sample/img/sample_person_a1.jpg", True, True),
    ("PPU Pipeline             (YOLOv7-PPU)", "PPU", "yolo_ppu", "object_detection/yolo_ppu", "yolo_ppu", "yolov7_640x640_ppu.dxnn", "assets/videos/snowboard.mp4", "sample/img/sample_street.jpg", True, False),
    ("Keypoint Detection       (SuperPoint)", "Keypoint & Pose", "superpoint", "keypoint_detection/superpoint", "superpoint", "superpoint_480x640.dxnn", "assets/videos/blackbox-city-road2.mov", "sample/img/sample_street.jpg", True, False),
    ("Object Pose Estimation   (DOPE)", "Keypoint & Pose", "dope", "object_pose_estimation/dope", "dope", "dope-hope-ketchup_480x640.dxnn", "assets/videos/snowboard.mp4", "sample/dope/000000.png", True, True),
    ("Panoptic Driving         (YOLOPv2)", "Driving & 3D", "yolopv2", "panoptic_driving_perception/yolopv2", "yolopv2", "yolopv2_384x640.dxnn", "assets/videos/blackbox-city-road.mp4", "sample/img/sample_parking.jpg", True, False),
    ("3D Object Detection      (SFA3D)", "Driving & 3D", "sfa3d", "3d_object_detection/sfa3d", "sfa3d", "sfa3d_608x608.dxnn", "assets/videos/blackbox-city-road.mp4", "sample/kitti/velodyne/000049.bin", True, True),
    ("Hand Detection           (MediaPipe Palm)", "Hand Detection", "mediapipe_hand_detector", "hand_detection/mediapipe_hand_detector", "mediapipe_hand_detector", "mediapipe-hand-detector_192x192.dxnn", "assets/videos/hand.mp4", "sample/img/sample_hand.jpg", True, False),
    # The four task categories DX Model Zoo added in 2_5_0. All four rank a query
    # against a gallery or produce a matte, so they are image-only: the comparison set
    # is a committed database (sample/gallery/*.npz), not a previous frame.
    ("Image Retrieval          (CLIP RN50)", "Retrieval & Matting", "clip-img_resnet50_224x224_openai", "image_retrieval/clip_rn50", "clip-img_resnet50_224x224_openai", "clip-img_resnet50_224x224_openai.dxnn", None, "sample/img/sample_person_a2.jpg", True, True),
    ("Visual Place Recognition (EigenPlaces R18)", "Retrieval & Matting", "eigenplaces-resnet18_512x512", "visual_place_recognition/eigenplaces", "eigenplaces-resnet18_512x512", "eigenplaces-resnet18_512x512.dxnn", None, "sample/vpr/queries/q1.jpg", True, True),
    ("Person Re-ID             (RepVGG-A0)", "Retrieval & Matting", "repvgg-a0-reid_256x128", "person_reid/repvgg_reid", "repvgg-a0-reid_256x128", "repvgg-a0-reid_256x128.dxnn", None, "sample/reid/queries/sample_person_a2.jpg", True, True),
    ("Image Matting            (PP-Matting HRNet-W48)", "Retrieval & Matting", "ppmatting-hrnet-w48-composition_512x512", "image_matting/ppmatting", "ppmatting-hrnet-w48-composition_512x512", "ppmatting-hrnet-w48-composition_512x512.dxnn", "assets/videos/person-pair-hallway.mp4", "sample/img/sample_person_b.jpg", True, True),
]

D_LABEL, D_GROUP, D_CPP, D_PYDIR, D_PYBASE, D_MODEL, D_VIDEO, D_IMAGE, D_PYASYNC, D_IMGONLY = range(10)


def cprint(text, color=""):
    print(f"{color}{text}{RESET}")


def banner():
    line = "=" * 63
    cprint(f"\n{line}", CYAN)
    cprint(f"  {BOLD}{CYAN}DX-APP Interactive Demo{RESET}")
    cprint(f"  Datexel NPU Inference  |  {len(DEMOS)} AI Tasks available", DIM)
    cprint(f"{line}\n", CYAN)


def select_menu(title, options, default=0):
    """Display a numbered menu and return the selected index."""
    print(f"\n  {BOLD}{CYAN}{title}{RESET}\n")
    prev_group = None
    for i, opt in enumerate(options):
        if isinstance(opt, tuple):
            group, label = opt
            if group != prev_group:
                print(f"\n  {YELLOW}[ {group} ]{RESET}")
                prev_group = group
            print(f"   {i:2d}: {label}")
        else:
            print(f"   {i+1}: {opt}")

    # Grouped menus are 0-indexed; flat menus are 1-indexed.
    grouped = isinstance(options[0], tuple)
    prompt_min = 0 if grouped else 1
    prompt_max = len(options) - 1 if grouped else len(options)

    while True:
        try:
            raw = input(f"\n  Select [{prompt_min}-{prompt_max}, default: {default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if not raw:
            return default if grouped else default - 1

        try:
            val = int(raw)
            if prompt_min <= val <= prompt_max:
                return val if grouped else val - 1
        except ValueError:
            pass
        cprint(f"  Invalid input: '{raw}'", RED)


def find_bin_dir():
    """Find the bin directory with executables."""
    bin_dir = DX_APP_PATH / "bin"
    if bin_dir.is_dir() and any(bin_dir.iterdir()):
        return bin_dir
    return None


def run_all(show_log=False):
    """Run EVERY demo non-interactively and print a PASS/FAIL/SKIP summary.

    Default matrix per task: C++ async + Python async (when available), image input,
    --no-display, save mode, no --show-log. Missing model/binary/input → SKIP (not fail).
    Returns process exit code (nonzero if any demo FAILED).
    """
    os.chdir(DX_APP_PATH)
    banner()
    cprint("  [--all] Running all demos: C++ & Python async, image, --no-display, save\n", CYAN)
    win = sys.platform == "win32"
    ok = fail = skip = 0
    failed_list = []
    line = "-" * 63
    for demo in DEMOS:
        model_path = str(resolve_model_path(demo[D_MODEL]))
        image = demo[D_IMAGE]
        model_ok = Path(model_path).is_file()
        input_ok = (DX_APP_PATH / image).exists()
        # Async variants: C++ async + Python async + Python async+C++postprocess.
        #   (lang, exec_name, display_label). Python variants only when D_PYASYNC.
        model_stem = Path(demo[D_MODEL]).stem
        variants = [("cpp", f"{model_stem}_async", "C++ async")]
        if demo[D_PYASYNC]:
            variants.append(("py", f"{model_stem}_async", "Python async"))
            variants.append(("py", f"{model_stem}_async_cpp_postprocess", "Python async+cpp_pp"))
        for lang, name, mode_label in variants:
            tag = f"{demo[D_LABEL].split('(')[0].strip()} [{mode_label}]"
            if not model_ok:
                cprint(f"  [SKIP] {tag}  (model missing: {model_path})", YELLOW); skip += 1; continue
            if not input_ok:
                cprint(f"  [SKIP] {tag}  (input missing: {image})", YELLOW); skip += 1; continue
            if lang == "cpp":
                exe = DX_APP_PATH / "bin" / (name + (".exe" if win else ""))
                if not exe.exists():
                    cprint(f"  [SKIP] {tag}  (binary missing: bin/{name} — build first)", YELLOW); skip += 1; continue
                cmd = [str(exe), "-m", model_path, "-i", image, "--no-display", "-s"]
            else:
                script = (
                    DX_APP_PATH / "src" / "python_example" / demo[D_PYDIR]
                    / model_stem / (name + ".py")
                )
                if not script.exists():
                    cprint(f"  [SKIP] {tag}  (example missing: {script.name})", YELLOW); skip += 1; continue
                cmd = [sys.executable, str(script), "--model", model_path, "--image", image, "--no-display", "--save"]
            if show_log:
                cmd.append("--show-log")
            print(f"  {DIM}$ {' '.join(cmd)}{RESET}")
            sys.stdout.flush()
            try:
                r = subprocess.run(cmd, cwd=str(DX_APP_PATH), capture_output=True, text=True, timeout=600)
            except subprocess.TimeoutExpired:
                cprint(f"  [FAIL] {tag}  (timeout)", RED); fail += 1; failed_list.append(tag); continue
            if r.returncode == 0:
                cprint(f"  [ OK ] {tag}", GREEN); ok += 1
            else:
                cprint(f"  [FAIL] {tag}  (exit {r.returncode})", RED)
                for ln in (r.stdout + r.stderr).strip().splitlines()[-12:]:
                    print(f"        {ln}")
                fail += 1; failed_list.append(tag)
    print(f"\n{CYAN}{line}{RESET}")
    cprint(f"  [--all] OK={ok}  FAIL={fail}  SKIP={skip}  ({ok + fail + skip} runs total)",
           GREEN if fail == 0 else RED)
    if failed_list:
        cprint("  Failed: " + ", ".join(failed_list), RED)
    print(f"{CYAN}{line}{RESET}")
    # 3-state exit: 1 if any FAIL / 0 if at least one ran OK (PASS) / 77 if nothing ran (SKIP)
    if fail:
        return 1
    return 0 if ok else 77


def main():
    parser = argparse.ArgumentParser(description="DX-APP Interactive Demo")
    parser.add_argument("--task", type=int, default=None,
                        help=f"Pre-select task (0-{len(DEMOS) - 1})")
    parser.add_argument("--mode", type=int, default=None, help="Pre-select mode (1-6)")
    parser.add_argument("--input", type=int, default=None, help="Pre-select input (1=video, 2=image)")
    parser.add_argument("--show-log", action="store_true", help="Enable verbose log")
    parser.add_argument("--all", action="store_true",
                        help="Run ALL demos non-interactively (C++ & Python async, image, "
                             "--no-display, save) and print a PASS/FAIL/SKIP summary")
    args = parser.parse_args()

    if args.all:
        sys.exit(run_all(show_log=args.show_log))

    os.chdir(DX_APP_PATH)

    banner()

    # Check bin directory
    if not find_bin_dir():
        cprint("  [WARN] bin/ not found. Run build.bat first to build the project.", YELLOW)
        cprint("         C++ demo modes will not work without building.\n", YELLOW)

    # ═══ Stage 1: Task Selection ═══
    if args.task is not None:
        if not (0 <= args.task < len(DEMOS)):
            cprint(f"  Invalid task: {args.task}", RED)
            sys.exit(1)
        task_idx = args.task
    else:
        task_options = [(d[D_GROUP], d[D_LABEL]) for d in DEMOS]
        task_idx = select_menu("[ Stage 1/3 ]  Select AI Task", task_options, default=0)

    demo = DEMOS[task_idx]
    cprint(f"\n  >> Task: {demo[D_LABEL]}", GREEN)

    # ═══ Stage 2: Mode Selection ═══
    modes = [
        ("C++ Sync", "cpp_sync"),
        ("C++ Async", "cpp_async"),
        ("Python Sync", "py_sync"),
    ]
    if demo[D_PYASYNC]:
        modes.append(("Python Async", "py_async"))
    modes.append(("Python Sync + C++ Postprocess", "py_sync_cpp_postprocess"))
    if demo[D_PYASYNC]:
        modes.append(("Python Async + C++ Postprocess", "py_async_cpp_postprocess"))

    if args.mode is not None:
        if not (1 <= args.mode <= len(modes)):
            cprint(f"  Invalid mode: {args.mode}", RED)
            sys.exit(1)
        mode_idx = args.mode - 1
    else:
        mode_labels = [m[0] for m in modes]
        mode_idx = select_menu("[ Stage 2/3 ]  Select Execution Mode", mode_labels, default=1)

    selected_mode = modes[mode_idx][1]
    cprint(f"  >> Mode: {modes[mode_idx][0]}", GREEN)

    # ═══ Stage 3: Input Selection ═══
    if demo[D_IMGONLY]:
        input_type = "image"
        cprint(f"  >> Input: image only (video not applicable)", GREEN)
    elif args.input is not None:
        input_type = "image" if args.input == 2 else "video"
    else:
        input_options = [
            f"Video  ({demo[D_VIDEO]})",
            f"Image  ({demo[D_IMAGE]})",
        ]
        input_idx = select_menu("[ Stage 3/3 ]  Select Input Type", input_options, default=1)
        input_type = "video" if input_idx == 0 else "image"

    input_file = demo[D_IMAGE] if input_type == "image" else demo[D_VIDEO]
    cprint(f"  >> Input: {input_type} ({input_file})", GREEN)

    # ═══ Build Command ═══
    model_file = resolve_model_path(demo[D_MODEL])
    model_path = str(model_file)

    # NOTE: "async" contains the substring "sync" — never test membership here.
    # Derive the suffix from the mode key by stripping its language prefix.
    if selected_mode.startswith("cpp_"):
        suffix = "_" + selected_mode[len("cpp_"):]
        exe_name = f"{Path(demo[D_MODEL]).stem}{suffix}"
        if sys.platform == "win32":
            exe_name += ".exe"
        exe_path = DX_APP_PATH / "bin" / exe_name
        cmd = [str(exe_path), "-m", model_path]
        if input_type == "video":
            cmd += ["-v", input_file]
        else:
            cmd += ["-i", input_file]
    else:
        model_stem = Path(demo[D_MODEL]).stem
        py_script_name = f"{model_stem}_{selected_mode[len('py_'):]}.py"
        py_script = (
            DX_APP_PATH / "src" / "python_example" / demo[D_PYDIR]
            / model_stem / py_script_name
        )
        cmd = [sys.executable, str(py_script), "--model", model_path]
        if input_type == "video":
            cmd += ["--video", input_file]
        else:
            cmd += ["--image", input_file]

    if args.show_log:
        cmd.append("--show-log")

    # ═══ Pre-flight Checks ═══
    model_full = model_file
    input_full = DX_APP_PATH / input_file

    if not model_full.exists():
        cprint(f"\n  [ERR] Model not found: {model_path}", RED)
        cprint(f"        Run: setup.bat to download models", YELLOW)
        sys.exit(1)

    if not input_full.exists():
        cprint(f"\n  [ERR] Input not found: {input_file}", RED)
        cprint(f"        Run: setup.bat to download sample assets", YELLOW)
        sys.exit(1)

    if selected_mode.startswith("cpp_"):
        if not exe_path.exists():
            cprint(f"\n  [ERR] Executable not found: {exe_path.name}", RED)
            cprint(f"        Run: build.bat to compile the project", YELLOW)
            sys.exit(1)

    # ═══ Execute ═══
    line = "-" * 63
    print(f"\n{CYAN}{line}{RESET}")
    print(f"  {BOLD}Task  :{RESET} {demo[D_LABEL]}")
    print(f"  {BOLD}Mode  :{RESET} {modes[mode_idx][0]}")
    print(f"  {BOLD}Input :{RESET} {input_type} ({input_file})")
    print(f"  {BOLD}Cmd   :{RESET} {' '.join(cmd)}")
    print(f"{CYAN}{line}{RESET}\n")

    try:
        result = subprocess.run(cmd, cwd=str(DX_APP_PATH))
        sys.exit(result.returncode)
    except FileNotFoundError:
        cprint(f"\n  [ERR] Could not execute: {cmd[0]}", RED)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
