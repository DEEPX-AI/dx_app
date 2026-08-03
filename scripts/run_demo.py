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


# ═══════════════════════════════════════════════════════════════
# Demo Registry
# ═══════════════════════════════════════════════════════════════
DEMOS = [
    # (label, group, cpp_base, py_dir, py_base, model, video, image, py_async, image_only)
    ("Object Detection         (YOLOv7)", "Detection", "yolov7", "object_detection/yolov7", "yolov7", "yolov7_640x640.dxnn", "assets/videos/snowboard.mp4", "sample/img/sample_street.jpg", True, False),
    ("Object Detection         (YOLOv11N)", "Detection", "yolov11n", "object_detection/yolov11n", "yolov11n", "yolo11-n_640x640.dxnn", "assets/videos/boat.mp4", "sample/img/sample_street.jpg", True, False),
    ("Face Detection           (SCRFD500M)", "Detection", "scrfd500m", "face_detection/scrfd500m", "scrfd500m", "scrfd-500m_640x640.dxnn", "assets/videos/dance-group.mov", "sample/img/sample_face.jpg", True, False),
    ("OBB Detection            (YOLO26N-OBB)", "Detection", "yolo26n_obb", "obb_detection/yolo26n_obb", "yolo26n_obb", "yolo26-n-obb_1024x1024.dxnn", "assets/videos/obb.mp4", "sample/img/sample_airport_satellite_view.png", True, False),
    ("Pose Estimation          (YOLOv8s-Pose)", "Pose & Landmark", "yolov8s_pose", "pose_estimation/yolov8s_pose", "yolov8s_pose", "yolov8-s-pose_640x640.dxnn", "assets/videos/dance-solo.mov", "sample/img/sample_people.jpg", True, False),
    ("Hand Landmark            (HandLandmarkLite)", "Pose & Landmark", "handlandmarklite_1", "hand_landmark/handlandmarklite_1", "handlandmarklite_1", "mediapipe-hands-lite_224x224.dxnn", "assets/videos/hand.mp4", "sample/img/sample_hand.jpg", True, False),
    ("Face Alignment           (3DDFA-V2)", "Pose & Landmark", "3ddfa_v2_mobilnetv1_120x120", "face_alignment/3ddfa_v2_mobilnetv1_120x120", "3ddfa_v2_mobilnetv1_120x120", "3ddfa-v2_mobilenetv1_120x120.dxnn", "assets/videos/face-alignment-closeup.mp4", "sample/img/face_pair/1_reference.jpg", True, False),
    ("Instance Segmentation    (YOLOv8N-Seg)", "Segmentation", "yolov8n_seg", "instance_segmentation/yolov8n_seg", "yolov8n_seg", "yolov8-n-seg_640x640.dxnn", "assets/videos/dogs.mp4", "sample/img/sample_street.jpg", True, False),
    ("Semantic Segmentation    (DeepLabV3+)", "Segmentation", "deeplabv3plusmobilenet", "semantic_segmentation/deeplabv3plusmobilenet", "deeplabv3plusmobilenet", "deeplabv3plus_mobilenetv1_512x512.dxnn", "assets/videos/blackbox-city-road.mp4", "sample/img/sample_parking.jpg", True, False),
    ("Classification           (ResNet50)", "Classification", "resnet50", "classification/resnet50", "resnet50", "resnet50_224x224.dxnn", "assets/videos/dogs.mp4", "sample/img/sample_dog.jpg", False, False),
    ("Depth Estimation         (Depth-Anything-V2-ViT-B)", "Depth Estimation", "depth_anything_v2_vitb", "depth_estimation/depth_anything_v2_vitb", "depth_anything_v2_vitb", "depthanythingv2-vitb_224x224.dxnn", "assets/videos/blackbox-city-road.mp4", "sample/img/sample_parking.jpg", True, False),
    ("Image Denoising          (DnCNN-50)", "Image Restoration", "dncnn_50", "image_denoising/dncnn_50", "dncnn_50", "dncnn-50_512x512.dxnn", "assets/videos/noisy_hand.mp4", "sample/img/sample_denoising.jpg", True, False),
    ("Super Resolution         (ESPCN-X4)", "Image Restoration", "espcn_x4", "super_resolution/espcn_x4", "espcn_x4", "espcn-x4_17x17.dxnn", "assets/videos/dance-group.mov", "sample/img/sample_lowres275x150.png", True, False),
    ("Image Enhancement        (Zero-DCE)", "Image Restoration", "zero_dce", "image_enhancement/zero_dce", "zero_dce", "zerodce_400x600.dxnn", "assets/videos/lowlight.mp4", "sample/img/sample_lowlight.jpg", True, False),
    ("Embedding                (ArcFace)", "Recognition", "arcface_mobilefacenet", "embedding/arcface_mobilefacenet", "arcface_mobilefacenet", "arcface_mobilefacenet_112x112.dxnn", "assets/videos/face-pair-sofa.mp4", "sample/img/face_pair", True, True),
    ("Attribute Recognition    (DeepMAR)", "Recognition", "deepmar_resnet50", "attribute_recognition/deepmar_resnet50", "deepmar_resnet50", "deepmar_resnet50_224x224.dxnn", "assets/videos/person-pair-hallway.mp4", "sample/img/sample_person_a1.jpg", True, True),
    ("Person Re-ID             (CasViT-T)", "Recognition", "casvit_t", "reid/casvit_t", "casvit_t", "casvit-t_224x224.dxnn", "assets/videos/person-pair-hallway.mp4", "sample/img/person_pair", True, True),
    ("PPU Pipeline             (YOLOv7-PPU)", "PPU", "yolov7_ppu", "ppu/yolov7_ppu", "yolov7_ppu", "yolov7_640x640_ppu.dxnn", "assets/videos/snowboard.mp4", "sample/img/sample_street.jpg", True, False),
    ("Keypoint Detection       (SuperPoint)", "Keypoint & Pose", "superpoint", "keypoint_detection/superpoint", "superpoint", "superpoint_480x640.dxnn", "assets/videos/blackbox-city-road2.mov", "sample/img/sample_street.jpg", True, False),
    ("Object Pose Estimation   (DOPE)", "Keypoint & Pose", "dope_hope_ketchup", "object_pose_estimation/dope_hope_ketchup", "dope_hope_ketchup", "dope-hope-ketchup_480x640.dxnn", "assets/videos/snowboard.mp4", "sample/dope/000000.png", True, True),
    ("Panoptic Driving         (YOLOPv2)", "Driving & 3D", "yolopv2", "panoptic_driving_perception/yolopv2", "yolopv2", "yolopv2_384x640.dxnn", "assets/videos/blackbox-city-road.mp4", "sample/img/sample_parking.jpg", True, False),
    ("3D Object Detection      (SFA3D)", "Driving & 3D", "sfa3d_608x608", "3d_object_detection/sfa3d_608x608", "sfa3d_608x608", "sfa3d_608x608.dxnn", "assets/videos/blackbox-city-road.mp4", "sample/kitti/velodyne/000049.bin", True, True),
    ("Hand Detection           (MediaPipe Palm)", "Hand Detection", "mediapipe_hand_detector", "hand_detection/mediapipe_hand_detector", "mediapipe_hand_detector", "mediapipe-hand-detector_192x192.dxnn", "assets/videos/hand.mp4", "sample/img/sample_person_a2.jpg", True, False),
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

    prompt_max = len(options) if isinstance(options[0], tuple) else len(options)
    prompt_min = 0 if isinstance(options[0], tuple) else 1

    while True:
        try:
            raw = input(f"\n  Select [{prompt_min}-{prompt_max - (1 if prompt_min == 0 else 0)}, default: {default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if not raw:
            return default if isinstance(options[0], tuple) else default - 1

        try:
            val = int(raw)
            if isinstance(options[0], tuple):
                if 0 <= val < len(options):
                    return val
            else:
                if 1 <= val <= len(options):
                    return val - 1
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
        model_path = f"assets/models/{demo[D_MODEL]}"
        image = demo[D_IMAGE]
        model_ok = (DX_APP_PATH / model_path).exists()
        input_ok = (DX_APP_PATH / image).exists()
        # Async variants: C++ async + Python async + Python async+C++postprocess.
        #   (lang, exec_name, display_label). Python variants only when D_PYASYNC.
        variants = [("cpp", f"{demo[D_CPP]}_async", "C++ async")]
        if demo[D_PYASYNC]:
            variants.append(("py", f"{demo[D_PYBASE]}_async", "Python async"))
            variants.append(("py", f"{demo[D_PYBASE]}_async_cpp_postprocess", "Python async+cpp_pp"))
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
                script = DX_APP_PATH / "src" / "python_example" / demo[D_PYDIR] / (name + ".py")
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
    parser.add_argument("--task", type=int, default=None, help="Pre-select task (0-22)")
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
    model_path = f"assets/models/{demo[D_MODEL]}"

    if selected_mode.startswith("cpp_"):
        suffix = "_sync" if "sync" in selected_mode else "_async"
        exe_name = f"{demo[D_CPP]}{suffix}"
        if sys.platform == "win32":
            exe_name += ".exe"
        exe_path = DX_APP_PATH / "bin" / exe_name
        cmd = [str(exe_path), "-m", model_path]
        if input_type == "video":
            cmd += ["-v", input_file]
        else:
            cmd += ["-i", input_file]
    else:
        py_script_name = f"{demo[D_PYBASE]}_{selected_mode.replace('py_', '')}.py"
        py_script = DX_APP_PATH / "src" / "python_example" / demo[D_PYDIR] / py_script_name
        cmd = [sys.executable, str(py_script), "--model", model_path]
        if input_type == "video":
            cmd += ["--video", input_file]
        else:
            cmd += ["--image", input_file]

    if args.show_log:
        cmd.append("--show-log")

    # ═══ Pre-flight Checks ═══
    model_full = DX_APP_PATH / model_path
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
        exe_check = DX_APP_PATH / "bin" / (f"{demo[D_CPP]}{'_sync' if 'sync' in selected_mode else '_async'}" + (".exe" if sys.platform == "win32" else ""))
        if not exe_check.exists():
            cprint(f"\n  [ERR] Executable not found: {exe_check.name}", RED)
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
