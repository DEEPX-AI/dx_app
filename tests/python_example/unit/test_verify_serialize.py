"""DXAPP_VERIFY writer and serializers (U-30): every result type, safe concurrent writes."""
import json
import sys
import threading
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[3] / "src" / "python_example"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.base.i_processor import DetectionResult, FaceAlignmentResult  # noqa: E402
from common.processors.dope_postprocessor import DopeResult  # noqa: E402
from common.processors.restoration_postprocessor import RestorationResult  # noqa: E402
from common.processors.sfa3d_postprocessor import Detection3DResult  # noqa: E402
from common.processors.superpoint_postprocessor import SuperPointResult  # noqa: E402
from common.processors.yolopv2_postprocessor import YOLOPv2Result  # noqa: E402
from common.runner import verify_serialize as vs  # noqa: E402


def _no_fallback(data):
    assert "repr" not in data and "result_type" not in data, data


def test_yolopv2_record_has_boxes_and_mask_stats():
    drivable = np.zeros((4, 8), dtype=np.uint8)
    drivable[:, :4] = 1
    result = YOLOPv2Result(
        detections=[DetectionResult(box=[1.0, 2.0, 3.0, 4.0], confidence=0.5, class_id=3, class_name="vehicle")],
        drivable_mask=drivable, lane_mask=np.zeros((4, 8), dtype=np.uint8))
    data = vs._serialize_results([result], (4, 8))
    _no_fallback(data)
    assert data["detections"] == [{"bbox": [1.0, 2.0, 3.0, 4.0], "conf": 0.5, "class_id": 3, "class_name": "vehicle"}]
    assert data["drivable_stats"]["mean"] == 0.5 and data["lane_stats"]["max"] == 0.0


def test_detection3d_record_mirrors_the_cpp_keys():
    d = Detection3DResult(class_id=1, class_name="Car", confidence=0.5, bev_x=1, bev_y=2, bev_w=3, bev_h=4,
                          x3d=5, y3d=6, z3d=7, dim_h=8, dim_w=9, dim_l=10, yaw=0.25)
    data = vs._serialize_results([d], (608, 608))
    _no_fallback(data)
    assert data["detections"] == [{"class_id": 1, "class_name": "Car", "conf": 0.5, "bev": [1.0, 2.0, 3.0, 4.0],
                                   "center": [5.0, 6.0, 7.0], "dims": [8.0, 9.0, 10.0], "yaw": 0.25}]


def test_superpoint_dope_face_alignment_restoration_are_serialized():
    sp = SuperPointResult(keypoints=[(1.0, 2.0)], scores=[0.5], descriptors=np.zeros((1, 256), dtype=np.float32))
    data = vs._serialize_results([sp], (10, 20))
    _no_fallback(data)
    assert data["detections"][0]["keypoints"] == [{"x": 1.0, "y": 2.0, "conf": 0.5}]
    assert data["descriptor_dim"] == 256
    dope = DopeResult(keypoints=np.full((9, 2), 0.5), centroid=np.array([0.5, 0.5]), confidence=0.9,
                      all_conf=np.full(9, 0.8), pose=None)
    data = vs._serialize_results([dope], (100, 200))
    _no_fallback(data)
    assert data["detections"][0]["keypoints"][0] == {"x": 100.0, "y": 50.0, "conf": 0.8}
    assert data["detections"][0]["has_pose"] is False
    face = FaceAlignmentResult(params=np.zeros(62), landmarks_2d=np.array([[1.0, 2.0]]), pose=[1.0, 2.0, 3.0])
    data = vs._serialize_results([face], (10, 10))
    _no_fallback(data)
    assert data["detections"] == [{"landmarks_2d": [{"x": 1.0, "y": 2.0}], "pose": [1.0, 2.0, 3.0], "params_size": 62}]
    data = vs._serialize_results([RestorationResult(output_image=np.zeros((4, 6, 3), dtype=np.uint8))], (2, 3))
    _no_fallback(data)
    assert data["output_shape"] == [4, 6, 3]


def test_raw_arrays_from_convertless_cpp_postprocess_are_stats():
    data = vs._serialize_results(np.ones((2, 3), dtype=np.float32), (2, 3))
    assert data["output_stats"]["shape"] == [2, 3]
    data = vs._serialize_results((np.ones((1, 6)), np.zeros((1, 4, 4))), (4, 4))
    assert [s["shape"] for s in data["output_stats_list"]] == [[1, 6], [1, 4, 4]]


def test_concurrent_dumps_are_whole_numbered_and_leave_no_temp(tmp_path, monkeypatch):
    monkeypatch.setenv("DXAPP_VERIFY", "1")
    monkeypatch.setenv("DXAPP_VERIFY_DIR", str(tmp_path))
    def writer(t):
        for i in range(25):
            det = DetectionResult(box=[float(t), float(i), 1.0, 1.0], confidence=0.5, class_id=t, class_name="p")
            vs.dump_verify_json([det], "img.jpg", "/models/py_threads_probe.dxnn", "object_detection", (10, 20))
    threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    lines = (tmp_path / "py_threads_probe.frames.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert sorted(r["frame"] for r in records) == list(range(200))
    last = json.loads((tmp_path / "py_threads_probe.json").read_text())
    assert "frame" not in last and last["model"] == "py_threads_probe.dxnn"
    assert not list(tmp_path.glob("*.tmp.*"))


def test_json_holds_the_last_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("DXAPP_VERIFY", "1")
    monkeypatch.setenv("DXAPP_VERIFY_DIR", str(tmp_path))
    for conf in (0.1, 0.9):
        det = DetectionResult(box=[0.0, 0.0, 1.0, 1.0], confidence=conf, class_id=0, class_name="p")
        vs.dump_verify_json([det], "v.mp4", "/models/py_last_probe.dxnn", "object_detection", (1, 1))
    assert json.loads((tmp_path / "py_last_probe.json").read_text())["detections"][0]["conf"] == 0.9


# Review Focus 5: an unusable DXAPP_VERIFY_DIR must not stop a run.
def test_unusable_verify_dir_warns_and_continues(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")
    monkeypatch.setenv("DXAPP_VERIFY", "1")
    monkeypatch.setenv("DXAPP_VERIFY_DIR", str(blocker))
    det = DetectionResult(box=[0.0, 0.0, 1.0, 1.0], confidence=0.5, class_id=0, class_name="p")
    assert vs.dump_verify_json([det], "i.jpg", "/models/py_blocked.dxnn", "object_detection", (1, 1)) is None
    assert "[DXAPP] [WARN] verify_serialize" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.tmp.*"))


# A frames.jsonl that cannot be opened must not use up a frame number: the next
# dump has to start the file, not append to an old run's file.
def test_frames_file_that_cannot_be_opened_is_started_by_the_next_dump(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DXAPP_VERIFY", "1")
    monkeypatch.setenv("DXAPP_VERIFY_DIR", str(tmp_path))
    frames = tmp_path / "py_frames_blocked.frames.jsonl"
    frames.mkdir()  # a directory: opening it as a file fails
    det = DetectionResult(box=[0.0, 0.0, 1.0, 1.0], confidence=0.5, class_id=0, class_name="p")
    vs.dump_verify_json([det], "i.jpg", "/models/py_frames_blocked.dxnn", "object_detection", (1, 1))
    assert "[DXAPP] [WARN] verify_serialize" in capsys.readouterr().out
    frames.rmdir()
    frames.write_text('{"frame": 0, "old_run": true}\n')
    written = vs.dump_verify_json([det], "i.jpg", "/models/py_frames_blocked.dxnn", "object_detection", (1, 1))
    assert written == str(tmp_path / "py_frames_blocked.json")
    records = [json.loads(line) for line in frames.read_text().splitlines()]
    assert [r["frame"] for r in records] == [0] and "old_run" not in records[0]
    assert not list(tmp_path.glob("*.tmp.*"))


def test_a_yolopv2_frame_without_boxes_still_has_a_payload():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from test_helpers.verify import payload_nonempty
    result = YOLOPv2Result(detections=[], drivable_mask=np.ones((4, 8), dtype=np.uint8),
                           lane_mask=np.zeros((4, 8), dtype=np.uint8))
    data = vs._serialize_results([result], (4, 8))
    assert data["detections"] == []
    assert payload_nonempty(data)
    assert not payload_nonempty({"image_height": 4, "image_width": 8, "detections": [],
                                 "drivable_stats": {}, "lane_stats": {"shape": []}})
