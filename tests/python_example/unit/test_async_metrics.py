"""Unit tests for async metric summary formatting."""

import sys
from pathlib import Path


_SRC = Path(__file__).resolve().parents[3] / "src" / "python_example"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utility.profiling import format_async_performance_summary_legacy


def _metrics(**overrides):
    metrics = {
        "sum_read": 0.10,
        "sum_preprocess": 0.20,
        "sum_inference": 0.30,
        "sum_postprocess": 0.40,
        "sum_render": 0.0,
        "sum_save": 0.0,
        "sum_display": 0.0,
        "infer_completed": 10,
        "render_completed": 0,
        "save_completed": 0,
        "display_completed": 0,
        "infer_first_ts": 100.0,
        "infer_last_ts": 101.0,
        "inflight_time_sum": 5.0,
        "inflight_max": 6,
    }
    metrics.update(overrides)
    return metrics


def test_async_summary_preserves_legacy_output_format_with_split_metrics():
    lines = format_async_performance_summary_legacy(
        _metrics(
            sum_inference=0.30,
            sum_inference_turnaround=0.30,
            sum_wait_block=0.05,
            sum_reqid_queue_wait=0.25,
            sum_reqid_enqueue_block=0.01,
        ),
        cnt=10,
        elapsed=1.2,
        display=False,
    )

    text = "\n".join(lines)

    assert "Inference          30.00 ms" in text
    assert "Inference Turnaround" not in text
    assert "Wait Block" not in text
    assert "ReqID Queue Wait" not in text
    assert "ReqID Enqueue Block" not in text
    assert "Throughput measured independently" in text


def test_async_summary_falls_back_to_legacy_inference_field():
    lines = format_async_performance_summary_legacy(
        _metrics(),
        cnt=10,
        elapsed=1.2,
        display=False,
    )

    text = "\n".join(lines)

    assert "Inference" in text
    assert "Inference Turnaround" not in text
    assert "Inference          30.00 ms" in text
    assert "Throughput measured independently" in text
