"""run_tiles_pipelined waits for every submitted job when a submit fails -
the Python twin of the C++ SP4 I2 fix. Without it the exception leaves those
jobs unwaited, and once the traceback is dropped `queue` frees the tensors
the engine may still be reading."""
import numpy as np
import pytest

from common.runner.sr_tiling import plan_tiles, run_tiles_pipelined


class FailingEngine:
    """run_async raises on its `fail_at`-th call (0-based); wait() records ids."""

    def __init__(self, fail_at):
        self.fail_at = fail_at
        self.calls = 0
        self.waited = []

    def run_async(self, tensors):
        if self.calls == self.fail_at:
            raise RuntimeError("submit failed")
        self.calls += 1
        return self.calls - 1

    def wait(self, job_id):
        self.waited.append(job_id)
        return [np.zeros((1, 1, 34, 34), np.float32)]


def test_a_failed_submit_waits_for_the_jobs_already_in_flight():
    _, _, plans = plan_tiles(17, 85, 17, 17, 0)  # five tiles in a row
    assert len(plans) == 5
    plane = np.zeros((17, 85), np.uint8)
    engine = FailingEngine(fail_at=2)
    with pytest.raises(RuntimeError, match="submit failed"):
        run_tiles_pipelined(engine, lambda tile: tile, plane, plans, 17, 17, inflight=16)
    assert engine.waited == [0, 1]


def test_the_success_path_is_unchanged():
    _, _, plans = plan_tiles(17, 85, 17, 17, 0)
    engine = FailingEngine(fail_at=-1)
    outputs = run_tiles_pipelined(engine, lambda tile: tile, np.zeros((17, 85), np.uint8),
                                  plans, 17, 17, inflight=2)
    assert engine.waited == [0, 1, 2, 3, 4] and all(o is not None for o in outputs)
