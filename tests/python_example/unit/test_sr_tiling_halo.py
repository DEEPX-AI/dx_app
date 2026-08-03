"""Unit tests for tiled-SR halo resolution (CLI > config.json > env > default)."""

import pytest

import numpy as np

from common.runner.sr_tiling import (
    DEFAULT_HALO,
    MAX_HALO,
    assemble_tiles,
    max_halo_for_tile,
    plan_tiles,
    resolve_halo,
    resolve_halo_with_source,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("DXAPP_SR_TILE_HALO", raising=False)


class TestPrecedence:
    """CLI > config.json > DXAPP_SR_TILE_HALO > default."""

    def test_default_when_nothing_given(self):
        assert resolve_halo() == DEFAULT_HALO
        assert resolve_halo_with_source() == (DEFAULT_HALO, "default")

    def test_env_still_honoured(self, monkeypatch):
        monkeypatch.setenv("DXAPP_SR_TILE_HALO", "0")
        assert resolve_halo() == 0
        assert resolve_halo_with_source() == (0, "env DXAPP_SR_TILE_HALO")

    def test_config_beats_env(self, monkeypatch):
        monkeypatch.setenv("DXAPP_SR_TILE_HALO", "1")
        assert resolve_halo(config={"sr_tile_halo": 2}) == 2
        assert resolve_halo_with_source(config={"sr_tile_halo": 2}) == (2, "config.json")

    def test_cli_beats_config_and_env(self, monkeypatch):
        monkeypatch.setenv("DXAPP_SR_TILE_HALO", "1")
        assert resolve_halo(cli=3, config={"sr_tile_halo": 2}) == 3
        assert resolve_halo_with_source(cli=3, config={"sr_tile_halo": 2}) == (
            3, "--sr-tile-halo")

    def test_cli_zero_is_honoured_not_treated_as_unset(self):
        """0 is a meaningful value (plain tiling), not 'unset'."""
        assert resolve_halo(cli=0, config={"sr_tile_halo": 4}) == 0

    def test_config_without_the_key_falls_through(self, monkeypatch):
        monkeypatch.setenv("DXAPP_SR_TILE_HALO", "3")
        assert resolve_halo(config={"score_threshold": 0.3}) == 3


class TestValidation:
    """Explicit (CLI / config) values are strict; env stays lenient."""

    @pytest.mark.parametrize("bad", [-1, "x", "4.5", 1.5, True])
    def test_invalid_cli_raises(self, bad):
        with pytest.raises(ValueError):
            resolve_halo(cli=bad)

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            resolve_halo(config={"sr_tile_halo": -3})

    def test_bad_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("DXAPP_SR_TILE_HALO", "abc")
        assert resolve_halo() == DEFAULT_HALO

    def test_negative_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("DXAPP_SR_TILE_HALO", "-2")
        assert resolve_halo() == DEFAULT_HALO

    def test_max_halo_for_tile(self):
        assert max_halo_for_tile(17) == 8
        assert max_halo_for_tile(1) == 0

    def test_halo_at_max_is_accepted(self):
        assert MAX_HALO == 4
        assert resolve_halo(cli=MAX_HALO, tile_h=17, tile_w=17) == 4

    def test_halo_over_max_raises(self):
        """Overlap beyond the receptive-field radius buys no accuracy, so 5+ is
        rejected even though a 17x17 tile could geometrically take 8."""
        with pytest.raises(ValueError) as exc:
            resolve_halo_with_source(cli=5, tile_h=17, tile_w=17)
        assert "0..4" in str(exc.value)

    def test_halo_over_max_raises_without_tile_size(self):
        with pytest.raises(ValueError) as exc:
            resolve_halo(cli=9)
        assert "0..4" in str(exc.value)

    def test_small_tile_limit_wins_over_max(self):
        """A 5x5 tile only allows 0..2, which is tighter than MAX_HALO."""
        with pytest.raises(ValueError) as exc:
            resolve_halo(cli=3, tile_h=5, tile_w=5)
        assert "0..2" in str(exc.value)
        assert resolve_halo(cli=2, tile_h=5, tile_w=5) == 2

    def test_env_over_max_still_raises(self, monkeypatch):
        """An out-of-range env value cannot be silently used — plan_tiles would
        raise deep inside the tiling loop, so reject it at resolution time."""
        monkeypatch.setenv("DXAPP_SR_TILE_HALO", "12")
        with pytest.raises(ValueError):
            resolve_halo(tile_h=17, tile_w=17)


class TestTileGeometry:
    def test_tile_counts_scale_with_halo(self):
        """275x150 LR plane, 17x17 tile: larger halo -> smaller stride -> more tiles."""
        counts = [len(plan_tiles(150, 275, 17, 17, halo)[2]) for halo in (0, 2, 4)]
        assert counts[0] < counts[1] < counts[2]
        assert (counts[0], counts[2]) == (153, 480)

    def test_missing_tile_outputs_are_skipped_and_counted(self, caplog):
        """A failed tile (None) and an output with no tensors ([]) both leave their
        region black; neither may raise, and both must be reported."""
        _, _, plans = plan_tiles(34, 34, 17, 17, 0)
        assert len(plans) == 4
        good = [np.full((1, 1, 17, 17), 0.5, dtype=np.float32)]
        outputs = [good, None, [], good]

        sr_y, tiles_done = assemble_tiles(plans, outputs, 34, 34, 1, 1)

        assert tiles_done == 2
        assert sr_y[0, 0] == 127 and sr_y[17, 17] == 127   # stitched tiles
        assert sr_y[0, 17] == 0 and sr_y[17, 0] == 0       # skipped tiles stay black
        assert "2 of 4 tiles produced no usable output" in caplog.text

    def test_all_tiles_present_logs_no_warning(self, caplog):
        _, _, plans = plan_tiles(17, 17, 17, 17, 0)
        outputs = [[np.full((1, 1, 17, 17), 1.0, dtype=np.float32)]]
        _, tiles_done = assemble_tiles(plans, outputs, 17, 17, 1, 1)
        assert tiles_done == 1
        assert "produced no usable output" not in caplog.text

    def test_halo_zero_is_plain_tiling(self):
        padded_h, padded_w, plans = plan_tiles(34, 51, 17, 17, 0)
        assert (padded_h, padded_w) == (34, 51)
        assert len(plans) == 2 * 3
        assert all(p.src_y == 0 and p.src_x == 0 for p in plans)
