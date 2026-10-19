"""yolo_ppu family factory.

One factory serves every variant of the ``yolo_ppu`` family: the per-variant
differences -- processor classes and their non-default arguments -- live in
``<variant>/config.json`` rather than in 14 near-identical factory files.

Variants needing computed arguments delegate to ``custom_ops.py``.
"""
from pathlib import Path

from common.base import IDetectionFactory
from common.variant_config import (
    build_processor,
    default_variant,
    load_variant_config,
)
import importlib.util as _ilu

_CUSTOM_OPS_PATH = Path(__file__).resolve().parent.parent / "custom_ops.py"
_co_spec = _ilu.spec_from_file_location(
    __name__.rsplit(".", 1)[0] + "_custom_ops", _CUSTOM_OPS_PATH)
custom_ops = _ilu.module_from_spec(_co_spec)
_co_spec.loader.exec_module(custom_ops)

_FAMILY_DIR = str(Path(__file__).resolve().parent.parent)


class YoloPpuFactory(IDetectionFactory):
    """Config-driven factory for the yolo_ppu family."""

    def __init__(self, config: dict = None, variant: str = None):
        self.variant = variant or default_variant(_FAMILY_DIR)
        self.spec = load_variant_config(_FAMILY_DIR, self.variant)
        # The variant config supplies the defaults; an explicit config overrides them.
        self.config = {**(self.spec.get("config") or {}), **(config or {})}

    def _build(self, role, input_width, input_height):
        # Variants whose arguments are computed rather than literal delegate to the
        # family's custom_ops.py, which holds their original factory verbatim.
        if self.variant in custom_ops.FACTORIES:
            delegate = custom_ops.FACTORIES[self.variant](self.config)
            if role == "visualizer":
                return delegate.create_visualizer()
            return getattr(delegate, "create_" + role)(input_width, input_height)
        return build_processor(
            self.spec[role],
            input_width=input_width,
            input_height=input_height,
            config=self.config,
        )

    def create_preprocessor(self, input_width: int, input_height: int):
        return self._build("preprocessor", input_width, input_height)

    def create_postprocessor(self, input_width: int, input_height: int):
        return self._build("postprocessor", input_width, input_height)

    def create_visualizer(self):
        return self._build("visualizer", self.spec["input_width"],
                           self.spec["input_height"])

    def get_model_name(self) -> str:
        return self.variant

    def get_task_type(self) -> str:
        return self.spec["task"]
