"""resnext family factory.

One factory serves every variant of the ``resnext`` family: the per-variant
differences -- processor classes and their non-default arguments -- live in
``<variant>/config.json`` rather than in 5 near-identical factory files.

"""
from pathlib import Path

from common.base import IClassificationFactory
from common.variant_config import (
    build_processor,
    default_variant,
    load_variant_config,
)
_FAMILY_DIR = str(Path(__file__).resolve().parent.parent)


class ResnextFactory(IClassificationFactory):
    """Config-driven factory for the resnext family."""

    def __init__(self, config: dict = None, variant: str = None):
        self.variant = variant or default_variant(_FAMILY_DIR)
        self.spec = load_variant_config(_FAMILY_DIR, self.variant)
        # The variant config supplies the defaults; an explicit config overrides them.
        self.config = {**(self.spec.get("config") or {}), **(config or {})}

    def _build(self, role, input_width, input_height):
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
