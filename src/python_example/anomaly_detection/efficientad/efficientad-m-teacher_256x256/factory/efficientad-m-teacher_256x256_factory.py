"""efficientad family factory.

One factory serves every variant of the ``efficientad`` family: the per-variant
differences -- processor classes and their non-default arguments -- live in
``<variant>/config.json`` rather than in 3 near-identical factory files.

"""
from pathlib import Path

from common.base import IAnomalyDetectionFactory
from common.variant_config import (
    build_processor,
    default_variant,
    load_variant_config,
)
_FAMILY_DIR = str(Path(__file__).resolve().parent.parent)


class EfficientadFactory(IAnomalyDetectionFactory):
    """Config-driven factory for the efficientad family."""

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

    def get_companion_models(self, primary_path: str):
            """The other two EfficientAD networks, beside the one ``-m`` named.

            EfficientAD scores a DISAGREEMENT -- the student's first 384 channels
            predict the teacher and its second 384 predict the autoencoder -- so no
            single network can produce the map. Declaring the set on the factory keeps
            the one-``-m`` CLI contract that run_demo.sh and every sweep rely on; the
            runner resolves these names in the primary model's own directory.

            The returned order matches ``config.roles`` in each variant config, which
            is the only thing that tells two same-shaped 384-channel maps apart.
            """
            from pathlib import Path as _Path
            roles = ("student", "teacher", "autoencoder")
            stem = _Path(primary_path).name
            primary = next((r for r in roles if f"-{r}_" in stem), None)
            if primary is None:
                raise ValueError(
                    f"{stem} does not name an EfficientAD role; expected one of "
                    f"{roles}.")
            return [(other, stem.replace(f"-{primary}_", f"-{other}_"))
                    for other in roles if other != primary]
