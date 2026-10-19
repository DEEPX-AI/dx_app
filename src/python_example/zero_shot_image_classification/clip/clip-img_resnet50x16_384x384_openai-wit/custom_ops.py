"""Family-specific processor construction that a variant config cannot express.

These variants pass COMPUTED arguments -- an ``imagenet_mean`` constant, a
``[m * 255.0 for m in mean]`` comprehension, a PPU anchor table -- so they cannot be
reduced to literal JSON. Their original factory modules are carried over verbatim
below and the family factory delegates to them, which preserves behaviour by
construction rather than by a re-derivation that could drift.
"""

from pathlib import Path

from common.processors import CLIPZeroShotPostprocessor
from common.visualizers import ClassificationVisualizer

# ---- carried over verbatim from src/python_example/embedding/vit_b_32_256_datacomp_s34b_b86k/factory/vit_b_32_256_datacomp_s34b_b86k_factory.py ----
"""
Vit_b_32_256_datacomp_s34b_b86k Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IEmbeddingFactory
from common.processors import CLIPImagePostprocessor, SimpleResizePreprocessor
from common.visualizers import EmbeddingVisualizer

# open_clip canonical normalization, on [0,1] pixels. These are the DEFAULT rather than
# an opt-in option because this .dxnn is an open_clip CLIP ViT-B/32 trained under exactly
# this transform: any other normalization silently relocates image embeddings into a
# different space from host-encoded text, so retrieval degrades without ever raising an
# error. Measured against host FP32 over 21 sample images, the previous x/255 stretch
# scored p5 cosine 0.4724 / mean 0.7901; with these constants, 0.8350 / 0.8935.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _check_zero_one_units(values, name: str) -> list:
    """Reject mean/std overrides given in 0-255 units.

    Several factories in this tree legitimately pass raw 0-255 means (e.g. RetinaFace
    passes [104, 117, 123]), so copying that shape here is an easy and completely silent
    mistake: SimpleResizePreprocessor would accept it and emit a plausible-looking but
    wrongly normalized tensor. Fail loudly instead.
    """
    values = [float(v) for v in values]
    if len(values) != 3:
        raise ValueError(f"{name} must have 3 per-channel values, got {len(values)}")
    for channel, value in enumerate(values):
        # A zero mean is meaningful (it is the identity); a zero std would divide by 0.
        too_small = value < 0.0 if name == "mean" else value <= 0.0
        if too_small or value > 1.0:
            raise ValueError(
                f"{name}[{channel}]={value!r} is outside [0,1]. This factory takes "
                f"{name} in open_clip [0,1] units (e.g. {name}="
                f"{CLIP_MEAN if name == 'mean' else CLIP_STD}), not 0-255 units; "
                f"divide a 0-255 value by 255."
            )
    return values


class Vit_b_32_256_datacomp_s34b_b86kFactory(IEmbeddingFactory):
    """Factory for creating Vit_b_32_256_datacomp_s34b_b86k components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        # SimpleResizePreprocessor applies mean/std to the RAW 0-255 resized image, so
        # the [0,1]-scale constants must be rescaled on the way in. The two forms are
        # algebraically identical: (v - 255m) / (255s) == (v/255 - m) / s. Config
        # overrides are given in [0,1] units too, matching the open_clip convention.
        mean = _check_zero_one_units(self.config.get("mean", CLIP_MEAN), "mean")
        std = _check_zero_one_units(self.config.get("std", CLIP_STD), "std")
        return SimpleResizePreprocessor(
            input_width, input_height,
            mean=[m * 255.0 for m in mean],
            std=[s * 255.0 for s in std],
        )

    # ---- zero-shot head (diverges from the carried-over factory ON PURPOSE) ----
    # The carried-over factory returned an image EMBEDDING and compared two images.
    # That is not what `zero_shot_image_classification` means, and the task cannot be
    # completed on device: the zoo's only text tower is an OpenAI RN50 whose .dxnn
    # emits the transformer's [1,77,512] hidden states, not a joint embedding -- a
    # different checkpoint AND an incomplete graph. So the text side is frozen at
    # build time into prompt_bank.json (scripts/build_clip_prompt_bank.py) and the
    # runtime does one numpy dot product against it. Every OTHER clip variant keeps
    # the embedding-comparison behaviour.
    # scripts/build_clip_prompt_bank.py writes the bank once per FAMILY directory,
    # where it is tracked; this file sits in a variant directory, so a bank there
    # overrides the family one.
    _BANK = next(
        (bank for bank in (Path(__file__).resolve().parent / "prompt_bank.json",
                           Path(__file__).resolve().parent.parent / "prompt_bank.json")
         if bank.is_file()),
        Path(__file__).resolve().parent.parent / "prompt_bank.json")

    def _bank_labels(self):
        import json
        return json.loads(self._BANK.read_text(encoding="utf-8"))["labels"]

    def create_postprocessor(self, input_width: int, input_height: int):
        return CLIPZeroShotPostprocessor(
            input_width, input_height,
            {**self.config, "prompt_bank": str(self._BANK),
             "top_k": int(self.config.get("top_k", 5))},
        )

    def create_visualizer(self):
        # ClassificationVisualizer indexes ITS OWN list by class_id, so it has to be
        # given the bank's labels in the bank's order -- an imagenet1000 default would
        # print a confident, entirely unrelated word.
        return ClassificationVisualizer(custom_labels=self._bank_labels())

    def get_model_name(self) -> str:
        return "vit_b_32_256_datacomp_s34b_b86k"

    def get_task_type(self) -> str:
        return "classification"

# ---- carried over verbatim from src/python_example/embedding/vit_l_14_datacomp_xl_s13b_b90k/factory/vit_l_14_datacomp_xl_s13b_b90k_factory.py ----
"""
Vit_l_14_datacomp_xl_s13b_b90k Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IEmbeddingFactory
from common.processors import CLIPImagePostprocessor, SimpleResizePreprocessor
from common.visualizers import EmbeddingVisualizer

# open_clip canonical normalization, on [0,1] pixels. These are the DEFAULT rather than
# an opt-in option because this .dxnn is an open_clip CLIP ViT-L/14 (datacomp_xl) whose
# input tensor is float32 [1,3,224,224] with NO normalization baked into the compiled
# graph: any other normalization silently relocates image embeddings into a different
# space from host-encoded text, so retrieval degrades without ever raising an error.
# The values are this checkpoint's own preprocess_cfg
# (laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K), which is the standard OpenAI CLIP pair.
# On-NPU A/B over 8 sample images moved the mean cross-image cosine from 0.4530 (the
# previous x/255 stretch) to 0.4022 -- the same direction and magnitude as the ViT-B/32
# sibling, where the host-FP32 p5 cosine went 0.4724 -> 0.8350.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _check_zero_one_units(values, name: str) -> list:
    """Reject mean/std overrides given in 0-255 units.

    Several factories in this tree legitimately pass raw 0-255 means (e.g. RetinaFace
    passes [104, 117, 123]), so copying that shape here is an easy and completely silent
    mistake: SimpleResizePreprocessor would accept it and emit a plausible-looking but
    wrongly normalized tensor. Fail loudly instead.
    """
    values = [float(v) for v in values]
    if len(values) != 3:
        raise ValueError(f"{name} must have 3 per-channel values, got {len(values)}")
    for channel, value in enumerate(values):
        # A zero mean is meaningful (it is the identity); a zero std would divide by 0.
        too_small = value < 0.0 if name == "mean" else value <= 0.0
        if too_small or value > 1.0:
            raise ValueError(
                f"{name}[{channel}]={value!r} is outside [0,1]. This factory takes "
                f"{name} in open_clip [0,1] units (e.g. {name}="
                f"{CLIP_MEAN if name == 'mean' else CLIP_STD}), not 0-255 units; "
                f"divide a 0-255 value by 255."
            )
    return values


class Vit_l_14_datacomp_xl_s13b_b90kFactory(IEmbeddingFactory):
    """Factory for creating Vit_l_14_datacomp_xl_s13b_b90k components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        # SimpleResizePreprocessor applies mean/std to the RAW 0-255 resized image, so
        # the [0,1]-scale constants must be rescaled on the way in. The two forms are
        # algebraically identical: (v - 255m) / (255s) == (v/255 - m) / s. Config
        # overrides are given in [0,1] units too, matching the open_clip convention.
        mean = _check_zero_one_units(self.config.get("mean", CLIP_MEAN), "mean")
        std = _check_zero_one_units(self.config.get("std", CLIP_STD), "std")
        return SimpleResizePreprocessor(
            input_width, input_height,
            mean=[m * 255.0 for m in mean],
            std=[s * 255.0 for s in std],
        )

    def create_postprocessor(self, input_width: int, input_height: int):
        return CLIPImagePostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return EmbeddingVisualizer()

    def get_model_name(self) -> str:
        return "vit_l_14_datacomp_xl_s13b_b90k"

    def get_task_type(self) -> str:
        return "embedding"


# ---- carried over verbatim from src/python_example/embedding/vit_l_14_quickgelu_dfn2b/factory/vit_l_14_quickgelu_dfn2b_factory.py ----
"""
Vit_l_14_quickgelu_dfn2b Factory
"""
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.

from common.base import IEmbeddingFactory
from common.processors import CLIPImagePostprocessor, SimpleResizePreprocessor
from common.visualizers import EmbeddingVisualizer

# open_clip canonical normalization, on [0,1] pixels. These are the DEFAULT rather than
# an opt-in option because this .dxnn is an open_clip CLIP ViT-L/14-quickgelu (DFN2B)
# whose input tensor is float32 [1,3,224,224] with NO normalization baked into the
# compiled graph: any other normalization silently relocates image embeddings into a
# different space from host-encoded text, so retrieval degrades without ever raising an
# error. The values are this checkpoint's own preprocess_cfg
# (apple/DFN2B-CLIP-ViT-L-14) -- DFN2B does not use an inception-style 0.5/0.5 pair, it
# carries the standard OpenAI CLIP one. On-NPU A/B over 8 sample images moved the mean
# cross-image cosine from 0.4997 (the previous x/255 stretch) to 0.4239 -- the same
# direction and magnitude as the ViT-B/32 sibling, where the host-FP32 p5 cosine went
# 0.4724 -> 0.8350.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _check_zero_one_units(values, name: str) -> list:
    """Reject mean/std overrides given in 0-255 units.

    Several factories in this tree legitimately pass raw 0-255 means (e.g. RetinaFace
    passes [104, 117, 123]), so copying that shape here is an easy and completely silent
    mistake: SimpleResizePreprocessor would accept it and emit a plausible-looking but
    wrongly normalized tensor. Fail loudly instead.
    """
    values = [float(v) for v in values]
    if len(values) != 3:
        raise ValueError(f"{name} must have 3 per-channel values, got {len(values)}")
    for channel, value in enumerate(values):
        # A zero mean is meaningful (it is the identity); a zero std would divide by 0.
        too_small = value < 0.0 if name == "mean" else value <= 0.0
        if too_small or value > 1.0:
            raise ValueError(
                f"{name}[{channel}]={value!r} is outside [0,1]. This factory takes "
                f"{name} in open_clip [0,1] units (e.g. {name}="
                f"{CLIP_MEAN if name == 'mean' else CLIP_STD}), not 0-255 units; "
                f"divide a 0-255 value by 255."
            )
    return values


class Vit_l_14_quickgelu_dfn2bFactory(IEmbeddingFactory):
    """Factory for creating Vit_l_14_quickgelu_dfn2b components."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def create_preprocessor(self, input_width: int, input_height: int):
        # SimpleResizePreprocessor applies mean/std to the RAW 0-255 resized image, so
        # the [0,1]-scale constants must be rescaled on the way in. The two forms are
        # algebraically identical: (v - 255m) / (255s) == (v/255 - m) / s. Config
        # overrides are given in [0,1] units too, matching the open_clip convention.
        mean = _check_zero_one_units(self.config.get("mean", CLIP_MEAN), "mean")
        std = _check_zero_one_units(self.config.get("std", CLIP_STD), "std")
        return SimpleResizePreprocessor(
            input_width, input_height,
            mean=[m * 255.0 for m in mean],
            std=[s * 255.0 for s in std],
        )

    def create_postprocessor(self, input_width: int, input_height: int):
        return CLIPImagePostprocessor(input_width, input_height, self.config)

    def create_visualizer(self):
        return EmbeddingVisualizer()

    def get_model_name(self) -> str:
        return "vit_l_14_quickgelu_dfn2b"

    def get_task_type(self) -> str:
        return "embedding"


# variant -> the factory class that builds it
FACTORIES = {
    "clip-img_vit-b32_256x256_datacomp-s34b-b86k": Vit_b_32_256_datacomp_s34b_b86kFactory,
    "clip-img_vit-l14_224x224_datacomp-xl-s13b-b90k": Vit_l_14_datacomp_xl_s13b_b90kFactory,
    "clip-img_vit-l14-quickgelu_224x224_dfn2b": Vit_l_14_quickgelu_dfn2bFactory,
}
