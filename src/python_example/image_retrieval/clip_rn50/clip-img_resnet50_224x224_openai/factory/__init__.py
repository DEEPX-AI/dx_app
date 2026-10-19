import importlib.util
from pathlib import Path

_FACTORY_FILE = Path(__file__).with_name('clip-img_resnet50_224x224_openai_factory.py')
_SPEC = importlib.util.spec_from_file_location(
    'dxapp_factory_clip_img_resnet50_224x224_openai_factory', _FACTORY_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'cannot load variant factory {_FACTORY_FILE}')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ClipFactory = getattr(_MODULE, 'ClipFactory')
__all__ = ['ClipFactory']
