import importlib.util
from pathlib import Path

_FACTORY_FILE = Path(__file__).with_name('repvgg-a0-reid_256x128_factory.py')
_SPEC = importlib.util.spec_from_file_location(
    'dxapp_factory_repvgg_a0_reid_256x128_factory', _FACTORY_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'cannot load variant factory {_FACTORY_FILE}')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RepvggReidFactory = getattr(_MODULE, 'RepvggReidFactory')
__all__ = ['RepvggReidFactory']
