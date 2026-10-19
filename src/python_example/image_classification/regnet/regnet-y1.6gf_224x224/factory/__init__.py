import importlib.util
from pathlib import Path

_FACTORY_FILE = Path(__file__).with_name('regnet-y1.6gf_224x224_factory.py')
_SPEC = importlib.util.spec_from_file_location(
    'dxapp_factory_regnet_y1_6gf_224x224_factory', _FACTORY_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'cannot load variant factory {_FACTORY_FILE}')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RegnetFactory = getattr(_MODULE, 'RegnetFactory')
__all__ = ['RegnetFactory']
