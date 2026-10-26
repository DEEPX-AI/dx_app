import importlib.util
from pathlib import Path

_FACTORY_FILE = Path(__file__).with_name('efficientnetv2-m_480x480_factory.py')
_SPEC = importlib.util.spec_from_file_location(
    'dxapp_factory_efficientnetv2_m_480x480_factory', _FACTORY_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'cannot load variant factory {_FACTORY_FILE}')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
EfficientnetFactory = getattr(_MODULE, 'EfficientnetFactory')
__all__ = ['EfficientnetFactory']
