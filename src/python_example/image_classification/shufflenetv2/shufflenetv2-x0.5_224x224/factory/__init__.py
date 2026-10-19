import importlib.util
from pathlib import Path

_FACTORY_FILE = Path(__file__).with_name('shufflenetv2-x0.5_224x224_factory.py')
_SPEC = importlib.util.spec_from_file_location(
    'dxapp_factory_shufflenetv2_x0_5_224x224_factory', _FACTORY_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'cannot load variant factory {_FACTORY_FILE}')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Shufflenetv2Factory = getattr(_MODULE, 'Shufflenetv2Factory')
__all__ = ['Shufflenetv2Factory']
