import importlib.util
from pathlib import Path

_FACTORY_FILE = Path(__file__).with_name('yolov6-n0_640x640_v0.1.0_factory.py')
_SPEC = importlib.util.spec_from_file_location(
    'dxapp_factory_yolov6_n0_640x640_v0_1_0_factory', _FACTORY_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'cannot load variant factory {_FACTORY_FILE}')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Yolov6Factory = getattr(_MODULE, 'Yolov6Factory')
__all__ = ['Yolov6Factory']
