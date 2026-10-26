import importlib.util
from pathlib import Path

_FACTORY_FILE = Path(__file__).with_name('yolov5-s6_1280x1280_v6.1_factory.py')
_SPEC = importlib.util.spec_from_file_location(
    'dxapp_factory_yolov5_s6_1280x1280_v6_1_factory', _FACTORY_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'cannot load variant factory {_FACTORY_FILE}')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Yolov5Factory = getattr(_MODULE, 'Yolov5Factory')
__all__ = ['Yolov5Factory']
