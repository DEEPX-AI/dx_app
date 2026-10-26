import importlib.util
from pathlib import Path

_FACTORY_FILE = Path(__file__).with_name('yolov12-seg-s_640x640_pre-optimized_factory.py')
_SPEC = importlib.util.spec_from_file_location(
    'dxapp_factory_yolov12_seg_s_640x640_pre_optimized_factory', _FACTORY_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'cannot load variant factory {_FACTORY_FILE}')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Yolov12SegFactory = getattr(_MODULE, 'Yolov12SegFactory')
__all__ = ['Yolov12SegFactory']
