import importlib.util
from pathlib import Path

_FACTORY_FILE = Path(__file__).with_name('yolo26-cls-s_224x224_factory.py')
_SPEC = importlib.util.spec_from_file_location(
    'dxapp_factory_yolo26_cls_s_224x224_factory', _FACTORY_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'cannot load variant factory {_FACTORY_FILE}')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Yolo26ClsFactory = getattr(_MODULE, 'Yolo26ClsFactory')
__all__ = ['Yolo26ClsFactory']
