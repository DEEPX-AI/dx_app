import importlib.util
from pathlib import Path

_FACTORY_FILE = Path(__file__).with_name('rtdetrv2-r18vd-120e_640x640_factory.py')
_SPEC = importlib.util.spec_from_file_location(
    'dxapp_factory_rtdetrv2_r18vd_120e_640x640_factory', _FACTORY_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'cannot load variant factory {_FACTORY_FILE}')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RtdetrFactory = getattr(_MODULE, 'RtdetrFactory')
__all__ = ['RtdetrFactory']
