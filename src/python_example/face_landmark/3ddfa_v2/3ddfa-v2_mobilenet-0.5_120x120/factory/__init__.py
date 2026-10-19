import importlib.util
from pathlib import Path

_FACTORY_FILE = Path(__file__).with_name('3ddfa-v2_mobilenet-0.5_120x120_factory.py')
_SPEC = importlib.util.spec_from_file_location(
    'dxapp_factory_3ddfa_v2_mobilenet_0_5_120x120_factory', _FACTORY_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'cannot load variant factory {_FACTORY_FILE}')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
N3ddfaV2Factory = getattr(_MODULE, 'N3ddfaV2Factory')
__all__ = ['N3ddfaV2Factory']
