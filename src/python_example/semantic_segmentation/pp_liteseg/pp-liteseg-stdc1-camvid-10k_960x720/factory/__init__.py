import importlib.util
from pathlib import Path

_FACTORY_FILE = Path(__file__).with_name('pp-liteseg-stdc1-camvid-10k_960x720_factory.py')
_SPEC = importlib.util.spec_from_file_location(
    'dxapp_factory_pp_liteseg_stdc1_camvid_10k_960x720_factory', _FACTORY_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'cannot load variant factory {_FACTORY_FILE}')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
PpLitesegFactory = getattr(_MODULE, 'PpLitesegFactory')
__all__ = ['PpLitesegFactory']
