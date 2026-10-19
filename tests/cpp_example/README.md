# C++ Example Tests

> **📄 This documentation has been moved to the main project docs.**
>
> **👉 [DX-APP C++ Example Tests](../../docs/source/docs/04_DX-APP_CPP_Example_Test.md)**

## Quick run

```bash
# from dx_app/, after ./build.sh (the tests run bin/)
python3 -m pytest tests/cpp_example
```

Headless machines need no display: without `DISPLAY` the harness sets `QT_QPA_PLATFORM=offscreen` for the examples.

On Windows the tests find the binaries in `bin/Release` (or another config dir), else `bin/`, with `.exe` names (`tests/test_helpers/platform_paths.py`).
