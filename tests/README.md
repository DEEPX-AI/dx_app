# dx_app tests

Five independent test suites live here. They have different prerequisites, so
pick the one that matches what you want to check before running anything.

| Suite | What it checks | Needs NPU? | Needs `./build.sh`? | Entry point |
|-------|----------------|:----------:|:-------------------:|-------------|
| [`python_example/`](python_example/) | `src/python_example` CLI contract + end-to-end inference | E2E only | no | `cd tests/python_example && pytest -m <marker>` |
| [`python_example/unit/`](python_example/unit/) | Postprocessor / framework logic, fully mocked | no | no | `cd tests/python_example && pytest unit` |
| [`cpp_example/`](cpp_example/) | `bin/` executables: CLI contract + end-to-end inference | E2E only | yes | `cd tests/cpp_example && pytest -m <marker>` |
| [`scripts/`](scripts/) | Repo tooling (`scripts/*.py`, `build.bat` generation, docs) | no | no | `pytest tests/scripts` |
| [`windows/`](windows/) | Windows-specific argument handling and pybind11 bindings | E2E only | yes | `tests\windows\run_tests.bat` |

Run each suite in its own pytest session, as `run_tc.sh` does (it runs
pytest inside each suite's directory). One session over `tests/cpp_example` and `tests/python_example`
together is not supported: 11 test modules have the same file name in both
(`test_e2e.py`, `test_cli_basic.py`, ...), and the C++ suite's
`from conftest import ...` then finds the Python suite's `conftest.py`, so
collection stops with 18 errors.

One helper directory is not a test suite:

- [`test_helpers/`](test_helpers/) — shared constants, discovery utilities, and
  the pytest wiring both example suites use.

## The usual entry point: `run_tc.sh`

Most people should not call `pytest` directly. [`run_tc.sh`](../run_tc.sh) at
the repo root wires up scope, markers, environment and reporting:

```bash
./run_tc.sh --cli                 # CLI contract tests, both languages
./run_tc.sh --python --e2e-quick  # Python examples, image inference only
./run_tc.sh --cpp --e2e           # C++ executables, video inference
./run_tc.sh --vis                 # visualization output checks
./run_tc.sh --coverage            # coverage-instrumented run
./run_tc.sh --camera --camera-index 0
./run_tc.sh --rtsp --rtsp-url rtsp://<addr>/stream
```

Scope flags (`--python`, `--cpp`) default to both. `./run_tc.sh --help` lists
every option.

## Marker dictionary

Both example suites run under `--strict-markers`, so an unregistered marker is
an error rather than a silent no-op. The two `pytest.ini` files share the
descriptions below — keep them in sync when you add a marker.

| Marker | Meaning |
|--------|---------|
| `cli` | CLI argument handling tests |
| `help` | `--help` / usage output tests |
| `e2e` | End-to-end inference tests (real NPU + models required) |
| `e2e_image` | E2E image inference tests |
| `e2e_stream` | E2E stream (video) inference tests |
| `e2e_camera` | E2E camera input tests (requires `--camera-index`) |
| `e2e_rtsp` | E2E RTSP input tests (requires `--rtsp-url`) |
| `e2e_short` | Reduced E2E set of representative models, at least one per task (`E2E_SHORT_MODELS`; used with `--e2e-short`, which runs image + stream for them) |
| `save_mode` | `--save` / `--save-dir` output tests |
| `dump_tensors` | `--dump-tensors` tensor debugging tests |
| `verify` | `DXAPP_VERIFY` numerical verification tests |
| `multi_loop` | `--loop N` multi-iteration tests |
| `signal_handling` | SIGINT graceful shutdown tests |
| `visualization` | Visualization output tests (run + image verification) |
| `sync_exec` / `async_exec` | Applied per case to sync / async examples |

Suite-specific markers: `contract` and `golden` (C++ only), `unit` (Python only).

`sync_exec`, `async_exec` and `e2e_short` are attached to parameters at
collection time rather than written as decorators, so `-m sync_exec` selects
cases across every module in the suite.

## Shared options

Registered once in [`test_helpers/pytest_support.py`](test_helpers/pytest_support.py)
and re-exported by both `conftest.py` files:

| Option | Default | Consumed by |
|--------|---------|-------------|
| `--loop N` | 1 (Python) / 20 (C++ image) | `loop_count` fixture |
| `--camera-index N` | unset → test skips | `e2e_camera` tests |
| `--rtsp-url URL` | unset → test skips | `e2e_rtsp` tests |
| `--stream-duration N` | 10 | camera / RTSP tests |

`--coverage` is C++-suite only (gcovr / lcov report). The autouse
`wait_for_temperature` fixture waits for the NPU to cool below 70 °C before
each `e2e` test and is a no-op everywhere else.

## Contract surface

External automation may drive these pytest suites directly, so the following
are a contract, not an implementation detail:

1. The directory names `tests/cpp_example` and `tests/python_example`, each
   runnable as `pytest -m <marker>` from inside it.
2. The marker names, and which tests each marker selects.
3. The file name `test_e2e_camera_rtsp.py` in both suites, plus the
   `--camera-index` and `--rtsp-url` option names.
4. `cpp_example/conftest.py::resolve_bin_dir()`.
5. The `test_helpers` symbols listed in
   [`test_helpers/__init__.py`](test_helpers/__init__.py).
6. `run_tc.sh`'s option surface.
7. pytest's exit code 5 ("no tests collected"), which means no model was
   installed rather than a failure.

## Running without an NPU

Collection always works, so the fast review loop needs no hardware:

```bash
cd tests/python_example && pytest unit           # mocked unit tests
pytest tests/scripts                             # repo tooling tests
cd tests/cpp_example && pytest -m contract       # source-level guards
cd tests/python_example && pytest --collect-only # what would run
```

## Adding a test

1. Put it in the suite that matches its prerequisites — mocked logic goes in
   `python_example/unit/`, anything needing hardware goes in an example suite
   under an `e2e*` marker.
2. Reuse an existing marker. If you truly need a new one, register it in
   **both** `pytest.ini` files and add it to the dictionary above, or
   `--strict-markers` will fail the run.
3. Share discovery, path, or option logic through `test_helpers/` rather than
   copying it into the second suite — that duplication is what this layout
   exists to prevent.
4. Before renaming a module, marker, or option, check the contract surface
   above.

## `tests/scripts` is hermetic

No test in `tests/scripts` configures, builds, installs or deletes anything
in the real repository, and none rewrites a tracked file.

- `hermetic_guard.py` snapshots `build/`, `build_*/`, `bin/`, `lib/`,
  `include/`, every tracked file and the names under `src/cpp_example/`. The
  autouse fixture in `tests/scripts/conftest.py` takes the snapshot before
  and after every test and fails the test that changed something.
- `build.sh` runs only through `build_sh_sandbox.sh` (driven by
  `build_sh_sandbox.py`): a copy of `build.sh` in a throwaway tree, with
  `cmake`, `ninja`, `python3` and `sudo` replaced by stubs that record their
  arguments.
- Never build while `tests/scripts` runs: the guard would, rightly, report
  the build.

So running `tests/scripts` never needs a reinstall of `bin/` afterwards.

## CI

`scripts/ci_checks.sh` is the one entry point for the repository checks
that need no NPU (those needing the dxrt headers SKIP where they are
missing), and it includes `pytest tests/scripts`. See *Repository checks*
in `docs/source/docs/04_DX-APP_CPP_Example_Test.md`.

There, `tests/scripts` runs with
`--known-failures tests/scripts/known_target_failures.txt`: each test listed
in that file (one `<test id> | <reason>` per line) is known to fail on the
release base and runs as a strict xfail, so the check stays green while it
fails and turns red the day it passes, until its line is deleted. A plain
`pytest tests/scripts` reads no list and shows those tests failing.

## Windows

The tests find the binaries with `tests/test_helpers/platform_paths.py`:
`bin/Release` (or `bin/RelWithDebInfo`, `bin/Debug`) when it exists, else
`bin/`, with `.exe` names. The Linux side of those rules is tested in
`tests/scripts/test_platform_paths.py`. The `tests/windows` suite runs only on
Windows (`tests\windows\run_tests.bat`).
