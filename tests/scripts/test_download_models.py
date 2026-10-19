"""scripts/download_models.py: model name matching (E1), CA bundle (E3) and
the exit status on failed downloads (R7)."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import ssl
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DOWNLOADER = ROOT / "scripts" / "download_models.py"
REGISTRY = ROOT / "config" / "model_registry.json"
MANIFEST = ROOT / "scripts" / "modelzoo_manifest.json"

# Registry model_names that genuinely have no manifest entry. Derived by
# running the bulk dry-run below, not guessed. Empty since 8d0b748: the one
# drift (deit_base384_distilled naming deit-b-distilled_384x384.dxnn) is now a
# registry alias_of row; the deit_base_distilled_2 alias was dropped, so it is on
# deit-b_384x384_distilled.dxnn, which the manifest publishes.
REGISTRY_DRIFT: set[str] = set()


def _load_module():
    spec = importlib.util.spec_from_file_location("download_models_under_test", DOWNLOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def dm():
    return _load_module()


@pytest.fixture(autouse=True)
def _x509_strict_as_python_sets_it(monkeypatch):
    """No test inherits the U-76 opt-in from the developer's shell: several
    write a placeholder "cert" bundle that a real SSLContext would reject."""
    monkeypatch.delenv("DXAPP_TLS_RELAX_X509_STRICT", raising=False)


def _dry_run(tmp_path: Path, *models: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DOWNLOADER), "--dry-run", "--no-json",
         "--output", str(tmp_path / "models"),
         "--models", *models],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def _registry() -> list[dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


# ── E1: model name matching ───────────────────────────────────────────────────

def test_every_registry_model_name_resolves_except_the_drift_list(tmp_path):
    names = [e["model_name"] for e in _registry() if e["model_name"] not in REGISTRY_DRIFT]
    supported = [e["model_name"] for e in _registry() if e.get("supported")]
    assert set(supported) <= set(names), "a supported registry entry is on the drift list"

    result = _dry_run(tmp_path, *names)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined[-3000:]
    assert "not found" not in combined, combined[-3000:]
    # One manifest entry per .dxnn file: an alias_of row names its target's file.
    files = {e["dxnn_file"].lower() for e in _registry() if e["model_name"] in names}
    assert f"Model whitelist: {len(files)} model(s) selected" in combined


def test_drift_entries_are_reported_and_fail(tmp_path):
    # With no drift in the data, a name no registry or manifest knows stands in.
    drift = sorted(REGISTRY_DRIFT) or ["no-such-registry-model"]
    result = _dry_run(tmp_path, *drift)
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    for name in drift:
        assert name in combined
    assert "not found in manifest" in combined


def test_manifest_name_still_matches_case_insensitively(tmp_path):
    result = _dry_run(tmp_path, "yolov8n", "RESNET50")
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Model whitelist: 2 model(s) selected" in combined


def test_dxnn_file_name_matches_with_and_without_extension(tmp_path):
    result = _dry_run(tmp_path, "yolov7-w6_1280x1280.dxnn", "retinaface_mobilenet-0.25_640x640")
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Model whitelist: 2 model(s) selected" in combined
    assert "YoloV7W6" in combined


def test_registry_model_name_alias_matches(tmp_path):
    # Neither is a manifest "name"; both are registry model_names.
    result = _dry_run(tmp_path, "yolo26n_obb", "centerpose_regnetx_1_6gf_fpn")
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Model whitelist: 2 model(s) selected" in combined


def test_one_model_named_twice_is_selected_once(dm):
    manifest = [
        {"name": "Foo", "category": "C", "dxnn_url": "https://h/x/foo_224x224.dxnn", "json_url": None},
        {"name": "Bar", "category": "C", "dxnn_url": "https://h/x/bar.dxnn", "json_url": None},
    ]
    aliases = {"foo_alias": "foo_224x224.dxnn"}
    selected, missing = dm.select_models(manifest, ["foo", "FOO_224X224.dxnn", "foo_alias"], aliases)
    assert [m["name"] for m in selected] == ["Foo"]
    assert missing == []
    selected, missing = dm.select_models(manifest, ["bar,foo_alias", "nope"], aliases)
    assert [m["name"] for m in selected] == ["Foo", "Bar"]  # manifest order
    assert missing == ["nope"]


def test_registry_load_is_best_effort(dm, tmp_path):
    assert dm.load_registry_aliases(tmp_path / "missing.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert dm.load_registry_aliases(broken) == {}
    odd = tmp_path / "odd.json"
    odd.write_text(json.dumps([{"model_name": "a"}, "junk", {"model_name": "b", "dxnn_file": "B.dxnn"}]),
                   encoding="utf-8")
    assert dm.load_registry_aliases(odd) == {"b": "b.dxnn"}


# ── E3: CA bundle ─────────────────────────────────────────────────────────────

def test_ca_bundle_order_env_first_then_system(dm, tmp_path):
    files = {}
    for key in ("requests", "curl", "sslcert", "sys1", "sys2"):
        files[key] = tmp_path / f"{key}.pem"
        files[key].write_text("cert", encoding="utf-8")
    system = [str(files["sys1"]), str(files["sys2"])]
    env = {"REQUESTS_CA_BUNDLE": str(files["requests"]), "CURL_CA_BUNDLE": str(files["curl"]),
           "SSL_CERT_FILE": str(files["sslcert"])}

    assert dm._resolve_ca_bundle(env, system) == str(files["requests"])
    del env["REQUESTS_CA_BUNDLE"]
    assert dm._resolve_ca_bundle(env, system) == str(files["curl"])
    del env["CURL_CA_BUNDLE"]
    assert dm._resolve_ca_bundle(env, system) == str(files["sslcert"])
    del env["SSL_CERT_FILE"]
    assert dm._resolve_ca_bundle(env, system) == str(files["sys1"])
    files["sys1"].unlink()
    assert dm._resolve_ca_bundle(env, system) == str(files["sys2"])
    files["sys2"].unlink()
    assert dm._resolve_ca_bundle(env, system) is None


def test_ca_bundle_skips_env_paths_that_do_not_exist(dm, tmp_path):
    sys1 = tmp_path / "sys1.pem"
    sys1.write_text("cert", encoding="utf-8")
    env = {"REQUESTS_CA_BUNDLE": str(tmp_path / "missing.pem"), "SSL_CERT_FILE": str(tmp_path)}
    assert dm._resolve_ca_bundle(env, [str(sys1)]) == str(sys1)


def test_session_uses_the_resolved_bundle(dm, tmp_path, monkeypatch, capsys):
    pytest.importorskip("requests")
    bundle = tmp_path / "corp.pem"
    bundle.write_text("cert", encoding="utf-8")
    monkeypatch.setattr(dm, "_resolve_ca_bundle", lambda *a, **k: str(bundle))
    session = dm._setup_session()
    assert session.verify == str(bundle)
    assert str(bundle) in capsys.readouterr().out

    monkeypatch.setattr(dm, "_resolve_ca_bundle", lambda *a, **k: None)
    session = dm._setup_session()
    assert session.verify is True  # certifi default, verification still on


def test_tls_verification_is_never_disabled():
    text = DOWNLOADER.read_text(encoding="utf-8")
    assert not re.search(r"verify\s*=\s*False", text)
    assert "CERT_NONE" not in text
    assert "CERT_OPTIONAL" not in text
    assert "_create_unverified_context" not in text
    assert "_create_default_https_context" not in text
    assert not re.search(r"check_hostname\s*=\s*(False|0)\b", text)
    assert not re.search(r"setattr\([^)]*[\"']check_hostname[\"']", text)
    assert not re.search(r"verify_mode\s*=", text)
    # U-76: the only verify_flags write clears X.509 strict mode, nothing else.
    assert re.findall(r"verify_flags\s*[-&|^]?=", text) == ["verify_flags &="]
    assert "verify_flags &= ~ssl.VERIFY_X509_STRICT" in text
    # Nor in the shell layers that install `requests` and run the downloader.
    for script in (DOWNLOADER, ROOT / "setup_sample_models.sh", ROOT / "setup.sh"):
        body = script.read_text(encoding="utf-8")
        for token in ("--trusted-host", "--break-system-packages", "PYTHONHTTPSVERIFY"):
            assert token not in body, f"{script.name} contains {token}"


# ── R7: exit status on failed downloads ───────────────────────────────────────

class _Response:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Session:
    """Stub session: url -> Response, or an exception to raise."""

    verify = "/stub/ca-bundle.pem"

    def __init__(self, routes):
        self.routes = routes
        self.verify_seen = []

    def get(self, url, stream=False, timeout=None, verify=None):
        self.verify_seen.append(verify)
        outcome = self.routes[url]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _write_manifest(tmp_path: Path, names: list[str]) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"name": n, "category": "Test", "dxnn_url": f"https://example.invalid/m/{n}.dxnn", "json_url": None}
        for n in names
    ]), encoding="utf-8")
    return manifest


def _run_main(dm, monkeypatch, tmp_path, session, *extra):
    monkeypatch.setattr(dm, "_setup_session", lambda: session)
    monkeypatch.setattr(sys, "argv", [
        "download_models.py", "--manifest", str(_write_manifest(tmp_path, ["good", "bad"])),
        "--output", str(tmp_path / "out"),
        "--no-json", "--workers", "1", *extra])
    try:
        dm.main()
    except SystemExit as exc:
        return exc.code
    return 0


def test_main_exits_nonzero_when_every_download_fails(dm, monkeypatch, tmp_path, capsys):
    err = ConnectionError("proxy refused")
    session = _Session({"https://example.invalid/m/good.dxnn": err,
                        "https://example.invalid/m/bad.dxnn": err})
    code = _run_main(dm, monkeypatch, tmp_path, session, "--all")
    out = capsys.readouterr()
    assert code not in (0, None)
    combined = out.out + out.err
    assert "2 of 2 file(s) failed" in combined
    assert "good" in combined and "bad" in combined and "proxy refused" in combined


def test_main_exits_nonzero_when_one_download_fails(dm, monkeypatch, tmp_path, capsys):
    session = _Session({"https://example.invalid/m/good.dxnn": _Response(200, b"x" * 10),
                        "https://example.invalid/m/bad.dxnn": _Response(403)})
    code = _run_main(dm, monkeypatch, tmp_path, session, "--all")
    combined = "".join(capsys.readouterr())
    assert code not in (0, None)
    assert "1 of 2 file(s) failed" in combined
    assert "HTTP 403" in combined
    assert (tmp_path / "out" / "good.dxnn").read_bytes() == b"x" * 10
    assert not (tmp_path / "out" / "bad.dxnn").exists()


def test_a_pending_row_403_is_pending_not_a_failure(dm, monkeypatch, tmp_path, capsys):
    """8d0b748's declared `pending: true` rows keep their meaning next to the
    exit status above: their 403 is counted as pending, while a 403 on any
    other row still fails the run."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"name": "good", "category": "Test", "json_url": None,
         "dxnn_url": "https://example.invalid/m/good.dxnn"},
        {"name": "soon", "category": "Test", "json_url": None, "pending": True,
         "dxnn_url": "https://example.invalid/m/soon.dxnn"},
    ]), encoding="utf-8")
    session = _Session({"https://example.invalid/m/good.dxnn": _Response(200, b"g"),
                        "https://example.invalid/m/soon.dxnn": _Response(403)})
    monkeypatch.setattr(dm, "_setup_session", lambda: session)
    monkeypatch.setattr(sys, "argv", [
        "download_models.py", "--manifest", str(manifest), "--all", "--no-json",
        "--output", str(tmp_path / "out"),
        "--workers", "1"])
    try:
        dm.main()
        code = 0
    except SystemExit as exc:
        code = exc.code
    combined = "".join(capsys.readouterr())
    assert code in (0, None), combined
    assert "Pending: 1 file(s) are not published yet" in combined
    assert "failed" not in combined
    assert not (tmp_path / "out" / "soon.dxnn").exists()
    assert not list((tmp_path / "out").glob("*.part"))


def test_main_exits_zero_when_all_downloads_succeed(dm, monkeypatch, tmp_path):
    session = _Session({"https://example.invalid/m/good.dxnn": _Response(200, b"g"),
                        "https://example.invalid/m/bad.dxnn": _Response(200, b"b")})
    assert _run_main(dm, monkeypatch, tmp_path, session, "--all") in (0, None)
    # every request names the session's bundle explicitly (see
    # test_request_uses_the_logged_bundle_not_the_environment)
    assert session.verify_seen == [_Session.verify, _Session.verify]


def test_truncated_download_fails_verification_and_leaves_no_file(dm, monkeypatch, tmp_path, capsys):
    session = _Session({
        "https://example.invalid/m/good.dxnn": _Response(200, b"g" * 4, {"Content-Length": "4"}),
        "https://example.invalid/m/bad.dxnn": _Response(200, b"b" * 3, {"Content-Length": "8"}),
    })
    code = _run_main(dm, monkeypatch, tmp_path, session, "--all")
    combined = "".join(capsys.readouterr())
    assert code not in (0, None)
    assert "expected 8 bytes, got 3" in combined
    assert not (tmp_path / "out" / "bad.dxnn").exists()
    assert not list((tmp_path / "out").glob("*.part"))


def test_empty_download_fails_verification(dm, monkeypatch, tmp_path, capsys):
    session = _Session({"https://example.invalid/m/good.dxnn": _Response(200, b"g"),
                        "https://example.invalid/m/bad.dxnn": _Response(200, b"")})
    code = _run_main(dm, monkeypatch, tmp_path, session, "--all")
    assert code not in (0, None)
    assert "empty file" in "".join(capsys.readouterr())


def test_requested_model_missing_from_manifest_fails(dm, monkeypatch, tmp_path, capsys):
    session = _Session({"https://example.invalid/m/good.dxnn": _Response(200, b"g")})
    code = _run_main(dm, monkeypatch, tmp_path, session, "--models", "good", "no_such_model")
    combined = "".join(capsys.readouterr())
    assert code not in (0, None)
    assert "no_such_model" in combined
    assert (tmp_path / "out" / "good.dxnn").exists()


# ── Fix round 1 ───────────────────────────────────────────────────────────────

def _run_demo_filenames() -> set[str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from run_demo import DEMOS, D_MODEL
    finally:
        sys.path.pop(0)
    return {demo[D_MODEL] for demo in DEMOS}


def test_demo_model_missing_from_manifest_fails(tmp_path):
    from urllib.parse import urlparse
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    demo = _run_demo_filenames()
    victim = sorted(demo)[0]
    trimmed = [m for m in manifest if Path(urlparse(m["dxnn_url"]).path).name != victim]
    assert len(trimmed) == len(manifest) - 1, victim
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(trimmed), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(DOWNLOADER), "--demo-models", "--dry-run", "--no-json",
         "--manifest", str(path), "--output", str(tmp_path / "models")],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    combined = result.stdout + result.stderr
    assert result.returncode == 1, combined[-2000:]
    assert f"Run demo model file(s) not found in manifest: {victim}" in combined
    assert f"1 requested model(s) not found in manifest: {victim}" in combined


def test_request_uses_the_logged_bundle_not_the_environment(dm, tmp_path, monkeypatch):
    requests = pytest.importorskip("requests")
    import io
    from requests.structures import CaseInsensitiveDict

    bundle = tmp_path / "system.pem"
    bundle.write_text("cert", encoding="utf-8")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/nonexistent")
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(dm, "SYSTEM_CA_BUNDLES", (str(bundle),))
    session = dm._setup_session()
    assert session.verify == str(bundle)

    seen = {}

    class Adapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            seen.update(kwargs)
            response = requests.Response()
            response.status_code = 200
            response.headers = CaseInsensitiveDict({"Content-Length": "3"})
            response.raw = io.BytesIO(b"abc")
            response.request = request
            response.url = request.url
            return response

        def close(self):
            pass

    session.mount("https://", Adapter())
    res = dm.download_file("https://example.invalid/m/x.dxnn", tmp_path / "x.dxnn", session)
    assert res["status"] == "ok", res
    assert seen["verify"] == str(bundle)  # not /nonexistent from the environment


def test_models_option_repeats_extend(tmp_path):
    result = _dry_run(tmp_path, "yolov8n", "--models", "resnet50")
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Model whitelist: 2 model(s) selected" in combined


def _write_one_model_manifest(tmp_path: Path) -> Path:
    return _write_manifest(tmp_path, ["good"])


def _run_one(dm, monkeypatch, tmp_path, out: Path):
    session = _Session({"https://example.invalid/m/good.dxnn": _Response(200, b"new")})
    monkeypatch.setattr(dm, "_setup_session", lambda: session)
    monkeypatch.setattr(sys, "argv", [
        "download_models.py", "--manifest", str(_write_one_model_manifest(tmp_path)),
        "--output", str(out),
        "--no-json", "--workers", "1", "--all"])
    try:
        dm.main()
    except SystemExit as exc:
        return exc.code
    return 0


def test_output_real_directory_is_used_in_place(dm, monkeypatch, tmp_path):
    out = tmp_path / "models"
    out.mkdir()
    (out / "earlier.dxnn").write_bytes(b"old")
    assert _run_one(dm, monkeypatch, tmp_path, out) in (0, None)
    assert (out / "earlier.dxnn").read_bytes() == b"old"
    assert (out / "good.dxnn").read_bytes() == b"new"


def test_output_valid_symlink_is_kept(dm, monkeypatch, tmp_path):
    target = tmp_path / "shared"
    target.mkdir()
    (target / "earlier.dxnn").write_bytes(b"old")
    out = tmp_path / "models"
    out.symlink_to(target)
    assert _run_one(dm, monkeypatch, tmp_path, out) in (0, None)
    assert out.is_symlink() and os.readlink(out) == str(target)
    assert (target / "earlier.dxnn").read_bytes() == b"old"
    assert (target / "good.dxnn").read_bytes() == b"new"


def test_output_dangling_symlink_is_an_error_and_kept(dm, monkeypatch, tmp_path, capsys):
    out = tmp_path / "models"
    out.symlink_to(tmp_path / "gone")
    code = _run_one(dm, monkeypatch, tmp_path, out)
    combined = "".join(capsys.readouterr())
    assert code not in (0, None)
    assert out.is_symlink() and os.readlink(out) == str(tmp_path / "gone")
    assert "dangling" in combined and "Fix the link" in combined
    assert not (tmp_path / "gone").exists()


# ── U-76: X.509 strict mode (Python 3.13+) behind a TLS-inspecting proxy ──────

AKI_ERROR = ("HTTPSConnectionPool(host='example.invalid', port=443): Max retries exceeded "
             "with url: /m/good.dxnn (Caused by SSLError(SSLCertVerificationError(1, "
             "'[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: Missing Authority "
             "Key Identifier (_ssl.c:1032)')))")


def _strict_default_context(cafile=None):
    """ssl.create_default_context() as Python 3.13+ builds it - on any Python,
    so the tests bite on the 3.12 venv too."""
    context = ssl.create_default_context(cafile=cafile)
    context.verify_flags |= ssl.VERIFY_X509_STRICT | ssl.VERIFY_X509_PARTIAL_CHAIN
    return context


def test_the_context_keeps_verification_on_by_default(dm):
    context = dm.build_ssl_context(None, relax_x509_strict=False, make_context=_strict_default_context)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.verify_flags & ssl.VERIFY_X509_STRICT


def test_relaxing_clears_only_x509_strict(dm):
    strict = _strict_default_context()
    relaxed = dm.build_ssl_context(None, relax_x509_strict=True, make_context=_strict_default_context)
    assert not relaxed.verify_flags & ssl.VERIFY_X509_STRICT
    # PARTIAL_CHAIN, TRUSTED_FIRST and every other flag stay as Python set them.
    assert relaxed.verify_flags == strict.verify_flags & ~ssl.VERIFY_X509_STRICT
    assert relaxed.verify_mode == ssl.CERT_REQUIRED
    assert relaxed.check_hostname is True
    # urllib3's own context pins TLS 1.2; Debian/Ubuntu's 3.12
    # create_default_context() leaves MINIMUM_SUPPORTED.
    assert relaxed.minimum_version >= ssl.TLSVersion.TLSv1_2


def test_the_real_default_context_loads_the_bundle(dm):
    certifi = pytest.importorskip("certifi")
    context = dm.build_ssl_context(certifi.where(), relax_x509_strict=True)
    assert context.cert_store_stats()["x509_ca"] > 0
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True
    assert not context.verify_flags & ssl.VERIFY_X509_STRICT


@pytest.mark.parametrize("value, expected", [
    (None, False), ("", False), ("0", False), ("1", True), ("yes", False)])
def test_the_relaxation_is_opt_in_with_exactly_1(dm, capsys, value, expected):
    env = {} if value is None else {dm.RELAX_X509_STRICT_ENV: value}
    assert dm.relax_x509_strict_requested(env) is expected
    if value == "yes":
        assert "is not 1" in capsys.readouterr().out


def test_the_session_uses_the_relaxed_context_only_when_asked(dm, monkeypatch):
    pytest.importorskip("requests")
    monkeypatch.setattr(dm, "_resolve_ca_bundle", lambda *a, **k: None)
    plain = dm._setup_session().get_adapter("https://example.invalid/")
    assert "ssl_context" not in plain.poolmanager.connection_pool_kw  # requests' own, as before

    monkeypatch.setenv(dm.RELAX_X509_STRICT_ENV, "1")
    adapter = dm._setup_session().get_adapter("https://example.invalid/")
    direct = adapter.poolmanager.connection_pool_kw["ssl_context"]
    proxied = adapter.proxy_manager_for("http://proxy.invalid:3128").connection_pool_kw["ssl_context"]
    assert proxied is direct  # a CONNECT tunnel through $HTTPS_PROXY gets the same context
    # The TLS connection to an https:// proxy itself keeps urllib3's own context.
    assert adapter.proxy_manager_for("https://proxy.invalid:3128").proxy_ssl_context is None
    assert not direct.verify_flags & ssl.VERIFY_X509_STRICT
    assert direct.verify_mode == ssl.CERT_REQUIRED and direct.check_hostname is True
    assert direct.cert_store_stats()["x509_ca"] > 0  # certifi, requests' default, when no bundle


def _one_ca_bundle(tmp_path: Path) -> Path:
    """A PEM holding exactly one CA, the first in certifi's bundle."""
    certifi = pytest.importorskip("certifi")
    pem = Path(certifi.where()).read_text(encoding="ascii")
    first = re.search(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----\n", pem, re.S)
    bundle = tmp_path / "one_ca.pem"
    bundle.write_text(first.group(0), encoding="ascii")
    return bundle


def test_the_relaxed_session_trusts_exactly_the_resolved_bundle(dm, monkeypatch, tmp_path):
    pytest.importorskip("requests")
    bundle = _one_ca_bundle(tmp_path)
    monkeypatch.setattr(dm, "_resolve_ca_bundle", lambda *a, **k: str(bundle))
    monkeypatch.setenv(dm.RELAX_X509_STRICT_ENV, "1")
    session = dm._setup_session()
    assert session.verify == str(bundle)
    context = session.get_adapter("https://example.invalid/").poolmanager.connection_pool_kw["ssl_context"]
    # Not certifi, not the OS store, not a union of them with the bundle.
    assert context.cert_store_stats()["x509_ca"] == 1


def test_an_x509_strict_failure_names_both_fixes(dm):
    hint = dm.x509_strict_hint(["proxy refused", AKI_ERROR], relaxed=False)
    assert hint is not None
    assert "Missing Authority Key Identifier" in hint
    assert "DXAPP_SETUP_PYTHON" in hint and "DXAPP_TLS_RELAX_X509_STRICT=1" in hint
    assert "the certificate chain and the host name are still verified" in hint
    assert "activate a Python 3.12-or-older virtualenv" in hint


def test_no_hint_for_other_failures_or_once_relaxed(dm):
    assert dm.x509_strict_hint(["proxy refused", "HTTP 403"], relaxed=False) is None
    assert dm.x509_strict_hint(
        ["certificate verify failed: self-signed certificate in certificate chain"],
        relaxed=False) is None  # a missing root: a CA-bundle problem, not strict mode
    assert dm.x509_strict_hint([AKI_ERROR], relaxed=True) is None


def test_main_prints_the_hint_when_a_download_fails_strict_mode(dm, monkeypatch, tmp_path, capsys):
    session = _Session({"https://example.invalid/m/good.dxnn": OSError(AKI_ERROR),
                        "https://example.invalid/m/bad.dxnn": _Response(200, b"x" * 10)})
    code = _run_main(dm, monkeypatch, tmp_path, session, "--all")
    err = capsys.readouterr().err
    assert code not in (0, None)
    assert "Missing Authority Key Identifier" in err
    assert "DXAPP_TLS_RELAX_X509_STRICT=1" in err


def test_help_names_the_tls_environment_variable():
    completed = subprocess.run([sys.executable, str(DOWNLOADER), "--help"], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                               timeout=60)
    assert completed.returncode == 0, completed.stdout
    assert "DXAPP_TLS_RELAX_X509_STRICT=1" in completed.stdout
    assert "still verified" in completed.stdout
    assert "https:// proxy itself stays strict" in completed.stdout
