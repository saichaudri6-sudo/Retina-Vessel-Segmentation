"""HTTP smoke tests over the whole app (TestClient, same cookie jar per module)."""

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from app.web.app import create_app

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "fundus_sample_1.png"
ALL_MODES = ("overlay", "prob", "mask", "orig", "green", "clahe", "frangi")


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


# --------------------------------------------------------------------- basics
def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["engine"] == "onnxruntime"


def test_index_serves_full_page(client):
    r = client.get("/")
    assert r.status_code == 200
    text = r.text
    assert "RetinaVessel" in text
    assert 'id="tray-card"' in text and 'oob' not in text  # inline, no swap-oob
    assert 'id="bay"' in text
    assert 'id="tuneform"' in text
    assert "set-cookie" in {k.lower() for k in r.headers.keys()}


def test_default_plate_rendered(client):
    r = client.get("/")
    assert "/media/view?mode=overlay" in r.text


# ------------------------------------------------------------- fragments flow
def test_case_switch(client):
    r = client.post("/ui/case", data={"case": "dense"})
    assert r.status_code == 200
    assert "Dense microvascular retina" in r.text
    assert 'id="tray-card" class="railsec tray" hx-swap-oob="true"' in r.text


def test_case_invalid_falls_back(client):
    r = client.post("/ui/case", data={"case": "hax"})
    assert "Benchmark retinal fundus" in r.text


def test_retune_carries_params(client):
    r = client.post("/ui/retune", data={"clip": "6.0", "cutoff": "0.7", "opacity": "0.9"})
    assert r.status_code == 200
    assert "&amp;clip=6.0" in r.text                      # href attrs are HTML-escaped
    assert "/media/view?mode=overlay&amp;clip=6.0&amp;cutoff=0.7&amp;opacity=0.9" in r.text


def test_mode_change_returns_viewer_fragment(client):
    r = client.post("/ui/view", data={"mode": "prob"})
    assert r.status_code == 200
    assert 'id="eyepiece"' in r.text
    assert "palette=magma" in r.text          # prob is a paletted mode
    # swatches visible in paletted modes
    assert "swatch" in r.text


# ------------------------------------------------------------------ media
@pytest.mark.parametrize("mode", ALL_MODES)
def test_media_plates_render(client, mode):
    r = client.get(f"/media/view?mode={mode}")
    assert r.status_code == 200
    assert len(r.content) > 500


def test_media_uses_browser_cache(client):
    r1 = client.get("/media/view?mode=orig")
    etag = r1.headers.get("ETag")
    assert etag
    r2 = client.get("/media/view?mode=orig", headers={"If-None-Match": etag})
    assert r2.status_code == 304


def test_thumb_for_samples(client):
    r = client.get("/media/thumb?which=bench")
    assert r.status_code == 200
    assert r.content[:4] == b"\x89PNG"


def test_upload_and_thumb(client):
    png = SAMPLE.read_bytes()
    r = client.post("/ui/upload", files={"file": ("my_fundus.png", png, "image/png")})
    assert r.status_code == 200
    assert "Uploaded fundus" in r.text or "my_fundus.png" in r.text
    t = client.get("/media/thumb?which=upload")
    assert t.status_code == 200
    assert t.content[:4] == b"\x89PNG"


def test_upload_rejects_junk(client):
    r = client.post("/ui/upload", files={"file": ("junk.png", b"definitely not an image", "image/png")})
    assert r.status_code == 200
    assert "could not be opened" in r.text


def test_upload_rejects_empty(client):
    r = client.post("/ui/upload")
    assert "Choose an image file first." in r.text


# ------------------------------------------------------------------- export
@pytest.mark.parametrize("kind", ("mask", "overlay", "prob"))
def test_export_kinds(client, kind):
    r = client.get(f"/export/{kind}?clip=3.0&cutoff=0.5&opacity=0.7&palette=magma")
    assert r.status_code == 200
    assert 'attachment; filename="retinal_vessel_' in r.headers["Content-Disposition"]
    assert r.content[:4] == b"\x89PNG"


def test_export_unknown_kind(client):
    r = client.get("/export/poster")
    assert r.status_code == 404
    assert "Unknown export kind." in r.text
