"""Session + store tests (uses the real ONNX weights; a few seconds total)."""


import app.sessions as sessions
from app.config import MAX_SESSIONS
from app.sessions import CaseSession


def test_session_stages_and_caching():
    s = CaseSession("test-1")
    assert s.case == "bench"
    assert (s.w == s.h > 0)

    first = s.pre(3.0)
    assert first[5] is False          # fresh preprocess
    assert s.pre(3.0)[5] is True      # cached preprocess

    prob = s.prob(3.0)
    assert prob[2] is False           # fresh inference
    assert s.prob(3.0)[2] is True     # cached inference

    assert s.pre(4.0)[5] is False     # new clip invalidates preprocess


def test_upload_sets_case_and_thumb():
    from pathlib import Path
    png = Path("samples", "fundus_sample_1.png").read_bytes()
    s = CaseSession("test-2")
    s.set_upload(png, "my scan.png")
    assert s.case == "upload"
    assert s.upload_name == "my scan.png"
    assert s.thumb_bytes[:4] == b"\x89PNG"
    assert (s.w > 0 and s.h > 0)


def test_store_lru_eviction():
    first = sessions.get_session("e-oldest")
    for i in range(MAX_SESSIONS):
        sessions.get_session(f"e-{i}")
    assert first.sid not in sessions._sessions
    assert len(sessions._sessions) <= MAX_SESSIONS
