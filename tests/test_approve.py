import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from motion_ai.approve.decision import compute_publish_decision, hold_acknowledged
from motion_ai.approve.media import MediaIndex, safe_basename
from motion_ai.approve.server import parse_range, serve
from motion_ai.approve.store import (ApprovalError, approvals_path, load_approvals,
                                     record_clip, record_package, save_approvals)


def manifest(holds=("wicket_missing", "quality_gate_failed")):
    return {
        "status": "hold_for_review",
        "package_holds": list(holds),
        "held_clip_ids": ["a1", "b2"],
        "clips": [
            {"id": "a1", "type": "wicket", "clip_file": "000070_wicket_a1.mp4", "hold_reasons": ["run_out"]},
            {"id": "b2", "type": "four", "clip_file": "000080_four_b2.mp4", "hold_reasons": ["no_cut_snap"]},
            {"id": "c3", "type": "six", "clip_file": "000090_six_c3.mp4", "hold_reasons": []},
        ],
        "withheld": [{"id": "w9", "type": "four", "reason": "same_ball_as_winning_runs"}],
        "publish_channel_allowed": False,
    }


def dec(decision, note="", reviewer="Asha"):
    return {"decision": decision, "reviewer": reviewer, "note": note, "decided_at": "t"}


ACK = "Ack quality_gate_failed (clip 4 re-encode ok) and wicket missing (179/7 is last ball)."


# ---------------------------------------------------------------- decision

def test_nothing_decided_blocks_everything():
    d = compute_publish_decision(manifest(), None)
    assert not d["package_approved"]
    assert "package_not_decided" in d["blockers"]
    assert "held_clip_undecided:a1" in d["blockers"] and "held_clip_undecided:b2" in d["blockers"]
    assert d["publish_clip_ids"] == []
    assert d["publish_channel_allowed"] is False


def test_full_approval_with_acknowledged_holds():
    ap = {"package": dec("approve", ACK), "clips": {"a1": dec("approve"), "b2": dec("reject")}}
    d = compute_publish_decision(manifest(), ap)
    assert d["package_approved"] and d["status"] == "approved" and d["blockers"] == []
    assert d["approved_clip_ids"] == ["a1"]
    assert d["rejected_clip_ids"] == ["b2"]
    assert d["default_included_clip_ids"] == ["c3"]
    assert d["publish_clip_ids"] == ["a1", "c3"]
    assert d["publish_channel_allowed"] is False


def test_hard_holds_must_be_named_in_note():
    ap = {"package": dec("approve", "looks fine"), "clips": {"a1": dec("approve"), "b2": dec("approve")}}
    d = compute_publish_decision(manifest(), ap)
    assert not d["package_approved"]
    assert set(d["unacknowledged_hard_holds"]) == {"wicket_missing", "quality_gate_failed"}
    ap["package"]["note"] = "ok despite quality_gate_failed"
    d = compute_publish_decision(manifest(), ap)
    assert d["blockers"] == ["hard_hold_not_acknowledged:wicket_missing"]


def test_soft_holds_need_no_acknowledgement():
    ap = {"package": dec("approve"), "clips": {"a1": dec("approve"), "b2": dec("approve")}}
    d = compute_publish_decision(manifest(holds=("over_cap",)), ap)
    assert d["package_approved"]


def test_undecided_held_clip_blocks():
    ap = {"package": dec("approve", ACK), "clips": {"a1": dec("approve")}}
    d = compute_publish_decision(manifest(), ap)
    assert d["blockers"] == ["held_clip_undecided:b2"]
    assert d["publish_clip_ids"] == []


def test_clip_with_hold_reasons_counts_as_held_even_if_not_listed():
    m = manifest(holds=())
    m["held_clip_ids"] = ["a1"]
    ap = {"package": dec("approve"), "clips": {"a1": dec("approve")}}
    assert "held_clip_undecided:b2" in compute_publish_decision(m, ap)["blockers"]


def test_package_rejected():
    ap = {"package": dec("reject", ACK), "clips": {"a1": dec("approve"), "b2": dec("approve")}}
    d = compute_publish_decision(manifest(), ap)
    assert d["status"] == "rejected" and "package_rejected" in d["blockers"]


def test_all_clips_rejected_blocks():
    m = manifest(holds=())
    m["held_clip_ids"] = ["a1", "b2", "c3"]
    m["clips"][2]["hold_reasons"] = ["x"]
    ap = {"package": dec("approve"), "clips": {k: dec("reject") for k in ("a1", "b2", "c3")}}
    assert "no_clips_to_publish" in compute_publish_decision(m, ap)["blockers"]


def test_invalid_decision_values_ignored():
    ap = {"package": dec("yes", ACK), "clips": {"a1": dec("maybe"), "b2": dec("approve")}}
    d = compute_publish_decision(manifest(), ap)
    assert "package_not_decided" in d["blockers"] and "held_clip_undecided:a1" in d["blockers"]


def test_withheld_never_published():
    ap = {"package": dec("approve", ACK), "clips": {"a1": dec("approve"), "b2": dec("approve"),
                                                    "w9": dec("approve")}}
    d = compute_publish_decision(manifest(), ap)
    assert "w9" not in d["publish_clip_ids"] and d["unknown_clip_ids"] == ["w9"]


def test_hold_acknowledged_forms():
    assert hold_acknowledged("quality_gate_failed", "Quality-gate failed: accepted")
    assert hold_acknowledged("wicket_missing", "WICKET_MISSING ok")
    assert not hold_acknowledged("wicket_missing", "wicket ok")


# ---------------------------------------------------------------- path safety

@pytest.mark.parametrize("name", ["../x.mp4", "a/b.mp4", "..\\x.mp4", "/etc/passwd", ".hidden.mp4",
                                  "x.json", "x..mp4", "", None, 5, "clip.mp4\x00.txt"])
def test_unsafe_basenames_rejected(name):
    assert not safe_basename(name)


def test_safe_basename_ok():
    assert safe_basename("000070_wicket_946f5e9f1748.mp4")


@pytest.fixture
def pkg(tmp_path):
    out = tmp_path / "output"
    clips = tmp_path / "clips"
    out.mkdir()
    clips.mkdir()
    (tmp_path / "secret.mp4").write_bytes(b"SECRET")
    (clips / "000070_wicket_a1.mp4").write_bytes(b"0123456789")
    (clips / "000090_six_c3.mp4").write_bytes(b"six")
    m = manifest()
    m["clips"].append({"id": "evil", "clip_file": "../secret.mp4", "hold_reasons": []})
    m["clips"].append({"id": "../x", "clip_file": "000090_six_c3.mp4"})
    mpath = out / "pkg.json"
    mpath.write_text(json.dumps(m))
    (out / "pkg.mp4").write_bytes(b"package")
    return mpath, clips, m


def test_media_index_whitelist(pkg):
    mpath, clips, m = pkg
    idx = MediaIndex(mpath, m, clips)
    assert idx.resolve("a1") == (clips / "000070_wicket_a1.mp4").resolve()
    assert idx.resolve("package") == mpath.with_suffix(".mp4").resolve()
    assert idx.resolve("b2") is None          # referenced but missing on disk
    assert idx.resolve("evil") is None        # traversal in clip_file
    assert idx.resolve("../x") is None        # bad id
    assert idx.resolve("000070_wicket_a1.mp4") is None  # filenames are not keys
    assert idx.resolve("../secret.mp4") is None


def test_media_index_rejects_symlink_escape(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (tmp_path / "outside.mp4").write_bytes(b"x")
    (clips / "000001_four_s1.mp4").symlink_to(tmp_path / "outside.mp4")
    m = {"clips": [{"id": "s1", "clip_file": "000001_four_s1.mp4"}]}
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(m))
    assert MediaIndex(mpath, m, clips).resolve("s1") is None


def test_parse_range():
    assert parse_range(None, 10) is None
    assert parse_range("bytes=0-", 10) == (0, 9)
    assert parse_range("bytes=2-4", 10) == (2, 4)
    assert parse_range("bytes=-3", 10) == (7, 9)
    assert parse_range("bytes=5-100", 10) == (5, 9)
    assert parse_range("bytes=10-", 10) == "invalid"
    assert parse_range("bytes=0-1,3-4", 10) == "invalid"
    assert parse_range("items=0-1", 10) == "invalid"


# ---------------------------------------------------------------- store

def test_store_never_touches_manifest(pkg):
    mpath, _, m = pkg
    before = mpath.read_bytes()
    ap = load_approvals(mpath)
    record_clip(m, ap, "a1", "approve", "Asha", "fine")
    record_package(ap, "approve", "Asha", ACK)
    save_approvals(mpath, m, ap)
    assert mpath.read_bytes() == before
    saved = json.loads(approvals_path(mpath).read_text())
    assert approvals_path(mpath).name == "pkg.approvals.json"
    assert saved["clips"]["a1"]["reviewer"] == "Asha"
    assert saved["package"]["decision"] == "approve"
    assert len(saved["history"]) == 2
    assert saved["publish_decision"]["publish_channel_allowed"] is False


def test_store_validation(pkg):
    mpath, _, m = pkg
    ap = load_approvals(mpath)
    with pytest.raises(ApprovalError):
        record_clip(m, ap, "w9", "approve", "Asha")   # withheld
    with pytest.raises(ApprovalError):
        record_clip(m, ap, "a1", "approve", "  ")     # no reviewer
    with pytest.raises(ApprovalError):
        record_package(ap, "publish", "Asha")


# ---------------------------------------------------------------- server

@pytest.fixture
def live(pkg):
    mpath, clips, m = pkg
    app, httpd = serve(mpath, clips, "127.0.0.1", 0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", mpath
    httpd.shutdown()
    httpd.server_close()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_server_page_media_and_traversal(live):
    base, mpath = live
    code, body, _ = _get(base + "/")
    assert code == 200 and b"/media/a1" in body and b"same_ball_as_winning_runs" in body
    assert _get(base + "/media/a1")[1] == b"0123456789"
    code, body, h = _get(base + "/media/a1", {"Range": "bytes=2-4"})
    assert code == 206 and body == b"234" and h["Content-Range"] == "bytes 2-4/10"
    for bad in ("/media/evil", "/media/../secret.mp4", "/media/%2e%2e/secret.mp4",
                "/media/000070_wicket_a1.mp4", "/../secret.mp4", "/media/b2"):
        code, body, _ = _get(base + bad)
        assert code == 404 and b"SECRET" not in body
    assert _get(base + "/", {"Host": "evil.example:80"})[0] == 403


def test_server_post_decisions(live):
    base, mpath = live
    before = mpath.read_bytes()

    def post(path, obj, ctype="application/json", origin=None):
        h = {"Content-Type": ctype}
        if origin:
            h["Origin"] = origin
        req = urllib.request.Request(base + path, data=json.dumps(obj).encode(), headers=h)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    code, res = post("/api/clip", {"clip_id": "a1", "decision": "approve", "reviewer": "Asha"})
    assert code == 200 and res["entry"]["decision"] == "approve"
    assert post("/api/clip", {"clip_id": "a1", "decision": "approve", "reviewer": "A"},
                ctype="text/plain")[0] == 415
    assert post("/api/clip", {"clip_id": "a1", "decision": "approve", "reviewer": "A"},
                origin="http://evil.example")[0] == 403
    assert post("/api/clip", {"clip_id": "zz", "decision": "approve", "reviewer": "A"})[0] == 400
    post("/api/clip", {"clip_id": "b2", "decision": "reject", "reviewer": "Asha"})
    code, res = post("/api/package", {"decision": "approve", "reviewer": "Asha", "note": ACK})
    assert code == 200 and res["decision"]["package_approved"]
    assert res["decision"]["publish_channel_allowed"] is False
    assert mpath.read_bytes() == before
    assert json.loads(approvals_path(mpath).read_text())["publish_decision"]["package_approved"]


def test_missing_clip_falls_back_to_package_span(pkg):
    from motion_ai.approve.page import _clip_video
    mpath, clips, m = pkg
    idx = MediaIndex(mpath, m, clips)
    html = _clip_video({"clip_file": "000080_four_b2.mp4", "start_in_package": 5.0, "duration": 24.9},
                       idx.url("b2"), idx)
    assert "/media/package#t=5.00,29.90" in html
    assert "not found" in _clip_video({"clip_file": "x.mp4"}, None, idx)
