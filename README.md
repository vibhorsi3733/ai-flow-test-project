# Motion-ai — IPL live highlights (toss → end of powerplay, or the full match)

Takes a live **SRT** feed and builds a highlights package automatically when the powerplay ends
(`HIGHLIGHT_SCOPE=powerplay`, default) or when the input ends (`HIGHLIGHT_SCOPE=match`: both innings, with
title / innings / break / result cards).
Clips are cut **during** the match, so at powerplay end only curation + joining remains.

## Architecture

```
SRT ─► ingest (2s segments) ─► buffer ─┬─► crowd/loudness excitement ──┐
                                       ├─► commentary: Whisper + Claude ┼─► fuser ─► clip per event
                                       └─► scoreboard OCR  (optional) ──┘              │
                                                                         powerplay end ▼
                                                     Claude curator ─► renderer ─► output/*.mp4 + .srt + .json
```

| Folder | What it does |
|---|---|
| `motion_ai/ingest/` | SRT → timestamped segments; auto-reconnect; rolling buffer |
| `motion_ai/detectors/` | Independent signal sources. **`__init__.py` is the only place that reads `OCR_AVAILABLE`** |
| `motion_ai/fusion/` | Combines sources into final events (rules differ for OCR on/off — see `fuser.py` docstring) |
| `motion_ai/highlights/` | Clip cutting, Claude curation (with rule-based fallback), rendering, powerplay trigger |

## OCR is optional

| `OCR_AVAILABLE` | Event sources | Notes |
|---|---|---|
| `false` (default) | Commentary (Whisper + Claude) + crowd audio | No OCR packages needed. Powerplay end from commentary, manual trigger or safety timer |
| `true` | + scoreboard OCR as the source of truth for runs/wickets/overs | Needs `config/scoreboard.json` and `pip install -r requirements-ocr.txt`. Fails at startup with a clear message if missing |

## Run

```bash
pip install -r requirements.txt          # + requirements-ocr.txt if OCR_AVAILABLE=true
cp .env.example .env                     # set SRT_URL, ANTHROPIC_API_KEY, OCR_AVAILABLE
cp config/match.example.json config/match.json   # teams + players for this match
python -m motion_ai
```

Requires `ffmpeg` built with SRT support on PATH, and an NVIDIA GPU for Whisper (or `ASR_DEVICE=cpu`).

**Manual powerplay trigger** (any time): `touch runtime/TRIGGER_POWERPLAY_END`

## Full match (`HIGHLIGHT_SCOPE=match`)

* Powerplay end is only logged; every clip of both innings is kept for curation; the package is built at end of input.
* 2nd innings = scoreboard drops >= 30 runs and >= 30 balls after >= 10 overs, held 6 reads / 3 s (a highlights
  video joins the chase mid-over, e.g. 31/0 in 2.4). `TARGET - n` is read from its own OCR region.
* Budget (`HIGHLIGHT_TARGET_SECONDS`, cards included, hard max 360 s): every wicket and the winning runs are always in;
  the rest is split 50/50 by innings (sixes, then fours from the last over back). Every dropped clip is listed in
  the manifest `withheld[]` with its reason.
* Needs `OCR_AVAILABLE=true`: only scoreboard-confirmed boundaries/wickets are shown, so startup fails without OCR.
* Cards show a number only from a confirmed read; `wicket_audit` compares wickets on the scoreboard with wickets
  clipped per innings, and any shortfall holds the package (`wicket_missing`, naming each score not clipped).
  A confirmed end-of-innings read (`innings_end`, e.g. 179/7) is authoritative for `wickets_on_board`, so a
  last-ball wicket that never showed as a score change is still named ("innings 1: 179/7 (20.0 ov) not clipped").

## Selected overs (`HIGHLIGHT_OVERS`)

Only some overs of each batting team, in both scopes. Empty/unset = every over.

```bash
HIGHLIGHT_OVERS=[{"team":"Triumph Knights","overs":"1-6"},{"team":"Eagle Thane Strikers","overs":[1,2,"16-20"]}]
```

* `overs`: an `"a-b"` range, a list of overs, or a mix (`["1-6", 20]`), each 1..20. A team not listed keeps all overs.
* Team names are checked against `config/match.json` `teams` at startup (unknown name = clear error). Innings 1 is
  batted by `"batting_first"` (add it to match.json, e.g. `"batting_first": "Triumph Knights"`; if missing, the
  first team key is used and a warning is logged), innings 2 by the other team.
* Over of a ball = from the scoreboard `o.b` at the ball (`event_overs`, else `overs`): `o.b` with b > 0 is over
  o+1 (`0.1` -> 1, `5.6` -> 6), `o.0` is over o (`6.0` -> 6, the ball that completed it). A commentary-only clip
  takes the next ball after the last board reading. Toss and innings-start clips are never filtered.
* Clips outside the selection go to `withheld[]` with reason `not_selected_over`; wickets outside it are listed in
  `wicket_audit.innings_N.not_selected_over` and never count as missing.
* Cards carry a soft crowd bed from the match's own feed (no dead air): back-to-back cards share one stretch taken
  right after the previous clip (the reaction), or just before the first clip for the opening cards; band-limited
  to 80-400 Hz so commentary is only a murmur, normalised to -36 LUFS (~16 dB under the programme), 0.4 s fades.
  Manifest `cards[].crowd_bed_t` = stream time used; no source audio -> silent card + a warning.

## Output quality

One encode only (ingest is `-c copy`): no upscale (a 1080p source is not scaled at all), source frame rate,
`libx264 -preset slow -crf 17 -profile:v high`, AAC 192k at the source rate. After rendering, a **binding quality
gate** compares every clip (before its caption) with the original source: VMAF mean >= 95, 5th percentile >= 90,
SSIM >= 0.985, plus 1080p / fps / High profile / bitrate >= source. One failure holds the package; the table is
in the manifest (`quality_gate`). Re-run it with `python -m motion_ai.quality output/<package>.json`.

**Iterate without re-processing** — every run records `runtime/events.jsonl` + `runtime/segments.jsonl`:

```bash
python -m motion_ai.replay --plan-only   # re-fuse + clip plans as JSON (~1s)
python -m motion_ai.replay               # re-render the package from cached segments
python -m motion_ai.rebuild              # re-curate already-rendered clips (no encoding)
python -m motion_ai.redetect ocr         # re-run one detector (ocr | commentary | audio) on the recorded segments
python -m motion_ai.redetect ocr --apply # ...and swap its events into events.jsonl
```

**Publishing safety** — the manifest carries `status` (`auto_ok` / `hold_for_review`), per-clip
`hold_reasons` (run_out, review, asr_only, no_call_anchor, no_cut_snap, slow_confirm, ocr_low_conf,
over_cap), `approve_by`, `on_timeout_result` and `publish_channel_allowed: false` (until the IPL
template and approve step exist). Held clips are never dropped and never auto-published.

## Approval

```bash
python -m motion_ai.approve output/<package>.json [--port 8765] [--clips-dir runtime_1080/clips]
# then open http://127.0.0.1:8765/
```

A local review page (stdlib only, bound to 127.0.0.1) shows the package summary (status, holds,
quality gate, wicket audit, time left until `approve_by`), every clip with a video player, caption,
score/overs, sources and hold reasons, and the withheld clips (read-only). Enter your reviewer name,
then Approve / Reject each clip (optional note) and the package.

- Decisions go to `<package>.approvals.json` next to the manifest (who, when, decision, note per
  clip and for the package, plus an append-only history). The manifest is never modified.
- The package counts as approved only when the reviewer approved it, every held clip has a
  decision, and every hard-blocker hold (`quality_gate_failed`, `wicket_missing`) is named in the
  package note. The result (approved / rejected / auto-included clip ids, blockers) is stored as
  `publish_decision` in the approvals file (`motion_ai.approve.compute_publish_decision`).
- `publish_channel_allowed` stays false: this step only records approval; publishing is separate.
- Only files the manifest references are served (package mp4 and each clip's `clip_file`, looked
  up by clip id). If a clip file is missing, the page plays that clip's span of the package mp4.

## Tests

```bash
pytest tests
```
