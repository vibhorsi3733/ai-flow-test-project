# Motion-ai — IPL live highlights (toss → end of powerplay, or the full match)

Turns a live cricket video feed into a ready-to-review highlights video, with no data feed: everything is read
from the broadcast itself (picture, commentary, crowd).

## How it works

```mermaid
flowchart TD
    SRC["Live feed: SRT stream<br/>(or a recorded file for testing)"] --> ING["1. Ingest<br/>ffmpeg → 2 s segments, no re-encode"]
    ING --> BUF[("2. Rolling buffer<br/>+ recorder: segments.jsonl / events.jsonl")]
    BUF --> AUD["3a. Crowd loudness<br/>excitement peaks"]
    BUF --> ASR["3b. Commentary<br/>Whisper speech-to-text → Claude extracts events"]
    BUF --> OCR["3c. Scoreboard OCR (optional)<br/>PaddleOCR reads score, overs, target, innings total"]
    AUD --> FUS["4. Fuser<br/>scoreboard confirms · commentary anchors · crowd supports"]
    ASR --> FUS
    OCR --> FUS
    FUS --> PLAN["5. Clip planner<br/>camera cuts, crowd roar, live dismissal + replay, caps"]
    PLAN --> CUR["6. Curator<br/>every wicket kept, length budget, team/over filter, wicket audit"]
    CUR --> REN["7. Renderer<br/>one encode, cards with crowd bed, peak limiter"]
    REN --> PKG[/"8. Package: .mp4 + .srt + manifest.json"/]
    PKG --> QG{"9. Quality gate<br/>VMAF / SSIM vs source"}
    QG --> APR["10. Human approval screen<br/>python -m motion_ai.approve"]
    APR --> PUB(["Publish: blocked until a rights-cleared feed<br/>(publish_channel_allowed = false)"])
```

### Steps, and why each one exists

1. **Ingest (SRT → 2-second segments).** ffmpeg copies the stream into short `.ts` files without re-encoding.
   *Why:* clips can be cut while the match is still live, the picture is never degraded before the single
   final encode, and a dropped feed reconnects on its own (stall watchdog) without losing what was recorded.
2. **Buffer + recorder.** Segments and every detected event are written to `runtime*/`.
   *Why:* a crash or restart resumes instead of starting over, and rules can be changed and re-checked in
   seconds with `python -m motion_ai.replay --plan-only` instead of re-processing the whole match.
3. **Three independent detectors read the same segments in parallel.**
   - *Crowd loudness* finds roars. *Why:* cheap, instant, and marks the exact second something happened.
   - *Commentary*: Whisper turns speech into text, Claude turns the text into events (wicket, six, review…);
     keyword rules are the fallback. *Why:* commentators name what happened, including run-outs and reviews
     that the scoreboard alone cannot explain.
   - *Scoreboard OCR* (optional, `OCR_AVAILABLE=true`): reads runs/wickets/overs, the chase target and the
     innings total. *Why:* the scoreboard is the ground truth: it confirms wickets and boundaries, detects
     the second innings and the result, and never mishears.
4. **Fuser.** Combines the three: the scoreboard confirms, commentary anchors the moment, crowd noise supports.
   Events are only finalised after the slowest detector has caught up. *Why:* each source alone is wrong
   sometimes (commentary talks over replays; the scoreboard updates late); together they are reliable.
5. **Clip planner.** Picks each clip's start on a camera cut, ends it when the crowd settles, keeps the live
   dismissal plus its replay, and skips known bad stretches (e.g. pop-ups). *Why:* clips that start mid-shot
   or end mid-celebration look automated; broadcast editors cut on cuts.
6. **Curator.** Keeps every wicket and the winning runs, fills the remaining time with sixes then fours,
   honours `HIGHLIGHT_OVERS` (chosen overs per team), and audits wickets against the scoreboard.
   *Why:* a highlights package that misses a wicket is wrong; length must stay inside the target.
7. **Renderer.** One H.264 encode at near-source quality, title / innings / break / result cards with a crowd
   bed, true-peak audio limiter. *Why:* one encode = no generation loss; cards give the story; the limiter
   keeps audio broadcast-safe.
8. **Package.** `.mp4` + `.srt` + `manifest.json` (every clip, why it was chosen or held, token cost).
   *Why:* the manifest is the audit trail a human (or a CMS) checks before anything goes out.
9. **Quality gate.** Every clip is compared with the source (VMAF ≥ 95, SSIM ≥ 0.985, 1080p, bitrate ≥ source).
   *Why:* "no quality loss" is measured, not assumed; one failing clip holds the whole package.
10. **Human approval.** Doubtful clips are held, never dropped; a person approves on a local page.
    *Why:* automation drafts, an editor decides; nothing is published automatically.

## Demo

| | Link |
|---|---|
| Source video (YouTube, used as a stand-in for the live SRT feed) | https://www.youtube.com/watch?v=F62YNMNfvA4 |
| Output highlights video (4:58, 1080p) | https://wpz5sl5dmq.ufs.sh/f/mY3PNsje3CtypowHhcdjIXq7xtJKiBkbU298L4yEwu1gCo6F |

Demo result: 14 clips + 6 cards, all 12 wickets clipped, quality gate 14/14 (VMAF ≥ 99), true peak −0.8 dBTP.
The source is a YouTube re-upload of a T20 Mumbai League broadcast: it is for internal demo only and must not be
published (`source_rights: unverified_youtube_reupload`).

## Overview

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
  to 80-400 Hz so commentary is only a murmur, normalised to -27 LUFS (~-31 dB mean vs ~-23 dB for clips on the
  demo match, i.e. ~8 dB under the programme), 0.4 s fades.
* Every clip and card goes through a true-peak limiter (4x oversampled `alimiter`, ceiling -2 dBFS, no auto-gain,
  delay-compensated) in the same single encode: the package stays <= -1 dBTP after AAC, programme loudness
  unchanged, same codec / rate / channels so the `-c copy` join still works.
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

## Branding template

Card and caption styling (colours, font, font sizes, logo, card lengths, caption box/position, safe margins)
comes from `config/template.json` if it exists, else from `config/template.example.json` — a neutral
placeholder equal to the built-in defaults. Every key is optional; a missing key keeps the default.
The template is validated at load (bad colour, unknown key, missing font/logo file -> a clear `TemplateError`).
Caption *text* is not part of the template, only its styling.

To brand the package, copy the example to `config/template.json` (git-ignored) and drop in the licensed files,
e.g. under `config/brand/` (relative paths are resolved against `config/`):

| File | Key | Notes |
|---|---|---|
| Logo, PNG with alpha (transparent background) | `logo.file` | Shown on every card (`logo.card`: position / width px / opacity); optional small corner bug on clips via `logo.clip_bug.enabled` |
| Font, `.ttf` or `.otf` | `font.file` | `null` = DejaVu Sans Bold; sizes per role in `font.sizes` (`headline`, `subhead`, `stat`, `body`, `small`, `caption`) |
| Colours | `colors.*` | `#rrggbb` or `#rrggbbaa` (or a basic name like `white`): `background`, `accent` (card bar, drawn when `cards.accent_bar_height` > 0), `text`, `caption_text`, `caption_border`, `caption_box` |

Other keys: `cards.seconds` (per card), `cards.line_spacing`, `cards.fade`, `caption.y` (top of the caption text
as a fraction of frame height; <= 0.75 so it never covers the broadcaster score bar), `caption.border_width`,
`caption.box` / `caption.box_padding` (top/bottom px) / `caption.box_padding_x` (left/right px), `caption.fade`
(s, fade in at the caption start and out at the clip end), `caption.ref_height` (the caption px values are for this
frame height and scale with the output), `safe_margins.x/y` (px, logo inset).
The default caption is a solid brand box (navy text on the gold accent, 88 px at 1080p) top centre at y 0.15,
clear of the broadcaster score bar (y 900-1024) and the corner logos, so it reads as ours, not the broadcaster's.
Positions: `top-left`, `top-right`, `bottom-left`, `bottom-right`, `top-center`, `bottom-center`.
Note: the curator still budgets the package with its own card lengths (`CARD_SECONDS`), so keep
`cards.seconds` close to them.

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

## Scoreboard calibration

OCR runs in its own venv (PaddleOCR 2.x needs numpy<2; the main `.venv` stays on numpy 2.x):

```bash
uv venv --python 3.10 .venv-ocr
uv pip install --python .venv-ocr/bin/python -r requirements-ocr.txt   # ffmpeg must be on PATH
```

`config/scoreboard.json` holds pixel boxes for one broadcaster's graphics. For a new broadcaster / video,
propose them automatically:

```bash
.venv-ocr/bin/python -m motion_ai.calibrate path/to/match.mp4                    # 5 middle frames (9, 17 if needed)
.venv-ocr/bin/python -m motion_ai.calibrate path/to/match.mp4 --at 100 470 400   # frames you picked
#   --out config/scoreboard.proposed.json (default)  --gpu  --preview-dir runtime/calibration
```

It runs PaddleOCR detection over the bottom 25% of each frame (native resolution), finds the score bar
(`73-1` + overs, merged across neighbouring boxes; the most-seen, tallest-text cluster wins over e.g. bowler
figures), `TARGET n` next to it, the `OVERS x TOTAL r-w` card footer and a second score-bar position
(`scoreline_alt`), pads the union of the boxes seen across frames, and writes them in the
`config/scoreboard.json` format. `scoreline` must be seen on >= 2 frames; `target` / `footer` on >= 1.

Then:

1. Look at `runtime/calibration/<video>/` — `frame_*.png` (all OCR boxes grey, proposed regions coloured) and
   `region_*.png` (each crop as the detector sees it).
2. Check the printed validation table (each region re-read with `ScoreboardOCRDetector` on every frame).
3. `target` only shows in the chase and `footer` only on the scorecards after an innings: if they are missing,
   re-run with `--at` times where they are on screen (also add a few score-bar times).
4. Copy the proposal to `config/scoreboard.json` (edit boxes by hand if a preview is off).
