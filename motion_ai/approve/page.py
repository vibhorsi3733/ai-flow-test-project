"""Render the review page. Plain HTML/CSS/JS inline; every manifest value is escaped."""

from __future__ import annotations

import json
from datetime import datetime
from html import escape

from .decision import HARD_BLOCKER_HOLDS


def _e(v) -> str:
    return escape("" if v is None else str(v), quote=True)


def seconds_left(approve_by, now: datetime | None = None):
    """Seconds until approve_by (ISO; naive means local time). None if missing/unparseable."""
    if not approve_by:
        return None
    try:
        dl = datetime.fromisoformat(str(approve_by))
    except ValueError:
        return None
    if now is None:
        now = datetime.now(dl.tzinfo) if dl.tzinfo else datetime.now()
    return (dl - now).total_seconds()


def _json_script(obj) -> str:
    # Safe inside <script>: no '</' sequences or HTML-significant chars survive.
    return (json.dumps(obj).replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026"))


def _quality_rows(qg: dict) -> str:
    rows = []
    for c in qg.get("clips") or []:
        ok = c.get("pass")
        rows.append(
            f"<tr class='{'ok' if ok else 'bad'}'><td>{_e(c.get('clip'))}</td><td>{_e(c.get('type'))}</td>"
            f"<td>{_e(c.get('vmaf_mean'))}</td><td>{_e(c.get('vmaf_p5'))}</td><td>{_e(c.get('ssim'))}</td>"
            f"<td>{'pass' if ok else 'FAIL'}</td></tr>")
    pkg = qg.get("package") or {}
    pkg_txt = ", ".join(f"{_e(k)}={_e(v)}" for k, v in pkg.items())
    return (f"<p class='small'>Reference: {_e(qg.get('reference'))}<br>Thresholds: "
            f"{_e(json.dumps(qg.get('thresholds') or {}))}<br>Package: {pkg_txt}</p>"
            "<table><tr><th>clip</th><th>type</th><th>VMAF mean</th><th>VMAF p5</th><th>SSIM</th><th></th></tr>"
            + "".join(rows) + "</table>")


def _wicket_audit(wa: dict) -> str:
    parts = []
    for key, inn in wa.items():
        if key == "wicket_missing" or not isinstance(inn, dict):
            continue
        end = inn.get("innings_end") or {}
        lis = "".join(f"<li>not clipped: {_e(x)}</li>" for x in inn.get("not_clipped") or [])
        lis += "".join(f"<li>commentary call not on scoreboard @ {_e(c.get('match_time_s'))}s: "
                       f"&ldquo;{_e(c.get('quote'))}&rdquo;</li>"
                       for c in inn.get("wicket_calls_not_on_scoreboard") or [])
        parts.append(f"<div><b>{_e(key)}</b>: {_e(inn.get('wickets_clipped'))}/"
                     f"{_e(inn.get('wickets_on_board'))} wickets clipped; end {_e(end.get('score'))} "
                     f"({_e(end.get('overs'))} ov)<ul>{lis}</ul></div>")
    missing = wa.get("wicket_missing") or []
    if missing:
        parts.append("<div class='bad'><b>wicket_missing</b><ul>"
                     + "".join(f"<li>{_e(m)}</li>" for m in missing) + "</ul></div>")
    return "".join(parts) or "<p>No wicket audit.</p>"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clip_video(clip: dict, url, media) -> str:
    """Clip mp4 if present; else the clip's span of the package mp4 (media fragment)."""
    if url:
        return f"<video controls preload='metadata' src='{_e(url)}'></video>"
    start, dur = _num(clip.get("start_in_package")), _num(clip.get("duration"))
    if media.has("package") and start is not None and dur:
        src = f"{media.url('package')}#t={start:.2f},{start + dur:.2f}"
        return (f"<video controls preload='metadata' src='{_e(src)}'></video>"
                f"<div class='small'>Clip file {_e(clip.get('clip_file'))} not found; playing "
                f"{start:.1f}s&ndash;{start + dur:.1f}s of the package video.</div>")
    return f"<div class='novideo'>clip file {_e(clip.get('clip_file'))} not found</div>"


def render_page(manifest: dict, approvals: dict, decision: dict, media, *,
                manifest_name: str, manifest_changed: bool) -> str:
    held = set(manifest.get("held_clip_ids") or [])
    holds = manifest.get("package_holds") or []
    qg = manifest.get("quality_gate") or {}
    wa = manifest.get("wicket_audit") or {}
    left = seconds_left(manifest.get("approve_by"))
    clip_decisions = approvals.get("clips") or {}
    pkg = approvals.get("package") or {}

    hold_html = "".join(
        f"<span class='tag {'hard' if h in HARD_BLOCKER_HOLDS else ''}'>{_e(h)}</span>" for h in holds
    ) or "<span class='tag okt'>none</span>"

    clip_cards = []
    for i, c in enumerate(manifest.get("clips") or []):
        cid = c.get("id") or ""
        url = media.url(cid)
        video = _clip_video(c, url, media)
        reasons = "".join(f"<span class='tag'>{_e(r)}</span>" for r in c.get("hold_reasons") or [])
        is_held = cid in held or bool(c.get("hold_reasons"))
        d = clip_decisions.get(cid) or {}
        quotes = "".join(f"<li>{_e(q)}</li>" for q in dict.fromkeys(c.get("quotes") or []))
        clip_cards.append(f"""
<section class="clip {'held' if is_held else ''}" data-clip="{_e(cid)}">
  <div class="vid">{video}</div>
  <div class="meta">
    <h3>#{i + 1} {_e(c.get('type'))} <small>{_e(cid)}</small>
      {"<span class='tag hard'>HELD</span>" if is_held else "<span class='tag okt'>auto ok</span>"}</h3>
    <p class="cap">{_e(c.get('caption'))}</p>
    <p>Score <b>{_e(c.get('score') or 'unknown')}</b> ({_e(c.get('overs') or '?')} ov){
       f" &middot; board {_e(c.get('board_score'))} ({_e(c.get('board_overs'))} ov)"
       if c.get('board_score') and c.get('board_score') != c.get('score') else ""} &middot; innings {_e(c.get('innings'))}
       &middot; match t={_e(c.get('match_time_s'))}s &middot; {_e(c.get('duration'))}s
       &middot; confidence {_e(c.get('confidence'))}</p>
    <p>Sources: {_e(', '.join(c.get('sources') or []))}</p>
    <p>Hold reasons: {reasons or 'none'}</p>
    {f"<details><summary>Commentary quotes</summary><ul>{quotes}</ul></details>" if quotes else ""}
    <textarea class="note" placeholder="Optional note">{_e(d.get('note'))}</textarea>
    <div class="row">
      <button class="approve" data-act="approve">Approve</button>
      <button class="reject" data-act="reject">Reject</button>
      <span class="status" data-status="{_e(cid)}"></span>
    </div>
  </div>
</section>""")

    withheld = "".join(
        f"<tr><td>{_e(w.get('id'))}</td><td>{_e(w.get('type'))}</td><td>{_e(w.get('innings'))}</td>"
        f"<td>{_e(w.get('score'))}</td><td>{_e(w.get('match_time_s'))}</td><td>{_e(w.get('reason'))}</td></tr>"
        for w in manifest.get("withheld") or [])
    withheld_html = ("<table><tr><th>id</th><th>type</th><th>inn</th><th>score</th><th>match t</th>"
                     f"<th>reason</th></tr>{withheld}</table>" if withheld else "<p>None.</p>")

    pkg_video = (f"<details><summary>Watch full package</summary><video controls preload='none' "
                 f"src='{_e(media.url('package'))}'></video></details>" if media.has("package") else
                 "<p class='small'>Package mp4 not found next to the manifest.</p>")
    hard_present = [h for h in holds if h in HARD_BLOCKER_HOLDS]
    ack_hint = (f"<p class='warn'>Hard blockers present: <b>{_e(', '.join(hard_present))}</b>. To approve, "
                "name each one in the package note (e.g. &ldquo;acknowledge quality_gate_failed: "
                "&hellip;&rdquo;).</p>" if hard_present else "")
    changed = ("<p class='warn'>The manifest changed since decisions were last saved. "
               "Re-check clips before approving.</p>" if manifest_changed else "")

    state = {"decision": decision, "clips": clip_decisions, "package": pkg or None,
             "seconds_left": left}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Package Review</title>
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --fg:#1d2330; --muted:#5b6475; --ok:#1e7a3c; --bad:#b3261e;
        --warn:#8a5a00; --line:#dde1e7; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,sans-serif; }}
main {{ max-width:1200px; margin:0 auto; padding:16px; }}
h1 {{ font-size:22px; margin:8px 0; }} h2 {{ font-size:18px; margin:24px 0 8px; }}
h3 {{ font-size:16px; margin:0 0 4px; }} h3 small {{ color:var(--muted); font-weight:400; }}
.card, .clip {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin:10px 0; }}
.clip {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px; }}
.clip.held {{ border-left:4px solid var(--bad); }}
@media (max-width:800px) {{ .clip {{ grid-template-columns:1fr; }} }}
video {{ width:100%; background:#000; border-radius:4px; }}
.novideo {{ padding:40px; text-align:center; background:#eee; color:var(--muted); }}
.meta p {{ margin:4px 0; }} .cap {{ font-weight:600; }}
.tag {{ display:inline-block; padding:1px 7px; margin:1px 2px; border-radius:10px; background:#eef0f4; font-size:12px; }}
.tag.hard {{ background:#fde7e5; color:var(--bad); font-weight:600; }}
.tag.okt {{ background:#e3f4e8; color:var(--ok); }}
.ok {{ color:var(--ok); }} .bad {{ color:var(--bad); }} .warn {{ color:var(--warn); font-weight:600; }}
.small {{ font-size:13px; color:var(--muted); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
td, th {{ border-bottom:1px solid var(--line); padding:4px 6px; text-align:left; }}
tr.bad td {{ color:var(--bad); }}
textarea {{ width:100%; min-height:44px; font:inherit; padding:6px; border:1px solid var(--line); border-radius:4px; }}
input[type=text] {{ font:inherit; padding:6px; border:1px solid var(--line); border-radius:4px; min-width:220px; }}
.row {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:6px; }}
button {{ font:inherit; padding:6px 16px; border-radius:4px; border:1px solid transparent; cursor:pointer; color:#fff; }}
button.approve {{ background:var(--ok); }} button.reject {{ background:var(--bad); }}
.status {{ font-size:13px; }}
.summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; }}
.big {{ font-size:18px; font-weight:700; }}
#decision ul {{ margin:4px 0; }}
.sticky {{ position:sticky; top:0; z-index:2; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); }}
</style></head>
<body><main>
<div class="sticky row"><label>Reviewer name <input type="text" id="reviewer" autocomplete="name"></label>
  <span class="small">Manifest: {_e(manifest_name)} (read-only). Decisions go to the .approvals.json next to it.</span></div>
<h1>{_e(manifest.get('title'))}</h1>
{changed}
<div class="summary">
  <div class="card"><div class="small">Status</div>
    <div class="big {'bad' if manifest.get('status') != 'auto_ok' else 'ok'}">{_e(manifest.get('status'))}</div>
    <div class="small">scope {_e(manifest.get('scope'))} &middot; {_e(manifest.get('duration'))}s &middot;
      rights: {_e(manifest.get('source_rights'))}</div></div>
  <div class="card"><div class="small">Package holds</div>{hold_html}</div>
  <div class="card"><div class="small">Quality gate</div>
    <div class="big {'ok' if qg.get('pass') else 'bad'}">{'PASS' if qg.get('pass') else 'FAIL'}</div></div>
  <div class="card"><div class="small">Approve by {_e(manifest.get('approve_by'))}</div>
    <div class="big" id="timeleft">&ndash;</div>
    <div class="small">on timeout: {_e(manifest.get('on_timeout'))} &rarr; {_e(manifest.get('on_timeout_result'))}</div></div>
</div>
<div class="card"><b>Held clips:</b> {len(held)} of {len(manifest.get('clips') or [])} &middot;
  <b>publish_channel_allowed:</b> false (publishing is not part of this step)</div>
<details class="card"><summary><b>Quality gate details</b></summary>{_quality_rows(qg)}</details>
<details class="card" open><summary><b>Wicket audit</b></summary>{_wicket_audit(wa)}</details>
{pkg_video}

<h2>Clips</h2>
{''.join(clip_cards)}

<h2>Withheld (not in package, read-only)</h2>
<div class="card">{withheld_html}</div>

<h2>Package decision</h2>
<div class="card">
  {ack_hint}
  <textarea id="pkgnote" placeholder="Package note (required to acknowledge hard blockers)">{_e(pkg.get('note'))}</textarea>
  <div class="row">
    <button class="approve" id="pkgapprove">Approve package</button>
    <button class="reject" id="pkgreject">Reject package</button>
    <span class="status" id="pkgstatus"></span>
  </div>
  <div id="decision"></div>
</div>
</main>
<script>
const STATE = {_json_script(state)};
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
try {{ $('#reviewer').value = localStorage.getItem('motion_ai_reviewer') || ''; }} catch (e) {{}}
$('#reviewer').addEventListener('change', () => {{ try {{ localStorage.setItem('motion_ai_reviewer', $('#reviewer').value); }} catch (e) {{}} }});
function fmt(d) {{ return d ? `${{d.decision.toUpperCase()}} by ${{esc(d.reviewer)}} at ${{esc(d.decided_at)}}` : 'undecided'; }}
function paintClip(id, d) {{
  const el = document.querySelector(`[data-status="${{CSS.escape(id)}}"]`);
  if (el) {{ el.innerHTML = fmt(d); el.className = 'status ' + (d ? (d.decision === 'approve' ? 'ok' : 'bad') : ''); }}
}}
function paintDecision(dec) {{
  const b = dec.blockers.length ? '<ul>' + dec.blockers.map(x => `<li>${{esc(x)}}</li>`).join('') + '</ul>' : '';
  $('#decision').innerHTML = `<p class="big ${{dec.package_approved ? 'ok' : 'bad'}}">${{dec.package_approved ? 'APPROVED (recorded)' : 'Not approved: ' + esc(dec.status)}}</p>`
    + `<p>Approved clips: ${{dec.approved_clip_ids.length}} &middot; rejected: ${{dec.rejected_clip_ids.length}} &middot; `
    + `auto-included (not held, undecided): ${{dec.default_included_clip_ids.length}} &middot; held undecided: ${{dec.undecided_held_clip_ids.length}}</p>`
    + (b ? '<p><b>Blockers</b></p>' + b : '') + '<p class="small">publish_channel_allowed stays false.</p>';
}}
function paintPkg(d) {{ $('#pkgstatus').innerHTML = fmt(d); }}
async function post(url, body) {{
  const r = await fetch(url, {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(body)}});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}}
function reviewer() {{
  const v = $('#reviewer').value.trim();
  if (!v) {{ alert('Enter your reviewer name at the top first.'); $('#reviewer').focus(); }}
  return v;
}}
async function decideClip(id, decision) {{
  const who = reviewer(); if (!who) return;
  const sec = document.querySelector(`section[data-clip="${{CSS.escape(id)}}"]`);
  try {{
    const j = await post('/api/clip', {{clip_id: id, decision, reviewer: who, note: sec.querySelector('.note').value}});
    paintClip(id, j.entry); paintDecision(j.decision);
  }} catch (e) {{ alert('Not saved: ' + e.message); }}
}}
async function decidePackage(decision) {{
  const who = reviewer(); if (!who) return;
  try {{
    const j = await post('/api/package', {{decision, reviewer: who, note: $('#pkgnote').value}});
    paintPkg(j.entry); paintDecision(j.decision);
  }} catch (e) {{ alert('Not saved: ' + e.message); }}
}}
document.querySelectorAll('section[data-clip] button[data-act]').forEach(btn =>
  btn.addEventListener('click', () => decideClip(btn.closest('section').dataset.clip, btn.dataset.act)));
$('#pkgapprove').addEventListener('click', () => decidePackage('approve'));
$('#pkgreject').addEventListener('click', () => decidePackage('reject'));
document.querySelectorAll('[data-status]').forEach(el => paintClip(el.dataset.status, STATE.clips[el.dataset.status]));
paintPkg(STATE.package); paintDecision(STATE.decision);
const t0 = Date.now();
function tick() {{
  if (STATE.seconds_left === null) {{ $('#timeleft').textContent = 'no deadline'; return; }}
  const s = Math.round(STATE.seconds_left - (Date.now() - t0) / 1000), a = Math.abs(s);
  const txt = `${{Math.floor(a / 3600)}}h ${{Math.floor(a % 3600 / 60)}}m ${{a % 60}}s`;
  $('#timeleft').textContent = s >= 0 ? txt + ' left' : 'overdue by ' + txt;
  $('#timeleft').className = 'big ' + (s >= 0 ? '' : 'bad');
}}
tick(); setInterval(tick, 1000);
</script>
</body></html>"""
