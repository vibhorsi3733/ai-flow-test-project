"""Score each detector (and the final package) against a hand-labelled answer key.

    python -m eval.score                                   # defaults: runtime_1080 events + newest manifest
    python -m eval.score --events runtime/events.jsonl --manifest output/x.json --labels eval/labels/m.json

For every source (asr = commentary, ocr = scoreboard, audio = crowd) and for the clips that made the
package, it reports per event type: hits, misses, false alarms, precision and recall. A detection counts as
a hit when its type matches and its time falls inside a label's [lo, hi] window, widened by --tol seconds.
Each label can be hit once per source. Results are also written to eval/results/<timestamp>.json so runs
can be compared before/after a change.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
from dataclasses import dataclass, field

BALL_TYPES = ("wicket", "four", "six")
SOURCES = ("asr", "ocr")


@dataclass
class Score:
    hits: int = 0
    misses: int = 0
    false_alarms: int = 0
    false_alarm_times: list[float] = field(default_factory=list)
    missed_labels: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        n = self.hits + self.false_alarms
        return self.hits / n if n else None

    @property
    def recall(self) -> float | None:
        n = self.hits + self.misses
        return self.hits / n if n else None


def match(labels: list[dict], detections: list[float], tol: float) -> Score:
    """One-to-one greedy match of detection times to label windows (all of one event type)."""
    s = Score()
    used = [False] * len(labels)
    for t in sorted(detections):
        best, best_d = None, None
        for i, lb in enumerate(labels):
            if used[i] or not (lb["lo"] - tol <= t <= lb["hi"] + tol):
                continue
            d = 0.0 if lb["lo"] <= t <= lb["hi"] else min(abs(t - lb["lo"]), abs(t - lb["hi"]))
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is None:
            s.false_alarms += 1
            s.false_alarm_times.append(round(t, 1))
        else:
            used[best] = True
            s.hits += 1
    for i, lb in enumerate(labels):
        if not used[i]:
            s.misses += 1
            s.missed_labels.append(f"{lb['type']} {lb.get('score') or ''} @{lb['lo']}-{lb['hi']}".strip())
    return s


# Withheld reasons where the ball is still on screen inside another clip of the package.
COVERED_REASONS = {"overlap", "same_ball_as_winning_runs"}


def package_stream(clips: list[dict], withheld: list[dict] = ()) -> list[dict]:
    kept = [{"type": c["type"], "t": c["match_time_s"]} for c in clips]
    covered = [{"type": w["type"], "t": w["match_time_s"]} for w in withheld if w.get("reason") in COVERED_REASONS]
    return kept + covered


def score_all(labels: list[dict], events: list[dict], clips: list[dict], tol: float,
              withheld: list[dict] = ()) -> dict[str, dict[str, Score]]:
    out: dict[str, dict[str, Score]] = {}
    streams = {src: [e for e in events if e.get("source") == src] for src in SOURCES}
    streams["package"] = package_stream(clips, withheld)
    for name, evs in streams.items():
        out[name] = {}
        for typ in BALL_TYPES:
            lbs = [lb for lb in labels if lb["type"] == typ]
            out[name][typ] = match(lbs, [e["t"] for e in evs if e["type"] == typ], tol)
    return out


def audio_report(labels: list[dict], events: list[dict], tol: float) -> dict:
    """Crowd peaks have no labels of their own; report how many sit near a labelled ball."""
    peaks = [e["t"] for e in events if e.get("source") == "audio"]
    near = sum(any(lb["lo"] - tol <= t <= lb["hi"] + tol for lb in labels) for t in peaks)
    return {"peaks": len(peaks), "near_a_labelled_ball": near}


def _pct(x: float | None) -> str:
    return "  -  " if x is None else f"{x * 100:4.0f}%"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default="eval/labels/t20_mumbai_1080p.json")
    ap.add_argument("--events", default="runtime_1080/events.jsonl")
    ap.add_argument("--manifest", default=None, help="package manifest JSON (default: newest in output/)")
    ap.add_argument("--tol", type=float, default=8.0, help="seconds a detection may fall outside a label window")
    ap.add_argument("--verified-only", action="store_true", help="ignore labels not yet checked by a human")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)

    doc = json.load(open(a.labels))
    labels = [lb for lb in doc["labels"] if lb.get("verified") or not a.verified_only]
    unverified = sum(not lb.get("verified") for lb in labels)
    events = [json.loads(line) for line in open(a.events) if line.strip()]
    manifest_path = a.manifest or max(glob.glob("output/*.json"), key=os.path.getmtime, default=None)
    manifest = json.load(open(manifest_path)) if manifest_path else {}
    clips, withheld = manifest.get("clips", []), manifest.get("withheld", [])

    res = score_all(labels, events, clips, a.tol, withheld)
    fallback = sum(1 for e in events if e.get("source") == "asr" and e.get("detail", {}).get("fallback"))
    asr_total = sum(1 for e in events if e.get("source") == "asr")

    print(f"labels: {len(labels)} ({unverified} unverified)  events: {a.events}  package: {manifest_path}")
    if unverified:
        print("WARNING: answer key is not fully human-verified; scores are provisional.")
    if asr_total and fallback == asr_total:
        print(f"WARNING: all {asr_total} commentary events came from the keyword fallback (Claude was not used).")
    print(f"\n{'source':<9}{'type':<8}{'hits':>5}{'miss':>6}{'false':>6}{'prec':>7}{'recall':>8}")
    for name, by_type in res.items():
        for typ, s in by_type.items():
            print(f"{name:<9}{typ:<8}{s.hits:>5}{s.misses:>6}{s.false_alarms:>6}{_pct(s.precision):>7}{_pct(s.recall):>8}")
    audio = audio_report(labels, events, a.tol)
    print(f"\naudio: {audio['peaks']} crowd peaks, {audio['near_a_labelled_ball']} near a labelled ball")

    report = {
        "when": dt.datetime.now().isoformat(timespec="seconds"),
        "labels": a.labels, "events": a.events, "manifest": manifest_path, "tol": a.tol,
        "unverified_labels": unverified, "asr_fallback_events": fallback, "asr_events": asr_total,
        "audio": audio,
        "scores": {n: {t: {"hits": s.hits, "misses": s.misses, "false_alarms": s.false_alarms,
                           "precision": s.precision, "recall": s.recall,
                           "false_alarm_times": s.false_alarm_times, "missed": s.missed_labels}
                       for t, s in bt.items()} for n, bt in res.items()},
    }
    if not a.no_save:
        os.makedirs("eval/results", exist_ok=True)
        path = f"eval/results/{report['when'].replace(':', '')}.json"
        json.dump(report, open(path, "w"), indent=1)
        print(f"saved {path}")
    return report


if __name__ == "__main__":
    main()
