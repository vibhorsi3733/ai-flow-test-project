# Evaluation

Scores every detector and the final package against a hand-labelled answer key, so any change
(prompt, threshold, new agent) gets a measurable before/after.

```bash
python -m eval.score                      # runtime_1080/events.jsonl + newest output/*.json
python -m eval.score --verified-only      # only labels a human has checked
```

Each run prints hits / misses / false alarms, precision and recall per source (`asr` = commentary,
`ocr` = scoreboard, `package` = final clips) and event type, and saves `eval/results/<time>.json`.

## Answer key: `labels/t20_mumbai_1080p.json`

**Drafted from the scoreboard OCR and not yet verified.** Until a human checks it, the OCR rows score
100% by construction, so the OCR numbers mean nothing yet. To verify: watch each `[lo, hi]` window, set
`verified: true`, fix wrong rows, and add balls the scoreboard missed.
