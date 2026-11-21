from eval.score import audio_report, match, score_all

LABELS = [
    {"type": "wicket", "lo": 100.0, "hi": 110.0},
    {"type": "wicket", "lo": 200.0, "hi": 205.0},
    {"type": "four", "lo": 300.0, "hi": 302.0},
]


def test_hit_inside_window():
    s = match(LABELS[:2], [105.0], tol=0)
    assert (s.hits, s.misses, s.false_alarms) == (1, 1, 0)


def test_tolerance_widens_window():
    assert match(LABELS[:1], [115.0], tol=8).hits == 1
    assert match(LABELS[:1], [115.0], tol=2).false_alarms == 1


def test_each_label_hit_once():
    s = match(LABELS[:1], [101.0, 102.0], tol=0)
    assert (s.hits, s.false_alarms) == (1, 1)


def test_precision_recall():
    s = match(LABELS[:2], [105.0, 150.0], tol=0)
    assert s.precision == 0.5 and s.recall == 0.5


def test_empty_gives_none_not_zero_division():
    s = match([], [], tol=8)
    assert s.precision is None and s.recall is None


def test_score_all_separates_sources_and_types():
    events = [
        {"type": "wicket", "t": 105.0, "source": "asr"},
        {"type": "wicket", "t": 250.0, "source": "asr"},   # false alarm
        {"type": "four", "t": 301.0, "source": "ocr"},
    ]
    clips = [{"type": "wicket", "match_time_s": 203.0}]
    res = score_all(LABELS, events, clips, tol=0)
    assert res["asr"]["wicket"].hits == 1 and res["asr"]["wicket"].false_alarms == 1
    assert res["ocr"]["four"].recall == 1.0
    assert res["package"]["wicket"].hits == 1


def test_overlap_withheld_counts_as_covered_but_dropped_does_not():
    withheld = [{"type": "four", "match_time_s": 301.0, "reason": "overlap"},
                {"type": "wicket", "match_time_s": 105.0, "reason": "not_on_scoreboard"}]
    res = score_all(LABELS, [], [], tol=0, withheld=withheld)
    assert res["package"]["four"].hits == 1
    assert res["package"]["wicket"].hits == 0


def test_dedupe_merges_repeats_of_same_source_and_type():
    events = [{"type": "wicket", "t": 105.0, "source": "asr"},
              {"type": "wicket", "t": 106.0, "source": "asr"},   # same call sent twice
              {"type": "wicket", "t": 106.0, "source": "ocr"}]   # other source: kept
    res = score_all(LABELS, events, [], tol=0, dedupe_s=10)
    assert res["asr"]["wicket"].false_alarms == 0 and res["asr"]["wicket"].hits == 1
    assert res["ocr"]["wicket"].hits == 1


def test_audio_report_counts_peaks_near_balls():
    events = [{"type": "excitement", "t": 104.0, "source": "audio"},
              {"type": "excitement", "t": 500.0, "source": "audio"}]
    assert audio_report(LABELS, events, tol=0) == {"peaks": 2, "near_a_labelled_ball": 1}
