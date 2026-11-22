"""Pure parts of the scoreboard calibration tool (no OCR / video needed)."""
import json

from motion_ai.calibrate import (Box, PREDICATES, build_config, choose_alt, choose_cluster, cluster, default_times,
                                 find_spans, group_lines, merge_boxes, near, pad_region, propose, union_box)

W, H = 1920, 1080


def sl(boxes):
    return find_spans(boxes, PREDICATES["scoreline"], 1.5)


# Boxes as PaddleOCR det+rec returned them on the demo (1080p, bottom 25%)
FRAME_100 = [Box("SIDHAANT", 348, 968, 532, 1006), Box("42", 683, 974, 731, 1005),
             Box("6-0", 858, 975, 964, 1019), Box("0.4", 984, 985, 1046, 1008),
             Box("0-60.4", 1558, 920, 1682, 949),   # bowler figures: also parse as a scoreline, but smaller
             Box("BOWLING FROM MEDIA CENTRE UNIVERSITY END", 1150, 977, 1660, 1004)]
FRAME_470 = [Box("ANISH", 352, 970, 462, 1008), Box("73-17.5", 880, 977, 1066, 1012),
             Box("0-190.5", 1544, 922, 1678, 951), Box("TARGET-180", 894, 943, 1022, 958)]
FRAME_277 = [Box("139-3 16.1", 870, 976, 1072, 1016), Box("0-101.1", 1550, 921, 1680, 950)]
FRAME_400 = [Box("EXTRAS11", 246, 983, 434, 1021), Box("OVERS 20", 832, 981, 1024, 1025),
             Box("TOTAL 179-7", 1362, 983, 1618, 1025)]
FRAME_608 = [Box("TARGET-180", 330, 943, 458, 958), Box("181-519.2", 306, 977, 516, 1012)]
FRAME_513 = [Box("TARGET-180", 330, 943, 458, 958), Box("112-212.2", 308, 978, 514, 1013)]


def test_score_and_overs_in_separate_boxes_are_merged():
    (b,) = sl(FRAME_100[:4])
    assert b.text == "6-0 0.4" and (b.x0, b.y0, b.x1, b.y1) == (858, 975, 1046, 1019)


def test_neighbours_too_far_apart_are_not_merged():
    assert sl([Box("6-0", 858, 975, 964, 1019), Box("0.4", 1400, 985, 1460, 1008)]) == []


def test_bare_score_without_overs_is_not_a_scoreline():
    assert sl([Box("73-1", 880, 977, 990, 1012)]) == []


def test_minimal_span_excludes_extra_neighbours():
    boxes = [Box("42", 830, 975, 850, 1010), Box("73-1", 860, 975, 960, 1010), Box("7.5", 970, 975, 1010, 1010),
             Box("P", 1020, 975, 1035, 1010)]
    assert [b.text for b in sl(boxes)] == ["73-1 7.5"]


def test_group_lines_and_merge():
    lines = group_lines(FRAME_470)
    assert [[b.text for b in line] for line in lines][0] == ["TARGET-180", "0-190.5"]   # overlap vs shorter box
    assert any([b.text for b in line] == ["ANISH", "73-17.5"] for line in lines)
    m = merge_boxes([Box("b", 50, 12, 60, 30), Box("a", 10, 10, 20, 28)])
    assert (m.text, m.x0, m.y0, m.x1, m.y1) == ("a b", 10, 10, 60, 30)


def test_footer_spans_far_apart_boxes_but_not_extras():
    (b,) = find_spans(FRAME_400, PREDICATES["footer"], 12.0)
    assert b.text == "OVERS 20 TOTAL 179-7" and b.x0 == 832 and b.x1 == 1618


def test_cluster_and_choose_prefers_main_bar_over_bowler_figures():
    cands = [(fi, b) for fi, f in enumerate([FRAME_100, FRAME_470, FRAME_277]) for b in sl(f)]
    groups = cluster(cands)
    assert len(groups) == 2   # main bar + bowler figures, both on 3 frames
    best = choose_cluster(groups)
    assert {b.text for _, b in best} == {"6-0 0.4", "73-17.5", "139-3 16.1"}


def test_choose_cluster_needs_min_frames():
    groups = cluster([(0, b) for b in sl(FRAME_100)])
    assert choose_cluster(groups, min_frames=2) is None
    assert choose_cluster(groups, min_frames=1) is not None


def test_union_and_padding_clamped():
    u = union_box([Box("", 858, 975, 1046, 1019), Box("", 880, 977, 1066, 1012)])
    assert (u.x0, u.y0, u.x1, u.y1) == (858, 975, 1066, 1019)
    assert pad_region(u, 40, W, H) == {"x": 838, "y": 965, "w": 248, "h": 64}
    assert pad_region(Box("", 5, 1070, 1915, 1080), 40, W, H) == {"x": 0, "y": 1060, "w": W, "h": 20}


def test_alt_layout_is_separate_cluster():
    frames = [FRAME_100, FRAME_470, FRAME_277, FRAME_608, FRAME_513]
    groups = cluster([(fi, b) for fi, f in enumerate(frames) for b in sl(f)])
    main = choose_cluster(groups)
    alt = choose_alt(groups, main)
    assert {b.text for _, b in alt} == {"181-519.2", "112-212.2"}


def test_near():
    assert near(Box("", 894, 943, 1022, 958), Box("", 858, 975, 1066, 1019), 120)
    assert not near(Box("", 330, 943, 458, 958), Box("", 858, 975, 1066, 1019), 120)


def test_propose_demo_like_frames():
    frames = [FRAME_100, FRAME_470, FRAME_277, FRAME_400, FRAME_608, FRAME_513]
    regions, info = propose(frames, W, H)
    assert set(regions) == {"scoreline", "scoreline_alt", "target", "footer"}
    s = regions["scoreline"]
    # current hand-measured config: {x: 850, y: 962, w: 235, h: 62}
    assert abs(s["x"] - 850) <= 25 and abs(s["y"] - 962) <= 10 and abs(s["w"] - 235) <= 40 and abs(s["h"] - 62) <= 10
    t = regions["target"]   # the TARGET next to the main bar (frame 470), not the compact layout's
    assert t["x"] < 894 and t["x"] + t["w"] > 1022 and info["target"]["frames"] == [1]
    assert regions["footer"]["scale"] == 2


def test_propose_without_target_near_scoreline():
    regions, _ = propose([FRAME_100, FRAME_277, FRAME_608], W, H)
    assert "target" not in regions and "scoreline" in regions


def test_default_times_nest_and_stay_in_middle():
    t5, t9 = default_times(600), default_times(600, 9)
    assert t5 == [90.0, 195.0, 300.0, 405.0, 510.0]
    assert set(t5) <= set(t9)


def test_build_config_format():
    cfg = build_config({"scoreline": {"x": 1, "y": 2, "w": 3, "h": 4}}, W, H, "demo/x.mp4")
    assert cfg["resolution"] == [W, H] and cfg["consensus_frames"] == 3
    assert "PROPOSAL" in cfg["_comment"] and set(cfg) >= {"resolution", "regions", "consensus_frames"}
    json.dumps(cfg)
