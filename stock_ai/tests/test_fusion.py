"""Late Fusion 엔진 단위 테스트 (DB/API 불필요) — 병합 시스템의 핵심 검증."""
import pytest

from src.modality.base import NEUTRAL, ModalitySignal
from src.modality.fusion import DEFAULT_WEIGHTS, LateFusion, verdict_of


def _sig(name, score, conf=1.0):
    return ModalitySignal(name=name, score=score, confidence=conf)


# ── 판정 라벨 ─────────────────────────────────────────────
@pytest.mark.parametrize("score,label", [
    (95, "매수 강력추천"), (60, "매수"), (50, "관망"),
    (35, "매도"), (10, "매도 강력추천"),
])
def test_verdict_thresholds(score, label):
    assert verdict_of(score) == label


# ── 가중 평균 ─────────────────────────────────────────────
def test_all_neutral_gives_neutral():
    f = LateFusion(confidence_weighted=False)
    sigs = {n: _sig(n, NEUTRAL) for n in DEFAULT_WEIGHTS}
    res = f.fuse(sigs)
    assert res.score == pytest.approx(50.0)
    assert res.direction == "중립"
    assert res.verdict == "관망"


def test_weighted_average_matches_manual():
    # confidence_weighted=False 로 순수 가중 평균 검증
    f = LateFusion(weights={"fundamental": 0.5, "chart": 0.5}, confidence_weighted=False)
    sigs = {"fundamental": _sig("fundamental", 80), "chart": _sig("chart", 40)}
    res = f.fuse(sigs)
    assert res.score == pytest.approx(60.0)        # (80+40)/2
    assert res.contributions == {"fundamental": 0.5, "chart": 0.5}


def test_confidence_weighting_shifts_toward_confident_signal():
    f = LateFusion(weights={"a": 0.5, "b": 0.5}, confidence_weighted=True)
    # 같은 가중치지만 a 가 더 확신 → 결과가 a(90) 쪽으로 당겨짐
    sigs = {"a": _sig("a", 90, conf=1.0), "b": _sig("b", 10, conf=0.2)}
    res = f.fuse(sigs)
    assert res.score > 50           # a 쪽으로 치우침
    assert res.contributions["a"] > res.contributions["b"]


# ── 결측 모달리티 재정규화 ────────────────────────────────
def test_missing_modalities_are_renormalized():
    f = LateFusion(weights={"fundamental": 0.4, "chart": 0.4, "news": 0.2},
                   confidence_weighted=False)
    sigs = {
        "fundamental": _sig("fundamental", 70),
        "chart": _sig("chart", 70),
        "news": ModalitySignal.unavailable("news"),   # 제외돼야 함
    }
    res = f.fuse(sigs)
    assert "news" in res.skipped
    assert set(res.used) == {"fundamental", "chart"}
    # 남은 둘로 재정규화 → 각 0.5
    assert res.contributions["fundamental"] == pytest.approx(0.5)
    assert res.score == pytest.approx(70.0)


def test_all_unavailable_returns_neutral_no_crash():
    f = LateFusion()
    sigs = {n: ModalitySignal.unavailable(n) for n in DEFAULT_WEIGHTS}
    res = f.fuse(sigs)
    assert res.score == NEUTRAL
    assert res.confidence == 0.0
    assert res.used == []
    assert res.verdict == "관망"


def test_empty_signals_dict_no_crash():
    res = LateFusion().fuse({})
    assert res.score == NEUTRAL
    assert res.used == []


# ── 충돌 감지 ─────────────────────────────────────────────
def test_conflict_detected_between_strong_opposite_signals():
    f = LateFusion(weights={"chart": 0.5, "news": 0.5})
    sigs = {"chart": _sig("chart", 90), "news": _sig("news", 10)}  # 강세 vs 약세
    res = f.fuse(sigs)
    assert res.conflict is True
    assert "충돌" in res.conflict_note


def test_no_conflict_when_aligned():
    f = LateFusion(weights={"chart": 0.5, "news": 0.5})
    sigs = {"chart": _sig("chart", 80), "news": _sig("news", 70)}  # 둘 다 강세
    res = f.fuse(sigs)
    assert res.conflict is False


def test_contributions_sum_to_one():
    f = LateFusion()
    sigs = {n: _sig(n, 60) for n in DEFAULT_WEIGHTS}
    res = f.fuse(sigs)
    assert sum(res.contributions.values()) == pytest.approx(1.0)
