"""ModalitySignal 표준 + 스케일 변환 헬퍼 단위 테스트 (DB/API 불필요)."""
import math

import pytest

from src.modality.base import (
    NEUTRAL,
    ModalitySignal,
    clamp_conf,
    clamp_score,
    direction_of,
    from_bipolar,
    from_probability,
    to_bipolar,
)


# ── 클램프 ────────────────────────────────────────────────
def test_clamp_score_bounds():
    assert clamp_score(-10) == 0.0
    assert clamp_score(150) == 100.0
    assert clamp_score(73.4) == 73.4


def test_clamp_score_none_and_nan_neutral():
    assert clamp_score(None) == NEUTRAL
    assert clamp_score(float("nan")) == NEUTRAL


def test_clamp_conf_bounds():
    assert clamp_conf(-1) == 0.0
    assert clamp_conf(2) == 1.0
    assert clamp_conf(None) == 0.0


# ── 양극 변환 ─────────────────────────────────────────────
@pytest.mark.parametrize("pm1,expected", [(-1, 0.0), (0.0, 50.0), (1.0, 100.0), (0.5, 75.0)])
def test_from_bipolar(pm1, expected):
    assert from_bipolar(pm1) == expected


def test_bipolar_roundtrip():
    for s in (0.0, 25.0, 50.0, 87.5, 100.0):
        assert math.isclose(from_bipolar(to_bipolar(s)), s, abs_tol=1e-9)


# ── 확률 변환 ─────────────────────────────────────────────
def test_from_probability_single():
    assert from_probability(0.8) == 80.0
    # 퍼센트로 들어와도 자동 정규화
    assert from_probability(80) == 80.0


def test_from_probability_up_down():
    assert from_probability(0.5, 0.5) == NEUTRAL          # 동률 → 중립
    assert from_probability(0.7, 0.1) > NEUTRAL           # 상승 우세 → 강세
    assert from_probability(0.1, 0.7) < NEUTRAL           # 하락 우세 → 약세


# ── 방향 ──────────────────────────────────────────────────
def test_direction_deadband():
    assert direction_of(50) == "중립"
    assert direction_of(54) == "중립"      # deadband 5 안
    assert direction_of(56) == "강세"
    assert direction_of(44) == "약세"


# ── ModalitySignal ────────────────────────────────────────
def test_signal_clamps_on_construction():
    sig = ModalitySignal(name="x", score=999, confidence=5)
    assert sig.score == 100.0
    assert sig.confidence == 1.0


def test_unavailable_signal_is_neutral_zeroconf():
    sig = ModalitySignal.unavailable("news", "키없음")
    assert sig.available is False
    assert sig.score == NEUTRAL
    assert sig.confidence == 0.0
    assert sig.detail["reason"] == "키없음"


def test_from_probability_classmethod_infers_confidence():
    sig = ModalitySignal.from_probability("chart", 0.9, 0.1)
    assert sig.score > 50
    assert sig.confidence > 0      # 방향 뚜렷 → 신뢰도 추정됨


def test_from_bipolar_classmethod():
    sig = ModalitySignal.from_bipolar("news", 1.0, confidence=0.5, label="호재")
    assert sig.score == 100.0
    assert sig.display_label == "호재"
