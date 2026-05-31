"""밸류에이션 모달리티 단위 테스트 — TTM 지표 계산 + 신호 변환 (DB/API 불필요)."""
import pytest

from src.modality.base import NEUTRAL
from src.modality.valuation import calculate_ttm_metrics, valuation_to_signal


def _rec(net, rev, op, equity=0, debt=0, shares=0):
    return {
        "net_income": net, "revenue": rev, "operating_income": op,
        "total_equity": equity, "total_debt": debt, "shares": shares,
    }


# ── TTM 지표 계산 ─────────────────────────────────────────
def test_empty_records_returns_empty():
    assert calculate_ttm_metrics([], 1000) == {}


def test_ttm_sums_recent_four_quarters():
    recs = [_rec(100, 1000, 200, equity=2000, shares=10) for _ in range(4)]
    m = calculate_ttm_metrics(recs, current_price=100)
    assert m["TTM_net_income"] == 400      # 100×4
    assert m["TTM_revenue"] == 4000
    assert m["EPS"] == pytest.approx(40.0)  # 400/10
    assert m["BPS"] == pytest.approx(200.0)  # 2000/10
    assert m["PER"] == pytest.approx(2.5)    # 100/40
    assert m["PBR"] == pytest.approx(0.5)    # 100/200
    assert m["ROE"] == pytest.approx(20.0)   # 400/2000×100
    assert m["op_margin"] == pytest.approx(20.0)  # 800/4000×100


def test_ttm_uses_only_first_four_records():
    recs = [_rec(100, 1000, 200, equity=2000, shares=10) for _ in range(6)]
    m = calculate_ttm_metrics(recs, 100)
    assert m["TTM_net_income"] == 400      # 6개 중 4개만


def test_per_none_when_loss_making():
    recs = [_rec(-100, 1000, -50, equity=2000, shares=10) for _ in range(4)]
    m = calculate_ttm_metrics(recs, 100)
    assert m["PER"] is None                # EPS<0 → PER None
    assert m["ROE"] == pytest.approx(-20.0)


def test_zero_shares_no_crash():
    recs = [_rec(100, 1000, 200, equity=2000, shares=0) for _ in range(4)]
    m = calculate_ttm_metrics(recs, 100)
    assert m["EPS"] == 0
    assert m["PER"] is None


def test_none_fields_treated_as_zero():
    recs = [{"net_income": None, "revenue": None, "operating_income": None,
             "total_equity": 1000, "shares": 10}] * 4
    m = calculate_ttm_metrics(recs, 100)   # None → 0, no crash
    assert m["TTM_net_income"] == 0


# ── 신호 변환 ─────────────────────────────────────────────
def test_valuation_signal_unavailable_on_empty():
    sig = valuation_to_signal({})
    assert sig.available is False


def test_cheap_high_roe_is_bullish():
    # 저PER + 고ROE → 강세(저평가)
    metrics = {"PER": 8.0, "ROE": 40.0, "PBR": 0.8}
    sig = valuation_to_signal(metrics)
    assert sig.available is True
    assert sig.score > NEUTRAL
    assert sig.label == "저평가"


def test_expensive_low_roe_is_bearish():
    metrics = {"PER": 55.0, "ROE": 5.0, "PBR": 4.0}
    sig = valuation_to_signal(metrics)
    assert sig.score < NEUTRAL
    assert sig.label == "고평가"


def test_more_metrics_higher_confidence():
    one = valuation_to_signal({"ROE": 25.0})
    three = valuation_to_signal({"PER": 10.0, "ROE": 25.0, "PBR": 1.0})
    assert three.confidence > one.confidence
