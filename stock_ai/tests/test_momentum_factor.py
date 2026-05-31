"""가격 모멘텀 팩터 단위 테스트 (DB 불필요 — _compute_scores 직접 호출)."""
import pandas as pd

from src.screener.rule_based import RuleBasedScreener, ScreenerWeights


def _base_df(mom_a, mom_b):
    """펀더멘털 동일, 모멘텀만 다른 2종목."""
    return pd.DataFrame({
        "ticker": ["A", "B"], "name": ["a", "b"], "market": ["KOSPI", "KOSPI"],
        "roe": [25.0, 25.0],
        "revenue_growth_yoy": [10.0, 10.0],
        "profit_growth_yoy": [10.0, 10.0],
        "per": [None, None],
        "price_momentum": [mom_a, mom_b],
    })


def test_higher_momentum_higher_score():
    scr = RuleBasedScreener()
    out = scr._compute_scores(_base_df(0.50, -0.20).copy(),
                              cap_bonus=False, momentum_penalty=False)
    a = out[out.ticker == "A"]["rule_score"].iloc[0]
    b = out[out.ticker == "B"]["rule_score"].iloc[0]
    assert a > b                       # 높은 모멘텀 → 높은 점수
    assert "momentum_score" in out.columns


def test_momentum_disabled_when_weight_zero():
    scr = RuleBasedScreener(weights=ScreenerWeights(price_momentum=0.0))
    out = scr._compute_scores(_base_df(0.50, -0.20).copy(),
                              cap_bonus=False, momentum_penalty=False)
    a = out[out.ticker == "A"]["rule_score"].iloc[0]
    b = out[out.ticker == "B"]["rule_score"].iloc[0]
    assert a == b                      # 가중 0 → 모멘텀 무시 → 동점


def test_no_momentum_column_is_safe():
    scr = RuleBasedScreener()
    df = _base_df(0.5, -0.2).drop(columns=["price_momentum"])
    out = scr._compute_scores(df.copy(), cap_bonus=False, momentum_penalty=False)
    # 컬럼 없으면 모멘텀 미적용 → 동일 펀더라 동점, 에러 없음
    assert out["rule_score"].nunique() == 1
