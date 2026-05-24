"""
6개월 리밸런싱 백테스트 (★ 사용자 핵심 요구)
================================================
당신이 원하는 시뮬레이션:
  - 2015년 1월부터 시작
  - 6개월마다 리밸런싱 (1월, 7월)
  - 매번 펀더멘털 점수 상위 10개 선정
  - 순위 가중 비중 (1위 가장 많이, 10위 가장 적게)
  - 매년 수익률 + KOSPI 대비 알파 측정

리밸런싱 규칙:
  - 신규 매수: 이전에 없던 종목 → 거래비용 부과
  - 보유 유지: 이전에도 있던 종목 → 비중 조정만 (작은 거래비용)
  - 매도: 이번 픽에 없으면 청산 → 거래비용 부과
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd

from src.config import CFG, DB_PATH

logger = logging.getLogger(__name__)


# ============================================================
# 비중 계산
# ============================================================
def rank_weighted(n: int) -> np.ndarray:
    """
    순위 가중: 1위가 가장 많고 N위가 가장 적게.
    공식: weight_i = (N - i + 1) / sum
    예: N=10 → [10, 9, 8, ..., 1] / 55
    """
    if n <= 0:
        return np.array([])
    raw = np.arange(n, 0, -1, dtype=float)   # [N, N-1, ..., 1]
    return raw / raw.sum()


def equal_weighted(n: int) -> np.ndarray:
    return np.ones(n) / n if n > 0 else np.array([])


def score_weighted(scores: np.ndarray) -> np.ndarray:
    """점수 비례. score가 음수가 나오면 0으로 클립."""
    s = np.clip(scores, 0, None)
    return s / s.sum() if s.sum() > 0 else equal_weighted(len(scores))


# ============================================================
# 결과 컨테이너
# ============================================================
@dataclass
class RebalanceResult:
    config: dict
    periods: pd.DataFrame      # 리밸런싱 구간별 수익률
    yearly: pd.DataFrame       # 연도별 집계
    holdings: pd.DataFrame     # 종목별 거래 기록
    metrics: dict = field(default_factory=dict)
    daily_equity: pd.DataFrame = field(default_factory=pd.DataFrame)  # 일별 자산곡선

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "=" * 70,
            "6개월 리밸런싱 백테스트 결과",
            "=" * 70,
            f"기간:           {m.get('start')} ~ {m.get('end')} ({m.get('n_periods')}개 6개월 구간)",
            f"리밸런싱 횟수:  {m.get('n_rebalances')}",
            "",
            f"총 수익률:      {m.get('total_return', 0)*100:>8.2f}%   (KOSPI200 {m.get('benchmark_total', 0)*100:>6.2f}%)",
            f"CAGR:           {m.get('cagr', 0)*100:>8.2f}%   (KOSPI200 {m.get('benchmark_cagr', 0)*100:>6.2f}%)",
            f"연평균 알파:    {m.get('alpha_annualized', 0)*100:>8.2f}%",
            "",
            f"변동성(연):     {m.get('volatility', 0)*100:>8.2f}%",
            f"샤프 지수:      {m.get('sharpe', 0):>8.2f}",
            f"최대 낙폭:      {m.get('mdd', 0)*100:>8.2f}%",
            "",
            f"승률(반기):     {m.get('win_rate_periods', 0)*100:>8.2f}%",
            f"승률(종목):     {m.get('win_rate_holdings', 0)*100:>8.2f}%",
            "=" * 70,
        ]
        return "\n".join(lines)


# ============================================================
# 백테스트 엔진
# ============================================================
PickerFn = Callable[[str, int], pd.DataFrame]
# (as_of_date, top_n) → DataFrame(ticker, rule_score, ...)


class RebalanceBacktest:
    """6개월마다 리밸런싱하는 백테스트."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield conn
        finally:
            conn.close()

    # ----- 거래일 헬퍼 -----
    def _trading_days_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
        with self._conn() as conn:
            return [pd.Timestamp(r[0]) for r in conn.execute("""
                SELECT DISTINCT date FROM ohlcv
                WHERE date BETWEEN ? AND ?
                ORDER BY date
            """, (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))).fetchall()]

    def _first_trading_day(self, target: pd.Timestamp) -> Optional[pd.Timestamp]:
        """target 이상의 첫 거래일."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT MIN(date) FROM ohlcv WHERE date >= ?
            """, (target.strftime("%Y-%m-%d"),)).fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None

    def _price_at(
        self, ticker: str, date: pd.Timestamp, field: str = "open"
    ) -> Optional[float]:
        """date 시점 가격(없으면 가장 가까운 다음 거래일)."""
        with self._conn() as conn:
            row = conn.execute(f"""
                SELECT {field} FROM ohlcv
                WHERE ticker=? AND date >= ? AND {field} IS NOT NULL
                ORDER BY date LIMIT 1
            """, (ticker, date.strftime("%Y-%m-%d"))).fetchone()
        return float(row[0]) if row and row[0] else None

    # ----- 리밸런싱 날짜 생성 -----
    @staticmethod
    def _rebalance_dates(start_year: int, end_year: int,
                         period_months: int = 6,
                         rebalance_day: int = 1) -> list[pd.Timestamp]:
        """
        리밸런싱 시점 생성.
        period_months=6: 1월 + 7월 (반기)
        period_months=4: 1월 + 5월 + 9월 (4개월)
        period_months=3: 1월 + 4월 + 7월 + 10월 (분기)
        rebalance_day: 각 월의 며칠에 리밸 (기본 1일). 해당 월에 그 일자가
            없으면 (예: 2월 30일) 그 월의 말일로 보정.
        """
        import calendar
        dates = []
        for y in range(start_year, end_year + 1):
            for m in range(1, 13, period_months):
                last = calendar.monthrange(y, m)[1]
                day = min(rebalance_day, last)
                dates.append(pd.Timestamp(f"{y}-{m:02d}-{day:02d}"))
        return dates

    # ----- 메인 실행 -----
    def run(
        self,
        picker: PickerFn,
        start_year: int = 2015,
        end_year: int = 2024,
        top_n: int = 10,
        weight_scheme: str = "rank",   # "rank" / "equal" / "score"
        txn_cost: Optional[float] = None,
        replacement_rule: str = "always",
        score_diff_pct: float = 15.0,
        period_months: int = 6,
        rebalance_day: int = 1,        # 각 월의 며칠에 리밸 (1~28)
        trend_stop_loss: bool = False,
        market_regime_filter: bool = False,
        # 현금 구간에 보유할 방어 자산 (KODEX 골드선물H 132030, 검증 채택 2026-05-15).
        # 빈 문자열이면 기존 0% 현금 수익. 현 시스템 기본값 = 132030.
        defensive_ticker: str = "132030",
        # ── Ichimoku Cloud + ADX/DMI 추세 강도 필터 (현 시스템, 2026-05-16 채택) ──
        # 약세 시 KODEX 골드선물H 132030 보유. MA200 레짐 대체.
        ichimoku_adx: bool = True,
        ia_tenkan: int = 9, ia_kijun: int = 26, ia_senkou_b: int = 52,
        ia_adx_period: int = 14,
        ia_adx_threshold: float = 25.0,
        # 분할 스위칭 (3-state: 100% / 50% / 0%) — 2026-05-16 채택
        ia_scaling: bool = True,
        # 부분 반기 종료일 (예: '2026-04-30'). None이면 end_year 정상 종료.
        end_date: Optional[str] = None,
        # ── 동적 레짐 파라미터 ──────────────────────────────────────────
        hysteresis_pct: float = 1.0,
        regime_cash_tiers: tuple = (
            (-2.0,          0.30),
            (-10.0,         0.80),
            (float("-inf"), 0.90),
        ),
        # ── ① 절대 모멘텀 오버라이드 ───────────────────────────────────
        momentum_days: int = 0,                # 0=비활성, 63≈3개월
        momentum_threshold_pct: float = 20.0,
        # ── ② MA 기준선 타입 ───────────────────────────────────────────
        ma_type: str = "ma200",                # "ma200" | "ma100" | "kama"
        # ── ③ 약세장 단기MA 현금 상한 ──────────────────────────────────
        bear_ma20_max_cash: float = 1.0,       # 1.0=비활성, 0.5=MA위면 50% 상한
        bear_ma20_days: int = 20,
        # ── (하위 호환) 단순 이진 모드용 ─────────────────────────────────
        cash_ratio_bearish: float = 0.5,
    ) -> RebalanceResult:
        """
        Parameters
        ----------
        replacement_rule : 종목 교체 규칙
            'always'      매번 처음부터 상위 N개 (기존 방식)
            'keep_simple' 기존 보유 통과하면 무조건 유지
            'score_diff'  새 후보 점수가 기존 최저 종목보다 score_diff_pct% 이상 높을 때만 교체
            'three_cond'  새 후보가 시총 ↑ AND PER ↓ AND ROE ↑ 모두 만족할 때만 교체
        score_diff_pct : score_diff 모드에서 교체 임계값 (기본 15%)
        period_months : 리밸런싱 주기 (6=반기, 3=분기)
        trend_stop_loss : True면 보유 중 60일선 이탈 종목 즉시 매도
        market_regime_filter : KOSPI200 200MA 기반 동적 현금 비중 활성화
        hysteresis_pct : 히스테리시스 데드존 크기 (%). 기본 1%.
            bear→bull 전환: price > MA200 × (1 + hysteresis_pct/100)
            bull→bear 전환: price < MA200 × (1 - hysteresis_pct/100)
        regime_cash_tiers : 이격도 단계별 (하한%, 현금비중) 튜플 목록.
            이격도 ≥ 0% → 현금 0% (풀 매수). bull 상태이면 항상 0%.
            bear 상태일 때 이격도에 따라 단계적 현금 비중 적용.
        cash_ratio_bearish : (deprecated) 단순 이진 필터용. hysteresis + tiers 사용 권장.
        """
        valid_rules = {"always", "keep_simple", "score_diff", "three_cond"}
        if replacement_rule not in valid_rules:
            raise ValueError(f"replacement_rule must be one of {valid_rules}")

        txn_cost = txn_cost if txn_cost is not None else CFG.backtest.txn_cost
        config = dict(start_year=start_year, end_year=end_year, top_n=top_n,
                      weight_scheme=weight_scheme, txn_cost=txn_cost,
                      replacement_rule=replacement_rule,
                      score_diff_pct=score_diff_pct,
                      benchmark=CFG.backtest.benchmark_ticker)

        # 리밸런싱 시점 → 다음 리밸런싱 시점 = 한 보유 구간
        rb_dates = self._rebalance_dates(start_year, end_year, period_months, rebalance_day)
        # 마지막 시점은 청산일이 없으니 제외 (종료일까지 보유로 계산)
        end_of_data = (pd.Timestamp(end_date) if end_date
                       else pd.Timestamp(f"{end_year}-12-31"))
        rb_actual: list[pd.Timestamp] = []
        for d in rb_dates:
            t = self._first_trading_day(d)
            if t is None or t > end_of_data:
                continue
            rb_actual.append(t)
        # end_date 지정 시 마지막 부분 반기를 위한 종료 거래일 추가
        # 벤치마크(069500) 기준 — 개별주 데이터가 후행할 수 있는 ETF 잔재 회피
        if end_date:
            bt_ticker = CFG.backtest.benchmark_ticker
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT MAX(date) FROM ohlcv WHERE ticker=? AND date <= ?",
                    (bt_ticker, end_date)).fetchone()
            last_td = pd.Timestamp(row[0]) if row and row[0] else None
            if last_td is not None and (not rb_actual or last_td > rb_actual[-1]):
                rb_actual.append(last_td)

        if len(rb_actual) < 2:
            raise RuntimeError("리밸런싱 시점 부족 — 데이터 수집 확인 필요")

        # ── 레짐 현금 비중 사전 계산 (리밸런싱 날짜 기준) ────────────────
        regime_map: dict[pd.Timestamp, tuple[float, float, bool]] = {}
        if market_regime_filter and not ichimoku_adx:
            regime_map = self._compute_regime_cash_ratios(
                rb_actual, hysteresis_pct, regime_cash_tiers,
                momentum_days, momentum_threshold_pct,
                ma_type,
                bear_ma20_max_cash, bear_ma20_days,
            )

        # Ichimoku + ADX 신호 사전 계산 (일별)
        # 바이너리: "STOCKS"/"GOLD"/None
        # 분할:   "BULL"/"CLOUD"/"BEAR_DEEP"/"WEAK"/None
        ichimoku_signals: dict[pd.Timestamp, Optional[str]] = {}
        if ichimoku_adx:
            all_days_ia = self._trading_days_in_range(rb_actual[0], rb_actual[-1])
            if ia_scaling:
                ichimoku_signals = self._compute_ichimoku_scaling_signals(
                    all_days_ia, ia_tenkan, ia_kijun, ia_senkou_b,
                    ia_adx_period, ia_adx_threshold,
                )
            else:
                ichimoku_signals = self._compute_ichimoku_adx_signals(
                    all_days_ia, ia_tenkan, ia_kijun, ia_senkou_b,
                    ia_adx_period, ia_adx_threshold,
                )

        period_rows: list[dict] = []
        holdings_rows: list[dict] = []
        daily_equity_rows: list[dict] = []      # 일별 자산곡선 → 일별 실측 MDD용

        port_value = 1.0
        bench_value = 1.0
        prev_holdings: dict[str, float] = {}    # {ticker: weight}
        txn_cost_accum = 0.0                    # 누적 거래비용
        # Ichimoku+ADX 상태기계 (이진)
        ia_state = "STOCKS"                     # 바이너리 "STOCKS" | "GOLD"
        ia_transitions: list[tuple] = []        # (date, from, to)
        ia_level = 1.0                          # 분할 스위칭 비중 (1.0 / 0.5 / 0.0)

        for i in range(len(rb_actual) - 1):
            entry_date = rb_actual[i]
            exit_date = rb_actual[i + 1]

            # 1) 종목 선정 (룩어헤드 방지: 진입 직전 시점 데이터로)
            decision_cutoff = entry_date - pd.Timedelta(days=1)
            try:
                picks = picker(decision_cutoff.strftime("%Y-%m-%d"), top_n)
            except Exception as e:
                logger.error("[%s] 선정 실패: %s", entry_date.date(), e)
                continue

            if picks.empty:
                logger.warning("[%s] 종목 없음 — 현금 유지", entry_date.date())
                # 현금 유지 → 그 구간 0% 수익
                period_rows.append({
                    "entry_date": entry_date, "exit_date": exit_date,
                    "n_picks": 0, "period_return": 0.0,
                    "benchmark_return": self._benchmark_return(entry_date, exit_date),
                    "portfolio_value": port_value,
                    "benchmark_value": bench_value * (1 + self._benchmark_return(entry_date, exit_date)),
                })
                # 현금 보유 구간 → 일별 자산곡선은 평탄
                for d in self._trading_days_in_range(entry_date, exit_date):
                    daily_equity_rows.append({
                        "date": d, "portfolio_value": port_value,
                    })
                bench_value *= (1 + self._benchmark_return(entry_date, exit_date))
                prev_holdings = {}
                continue

            # 1.5) 종목 교체 규칙 적용
            #      - always: 그대로 두고 점수순 N개
            #      - keep_simple: 기존 보유 + 새 종목 합쳐서 기존 우선
            #      - score_diff: 새 후보가 기존 최저보다 X% 이상 높아야 교체 (A)
            #      - three_cond: 시총↑ AND PER↓ AND ROE↑ 동시 만족 시만 교체 (B)
            score_col = "rule_score" if "rule_score" in picks.columns else "final_score"

            if replacement_rule == "always" or not prev_holdings:
                pass  # 기본 동작 유지
            elif replacement_rule == "keep_simple":
                kept = picks[picks["ticker"].isin(prev_holdings)]
                new_ = picks[~picks["ticker"].isin(prev_holdings)]
                picks = pd.concat([
                    kept.sort_values(score_col, ascending=False),
                    new_.sort_values(score_col, ascending=False),
                ], ignore_index=True)
                logger.info("[%s] keep_simple: %d유지 / %d신규",
                            entry_date.date(), len(kept), top_n - len(kept))
            elif replacement_rule == "score_diff":
                picks = self._apply_score_diff(picks, prev_holdings, top_n,
                                                 score_col, score_diff_pct,
                                                 entry_date)
            elif replacement_rule == "three_cond":
                picks = self._apply_three_cond(picks, prev_holdings, top_n,
                                                 score_col, entry_date)

            # 2) 비중 계산
            picks = picks.head(top_n).reset_index(drop=True)
            if weight_scheme == "rank":
                weights = rank_weighted(len(picks))
            elif weight_scheme == "equal":
                weights = equal_weighted(len(picks))
            elif weight_scheme == "score":
                score_col = "rule_score" if "rule_score" in picks.columns else "final_score"
                weights = score_weighted(picks[score_col].to_numpy())
            else:
                raise ValueError(f"알 수 없는 weight_scheme: {weight_scheme}")

            # 3) 진입가/청산가 조회
            entry_prices: dict[str, float] = {}
            exit_prices: dict[str, float] = {}
            valid_idx = []
            for idx, row in picks.iterrows():
                tk = row["ticker"]
                ep = self._price_at(tk, entry_date, "open")
                xp = self._price_at(tk, exit_date, "open")
                if ep is None or xp is None or ep <= 0 or xp <= 0:
                    logger.debug("[%s] %s: 가격 없음", entry_date.date(), tk)
                    continue
                # ★ trend_stop_loss: 보유 기간 중 60일선 깬 시점 찾으면 거기서 매도
                if trend_stop_loss:
                    sl = self._find_trend_break(tk, entry_date, exit_date)
                    if sl is not None:
                        sl_date, sl_price = sl
                        if sl_price > 0:
                            xp = sl_price  # 그 시점 종가로 매도 처리
                            logger.debug("[%s] %s: 추세 손절 %s @ %.0f",
                                         entry_date.date(), tk, sl_date.date(), sl_price)
                entry_prices[tk] = ep
                exit_prices[tk] = xp
                valid_idx.append(idx)

            if not valid_idx:
                logger.warning("[%s] 유효 종목 없음", entry_date.date())
                continue

            # 유효 종목만 필터
            picks_valid = picks.loc[valid_idx].reset_index(drop=True)
            if weight_scheme == "rank":
                weights = rank_weighted(len(picks_valid))
            elif weight_scheme == "equal":
                weights = equal_weighted(len(picks_valid))
            elif weight_scheme == "score":
                score_col = "rule_score" if "rule_score" in picks_valid.columns else "final_score"
                weights = score_weighted(picks_valid[score_col].to_numpy())

            # 4) 거래비용 계산 (turnover 기반)
            curr_holdings = {row["ticker"]: weights[i] for i, (_, row) in enumerate(picks_valid.iterrows())}
            turnover = self._compute_turnover(prev_holdings, curr_holdings)
            cost = turnover * txn_cost

            # 5) 종목별 수익률 → 가중 합산
            tickers = picks_valid["ticker"].tolist()
            rets = np.array([
                exit_prices[tk] / entry_prices[tk] - 1.0 for tk in tickers
            ])
            gross_period_ret = float((weights * rets).sum())

            # 벤치마크
            bench_ret = self._benchmark_return(entry_date, exit_date)
            period_start_value = port_value

            if ichimoku_adx:
                # ── Ichimoku Cloud + ADX 일별 워크 (binary 또는 분할 scaling) ──
                basket = self._basket_daily_path(
                    tickers, weights, entry_prices, exit_prices,
                    entry_date, exit_date,
                )
                _, gold_cumret = self._defensive_path(
                    defensive_ticker or "132030", entry_date, exit_date)

                if ia_scaling:
                    # 3-state 분할 (1.0 / 0.5 / 0.0)
                    port_value *= (1 - turnover * txn_cost * ia_level)
                    txn_cost_accum += turnover * txn_cost * ia_level
                    if basket is None:
                        port_value *= (1 + ia_level * gross_period_ret)
                        net_period_ret = (port_value / period_start_value - 1.0
                                          if period_start_value else 0.0)
                    else:
                        basket_prev, gold_prev = 1.0, 1.0
                        for d in basket.index:
                            basket_now = float(basket.loc[d])
                            if gold_cumret is not None and d in gold_cumret.index:
                                gold_now = 1.0 + float(gold_cumret.loc[d])
                            else:
                                gold_now = gold_prev
                            stock_ret = basket_now / basket_prev - 1.0
                            gold_ret = gold_now / gold_prev - 1.0
                            port_ret = ia_level * stock_ret + (1.0 - ia_level) * gold_ret
                            port_value *= (1 + port_ret)
                            basket_prev, gold_prev = basket_now, gold_now
                            # 상태 전환 — 분할 스위칭 규칙
                            sig = ichimoku_signals.get(d)
                            target = ia_level
                            if sig == "BULL":
                                target = 1.0
                            elif sig == "BEAR_DEEP":
                                target = 0.0
                            elif sig == "CLOUD" and ia_level == 1.0:
                                target = 0.5
                            if abs(target - ia_level) > 1e-9:
                                diff = abs(target - ia_level)
                                port_value *= (1 - diff * txn_cost)
                                txn_cost_accum += diff * txn_cost
                                ia_transitions.append((d, ia_level, target))
                                ia_level = target
                            daily_equity_rows.append({
                                "date": d, "portfolio_value": port_value,
                                "stock_pct": ia_level,
                            })
                        net_period_ret = (port_value / period_start_value - 1.0
                                          if period_start_value else 0.0)
                    cash_ratio = 1.0 - ia_level
                    invest_ratio = ia_level
                    bull_regime = ia_level >= 0.5
                else:
                    # 바이너리 (기존 채택본)
                    if ia_state == "STOCKS":
                        port_value *= (1 - turnover * txn_cost)
                        txn_cost_accum += turnover * txn_cost
                    if basket is None:
                        if ia_state == "STOCKS":
                            port_value *= (1 + gross_period_ret)
                        net_period_ret = (port_value / period_start_value - 1.0
                                          if period_start_value else 0.0)
                    else:
                        basket_prev, gold_prev = 1.0, 1.0
                        for d in basket.index:
                            basket_now = float(basket.loc[d])
                            if gold_cumret is not None and d in gold_cumret.index:
                                gold_now = 1.0 + float(gold_cumret.loc[d])
                            else:
                                gold_now = gold_prev
                            if ia_state == "STOCKS":
                                r_d = basket_now / basket_prev - 1.0
                            else:
                                r_d = gold_now / gold_prev - 1.0
                            port_value *= (1 + r_d)
                            basket_prev, gold_prev = basket_now, gold_now
                            sig = ichimoku_signals.get(d)
                            if sig is not None and sig != ia_state:
                                port_value *= (1 - txn_cost)
                                txn_cost_accum += txn_cost
                                ia_transitions.append((d, ia_state, sig))
                                ia_state = sig
                            daily_equity_rows.append({
                                "date": d, "portfolio_value": port_value,
                            })
                        net_period_ret = (port_value / period_start_value - 1.0
                                          if period_start_value else 0.0)
                    cash_ratio = 0.0 if ia_state == "STOCKS" else 1.0
                    invest_ratio = 1.0 - cash_ratio
                    bull_regime = (ia_state == "STOCKS")
                disparity_pct = 0.0
            else:
                # 5.5) 현금 비중 결정 (레짐 필터: 이격도 단계별 현금 비중)
                cash_ratio = 0.0
                disparity_pct = 0.0
                bull_regime = True
                if market_regime_filter and entry_date in regime_map:
                    cash_ratio, disparity_pct, bull_regime = regime_map[entry_date]

                invest_ratio = 1.0 - cash_ratio

                # 방어 자산 (현금 구간 보유). 빈 문자열이면 0% 수익.
                defensive_ret = 0.0
                defensive_cumret = None
                if defensive_ticker and cash_ratio > 0:
                    defensive_ret, defensive_cumret = self._defensive_path(
                        defensive_ticker, entry_date, exit_date)

                net_period_ret = (
                    invest_ratio * ((1 + gross_period_ret) * (1 - cost) - 1)
                    + cash_ratio * defensive_ret
                )
                port_value *= (1 + net_period_ret)
                txn_cost_accum += cost

                # 일별 자산곡선 — 투자 부분(주식) + 현금 부분(방어자산)
                basket = self._basket_daily_path(
                    tickers, weights, entry_prices, exit_prices,
                    entry_date, exit_date,
                )
                if basket is not None:
                    for d in basket.index:
                        invest_part = invest_ratio * (float(basket.loc[d]) * (1 - cost) - 1)
                        if defensive_cumret is not None and d in defensive_cumret.index:
                            cash_part = cash_ratio * float(defensive_cumret.loc[d])
                        else:
                            cash_part = 0.0
                        pv_d = period_start_value * (1 + invest_part + cash_part)
                        daily_equity_rows.append({
                            "date": d, "portfolio_value": pv_d,
                        })

            if market_regime_filter and not ichimoku_adx:
                regime_tag = "🟢강세" if bull_regime else f"🔴약세({disparity_pct:+.1f}%)"
                logger.info(
                    "[%s] 레짐=%s 현금=%.0f%% 투자=%.0f%%",
                    entry_date.date(), regime_tag,
                    cash_ratio * 100, invest_ratio * 100,
                )

            bench_value *= (1 + bench_ret)

            # 6) 기록
            period_rows.append({
                "entry_date": entry_date, "exit_date": exit_date,
                "n_picks": len(picks_valid),
                "gross_return": gross_period_ret,
                "txn_cost": cost,
                "period_return": net_period_ret,
                "benchmark_return": bench_ret,
                "alpha": net_period_ret - bench_ret,
                "turnover": turnover,
                "portfolio_value": port_value,
                "benchmark_value": bench_value,
                "bull_regime": bull_regime,
                "disparity_pct": round(disparity_pct, 2),
                "cash_ratio": round(cash_ratio, 2),
            })
            for j, tk in enumerate(tickers):
                holdings_rows.append({
                    "entry_date": entry_date, "exit_date": exit_date,
                    "ticker": tk,
                    "name": picks_valid.iloc[j].get("name"),
                    "weight": float(weights[j]),
                    "entry_price": entry_prices[tk],
                    "exit_price": exit_prices[tk],
                    "return_pct": float(rets[j]),
                    "score": float(picks_valid.iloc[j].get(
                        "rule_score", picks_valid.iloc[j].get("final_score", 0)
                    )),
                })

            logger.info(
                "[%s ~ %s] 포트 %+.2f%% (gross %+.2f%% - 비용 %.2f%%) | KOSPI200 %+.2f%% | α %+.2f%% | TO %.0f%%",
                entry_date.date(), exit_date.date(),
                net_period_ret * 100, gross_period_ret * 100, cost * 100,
                bench_ret * 100, (net_period_ret - bench_ret) * 100, turnover * 100,
            )

            prev_holdings = curr_holdings

        # 결과 집계
        periods_df = pd.DataFrame(period_rows)
        holdings_df = pd.DataFrame(holdings_rows)
        yearly_df = self._aggregate_yearly(periods_df) if not periods_df.empty else pd.DataFrame()
        metrics = self._compute_metrics(periods_df, yearly_df, holdings_df) if not periods_df.empty else {}

        # 일별 자산곡선 + 일별 MDD (구간 경계 날짜는 마지막 기록 유지)
        daily_equity_df = pd.DataFrame(daily_equity_rows)
        if not daily_equity_df.empty:
            daily_equity_df = (
                daily_equity_df
                .drop_duplicates(subset="date", keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
            eq = daily_equity_df["portfolio_value"].to_numpy()
            peaks = np.maximum.accumulate(eq)
            metrics["mdd_daily"] = float((eq / peaks - 1).min()) if len(eq) else 0.0

        if metrics:
            metrics["total_txn_cost"] = txn_cost_accum
            if ichimoku_adx:
                metrics["ia_transitions"] = ia_transitions
                if ia_scaling:
                    metrics["ia_final_level"] = ia_level
                else:
                    metrics["ia_final_state"] = ia_state

        return RebalanceResult(
            config=config,
            periods=periods_df,
            yearly=yearly_df,
            holdings=holdings_df,
            metrics=metrics,
            daily_equity=daily_equity_df,
        )

    # ----- 거래비용 (turnover) -----
    @staticmethod
    def _compute_turnover(prev: dict, curr: dict) -> float:
        """
        turnover = 0.5 × Σ |w_curr - w_prev|
        prev이 비어있으면 turnover=1.0 (전부 신규 매수).
        """
        if not prev:
            return 1.0
        all_tk = set(prev) | set(curr)
        diff = sum(abs(curr.get(tk, 0) - prev.get(tk, 0)) for tk in all_tk)
        return diff / 2.0

    # ----- 종목 교체 규칙 헬퍼 -----
    def _apply_score_diff(
        self,
        picks: pd.DataFrame,
        prev_holdings: dict,
        top_n: int,
        score_col: str,
        threshold_pct: float,
        entry_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        규칙 A: 새 후보 점수가 기존 최저보다 threshold_pct% 이상 높을 때만 교체.
        그 외에는 기존 보유 우선.
        """
        kept_tk = [tk for tk in prev_holdings if tk in set(picks["ticker"])]
        kept_df = picks[picks["ticker"].isin(kept_tk)].sort_values(score_col, ascending=False)
        new_df = picks[~picks["ticker"].isin(kept_tk)].sort_values(score_col, ascending=False)

        if len(kept_df) >= top_n:
            return kept_df.head(top_n).reset_index(drop=True)

        # 기존 보유 + 부족분 채우면서, 신규는 임계값 넘는 것만
        slots = top_n - len(kept_df)
        # "기존 최저 점수" 기준
        if not kept_df.empty:
            min_kept_score = kept_df[score_col].min()
            threshold = min_kept_score * (1 + threshold_pct / 100.0)
            promotable = new_df[new_df[score_col] >= threshold].head(slots)
            # 임계 못 넘은 신규는 그래도 빈 자리는 채워야 함
            backup = new_df[~new_df["ticker"].isin(promotable["ticker"])].head(slots - len(promotable))
            chosen_new = pd.concat([promotable, backup])
        else:
            chosen_new = new_df.head(slots)

        replaced = len(chosen_new) - (top_n - len(kept_df) - len(chosen_new) if False else 0)
        logger.info("[%s] score_diff: %d유지 / %d신규 (임계 %.0f%%)",
                    entry_date.date(), len(kept_df), len(chosen_new), threshold_pct)
        return pd.concat([kept_df, chosen_new], ignore_index=True).head(top_n)

    def _apply_three_cond(
        self,
        picks: pd.DataFrame,
        prev_holdings: dict,
        top_n: int,
        score_col: str,
        entry_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        규칙 B: 새 후보가 (시총↑) AND (PER↓) AND (ROE↑) 모두 만족할 때만
        기존 포트폴리오에서 시총 가장 작은 종목과 교체.
        PER이 None이면 PER 조건 무시.
        """
        kept_tk = [tk for tk in prev_holdings if tk in set(picks["ticker"])]
        kept_df = picks[picks["ticker"].isin(kept_tk)].copy()
        new_df = picks[~picks["ticker"].isin(kept_tk)].sort_values(score_col, ascending=False).copy()

        # 빈 자리 채우기 — 충분히 비어있으면 새 종목으로 그대로 채움
        slots = top_n - len(kept_df)
        if slots > 0:
            fillers = new_df.head(slots)
            kept_df = pd.concat([kept_df, fillers], ignore_index=True)
            new_df = new_df[~new_df["ticker"].isin(fillers["ticker"])]

        # 이제 가득 찬 상태(top_n개). 남은 신규 후보들로 1:1 교체 시도
        if "market_cap" not in kept_df.columns or kept_df.empty or new_df.empty:
            return kept_df.head(top_n).reset_index(drop=True)

        replaced = 0
        for _, candidate in new_df.iterrows():
            # 기존 중 시총 가장 작은 종목
            kept_df_sorted = kept_df.sort_values("market_cap")
            smallest = kept_df_sorted.iloc[0]
            cond_cap = candidate["market_cap"] > smallest["market_cap"]
            cond_roe = candidate["roe"] > smallest["roe"]
            # PER: 둘 다 있으면 비교, 없으면 통과
            if pd.notna(candidate.get("per")) and pd.notna(smallest.get("per")):
                cond_per = candidate["per"] < smallest["per"]
            else:
                cond_per = True
            if cond_cap and cond_per and cond_roe:
                # 교체
                kept_df = kept_df[kept_df["ticker"] != smallest["ticker"]]
                kept_df = pd.concat([kept_df, pd.DataFrame([candidate])], ignore_index=True)
                replaced += 1

        logger.info("[%s] three_cond: %d교체",
                    entry_date.date(), replaced)
        return kept_df.head(top_n).reset_index(drop=True)


    # ----- 시장 레짐 (동적 현금 비중 + 히스테리시스) -----
    @staticmethod
    def _kama(prices: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        """Kaufman's Adaptive Moving Average — 변동성 조절형 이평선."""
        fast_sc = 2.0 / (fast + 1)
        slow_sc = 2.0 / (slow + 1)
        vals = prices.to_numpy(dtype=float)
        out = np.full(len(vals), np.nan)
        # warm-up: n번째부터 시작
        if len(vals) <= n:
            return pd.Series(out, index=prices.index)
        out[n - 1] = vals[n - 1]
        for i in range(n, len(vals)):
            direction = abs(vals[i] - vals[i - n])
            noise = np.sum(np.abs(np.diff(vals[i - n: i + 1])))
            er = direction / noise if noise > 0 else 0.0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            out[i] = out[i - 1] + sc * (vals[i] - out[i - 1])
        return pd.Series(out, index=prices.index)

    def _compute_regime_cash_ratios(
        self,
        dates: list[pd.Timestamp],
        hysteresis_pct: float = 1.0,
        cash_tiers: tuple = (
            (-2.0,          0.30),
            (-10.0,         0.80),
            (float("-inf"), 0.90),
        ),
        # ① 절대 모멘텀 오버라이드
        momentum_days: int = 0,                 # 0=비활성, 63≈3개월
        momentum_threshold_pct: float = 20.0,   # bear 중 N일 수익률 ≥ 임계값 → 강제 강세
        # ② MA 타입 (기준선)
        ma_type: str = "ma200",                 # "ma200" | "ma100" | "kama"
        # ③ 약세장 MA20 캡
        bear_ma20_max_cash: float = 1.0,        # 1.0=비활성, 0.5=MA20위면 현금 50% 상한
        bear_ma20_days: int = 20,
    ) -> dict:
        """
        각 리밸런싱 날짜에 대해 (현금비중, 이격도%, bull여부) 계산.

        ① 절대 모멘텀: bear 상태에서도 momentum_days 수익률 ≥ momentum_threshold_pct% 이면 강제 강세 전환.
        ② MA 기준선: ma_type으로 "ma200"(기본)/"ma100"/"kama" 선택.
        ③ 약세 MA20 캡: bear이지만 단기 MA 위에 있으면 현금을 bear_ma20_max_cash 이하로 제한.
        """
        from datetime import timedelta

        if not dates:
            return {}

        ma_period = 200 if ma_type == "ma200" else 100
        load_days = max(ma_period * 2, 420) + (momentum_days or 0) + 10
        load_start = (min(dates) - timedelta(days=load_days)).strftime("%Y-%m-%d")
        load_end   = max(dates).strftime("%Y-%m-%d")

        with self._conn() as conn:
            kospi = pd.read_sql_query(
                "SELECT date, close FROM ohlcv "
                "WHERE ticker='069500' AND date BETWEEN ? AND ? ORDER BY date",
                conn, params=[load_start, load_end], parse_dates=["date"],
            )

        if kospi.empty:
            logger.warning("KOSPI200 ETF 데이터 없음 → 레짐 필터 비활성")
            return {d: (0.0, 0.0, True) for d in dates}

        kospi = kospi.set_index("date").sort_index()

        # ② MA 기준선 계산
        if ma_type == "kama":
            kospi["trend_ma"] = self._kama(kospi["close"])
        else:
            kospi["trend_ma"] = kospi["close"].rolling(ma_period, min_periods=ma_period).mean()

        # ③ 단기 MA (MA20 캡용)
        if bear_ma20_max_cash < 1.0:
            kospi["ma_short"] = kospi["close"].rolling(bear_ma20_days, min_periods=bear_ma20_days).mean()

        # ① 모멘텀용 과거가 shift
        if momentum_days > 0:
            kospi["price_n_ago"] = kospi["close"].shift(momentum_days)

        sorted_tiers = sorted(cash_tiers, key=lambda x: x[0], reverse=True)
        is_bull = True
        result: dict = {}

        for date in dates:
            available = kospi.loc[:date]
            if available.empty or pd.isna(available["trend_ma"].iloc[-1]):
                result[date] = (0.0, 0.0, True)
                continue

            row_k = available.iloc[-1]
            current_price = float(row_k["close"])
            trend_ma_val  = float(row_k["trend_ma"])
            disparity_pct = (current_price / trend_ma_val - 1) * 100

            bull_threshold = trend_ma_val * (1 + hysteresis_pct / 100)
            bear_threshold = trend_ma_val * (1 - hysteresis_pct / 100)

            # 히스테리시스 전환 (데드존 안에서는 이전 상태 유지)
            if is_bull and current_price < bear_threshold:
                is_bull = False
                logger.info(
                    "[%s] 레짐 전환: 강세→약세 | KOSPI=%.0f %s=%.0f 이격도=%+.1f%%",
                    date.date(), current_price, ma_type.upper(), trend_ma_val, disparity_pct,
                )
            elif not is_bull and current_price > bull_threshold:
                is_bull = True
                logger.info(
                    "[%s] 레짐 전환: 약세→강세 | KOSPI=%.0f %s=%.0f 이격도=%+.1f%%",
                    date.date(), current_price, ma_type.upper(), trend_ma_val, disparity_pct,
                )

            # ① 절대 모멘텀 오버라이드 (bear 중에만 체크)
            if not is_bull and momentum_days > 0:
                price_n_ago = float(row_k["price_n_ago"]) if not pd.isna(row_k.get("price_n_ago", float("nan"))) else None
                if price_n_ago and price_n_ago > 0:
                    momentum_ret = (current_price / price_n_ago - 1) * 100
                    if momentum_ret >= momentum_threshold_pct:
                        is_bull = True
                        logger.info(
                            "[%s] 모멘텀 오버라이드: %dd수익률=%+.1f%% ≥ %.0f%% → 강제 강세",
                            date.date(), momentum_days, momentum_ret, momentum_threshold_pct,
                        )

            # 현금 비중 결정
            if is_bull:
                cash_ratio = 0.0
            else:
                cash_ratio = sorted_tiers[-1][1]
                for lower_bound, ratio in sorted_tiers:
                    if disparity_pct >= lower_bound:
                        cash_ratio = ratio
                        break

                # ③ 약세장 MA20 캡: 단기 MA 위에 있으면 현금 상한 적용
                if bear_ma20_max_cash < 1.0:
                    ma_short_val = float(row_k.get("ma_short", float("nan")))
                    if not pd.isna(ma_short_val) and current_price > ma_short_val:
                        if cash_ratio > bear_ma20_max_cash:
                            logger.info(
                                "[%s] MA%d 캡: 현금 %.0f%%→%.0f%% (단기MA위)",
                                date.date(), bear_ma20_days, cash_ratio * 100, bear_ma20_max_cash * 100,
                            )
                            cash_ratio = bear_ma20_max_cash

            result[date] = (cash_ratio, disparity_pct, is_bull)

        return result

    # ----- 일별 자산곡선용 헬퍼 -----
    def _load_daily_closes(
        self, tickers: list[str], start: pd.Timestamp, end: pd.Timestamp,
    ) -> pd.DataFrame:
        """tickers의 start~end 일별 종가 → (date 인덱스, ticker 컬럼) DataFrame."""
        if not tickers:
            return pd.DataFrame()
        ph = ",".join("?" * len(tickers))
        with self._conn() as conn:
            df = pd.read_sql_query(
                f"SELECT date, ticker, close FROM ohlcv "
                f"WHERE ticker IN ({ph}) AND date BETWEEN ? AND ? "
                f"AND close IS NOT NULL ORDER BY date",
                conn,
                params=[*tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
                parse_dates=["date"],
            )
        if df.empty:
            return pd.DataFrame()
        return df.pivot(index="date", columns="ticker", values="close")

    def _basket_daily_path(
        self,
        tickers: list[str],
        weights: np.ndarray,
        entry_prices: dict[str, float],
        exit_prices: dict[str, float],
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
    ) -> Optional[pd.Series]:
        """
        보유 바스켓의 buy-and-hold 일별 가치 경로.
        - 시작 기준: entry_date 시가 (값 1.0 직전)
        - 마지막 거래일(exit_date): 시가로 청산 → basket[-1] == 1 + gross_period_ret 보장
        - 중간 거래일: 종가
        반환: date 인덱스 Series. 데이터 부족 시 None.
        """
        days = self._trading_days_in_range(entry_date, exit_date)
        if len(days) < 2:
            return None
        closes = self._load_daily_closes(tickers, entry_date, exit_date)
        if closes.empty:
            return None
        closes = closes.reindex(index=days, columns=tickers)
        cumret = pd.DataFrame(index=days, dtype=float)
        for tk in tickers:
            col = closes[tk].astype(float).copy()
            col.iloc[-1] = exit_prices[tk]          # 마지막 거래일 = 청산 시가
            if pd.isna(col.iloc[0]):
                col.iloc[0] = entry_prices[tk]
            col = col.ffill().bfill()
            cumret[tk] = col / entry_prices[tk]
        w = pd.Series(weights, index=tickers)
        basket = (cumret * w).sum(axis=1)
        return basket

    # ----- Ichimoku Cloud + ADX/DMI 신호 -----
    def _compute_ichimoku_adx_signals(
        self, dates: list[pd.Timestamp],
        tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
        adx_period: int = 14, adx_threshold: float = 25.0,
    ) -> dict[pd.Timestamp, Optional[str]]:
        """KOSPI200 (069500) 일봉 Ichimoku Cloud + ADX/DMI.
        - STOCKS: close > cloud_top AND ADX > threshold AND +DI > -DI
        - GOLD  : close < cloud_bot AND ADX > threshold AND -DI > +DI
        - 그 외 (ADX 약함/cloud 안) : None (유지)
        Senkou span은 kijun 일수만큼 forward shift → 룩어헤드 안전.
        """
        if not dates:
            return {}
        from datetime import timedelta
        warmup = max(senkou_b, kijun, adx_period) * 4 + 60
        load_start = (min(dates) - timedelta(days=warmup * 2)).strftime("%Y-%m-%d")
        load_end = max(dates).strftime("%Y-%m-%d")
        with self._conn() as conn:
            df = pd.read_sql_query(
                "SELECT date, high, low, close FROM ohlcv "
                "WHERE ticker='069500' AND date BETWEEN ? AND ? ORDER BY date",
                conn, params=[load_start, load_end], parse_dates=["date"])
        if df.empty:
            return {d: None for d in dates}
        df = df.set_index("date").sort_index()
        high, low, close = df["high"], df["low"], df["close"]

        # Ichimoku
        tenkan_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        kijun_line  = (high.rolling(kijun).max()  + low.rolling(kijun).min())  / 2
        span_a = ((tenkan_line + kijun_line) / 2).shift(kijun)
        span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)

        # Wilder's ADX/DMI
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm  = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        alpha = 1.0 / adx_period
        atr = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
        plus_di  = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        out: dict[pd.Timestamp, Optional[str]] = {}
        for d in dates:
            if d not in df.index:
                out[d] = None
                continue
            if pd.isna(cloud_top.loc[d]) or pd.isna(adx.loc[d]):
                out[d] = None
                continue
            c  = float(close.loc[d])
            ct = float(cloud_top.loc[d]); cb = float(cloud_bot.loc[d])
            a  = float(adx.loc[d])
            pdi = float(plus_di.loc[d]); mdi = float(minus_di.loc[d])
            if a <= adx_threshold:
                out[d] = None
            elif c > ct and pdi > mdi:
                out[d] = "STOCKS"
            elif c < cb and mdi > pdi:
                out[d] = "GOLD"
            else:
                out[d] = None
        return out

    # ----- Ichimoku 분할 스위칭 신호 (3-state) -----
    def _compute_ichimoku_scaling_signals(
        self, dates: list[pd.Timestamp],
        tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
        adx_period: int = 14, adx_threshold: float = 25.0,
    ) -> dict[pd.Timestamp, Optional[str]]:
        """KOSPI200 일봉 Ichimoku Cloud + ADX 분할 스위칭 신호.
        - BULL      : close > cloud_top AND ADX > threshold AND +DI > -DI
        - CLOUD     : close ≤ cloud_top AND close ≥ cloud_bot (구름 안/하단까지)
        - BEAR_DEEP : close < cloud_bot
        - WEAK      : close > cloud_top 이지만 ADX 약함 또는 +DI ≤ -DI
        호출자에서 state-dependent 전환 결정 (1.0/0.5/0.0).
        """
        if not dates:
            return {}
        from datetime import timedelta
        warmup = max(senkou_b, kijun, adx_period) * 4 + 60
        load_start = (min(dates) - timedelta(days=warmup * 2)).strftime("%Y-%m-%d")
        load_end = max(dates).strftime("%Y-%m-%d")
        with self._conn() as conn:
            df = pd.read_sql_query(
                "SELECT date, high, low, close FROM ohlcv "
                "WHERE ticker='069500' AND date BETWEEN ? AND ? ORDER BY date",
                conn, params=[load_start, load_end], parse_dates=["date"])
        if df.empty:
            return {d: None for d in dates}
        df = df.set_index("date").sort_index()
        high, low, close = df["high"], df["low"], df["close"]
        tenkan_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        kijun_line  = (high.rolling(kijun).max()  + low.rolling(kijun).min())  / 2
        span_a = ((tenkan_line + kijun_line) / 2).shift(kijun)
        span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
        # Wilder ADX/DMI
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm  = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        alpha = 1.0 / adx_period
        atr = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
        plus_di  = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        out: dict[pd.Timestamp, Optional[str]] = {}
        for d in dates:
            if d not in df.index:
                out[d] = None
                continue
            if pd.isna(cloud_top.loc[d]) or pd.isna(adx.loc[d]):
                out[d] = None
                continue
            c  = float(close.loc[d])
            ct = float(cloud_top.loc[d]); cb = float(cloud_bot.loc[d])
            a  = float(adx.loc[d])
            pdi = float(plus_di.loc[d]); mdi = float(minus_di.loc[d])
            if c > ct and a > adx_threshold and pdi > mdi:
                out[d] = "BULL"
            elif c < cb:
                out[d] = "BEAR_DEEP"
            elif c <= ct:
                out[d] = "CLOUD"
            else:           # close > cloud_top but ADX weak or DI 역전
                out[d] = "WEAK"
        return out

    # ----- 방어 자산 경로 (현금 대신 보유) -----
    def _defensive_path(
        self, ticker: str, entry_date: pd.Timestamp, exit_date: pd.Timestamp,
    ) -> tuple[float, Optional[pd.Series]]:
        """
        방어 자산의 (구간수익률, 일별 누적수익률 series).
        구간수익률 = 청산시가 / 진입시가 - 1.
        일별 누적수익률은 entry_open 대비 각 거래일의 (close 또는 마지막날 open) / entry_open - 1.
        데이터 없으면 (0.0, None).
        """
        ep = self._price_at(ticker, entry_date, "open")
        xp = self._price_at(ticker, exit_date, "open")
        if ep is None or xp is None or ep <= 0:
            return 0.0, None
        period_ret = xp / ep - 1.0
        days = self._trading_days_in_range(entry_date, exit_date)
        if len(days) < 2:
            return period_ret, None
        closes = self._load_daily_closes([ticker], entry_date, exit_date)
        if closes.empty or ticker not in closes.columns:
            return period_ret, None
        s = closes.reindex(index=days)[ticker].astype(float).copy()
        s.iloc[-1] = xp           # 청산일은 시가로 마무리
        if pd.isna(s.iloc[0]):
            s.iloc[0] = ep
        s = s.ffill().bfill()
        return period_ret, (s / ep - 1.0)

    def _benchmark_return(self, entry: pd.Timestamp, exit_: pd.Timestamp) -> float:
        bt_ticker = CFG.backtest.benchmark_ticker
        ep = self._price_at(bt_ticker, entry, "open")
        xp = self._price_at(bt_ticker, exit_, "open")
        if ep is None or xp is None:
            return 0.0
        return xp / ep - 1.0

    def _find_trend_break(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp,
    ) -> Optional[tuple[pd.Timestamp, float]]:
        """
        start ~ end 기간 중에 60일선을 깬 첫 시점을 찾음.
        반환: (날짜, 종가) 또는 None.

        60일선 = 최근 60거래일 평균. 종가가 60일선 아래로 내려간 첫 날.
        """
        # 60일선 계산을 위해 60일 전부터 데이터 필요
        from datetime import timedelta
        load_start = (start - timedelta(days=130)).strftime("%Y-%m-%d")
        load_end = end.strftime("%Y-%m-%d")
        with self._conn() as conn:
            df = pd.read_sql_query("""
                SELECT date, close FROM ohlcv
                WHERE ticker = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """, conn, params=[ticker, load_start, load_end], parse_dates=["date"])
        if len(df) < 60:
            return None
        df = df.sort_values("date").reset_index(drop=True)
        df["ma60"] = df["close"].rolling(60).mean()
        # start 이후의 행만 검사
        df = df[df["date"] >= start]
        if df.empty:
            return None
        broken = df[df["close"] < df["ma60"]]
        if broken.empty:
            return None
        first = broken.iloc[0]
        return (first["date"], float(first["close"]))

    # ----- 연도별 집계 -----
    @staticmethod
    def _aggregate_yearly(periods: pd.DataFrame) -> pd.DataFrame:
        if periods.empty:
            return pd.DataFrame()

        df = periods.copy()
        df["year"] = pd.to_datetime(df["entry_date"]).dt.year

        yearly = df.groupby("year").agg(
            n_periods=("period_return", "count"),
            portfolio_return=("period_return", lambda x: float(np.prod(1 + x) - 1)),
            benchmark_return=("benchmark_return", lambda x: float(np.prod(1 + x) - 1)),
        ).reset_index()
        yearly["alpha"] = yearly["portfolio_return"] - yearly["benchmark_return"]
        return yearly

    # ----- 성과 지표 -----
    @staticmethod
    def _compute_metrics(periods: pd.DataFrame, yearly: pd.DataFrame,
                         holdings: pd.DataFrame) -> dict:
        n_periods = len(periods)
        port = periods["period_return"].to_numpy()
        bench = periods["benchmark_return"].to_numpy()

        total_return = float(np.prod(1 + port) - 1)
        bench_total = float(np.prod(1 + bench) - 1)

        # 연도 수
        n_years = max(1, n_periods / 2)   # 6개월 구간이므로
        cagr = (1 + total_return) ** (1 / n_years) - 1
        bench_cagr = (1 + bench_total) ** (1 / n_years) - 1

        # 변동성: 6개월 수익률 → 연환산 (sqrt(2))
        vol_half = float(np.std(port, ddof=1)) if n_periods > 1 else 0.0
        vol_annual = vol_half * np.sqrt(2)
        sharpe = (cagr / vol_annual) if vol_annual > 0 else 0.0

        # MDD
        eq = periods["portfolio_value"].to_numpy()
        peaks = np.maximum.accumulate(eq)
        mdd = float((eq / peaks - 1).min()) if len(eq) else 0.0

        return {
            "start": periods["entry_date"].min().strftime("%Y-%m-%d"),
            "end": periods["exit_date"].max().strftime("%Y-%m-%d"),
            "n_periods": n_periods,
            "n_rebalances": n_periods - 1,    # 첫 진입은 리밸런싱 아님
            "total_return": total_return,
            "benchmark_total": bench_total,
            "cagr": cagr,
            "benchmark_cagr": bench_cagr,
            "volatility": vol_annual,
            "sharpe": float(sharpe),
            "mdd": mdd,
            "win_rate_periods": float((port > 0).mean()),
            "win_rate_holdings": float((holdings["return_pct"] > 0).mean()) if not holdings.empty else 0.0,
            "alpha_annualized": float(np.mean(yearly["alpha"])) if not yearly.empty else 0.0,
        }


# ============================================================
# CLI 실행 — 규칙 기반 백테스트
# ============================================================
def run_rule_based_backtest(
    start_year: int = 2015,
    end_year: int = 2024,
    top_n: int = 10,
    weight_scheme: str = "rank",
    replacement_rule: str = "always",
    score_diff_pct: float = 15.0,
    market_split: bool = False,
    trend_filter: bool = False,
    period_months: int = 6,
    rebalance_day: int = 1,
    trend_stop_loss: bool = False,
    trend_bonus: float = 0.0,
    market_cap_min: Optional[float] = None,
    market_cap_percentile: Optional[float] = None,
    market_ratio: Optional[str] = None,
    txn_cost: Optional[float] = None,
    market_regime_filter: bool = False,
    defensive_ticker: str = "132030",
    ichimoku_adx: bool = True,
    ia_tenkan: int = 9,
    ia_kijun: int = 26,
    ia_senkou_b: int = 52,
    ia_adx_period: int = 14,
    ia_adx_threshold: float = 25.0,
    ia_scaling: bool = True,
    end_date: Optional[str] = None,
    hysteresis_pct: float = 1.0,
    regime_cash_tiers: tuple = (
        (-2.0,          0.30),
        (-10.0,         0.80),
        (float("-inf"), 0.90),
    ),
    momentum_days: int = 0,
    momentum_threshold_pct: float = 20.0,
    ma_type: str = "ma200",
    bear_ma20_max_cash: float = 1.0,
    bear_ma20_days: int = 20,
    use_ttm_per: bool = True,        # 2026-05-16 채택: DART 자체 TTM PER (KRX 갱신지연 우회)
    use_ttm_fundamentals: bool = False,  # ROE/영업이익률/성장률도 자체 TTM 계산
    screener_extra: Optional[dict] = None,  # select_top_n 에 추가 전달 (실험용)
) -> RebalanceResult:
    """규칙 기반 백테스트 실행 헬퍼."""
    from src.screener.rule_based import RuleBasedScreener

    screener = RuleBasedScreener()
    extra = screener_extra or {}

    def picker(as_of: str, n: int) -> pd.DataFrame:
        return screener.select_top_n(
            as_of=as_of, top_n=n,
            market_split=market_split,
            trend_filter=trend_filter,
            trend_bonus=trend_bonus,
            market_cap_min=market_cap_min,
            market_cap_percentile=market_cap_percentile,
            market_ratio=market_ratio,
            use_ttm_per=use_ttm_per,
            use_ttm_fundamentals=use_ttm_fundamentals,
            **extra,
        )

    bt = RebalanceBacktest()
    return bt.run(
        picker=picker,
        start_year=start_year,
        end_year=end_year,
        top_n=top_n,
        weight_scheme=weight_scheme,
        replacement_rule=replacement_rule,
        score_diff_pct=score_diff_pct,
        period_months=period_months,
        rebalance_day=rebalance_day,
        trend_stop_loss=trend_stop_loss,
        txn_cost=txn_cost,
        market_regime_filter=market_regime_filter,
        defensive_ticker=defensive_ticker,
        ichimoku_adx=ichimoku_adx,
        ia_tenkan=ia_tenkan,
        ia_kijun=ia_kijun,
        ia_senkou_b=ia_senkou_b,
        ia_adx_period=ia_adx_period,
        ia_adx_threshold=ia_adx_threshold,
        ia_scaling=ia_scaling,
        end_date=end_date,
        hysteresis_pct=hysteresis_pct,
        regime_cash_tiers=regime_cash_tiers,
        momentum_days=momentum_days,
        momentum_threshold_pct=momentum_threshold_pct,
        ma_type=ma_type,
        bear_ma20_max_cash=bear_ma20_max_cash,
        bear_ma20_days=bear_ma20_days,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(">>> 규칙 기반 6개월 리밸런싱 백테스트 (2015~2024)")
    result = run_rule_based_backtest(start_year=2015, end_year=2024, top_n=10)
    print(result.summary())

    if not result.yearly.empty:
        print("\n[연도별]")
        yr = result.yearly.copy()
        for c in ("portfolio_return", "benchmark_return", "alpha"):
            yr[c] = (yr[c] * 100).round(2)
        print(yr.to_string(index=False))
