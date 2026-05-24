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
    @staticmethod
    def _rebalance_dates(start_year: int, end_year: int,
                         period_months: int = 6) -> list[pd.Timestamp]:
        """
        리밸런싱 시점 생성.
        period_months=6: 1월 + 7월 (반기)
        period_months=4: 1월 + 5월 + 9월 (4개월)
        period_months=3: 1월 + 4월 + 7월 + 10월 (분기)
        """
        dates = []
        for y in range(start_year, end_year + 1):
            for m in range(1, 13, period_months):
                dates.append(pd.Timestamp(f"{y}-{m:02d}-01"))
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
        trend_stop_loss: bool = False,
    ) -> RebalanceResult:
        """
        Parameters
        ----------
        replacement_rule : 종목 교체 규칙
            'always'      매번 처음부터 상위 N개 (기존 방식)
            'keep_simple' 기존 보유 통과하면 무조건 유지
            'score_diff'  새 후보 점수가 기존 최저 종목보다 score_diff_pct% 이상 높을 때만 교체 (A안)
            'three_cond'  새 후보가 시총 ↑ AND PER ↓ AND ROE ↑ 모두 만족할 때만 교체 (B안)
        score_diff_pct : score_diff 모드에서 교체 임계값 (기본 15%)
        period_months : 리밸런싱 주기 (6=반기, 4=4개월, 3=분기)
        trend_stop_loss : True면 보유 기간 중에도 매월 1일 추세 점검 → 60일선 깬 종목 즉시 매도
                          (사용자 의도: 추세 꺾이면 즉시 손절)
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
        rb_dates = self._rebalance_dates(start_year, end_year, period_months)
        # 마지막 시점은 청산일이 없으니 제외 (종료일까지 보유로 계산)
        end_of_data = pd.Timestamp(f"{end_year}-12-31")
        rb_actual: list[pd.Timestamp] = []
        for d in rb_dates:
            t = self._first_trading_day(d)
            if t is None or t > end_of_data:
                continue
            rb_actual.append(t)

        if len(rb_actual) < 2:
            raise RuntimeError("리밸런싱 시점 부족 — 데이터 수집 확인 필요")

        period_rows: list[dict] = []
        holdings_rows: list[dict] = []

        port_value = 1.0
        bench_value = 1.0
        prev_holdings: dict[str, float] = {}    # {ticker: weight}

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
            net_period_ret = (1 + gross_period_ret) * (1 - cost) - 1

            # 벤치마크
            bench_ret = self._benchmark_return(entry_date, exit_date)

            port_value *= (1 + net_period_ret)
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

        return RebalanceResult(
            config=config,
            periods=periods_df,
            yearly=yearly_df,
            holdings=holdings_df,
            metrics=metrics,
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


    # ----- 벤치마크 수익률 -----
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
    trend_stop_loss: bool = False,
    trend_bonus: float = 0.0,
    market_cap_min: Optional[float] = None,
    market_cap_percentile: Optional[float] = None,
) -> RebalanceResult:
    """규칙 기반 백테스트 실행 헬퍼."""
    from src.screener.rule_based import RuleBasedScreener

    screener = RuleBasedScreener()

    def picker(as_of: str, n: int) -> pd.DataFrame:
        return screener.select_top_n(
            as_of=as_of, top_n=n,
            market_split=market_split,
            trend_filter=trend_filter,
            trend_bonus=trend_bonus,
            market_cap_min=market_cap_min,
            market_cap_percentile=market_cap_percentile,
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
        trend_stop_loss=trend_stop_loss,
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
