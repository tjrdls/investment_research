"""
Walk-Forward 백테스트
=======================
매년 1월 첫 거래일에 AI가 종목 선정 → 12월 마지막 거래일에 평가.
2010~2024년 반복해 모델 성능 검증.

룩어헤드 방지: 의사결정 시점은 entry_date - 1d. 이후 데이터는 보지 않음.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from src.config import CFG, DB_PATH
from src.recommend.recommender import Recommender

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    config: dict
    yearly: pd.DataFrame
    holdings: pd.DataFrame
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "=" * 65,
            "백테스트 결과",
            "=" * 65,
            f"기간:         {m.get('start_year')} ~ {m.get('end_year')} ({m.get('n_years')}년)",
            f"총 수익률:    {m.get('total_return', 0)*100:>8.2f}%   (KOSPI200 {m.get('benchmark_total', 0)*100:>6.2f}%)",
            f"CAGR:         {m.get('cagr', 0)*100:>8.2f}%   (KOSPI200 {m.get('benchmark_cagr', 0)*100:>6.2f}%)",
            f"변동성:       {m.get('volatility', 0)*100:>8.2f}%",
            f"샤프:         {m.get('sharpe', 0):>8.2f}",
            f"최대 낙폭:    {m.get('mdd', 0)*100:>8.2f}%",
            f"승률(연):     {m.get('win_rate_yearly', 0)*100:>8.2f}%",
            f"승률(종목):   {m.get('win_rate_holdings', 0)*100:>8.2f}%",
            f"연평균 알파: {m.get('alpha_annualized', 0)*100:>8.2f}%",
            "=" * 65,
        ]
        return "\n".join(lines)


class WalkForwardBacktest:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.recommender = Recommender(db_path=db_path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield conn
        finally:
            conn.close()

    def _trading_days(self, year: int) -> list[pd.Timestamp]:
        with self._conn() as conn:
            return [pd.Timestamp(r[0]) for r in conn.execute("""
                SELECT DISTINCT date FROM ohlcv
                WHERE date BETWEEN ? AND ?
                ORDER BY date
            """, (f"{year}-01-01", f"{year}-12-31")).fetchall()]

    def _price(self, ticker: str, target: pd.Timestamp,
               op: str = ">=", field: str = "open") -> Optional[tuple[pd.Timestamp, float]]:
        cmp = "AND date >= ? ORDER BY date" if op == ">=" else "AND date <= ? ORDER BY date DESC"
        sql = f"SELECT date, {field} FROM ohlcv WHERE ticker=? {cmp} LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(sql, (ticker, target.strftime("%Y-%m-%d"))).fetchone()
        return (pd.Timestamp(row[0]), float(row[1])) if row and row[1] else None

    def run(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        top_n: Optional[int] = None,
    ) -> BacktestResult:
        bt = CFG.backtest
        start_year = start_year or bt.start_year
        end_year = end_year or bt.end_year
        top_n = top_n or CFG.recommend.final_top_n

        config = dict(start_year=start_year, end_year=end_year, top_n=top_n,
                       txn_cost=bt.txn_cost, benchmark=bt.benchmark_ticker)

        yearly_rows, holdings_rows = [], []
        port_value = bench_value = 1.0

        for year in range(start_year, end_year + 1):
            tdays = self._trading_days(year)
            if len(tdays) < 100:
                logger.warning("[%d] 거래일 부족 — 건너뜀", year)
                continue

            entry_date = tdays[0]
            exit_date = tdays[-1]
            decision_cutoff = entry_date - pd.Timedelta(days=1)

            try:
                picks = self.recommender.recommend(as_of=decision_cutoff.strftime("%Y-%m-%d"))
            except Exception as e:
                logger.error("[%d] 추천 실패: %s", year, e)
                continue

            if picks.empty:
                logger.warning("[%d] 추천 종목 없음", year)
                continue

            picks = picks.head(top_n)

            holdings_returns = []
            for _, row in picks.iterrows():
                tk = row["ticker"]
                entry = self._price(tk, entry_date, ">=", "open")
                exit_ = self._price(tk, exit_date, "<=", "close")
                if not entry or not exit_:
                    continue
                gross = exit_[1] / entry[1] - 1.0
                net = (1 + gross) * (1 - bt.txn_cost) ** 2 - 1
                holdings_returns.append(net)
                holdings_rows.append({
                    "year": year, "ticker": tk, "name": row.get("name"),
                    "entry_price": entry[1], "exit_price": exit_[1],
                    "return_pct": net, "ai_score": row.get("final_score"),
                })

            if not holdings_returns:
                continue

            port_ret = float(np.mean(holdings_returns))

            bench_ret = 0.0
            b_e = self._price(bt.benchmark_ticker, entry_date, ">=", "open")
            b_x = self._price(bt.benchmark_ticker, exit_date, "<=", "close")
            if b_e and b_x:
                bench_ret = b_x[1] / b_e[1] - 1.0

            port_value *= (1 + port_ret)
            bench_value *= (1 + bench_ret)

            yearly_rows.append({
                "year": year,
                "portfolio_return": port_ret,
                "benchmark_return": bench_ret,
                "alpha": port_ret - bench_ret,
                "n_picks": len(holdings_returns),
                "portfolio_value": port_value,
                "benchmark_value": bench_value,
            })

            logger.info(
                "[%d] AI %+.2f%%  KOSPI200 %+.2f%%  α %+.2f%%  (%d종목)",
                year, port_ret * 100, bench_ret * 100,
                (port_ret - bench_ret) * 100, len(holdings_returns),
            )

        yearly_df = pd.DataFrame(yearly_rows)
        holdings_df = pd.DataFrame(holdings_rows)
        metrics = self._compute_metrics(yearly_df, holdings_df) if not yearly_df.empty else {}

        return BacktestResult(config=config, yearly=yearly_df,
                              holdings=holdings_df, metrics=metrics)

    @staticmethod
    def _compute_metrics(yearly: pd.DataFrame, holdings: pd.DataFrame) -> dict:
        n = len(yearly)
        port = yearly["portfolio_return"].to_numpy()
        bench = yearly["benchmark_return"].to_numpy()

        total = float(np.prod(1 + port) - 1)
        bench_total = float(np.prod(1 + bench) - 1)
        cagr = (1 + total) ** (1 / n) - 1 if n else 0.0
        bench_cagr = (1 + bench_total) ** (1 / n) - 1 if n else 0.0

        vol = float(np.std(port, ddof=1)) if n > 1 else 0.0
        sharpe = float(np.mean(port) / vol) if vol > 0 else 0.0

        eq = yearly["portfolio_value"].to_numpy()
        peaks = np.maximum.accumulate(eq)
        mdd = float((eq / peaks - 1).min()) if len(eq) else 0.0

        return {
            "start_year": int(yearly["year"].min()),
            "end_year": int(yearly["year"].max()),
            "n_years": n,
            "total_return": total,
            "benchmark_total": bench_total,
            "cagr": cagr,
            "benchmark_cagr": bench_cagr,
            "volatility": vol,
            "sharpe": sharpe,
            "mdd": mdd,
            "win_rate_yearly": float((port > 0).mean()),
            "win_rate_holdings": float((holdings["return_pct"] > 0).mean()) if not holdings.empty else 0.0,
            "alpha_annualized": float(np.mean(port - bench)),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    bt = WalkForwardBacktest()
    result = bt.run()
    print(result.summary())
    print("\n[연도별]")
    print(result.yearly.to_string(index=False))
