"""Production validation: 블랙스완 스트레스 테스트 + 몬테카를로 부트스트랩.

Inputs:
  - RebalanceResult (이미 risk_overlay 적용된 daily PV 도 받음)

Outputs:
  - blackswan_table: pd.DataFrame (이벤트별 시스템/벤치 MDD + 알파)
  - mc_results: dict (표준 / 1.5x / 2.0x 시나리오)
  - verdict: dict (합격 체크리스트 + 최종 판정)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DB_PATH


# 역사적 블랙스완 이벤트 정의
BLACKSWAN_EVENTS = [
    ("2018 미중 무역전쟁/금리쇼크", "2018-10-01", "2018-12-31"),
    ("2020 COVID-19 팬데믹",        "2020-02-15", "2020-04-15"),
    ("2022 인플레이션 대폭락",      "2022-01-01", "2022-10-31"),
]


def _load_benchmark_returns(start: str, end: str, ticker: str = "069500") -> pd.Series:
    """KODEX200(069500) 일별 수익률. 못 찾으면 빈 Series."""
    with sqlite3.connect(DB_PATH) as c:
        df = pd.read_sql_query(
            "SELECT date, close FROM ohlcv WHERE ticker=? AND date BETWEEN ? AND ? ORDER BY date",
            c, params=[ticker, start, end], parse_dates=["date"],
        )
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("date")["close"].pct_change().fillna(0)


def blackswan_analysis(daily_pv: pd.DataFrame) -> pd.DataFrame:
    """daily_pv: columns=[date, pv]. 각 이벤트별 시스템/벤치 MDD + 수익률 비교."""
    df = daily_pv.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    bench_ret = _load_benchmark_returns(
        df["date"].min().strftime("%Y-%m-%d"),
        df["date"].max().strftime("%Y-%m-%d"),
    )

    rows = []
    for name, start, end in BLACKSWAN_EVENTS:
        s_ts, e_ts = pd.Timestamp(start), pd.Timestamp(end)
        sub = df[(df["date"] >= s_ts) & (df["date"] <= e_ts)]
        if sub.empty:
            rows.append(dict(event=name, period=f"{start} ~ {end}",
                              sys_mdd=None, bench_mdd=None, defense=None,
                              sys_ret=None, bench_ret=None, alpha=None))
            continue
        eq = sub["pv"].to_numpy()
        pk = np.maximum.accumulate(eq)
        sys_mdd = float((eq / pk - 1).min())
        sys_ret = float(eq[-1] / eq[0] - 1)
        bs = bench_ret.reindex(sub["date"]).fillna(0)
        if bs.empty or bs.sum() == 0:
            bench_pv_arr = np.ones(len(sub))
        else:
            bench_pv_arr = (1 + bs).cumprod().to_numpy()
        bpk = np.maximum.accumulate(bench_pv_arr)
        bench_mdd = float((bench_pv_arr / bpk - 1).min())
        bench_total = float(bench_pv_arr[-1] - 1)
        defense = (1 - sys_mdd / bench_mdd) * 100 if bench_mdd < 0 else 0
        rows.append(dict(
            event=name, period=f"{start} ~ {end}",
            sys_mdd=sys_mdd, bench_mdd=bench_mdd, defense=defense,
            sys_ret=sys_ret, bench_ret=bench_total,
            alpha=sys_ret - bench_total,
        ))
    return pd.DataFrame(rows)


def monte_carlo(daily_pv: pd.DataFrame, n_sim: int = 10000,
                block: int = 5, seed: int = 42) -> dict:
    """일별 수익률 block bootstrap. 표준/1.5x/2.0x 스트레스 시나리오 모두 산출.
    반환: {scenario_label: dict(ruin_1, ruin_05, cagr_median, cagr_p1, mdd_p1)}
    """
    df = daily_pv.copy().sort_values("date").reset_index(drop=True)
    ret = df["pv"].astype(float).pct_change().fillna(0).iloc[1:].to_numpy()
    n_days = len(ret)
    yrs = n_days / 252
    rng = np.random.default_rng(seed)

    def _run(stress_mult: float) -> dict:
        n_blocks = n_days // block + 1
        final_pv = np.empty(n_sim)
        max_dd = np.empty(n_sim)
        cagr_arr = np.empty(n_sim)
        for i in range(n_sim):
            starts = rng.integers(0, len(ret) - block, n_blocks)
            sampled = np.concatenate([ret[s:s + block] for s in starts])[:n_days]
            if stress_mult != 1.0:
                sampled = np.where(sampled < 0, sampled * stress_mult, sampled)
            pv = np.cumprod(1 + sampled)
            final_pv[i] = pv[-1]
            peaks = np.maximum.accumulate(pv)
            max_dd[i] = (pv / peaks - 1).min()
            cagr_arr[i] = pv[-1] ** (1 / yrs) - 1
        return dict(
            ruin_1=float((final_pv < 1.0).mean()),
            ruin_05=float((final_pv < 0.5).mean()),
            cagr_median=float(np.median(cagr_arr)),
            cagr_p5=float(np.percentile(cagr_arr, 5)),
            cagr_p1=float(np.percentile(cagr_arr, 1)),
            mdd_median=float(np.median(max_dd)),
            mdd_p5=float(np.percentile(max_dd, 5)),
            mdd_p1=float(np.percentile(max_dd, 1)),
        )

    return {
        "standard": _run(1.0),
        "stress_1_5x": _run(1.5),
        "extreme_2_0x": _run(2.0),
    }


def verdict(realized: dict, mc: dict) -> dict:
    """realized = result.metrics. mc = monte_carlo 결과."""
    checks = [
        ("실측 Sharpe > 1.0", realized.get("sharpe", 0) > 1.0),
        ("실측 daily MDD > -30%", realized.get("mdd_daily", -1) > -0.30),
        ("MC 표준 파산확률 < 5%", mc["standard"]["ruin_1"] * 100 < 5),
        ("MC 표준 1% 최악 CAGR > 0%", mc["standard"]["cagr_p1"] > 0),
        ("MC 표준 1% 최악 MDD > -50%", mc["standard"]["mdd_p1"] > -0.50),
        ("MC 1.5x 파산확률 < 50% (참고)", mc["stress_1_5x"]["ruin_1"] * 100 < 50),
    ]
    passed = sum(1 for _, ok in checks if ok)
    label = "✅ 합격" if passed >= 5 else ("⚠️ 조건부 합격" if passed >= 4 else "❌ 불합격")
    return dict(checks=checks, passed=passed, total=len(checks), label=label)
