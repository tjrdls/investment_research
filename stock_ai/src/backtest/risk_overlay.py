"""Risk overlay post-processing for RebalanceResult.

Applies two policies to a RebalanceResult that the engine itself does not yet
handle natively:
  1. Single-position weight cap: each holding clipped to `weight_cap`; excess
     allocated to a bond yielding `bond_yield` annualised (daily compounded).
  2. Bond fallback for n_picks==0 quarters: instead of flat cash, accrue the
     same daily bond yield.

Stock-holding quarters still respect the engine's `cash_ratio` (defensive
ticker share), which is multiplied with the gold ETF price path.

The function returns a metrics dict matching `RebalanceResult.metrics` keys
(total_return, cagr, sharpe, mdd_daily, mdd, win_rate_periods,
alpha_annualized) so callers can swap it into existing UI/reporting code.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DB_PATH


def _load_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    ph = ",".join("?" * len(tickers))
    with sqlite3.connect(DB_PATH) as c:
        df = pd.read_sql_query(
            f"SELECT ticker, date, close FROM ohlcv "
            f"WHERE ticker IN ({ph}) AND date BETWEEN ? AND ? ORDER BY date",
            c, params=[*tickers, start, end], parse_dates=["date"],
        )
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="date", columns="ticker", values="close")


def apply_risk_overlay(
    result,
    weight_cap: Optional[float] = 0.25,
    bond_yield: float = 0.03,
    defensive_ticker: str = "132030",
) -> dict:
    """Re-simulate daily PV with weight cap + bond fallback. Returns metrics dict."""
    bond_d = (1 + bond_yield) ** (1 / 365) - 1
    daily_pv: list[tuple] = []
    pv = 1.0
    de = result.daily_equity.copy()
    de["date"] = pd.to_datetime(de["date"])

    for _, row in result.periods.iterrows():
        entry = pd.Timestamp(row["entry_date"])
        exit_ = pd.Timestamp(row["exit_date"])
        n_picks = int(row["n_picks"])
        cash_r = row.get("cash_ratio", 0) or 0
        if pd.isna(cash_r):
            cash_r = 0
        stock_r = 1.0 - cash_r
        es = entry.strftime("%Y-%m-%d")
        xs = exit_.strftime("%Y-%m-%d")
        period_dates = pd.to_datetime(
            de[(de["date"] >= entry) & (de["date"] < exit_)]["date"].unique()
        )
        if len(period_dates) == 0:
            continue

        gold = _load_prices([defensive_ticker], es, xs)
        if gold.empty:
            gold_norm = pd.Series(1.0, index=period_dates)
        else:
            gold_norm = gold.iloc[:, 0] / gold.iloc[0, 0]
        gold_norm = gold_norm.reindex(period_dates).ffill().fillna(1.0)

        if n_picks > 0:
            hq = result.holdings[
                (result.holdings["entry_date"] == row["entry_date"])
                & (result.holdings["exit_date"] == row["exit_date"])
            ].copy()
            if hq.empty:
                days_arr = np.array([(d - period_dates[0]).days for d in period_dates])
                mix = pd.Series((1 + bond_d) ** days_arr, index=period_dates)
            else:
                hq["w_new"] = (
                    hq["weight"].clip(upper=weight_cap)
                    if weight_cap is not None
                    else hq["weight"]
                )
                bond_w = max(0.0, 1.0 - hq["w_new"].sum())
                prices = (
                    _load_prices(hq["ticker"].tolist(), es, xs)
                    .reindex(period_dates)
                    .ffill()
                    .bfill()
                )
                if prices.empty:
                    days_arr = np.array([(d - period_dates[0]).days for d in period_dates])
                    stock_part = pd.Series((1 + bond_d) ** days_arr, index=period_dates)
                else:
                    norm = prices / prices.iloc[0]
                    ws = hq.set_index("ticker")["w_new"]
                    stock_norm = (norm * ws).sum(axis=1)
                    days_arr = np.array([(d - period_dates[0]).days for d in period_dates])
                    bond_norm = pd.Series((1 + bond_d) ** days_arr, index=period_dates)
                    stock_part = stock_norm + bond_w * bond_norm
                mix = stock_r * stock_part + cash_r * gold_norm
        else:
            days_arr = np.array([(d - period_dates[0]).days for d in period_dates])
            mix = pd.Series((1 + bond_d) ** days_arr, index=period_dates)

        for d, val in mix.items():
            daily_pv.append((d, pv * float(val)))
        pv = pv * float(mix.iloc[-1])

    df = (
        pd.DataFrame(daily_pv, columns=["date", "pv"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if df.empty:
        return {}

    eq = df["pv"].to_numpy()
    ret = df["pv"].pct_change().fillna(0).iloc[1:].to_numpy()
    yrs = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    tr = float(eq[-1] - 1)
    cagr = (1 + tr) ** (1 / yrs) - 1 if yrs > 0 else 0
    sh = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    pk = np.maximum.accumulate(eq)
    mdd_d = float((eq / pk - 1).min())
    win = total = 0
    for _, p in result.periods.iterrows():
        e = pd.Timestamp(p["entry_date"])
        x = pd.Timestamp(p["exit_date"])
        sub = df[(df["date"] >= e) & (df["date"] < x)]
        if len(sub) >= 2:
            qr = sub["pv"].iloc[-1] / sub["pv"].iloc[0] - 1
            if qr > 0:
                win += 1
            total += 1
    wr = win / total if total > 0 else 0
    bt = 1.0
    for br in result.periods["benchmark_return"]:
        if pd.notna(br):
            bt *= 1 + br
    bcg = bt ** (1 / yrs) - 1 if yrs > 0 else 0
    return dict(
        total_return=tr,
        cagr=cagr,
        sharpe=sh,
        mdd_daily=mdd_d,
        mdd=mdd_d,
        win_rate_periods=wr,
        alpha_annualized=cagr - bcg,
        benchmark_cagr=bcg,
        daily=df,
    )
