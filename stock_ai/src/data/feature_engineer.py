"""
피처 엔지니어링 — 3-인코더 분리 입력
=====================================
6:2:2 가중치를 위해 피처를 3그룹으로 분리:
  1. 펀더멘털 (4): ROE, 매출성장률 YoY, 순이익성장률 YoY, PER 매력도
  2. 차트 (6):    RSI, 볼린저폭, MA이격, 변동성, 거래량 z, MACD
  3. 시장 (3):    KOSPI 수익률, KOSPI 변동성, 금리 프록시
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from src.config import CFG, DB_PATH

logger = logging.getLogger(__name__)


class FeatureEngineer:
    FUNDAMENTAL_COLS = ["roe", "revenue_growth_yoy", "profit_growth_yoy", "per_score"]
    CHART_COLS = ["rsi_14", "bb_width", "ma_dev", "volatility_20", "volume_z", "macd_z",
                  "above_ma60", "ma60_slope"]
    MARKET_COLS = ["kospi_return", "kospi_vol", "rate_proxy"]

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self._kospi_cache: Optional[pd.DataFrame] = None

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield conn
        finally:
            conn.close()

    # 차트 피처
    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _macd_hist(close: pd.Series) -> pd.Series:
        fast = close.ewm(span=12, adjust=False).mean()
        slow = close.ewm(span=26, adjust=False).mean()
        macd = fast - slow
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal

    def _chart_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        out = pd.DataFrame(index=df.index)
        out["rsi_14"] = self._rsi(close, 14) / 100.0
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        out["bb_width"] = (4 * std20) / (ma20 + 1e-9)
        ma60 = close.rolling(60).mean()
        out["ma_dev"] = (close - ma60) / (ma60 + 1e-9)
        log_ret = np.log(close / close.shift(1))
        out["volatility_20"] = log_ret.rolling(20).std()
        log_vol = np.log(volume.replace(0, np.nan) + 1)
        out["volume_z"] = (log_vol - log_vol.rolling(60).mean()) / (log_vol.rolling(60).std() + 1e-9)
        macd_h = self._macd_hist(close)
        out["macd_z"] = (macd_h - macd_h.rolling(60).mean()) / (macd_h.rolling(60).std() + 1e-9)

        # ★ 추세 피처 (사용자 의도: 60일선 위에 있으면 좋다)
        # above_ma60: 0 또는 1 (정배열 여부)
        out["above_ma60"] = (close > ma60).astype(float)
        # ma60_slope: 60일선 자체가 오르고 있는지 (10일 전 대비 변화율)
        out["ma60_slope"] = (ma60 - ma60.shift(10)) / (ma60.shift(10) + 1e-9)

        return out[self.CHART_COLS]

    # 펀더멘털 피처
    def _load_fundamentals(self, ticker: str) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query("""
                SELECT period_end, roe, revenue_growth_yoy, profit_growth_yoy, debt_ratio
                FROM fundamentals WHERE ticker=?
                ORDER BY period_end
            """, conn, params=[ticker], parse_dates=["period_end"])

    def _load_per(self, ticker: str) -> pd.DataFrame:
        with self._conn() as conn:
            try:
                return pd.read_sql_query("""
                    SELECT date, per FROM per_history WHERE ticker=?
                    ORDER BY date
                """, conn, params=[ticker], parse_dates=["date"]).set_index("date")
            except Exception:
                return pd.DataFrame()

    def _fundamental_features(self, ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        out = pd.DataFrame(index=dates, columns=self.FUNDAMENTAL_COLS, dtype=float)

        funds = self._load_fundamentals(ticker)
        if not funds.empty:
            funds = funds.set_index("period_end").sort_index()
            for col in ("roe", "revenue_growth_yoy", "profit_growth_yoy"):
                if col in funds.columns:
                    out[col] = funds[col].reindex(dates, method="ffill").values

        per_df = self._load_per(ticker)
        if not per_df.empty:
            per_s = per_df["per"].reindex(dates, method="ffill")
            out["per_score"] = 1.0 / (1 + per_s.fillna(50) / 30.0)
        else:
            out["per_score"] = 0.0
        return out.fillna(0.0)

    # 시장 피처
    def _load_kospi(self) -> pd.DataFrame:
        if self._kospi_cache is not None:
            return self._kospi_cache
        with self._conn() as conn:
            df = pd.read_sql_query("""
                SELECT date, close FROM ohlcv WHERE ticker='069500'
                ORDER BY date
            """, conn, parse_dates=["date"])
        if df.empty:
            logger.warning("KOSPI 프록시(069500) 없음")
            return pd.DataFrame()
        self._kospi_cache = df.set_index("date")
        return self._kospi_cache

    def _market_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        out = pd.DataFrame(index=dates, columns=self.MARKET_COLS, dtype=float)
        kospi = self._load_kospi()
        if not kospi.empty:
            close = kospi["close"].reindex(dates, method="ffill")
            log_ret = np.log(close / close.shift(1))
            out["kospi_return"] = log_ret.values
            out["kospi_vol"] = log_ret.rolling(20).std().values
        else:
            out["kospi_return"] = 0.0
            out["kospi_vol"] = 0.0
        out["rate_proxy"] = 1.0 / (1 + out["kospi_vol"].fillna(0.02))
        return out.fillna(0.0)

    # 통합
    def build_for_ticker(self, ticker: str, start: str, end: str) -> Optional[dict]:
        with self._conn() as conn:
            ohlcv = pd.read_sql_query("""
                SELECT date, open, high, low, close, volume FROM ohlcv
                WHERE ticker=? AND date BETWEEN ? AND ?
                ORDER BY date
            """, conn, params=[ticker, start, end], parse_dates=["date"])

        if len(ohlcv) < CFG.model.seq_len + CFG.model.horizon + 30:
            return None

        ohlcv = ohlcv.set_index("date")
        chart = self._chart_features(ohlcv)
        funds = self._fundamental_features(ticker, ohlcv.index)
        market = self._market_features(ohlcv.index)

        valid = chart.dropna().index
        return {
            "fundamental": funds.loc[valid],
            "chart": chart.loc[valid],
            "market": market.loc[valid],
            "close": ohlcv.loc[valid, "close"],
        }

    def build_dataset(
        self,
        start: str = "2015-01-01",
        end: Optional[str] = None,
        market_cap_min: float = None,
    ) -> dict:
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        market_cap_min = market_cap_min or CFG.hard_filter.market_cap_min_krw

        with self._conn() as conn:
            tickers = [r[0] for r in conn.execute("""
                SELECT DISTINCT ticker FROM market_cap WHERE market_cap >= ?
            """, (market_cap_min,)).fetchall()]

        if not tickers:
            raise RuntimeError(
                f"시총 {market_cap_min/1e12:.1f}조+ 종목 없음. 데이터 수집 먼저 실행."
            )
        logger.info("학습 대상: %d개 종목", len(tickers))

        T = CFG.model.seq_len
        H = CFG.model.horizon
        stride = CFG.train.sample_stride

        funds, charts, markets, ys, metas = [], [], [], [], []

        for tk in tickers:
            data = self.build_for_ticker(tk, start, end)
            if data is None:
                continue
            f = data["fundamental"].to_numpy(dtype=np.float32)
            c = data["chart"].to_numpy(dtype=np.float32)
            m = data["market"].to_numpy(dtype=np.float32)
            close = data["close"].to_numpy(dtype=np.float32)
            dates = data["chart"].index

            for i in range(T, len(close) - H, stride):
                p_now = close[i - 1]
                p_future = close[i - 1 + H]
                if p_now <= 0 or p_future <= 0:
                    continue
                target = float(np.clip(np.log(p_future / p_now), -1.1, 1.1))

                funds.append(f[i - T:i])
                charts.append(c[i - T:i])
                markets.append(m[i - T:i])
                ys.append(target)
                metas.append({
                    "ticker": tk,
                    "decision_date": dates[i - 1].strftime("%Y-%m-%d"),
                    "future_date": dates[i - 1 + H].strftime("%Y-%m-%d"),
                })

        if not funds:
            raise RuntimeError("샘플 생성 실패")

        result = {
            "fund": np.stack(funds),
            "chart": np.stack(charts),
            "market": np.stack(markets),
            "y": np.array(ys, dtype=np.float32),
            "meta": pd.DataFrame(metas),
        }
        logger.info("데이터셋: fund=%s chart=%s market=%s y=%s",
                    result["fund"].shape, result["chart"].shape,
                    result["market"].shape, result["y"].shape)
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    fe = FeatureEngineer()
    ds = fe.build_dataset()
    print(f"샘플: {len(ds['y']):,}")
    print(f"종목: {ds['meta']['ticker'].nunique()}")
