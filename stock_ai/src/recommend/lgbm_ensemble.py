"""LightGBM Hybrid Ensemble Screener.

v3 (차트 10 피처) + v4 (펀더멘털 13 피처) 동시 로드.
  - KR 종목: final_score = 0.6 × v3 + 0.4 × v4 (v3 의 IC + v4 의 Spread 시너지)
  - US 종목: v3 단독 (펀더멘털 데이터 없음)

최종 ensemble_score = ai_weight × hybrid_ai_score + (1 - ai_weight) × rule_score
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DB_PATH, MODEL_DIR
from src.screener.rule_based import RuleBasedScreener

logger = logging.getLogger(__name__)

# 인덱스 매핑 (KR=KOSPI 069500, US=NASDAQ QQQ)
INDEX_TICKER = {"KR": "069500", "US": "QQQ"}

# v3 피처 순서 (10개) — trend_lgbm_v2.FEATURE_COLS 에서 rsi_signal_ratio 제거
V3_FEATURES = [
    "price_to_ma200", "ma60_slope", "ma20_to_ma60", "bb_position",
    "volume_ratio", "macd_hist_ratio",
    "stoch_60_pos", "obv_slope_20",
    "index_ma60_slope", "index_mdd_20",
]

# v4 피처 순서 (13개) — v3 + 펀더멘털 3
V4_FEATURES = V3_FEATURES + ["roe_latest", "op_margin_growth", "per_inverse"]

# 하이브리드 가중치 (KR 시장)
HYBRID_W_V3 = 0.6
HYBRID_W_V4 = 0.4


class LGBMEnsembleScreener:
    """Hybrid v3 + v4 LGBM 앙상블 (KR) + v3 단독 (US).

    Parameters
    ----------
    ai_weight : float
        Rule 대비 AI 비중. ensemble_score = ai_weight × hybrid_ai + (1-ai_weight) × rule
    model_v3_path : Path
        v3 모델 파일 경로 (10 피처)
    model_v4_path : Path
        v4 모델 파일 경로 (13 피처)
    """

    def __init__(
        self,
        ai_weight: float = 0.5,
        model_v3_path: Path = MODEL_DIR / "trend_lgbm_v3.txt",
        model_v4_path: Path = MODEL_DIR / "trend_lgbm_v4.txt",
    ):
        import lightgbm as lgb
        assert 0.0 <= ai_weight <= 1.0
        self.ai_weight = ai_weight
        self.model_v3 = lgb.Booster(model_file=str(model_v3_path)) if model_v3_path.exists() else None
        self.model_v4 = lgb.Booster(model_file=str(model_v4_path)) if model_v4_path.exists() else None
        if self.model_v3 is None and self.model_v4 is None:
            raise FileNotFoundError(f"v3 ({model_v3_path}) / v4 ({model_v4_path}) 둘 다 없음")
        self.rule = RuleBasedScreener()
        # 캐시
        self._index_cache: dict[str, pd.DataFrame] = {}
        self._fund_cache: Optional[pd.DataFrame] = None
        self._per_cache: Optional[pd.DataFrame] = None
        logger.info("LGBM Hybrid Ensemble: ai_weight=%.2f, v3=%s, v4=%s, KR weights=(v3 %.0f%%, v4 %.0f%%)",
                    ai_weight,
                    "OK" if self.model_v3 else "X",
                    "OK" if self.model_v4 else "X",
                    HYBRID_W_V3 * 100, HYBRID_W_V4 * 100)

    # ─── 인덱스 매크로 캐시 ─────────────────────────────────
    def _load_index(self, idx_ticker: str, as_of: str) -> pd.DataFrame:
        key = f"{idx_ticker}::{as_of}"
        if key in self._index_cache:
            return self._index_cache[key]
        as_of_dt = pd.Timestamp(as_of)
        start = (as_of_dt - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        with sqlite3.connect(DB_PATH) as c:
            idx = pd.read_sql_query(
                "SELECT date, close FROM ohlcv WHERE ticker=? AND date BETWEEN ? AND ? ORDER BY date",
                c, params=[idx_ticker, start, as_of], parse_dates=["date"])
        if idx.empty:
            return idx
        ma60 = idx["close"].rolling(60).mean()
        idx["index_ma60_slope"] = (ma60 - ma60.shift(5)) / ma60.shift(5)
        roll_max = idx["close"].rolling(20).max()
        idx["index_mdd_20"] = (idx["close"] - roll_max) / roll_max
        self._index_cache[key] = idx
        return idx

    # ─── 펀더멘털 + PER 캐시 (v4 inference 용) ─────────────
    def _load_fundamentals(self) -> dict:
        """ticker → (publish_dates np.array, roe arr, op_g arr) 로 미리 변환 (lookup O(log n))."""
        if self._fund_cache is not None:
            return self._fund_cache
        with sqlite3.connect(DB_PATH) as c:
            f = pd.read_sql_query("""
                SELECT ticker AS code, period_end, roe AS roe_latest, operating_margin
                FROM fundamentals ORDER BY ticker, period_end
            """, c, parse_dates=["period_end"])
        if f.empty:
            self._fund_cache = {}
            return self._fund_cache
        m = f["period_end"].dt.month
        f["publish_date"] = f["period_end"] + pd.to_timedelta(np.where(m == 12, 90, 45), unit="D")
        f = f.sort_values(["code", "period_end"]).reset_index(drop=True)
        f["op_margin_growth"] = f.groupby("code")["operating_margin"].diff()
        f["roe_latest"] = f["roe_latest"].clip(-200, 200)
        f["op_margin_growth"] = f["op_margin_growth"].clip(-50, 50)
        # ticker 별 dict 변환 (numpy 배열 — searchsorted 가능)
        cache = {}
        for code, g in f.groupby("code"):
            pub = g["publish_date"].to_numpy()
            cache[code] = (
                pub,
                g["roe_latest"].to_numpy(),
                g["op_margin_growth"].to_numpy(),
            )
        self._fund_cache = cache
        return cache

    def _load_per(self) -> dict:
        """ticker → (dates np.array, per_inv arr)."""
        if self._per_cache is not None:
            return self._per_cache
        with sqlite3.connect(DB_PATH) as c:
            p = pd.read_sql_query(
                "SELECT ticker AS code, date, per FROM per_history "
                "WHERE per IS NOT NULL ORDER BY ticker, date",
                c, parse_dates=["date"])
        if p.empty:
            self._per_cache = {}
            return self._per_cache
        inv = np.where(p["per"] > 0, 1.0 / p["per"].clip(lower=2.0), 0.0)
        p["per_inverse"] = np.clip(inv, -0.5, 0.5)
        cache = {}
        for code, g in p.groupby("code"):
            cache[code] = (g["date"].to_numpy(), g["per_inverse"].to_numpy())
        self._per_cache = cache
        return cache

    def _latest_fund_per(self, ticker: str, as_of: str) -> tuple:
        """O(log n) binary search 로 as_of 이전 가장 최근 값 반환."""
        fund_dict = self._load_fundamentals()
        per_dict = self._load_per()
        as_of_np = np.datetime64(pd.Timestamp(as_of))
        roe = op_g = per_inv = np.nan
        if ticker in fund_dict:
            pub, roe_arr, op_arr = fund_dict[ticker]
            i = np.searchsorted(pub, as_of_np, side="right") - 1
            if i >= 0:
                roe = float(roe_arr[i]) if not np.isnan(roe_arr[i]) else np.nan
                op_g = float(op_arr[i]) if not np.isnan(op_arr[i]) else np.nan
        if ticker in per_dict:
            dts, per_arr = per_dict[ticker]
            i = np.searchsorted(dts, as_of_np, side="right") - 1
            if i >= 0:
                per_inv = float(per_arr[i])
        return roe, op_g, per_inv

    # ─── 피처 계산 (13 피처, 모두 산출 후 v3/v4 분리 사용) ──
    def _compute_features(self, tickers: list, markets: dict, as_of: str) -> dict:
        """각 ticker 의 13 피처 (v4) 계산. v3 inference 시 처음 10개만 사용.
        펀더멘털은 KR 종목만 채워지고 US 는 NaN.
        """
        if not tickers:
            return {}
        as_of_dt = pd.Timestamp(as_of)
        start_load = (as_of_dt - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        ph = ",".join("?" * len(tickers))
        with sqlite3.connect(DB_PATH) as c:
            df = pd.read_sql_query(
                f"SELECT ticker, date, close, high, low, volume FROM ohlcv "
                f"WHERE ticker IN ({ph}) AND date BETWEEN ? AND ? ORDER BY ticker, date",
                c, params=[*tickers, start_load, as_of], parse_dates=["date"])
        if df.empty:
            return {}

        idx_kr = self._load_index("069500", as_of)
        idx_us = self._load_index("QQQ", as_of)
        idx_kr_last = idx_kr.iloc[-1] if not idx_kr.empty else None
        idx_us_last = idx_us.iloc[-1] if not idx_us.empty else None

        out = {}
        for tk, g in df.groupby("ticker"):
            g = g.sort_values("date").reset_index(drop=True)
            if len(g) < 200:
                continue
            close = g["close"]; high = g["high"]; low = g["low"]; vol = g["volume"]
            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()
            ma200 = close.rolling(200).mean()

            price_to_ma200 = (close / ma200).iloc[-1]
            ma60_slope = ((ma60 - ma60.shift(5)) / ma60.shift(5)).iloc[-1]
            ma20_to_ma60 = (ma20 / ma60).iloc[-1]
            std20 = close.rolling(20).std()
            upper = ma20 + 2 * std20; lower = ma20 - 2 * std20
            bb_position = ((close - lower) / (upper - lower + 1e-9)).iloc[-1]
            vol_ma20 = vol.rolling(20).mean()
            volume_ratio = (vol / (vol_ma20 + 1e-9)).iloc[-1]
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            sig = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist_ratio = ((macd_line - sig) / close).iloc[-1]
            low60 = low.rolling(60).min()
            high60 = high.rolling(60).max()
            stoch_60_pos = ((close - low60) / (high60 - low60 + 1e-9)).iloc[-1]
            obv_diff = np.sign(close.diff().fillna(0)) * vol
            obv = obv_diff.cumsum()
            obv_prev = obv.shift(20)
            obv_slope_20 = ((obv - obv_prev) / (obv_prev.abs() + 1e-9)).iloc[-1]

            mk = markets.get(tk, "KOSPI")
            idx_row = idx_kr_last if mk in ("KOSPI", "KOSDAQ") else idx_us_last
            if idx_row is None: continue
            idx_slope = idx_row["index_ma60_slope"]
            idx_mdd = idx_row["index_mdd_20"]

            # 펀더멘털 (KR 만 시도)
            if mk in ("KOSPI", "KOSDAQ"):
                roe, op_g, per_inv = self._latest_fund_per(tk, as_of)
            else:
                roe = op_g = per_inv = np.nan

            feats13 = np.array([
                price_to_ma200, ma60_slope, ma20_to_ma60, bb_position,
                volume_ratio, macd_hist_ratio,
                stoch_60_pos, obv_slope_20,
                idx_slope, idx_mdd,
                roe, op_g, per_inv,
            ], dtype=np.float32)
            # 핵심 10 피처 (인덱스 0~9) 중 NaN 있으면 제외
            if np.isnan(feats13[:10]).any():
                continue
            out[tk] = feats13
        return out

    # ─── 하이브리드 스코어 계산 ────────────────────────────
    def _hybrid_score(self, ticker: str, feats13: np.ndarray, market: str) -> float:
        """KR: 0.6×v3 + 0.4×v4. US: v3 단독. 결과 0~100."""
        is_kr = market in ("KOSPI", "KOSDAQ")

        v3_score = None
        if self.model_v3 is not None:
            v3_prob = float(self.model_v3.predict(feats13[:10].reshape(1, -1))[0])
            v3_score = v3_prob * 100.0

        if not is_kr or self.model_v4 is None:
            return v3_score if v3_score is not None else 50.0

        # KR 종목: v4 도 계산 (펀더 NaN 이면 LightGBM 자동 처리)
        v4_prob = float(self.model_v4.predict(feats13.reshape(1, -1))[0])
        v4_score = v4_prob * 100.0
        # v3 없으면 v4 단독
        if v3_score is None:
            return v4_score
        return HYBRID_W_V3 * v3_score + HYBRID_W_V4 * v4_score

    # ─── 메인 API ──────────────────────────────────────────
    def select_top_n(
        self,
        as_of: str,
        top_n: int = 10,
        market_split: bool = False,
        trend_filter: bool = False,
        market_cap_min: Optional[float] = None,
        market_cap_percentile: Optional[float] = None,
        market_ratio: Optional[str] = None,
        use_ttm_per: bool = True,
        trend_bonus: float = 0.0,
    ) -> pd.DataFrame:
        # 1) Rule 후보 풀
        candidates_n = max(top_n * 3, 30)
        picks = self.rule.select_top_n(
            as_of=as_of, top_n=candidates_n,
            market_split=False, trend_filter=trend_filter,
            market_cap_min=market_cap_min,
            market_cap_percentile=market_cap_percentile,
            trend_bonus=trend_bonus,
            use_ttm_per=use_ttm_per,
        )
        if picks.empty:
            return picks

        # 2) 13 피처 계산 + 하이브리드 점수
        markets = dict(zip(picks["ticker"], picks["market"]))
        feats_map = self._compute_features(picks["ticker"].tolist(), markets, as_of)
        v3_scores, v4_scores, hybrid_scores = [], [], []
        for tk in picks["ticker"]:
            if tk not in feats_map:
                hybrid_scores.append(50.0); v3_scores.append(50.0); v4_scores.append(50.0)
                continue
            feats = feats_map[tk]
            mk = markets[tk]
            # v3 단독
            v3_only = float(self.model_v3.predict(feats[:10].reshape(1, -1))[0]) * 100 if self.model_v3 else 50.0
            v3_scores.append(v3_only)
            # v4 단독 (KR only — US 는 모두 NaN 이라 의미 없지만 일관성 위해 계산)
            if self.model_v4 and mk in ("KOSPI", "KOSDAQ"):
                v4_only = float(self.model_v4.predict(feats.reshape(1, -1))[0]) * 100
            else:
                v4_only = v3_only  # US 는 v3 와 동일
            v4_scores.append(v4_only)
            # Hybrid
            hybrid_scores.append(self._hybrid_score(tk, feats, mk))

        picks["ai_v3_score"] = v3_scores
        picks["ai_v4_score"] = v4_scores
        picks["ai_score"] = hybrid_scores

        # 3) Ensemble = ai_weight × hybrid + (1-w) × rule
        picks["ensemble_score"] = (
            self.ai_weight * picks["ai_score"]
            + (1 - self.ai_weight) * picks["rule_score"]
        ).round(2)
        picks["rule_score"] = picks["ensemble_score"]  # 백테스트 호환

        # 4) 시장 분배 + top_n
        if market_ratio is not None:
            try:
                a, b = map(int, market_ratio.split(":"))
                n_kospi = round(top_n * a / (a + b)); n_kosdaq = top_n - n_kospi
            except Exception:
                n_kospi = n_kosdaq = top_n // 2
            kospi = picks[picks["market"] == "KOSPI"].nlargest(n_kospi, "ensemble_score")
            kosdaq = picks[picks["market"] == "KOSDAQ"].nlargest(n_kosdaq, "ensemble_score")
            return pd.concat([kospi, kosdaq]).sort_values("ensemble_score", ascending=False).reset_index(drop=True)
        if market_split:
            n = top_n // 2
            kospi = picks[picks["market"] == "KOSPI"].nlargest(n, "ensemble_score")
            kosdaq = picks[picks["market"] == "KOSDAQ"].nlargest(n, "ensemble_score")
            return pd.concat([kospi, kosdaq]).sort_values("ensemble_score", ascending=False).reset_index(drop=True)
        return picks.nlargest(top_n, "ensemble_score").reset_index(drop=True)
