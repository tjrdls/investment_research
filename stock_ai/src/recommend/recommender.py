"""
추천 엔진 (★ 사용자 직접 사용)
================================
당신의 워크플로우:
  1. 시총 1조+ 종목 (하드 필터)
  2. ROE ≥ 30%, PER ≤ 50, 매출/순이익 성장 (하드 필터)
  3. 통과한 종목들에 LSTM 모델 적용 (6:2:2 가중)
  4. 상위 20개 → 최종 10개 추천
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch

from src.config import CFG, DB_PATH, MODEL_PATH, get_device
from src.data.feature_engineer import FeatureEngineer
from src.model.lstm import MultiEncoderLSTM
from src.model.trainer import GroupScaler

logger = logging.getLogger(__name__)


class Recommender:
    """하드 필터 + 딥러닝 점수 + 상위 N개."""

    def __init__(self, db_path: Path = DB_PATH, model_path: Path = MODEL_PATH):
        self.db_path = Path(db_path)
        self.model_path = Path(model_path)
        self.fe = FeatureEngineer(db_path)
        self._model: Optional[MultiEncoderLSTM] = None
        self._scaler: Optional[GroupScaler] = None
        self._device = get_device()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield conn
        finally:
            conn.close()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"모델 없음: {self.model_path}\n"
                "먼저 `python main.py train` 실행하세요."
            )
        ckpt = torch.load(self.model_path, map_location=self._device, weights_only=False)

        self._model = MultiEncoderLSTM().to(self._device)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

        self._scaler = GroupScaler()
        s = ckpt["scaler_stats"]
        self._scaler.stats = {
            "fund": (s["fund_mean"], s["fund_std"]),
            "chart": (s["chart_mean"], s["chart_std"]),
            "market": (s["market_mean"], s["market_std"]),
        }
        logger.info("✓ 모델 로드: %s (디바이스: %s)", self.model_path.name, self._device)

    def hard_filter(self, as_of: str) -> pd.DataFrame:
        """당신이 정한 모든 하드 필터 적용."""
        hf = CFG.hard_filter
        rev_col, prof_col = ("revenue_growth_yoy", "profit_growth_yoy") \
            if hf.growth_period == "yoy" else ("revenue_growth_qoq", "profit_growth_qoq")

        with self._conn() as conn:
            try:
                df = pd.read_sql_query("""
                    WITH latest_cap AS (
                        SELECT ticker, market_cap FROM market_cap m
                        WHERE date = (SELECT MAX(date) FROM market_cap WHERE ticker=m.ticker AND date<=?)
                    ),
                    latest_fund AS (
                        SELECT * FROM fundamentals f
                        WHERE period_end = (SELECT MAX(period_end) FROM fundamentals WHERE ticker=f.ticker AND period_end<=?)
                    ),
                    latest_per AS (
                        SELECT ticker, per FROM per_history p
                        WHERE date = (SELECT MAX(date) FROM per_history WHERE ticker=p.ticker AND date<=?)
                    )
                    SELECT t.ticker, t.name, t.market,
                           c.market_cap, f.roe, p.per,
                           f.revenue_growth_yoy, f.profit_growth_yoy,
                           f.revenue_growth_qoq, f.profit_growth_qoq,
                           f.debt_ratio
                    FROM tickers t
                    JOIN latest_cap c ON c.ticker = t.ticker
                    LEFT JOIN latest_fund f ON f.ticker = t.ticker
                    LEFT JOIN latest_per p ON p.ticker = t.ticker
                    WHERE c.market_cap >= ?
                """, conn, params=[as_of, as_of, as_of, hf.market_cap_min_krw])
            except sqlite3.OperationalError as e:
                logger.warning("PER 테이블 없음 — PER 필터 생략: %s", e)
                df = pd.read_sql_query("""
                    WITH latest_cap AS (
                        SELECT ticker, market_cap FROM market_cap m
                        WHERE date = (SELECT MAX(date) FROM market_cap WHERE ticker=m.ticker AND date<=?)
                    ),
                    latest_fund AS (
                        SELECT * FROM fundamentals f
                        WHERE period_end = (SELECT MAX(period_end) FROM fundamentals WHERE ticker=f.ticker AND period_end<=?)
                    )
                    SELECT t.ticker, t.name, t.market,
                           c.market_cap, f.roe, NULL AS per,
                           f.revenue_growth_yoy, f.profit_growth_yoy,
                           f.revenue_growth_qoq, f.profit_growth_qoq,
                           f.debt_ratio
                    FROM tickers t
                    JOIN latest_cap c ON c.ticker=t.ticker
                    LEFT JOIN latest_fund f ON f.ticker=t.ticker
                    WHERE c.market_cap >= ?
                """, conn, params=[as_of, as_of, hf.market_cap_min_krw])

        if df.empty:
            logger.warning("시총 1조+ 종목 없음 (as_of=%s)", as_of)
            return df

        before = len(df)

        # ROE ≥ 30%
        df = df[df["roe"].fillna(-999) >= hf.roe_min]
        # PER ≤ 50
        if "per" in df.columns and df["per"].notna().any():
            df = df[(df["per"].fillna(999) <= hf.per_max) & (df["per"].fillna(-1) > hf.per_min)]
        # 매출/순이익 성장
        if hf.revenue_growth_required:
            df = df[df[rev_col].fillna(-999) > 0]
        if hf.profit_growth_required:
            df = df[df[prof_col].fillna(-999) > 0]

        logger.info("하드 필터: %d → %d개", before, len(df))
        return df.reset_index(drop=True)

    @torch.no_grad()
    def ai_score(self, tickers: list[str], as_of: str) -> pd.DataFrame:
        self._load_model()

        seq_len = CFG.model.seq_len
        load_start = (
            pd.Timestamp(as_of) - pd.Timedelta(days=seq_len * 4 + 60)
        ).strftime("%Y-%m-%d")

        funds, charts, markets, valid_tk = [], [], [], []
        for tk in tickers:
            data = self.fe.build_for_ticker(tk, load_start, as_of)
            if data is None or len(data["chart"]) < seq_len:
                continue
            funds.append(data["fundamental"].iloc[-seq_len:].to_numpy(dtype=np.float32))
            charts.append(data["chart"].iloc[-seq_len:].to_numpy(dtype=np.float32))
            markets.append(data["market"].iloc[-seq_len:].to_numpy(dtype=np.float32))
            valid_tk.append(tk)

        if not funds:
            return pd.DataFrame(columns=[
                "ticker", "final_score", "fund_score",
                "chart_score", "market_score", "pred_return",
            ])

        F_arr = self._scaler.transform("fund", np.stack(funds))
        C_arr = self._scaler.transform("chart", np.stack(charts))
        M_arr = self._scaler.transform("market", np.stack(markets))

        ft = torch.from_numpy(F_arr).to(self._device)
        ct = torch.from_numpy(C_arr).to(self._device)
        mt = torch.from_numpy(M_arr).to(self._device)
        out = self._model(ft, ct, mt)

        return pd.DataFrame({
            "ticker": valid_tk,
            "final_score": out["final_score"].cpu().numpy() * 100,
            "fund_score": out["fund_score"].cpu().numpy() * 100,
            "chart_score": out["chart_score"].cpu().numpy() * 100,
            "market_score": out["market_score"].cpu().numpy() * 100,
            "pred_return": out["pred_return"].cpu().numpy(),
        })

    def recommend(self, as_of: Optional[str] = None) -> pd.DataFrame:
        as_of = as_of or pd.Timestamp.today().strftime("%Y-%m-%d")
        logger.info("=== 추천 시작 (as_of=%s) ===", as_of)

        candidates = self.hard_filter(as_of)
        if candidates.empty:
            logger.warning("하드 필터 통과 종목 없음")
            return pd.DataFrame()

        ai_df = self.ai_score(candidates["ticker"].tolist(), as_of)
        if ai_df.empty:
            logger.warning("AI 점수 계산 실패")
            return candidates

        merged = candidates.merge(ai_df, on="ticker", how="inner")
        merged = merged.sort_values("final_score", ascending=False).reset_index(drop=True)

        # 상위 20개 분석 → 최종 10개 (당신 요구)
        top20 = merged.head(CFG.recommend.analyze_top_n)
        logger.info("상위 %d개 분석 완료", len(top20))
        final = top20.head(CFG.recommend.final_top_n)

        cols = [
            "ticker", "name", "market",
            "final_score", "fund_score", "chart_score", "market_score",
            "pred_return", "roe", "per",
            "revenue_growth_yoy", "profit_growth_yoy",
            "market_cap",
        ]
        cols = [c for c in cols if c in final.columns]
        return final[cols].reset_index(drop=True)


def format_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "추천 종목 없음"

    fmt = df.copy()
    if "market_cap" in fmt.columns:
        fmt["시총(조)"] = (fmt["market_cap"] / 1e12).round(2)
        fmt = fmt.drop(columns=["market_cap"])
    if "pred_return" in fmt.columns:
        fmt["6M 예측"] = (fmt["pred_return"] * 100).round(1).astype(str) + "%"
        fmt = fmt.drop(columns=["pred_return"])

    for c in ("final_score", "fund_score", "chart_score", "market_score"):
        if c in fmt.columns:
            fmt[c] = fmt[c].round(1)
    for c in ("roe", "revenue_growth_yoy", "profit_growth_yoy"):
        if c in fmt.columns:
            fmt[c] = fmt[c].round(1).astype(str) + "%"
    if "per" in fmt.columns:
        fmt["per"] = fmt["per"].round(1)

    return fmt.to_string(index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rec = Recommender()
    result = rec.recommend()
    print()
    print("=" * 80)
    print(f"AI 추천 종목 ({len(result)}개)")
    print("=" * 80)
    print(format_table(result))
