"""
앙상블 추천 모델 (규칙 + 차트 AI)
================================================
규칙 기반 점수와 차트 AI 신뢰도를 가중 평균.

최종점수 = ai_weight × AI신뢰도(0~100) + (1 - ai_weight) × 규칙점수

ai_weight 0.0 = 순수 규칙 (기존 시스템)
ai_weight 0.2 = 80:20 (사용자 제안)
ai_weight 0.3 = 70:30
ai_weight 0.5 = 50:50
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.config import DB_PATH, MODEL_DIR
from src.model.chart_lstm import ChartLSTM
from src.screener.rule_based import RuleBasedScreener

logger = logging.getLogger(__name__)


class EnsembleScreener:
    """규칙 기반 + 차트 AI 앙상블."""

    def __init__(
        self,
        ai_weight: float = 0.2,
        seq_len: int = 60,
        model_path: Optional[Path] = None,
        db_path: Path = DB_PATH,
    ):
        assert 0.0 <= ai_weight <= 1.0, "ai_weight는 0~1 사이"
        self.ai_weight = ai_weight
        self.seq_len = seq_len
        self.db_path = db_path
        self.rule_screener = RuleBasedScreener(db_path=db_path)

        # 디바이스
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 차트 모델 로드
        model_path = model_path or (MODEL_DIR / "chart_lstm.pt")
        if not model_path.exists():
            raise FileNotFoundError(
                f"차트 모델 없음: {model_path}\n"
                "먼저 'python main.py train-chart' 실행 필요"
            )
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = ChartLSTM(chart_dim=ckpt["chart_dim"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        logger.info("✓ 차트 모델 로드 (val IC: %.4f, AI 비중: %.0f%%)",
                    ckpt["val_ic"], ai_weight * 100)

    def _compute_chart_features(self, ohlcv: pd.DataFrame) -> Optional[np.ndarray]:
        """단일 종목의 OHLCV → 최근 seq_len일 차트 피처 (seq_len, 8)."""
        df = ohlcv.sort_values("date").copy()
        if len(df) < self.seq_len + 60:
            return None

        df["rsi_14"] = self._rsi(df["close"], 14)
        ma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        df["bb_width"] = (4 * std20) / ma20
        df["ma_dev"] = (df["close"] - ma20) / ma20
        df["volatility_20"] = df["close"].pct_change().rolling(20).std()
        df["volume_z"] = (df["volume"] - df["volume"].rolling(60).mean()) / (df["volume"].rolling(60).std() + 1e-9)
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        macd = ema12 - ema26
        df["macd_z"] = (macd - macd.rolling(60).mean()) / (macd.rolling(60).std() + 1e-9)
        ma60 = df["close"].rolling(60).mean()
        df["above_ma60"] = (df["close"] > ma60).astype(float)
        df["ma60_slope"] = ma60.pct_change(5)

        chart_cols = ["rsi_14","bb_width","ma_dev","volatility_20",
                      "volume_z","macd_z","above_ma60","ma60_slope"]
        df = df.dropna(subset=chart_cols).reset_index(drop=True)
        if len(df) < self.seq_len:
            return None
        return df[chart_cols].iloc[-self.seq_len:].values.astype(np.float32)

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)

    def _get_ai_scores(self, tickers: list[str], as_of: str) -> dict[str, float]:
        """대상 종목들의 AI 신뢰도 점수(0~100) 일괄 계산."""
        if not tickers:
            return {}

        # OHLCV 일괄 로드
        from datetime import datetime, timedelta
        start = (datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=200)).strftime("%Y-%m-%d")
        placeholders = ",".join("?" * len(tickers))
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"""
                SELECT ticker, date, open, high, low, close, volume FROM ohlcv
                WHERE ticker IN ({placeholders})
                  AND date BETWEEN ? AND ?
                ORDER BY ticker, date
            """, conn, params=[*tickers, start, as_of])

        # 종목별 피처 계산
        batch = []
        batch_tickers = []
        for tk, g in df.groupby("ticker"):
            feats = self._compute_chart_features(g)
            if feats is None:
                continue
            batch.append(feats)
            batch_tickers.append(tk)

        if not batch:
            return {}

        # 배치 추론
        x = torch.tensor(np.array(batch), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            conf = out["trend_confidence"].cpu().numpy()

        # 0~1 → 0~100 (규칙 점수와 동일 스케일)
        return {tk: float(c) * 100 for tk, c in zip(batch_tickers, conf)}

    def select_top_n(
        self,
        as_of: str,
        top_n: int = 10,
        market_split: bool = False,
        trend_filter: bool = False,
        market_cap_min: Optional[float] = None,
        trend_bonus: float = 0.0,
        market_ratio: Optional[str] = None,
        use_ttm_per: bool = True,  # 포트폴리오/성과와 동일 default
    ) -> pd.DataFrame:
        """앙상블 추천."""
        # 1) 규칙 기반 후보 풀 (확장: top_n × 3 정도 가져와서 AI 재정렬)
        candidates_n = max(top_n * 3, 30)
        rule_picks = self.rule_screener.select_top_n(
            as_of=as_of, top_n=candidates_n,
            market_split=False,  # 일단 다 받음
            trend_filter=trend_filter,
            market_cap_min=market_cap_min,
            trend_bonus=trend_bonus,
            use_ttm_per=use_ttm_per,
        )
        if rule_picks.empty:
            return rule_picks

        # 2) AI 신뢰도 점수 일괄 계산
        ai_scores = self._get_ai_scores(rule_picks["ticker"].tolist(), as_of)
        rule_picks["ai_score"] = rule_picks["ticker"].map(ai_scores).fillna(50.0)  # 없으면 중립

        # 3) 앙상블 점수
        rule_picks["ensemble_score"] = (
            self.ai_weight * rule_picks["ai_score"]
            + (1 - self.ai_weight) * rule_picks["rule_score"]
        ).round(2)

        # 4) 시장 분리 적용 (market_ratio 우선, market_split 후순위)
        if market_ratio is not None:
            try:
                parts = market_ratio.split(":")
                r_kospi, r_kosdaq = int(parts[0]), int(parts[1])
                total_r = r_kospi + r_kosdaq
                n_kospi = round(top_n * r_kospi / total_r)
                n_kosdaq = top_n - n_kospi
            except Exception:
                n_kospi = n_kosdaq = top_n // 2
            kospi = rule_picks[rule_picks["market"] == "KOSPI"].sort_values(
                "ensemble_score", ascending=False).head(n_kospi)
            kosdaq = rule_picks[rule_picks["market"] == "KOSDAQ"].sort_values(
                "ensemble_score", ascending=False).head(n_kosdaq)
            result = pd.concat([kospi, kosdaq], ignore_index=True)
        elif market_split:
            n_per_market = top_n // 2
            kospi = rule_picks[rule_picks["market"] == "KOSPI"].sort_values(
                "ensemble_score", ascending=False).head(n_per_market)
            kosdaq = rule_picks[rule_picks["market"] == "KOSDAQ"].sort_values(
                "ensemble_score", ascending=False).head(n_per_market)
            result = pd.concat([kospi, kosdaq], ignore_index=True)
        else:
            result = rule_picks.sort_values(
                "ensemble_score", ascending=False).head(top_n)

        # ensemble_score 를 rule_score 컬럼에도 복사 (백테스트 호환)
        result = result.copy()
        result["rule_score"] = result["ensemble_score"]
        return result.reset_index(drop=True)


def run_ensemble_backtest(
    ai_weight: float = 0.2,
    start_year: int = 2015,
    end_year: int = 2024,
    top_n: int = 10,
    replacement_rule: str = "keep_simple",
    market_split: bool = True,
    trend_filter: bool = True,
    period_months: int = 6,
    market_cap_min: Optional[float] = 1e12,
    trend_bonus: float = 0.0,
    model_path: Optional[Path] = None,
    market_ratio: Optional[str] = None,
):
    """앙상블 백테스트 실행."""
    from src.backtest.rebalance import RebalanceBacktest

    screener = EnsembleScreener(ai_weight=ai_weight, model_path=model_path)

    def picker(as_of: str, n: int) -> pd.DataFrame:
        return screener.select_top_n(
            as_of=as_of, top_n=n,
            market_split=market_split,
            trend_filter=trend_filter,
            market_cap_min=market_cap_min,
            trend_bonus=trend_bonus,
            market_ratio=market_ratio,
        )

    bt = RebalanceBacktest()
    return bt.run(
        picker=picker,
        start_year=start_year, end_year=end_year,
        top_n=top_n, weight_scheme="rank",
        replacement_rule=replacement_rule,
        period_months=period_months,
    )
