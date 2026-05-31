"""학습된 Fusion 게이트용 학습 데이터 빌더.

날짜별로 LGBM 앙상블 후보 풀을 **한 번만** 평가해 (chart=ai_score, fundamental=rule_score)
를 얻고, 종목별 valuation·macro 신호를 붙인 뒤, 42거래일 후 횡단면 수익률 백분위를
타깃으로 만든다. 뉴스 모달은 과거 재구성 불가라 제외(학습에서 NEWS 는 unavailable).

반환: (X[N,12], mask[N,4], score[N,4], y[N]) — learned_fusion.LearnedFusion.fit 입력.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import DB_PATH
from src.modality.analyzer import MultimodalAnalyzer
from src.modality.base import ModalitySignal
from src.modality.learned_fusion import signals_to_features

logger = logging.getLogger(__name__)


def month_grid(start: str, end: str, day: int = 9) -> List[str]:
    """start~end 사이 매월 `day` 일 날짜 리스트 (리밸 시점과 유사)."""
    dates = pd.date_range(start=start, end=end, freq="MS")
    out = []
    for d in dates:
        cand = d.replace(day=min(day, 28))
        if pd.Timestamp(start) <= cand <= pd.Timestamp(end):
            out.append(cand.strftime("%Y-%m-%d"))
    return out


def _forward_pct(db_path: Path, tickers: list, as_of: str, horizon: int) -> dict:
    """as_of 기준 horizon 거래일 후 수익률 → 횡단면 백분위(0~1) dict."""
    if not tickers:
        return {}
    ph = ",".join("?" * len(tickers))
    with sqlite3.connect(db_path) as c:
        df = pd.read_sql_query(
            f"SELECT ticker, date, close FROM ohlcv "
            f"WHERE ticker IN ({ph}) AND date >= ? ORDER BY ticker, date",
            c, params=[*tickers, as_of])
    rets = {}
    for tk, g in df.groupby("ticker"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        entry = g["close"].iloc[0]
        exit_ = g["close"].iloc[min(horizon, len(g) - 1)]
        if entry and entry > 0:
            rets[tk] = exit_ / entry - 1.0
    if not rets:
        return {}
    s = pd.Series(rets)
    pct = s.rank(pct=True)          # 0~1 횡단면 백분위
    return pct.to_dict()


def build_samples(
    as_of_dates: List[str],
    db_path: Path = DB_PATH,
    horizon: int = 42,
    pool_n: int = 300,
    ai_weight: float = 0.2,
    market_cap_min: Optional[float] = None,
    progress: bool = True,
):
    """학습 샘플 (X, mask, score, y) 생성."""
    analyzer = MultimodalAnalyzer(db_path=db_path, ai_weight=ai_weight)
    # 앙상블 1회 로드
    from src.recommend.lgbm_ensemble import LGBMEnsembleScreener
    ens = LGBMEnsembleScreener(ai_weight=ai_weight)
    analyzer._ensemble = ens

    name_map = {}
    with sqlite3.connect(db_path) as c:
        for tk, nm in c.execute("SELECT ticker, name FROM tickers"):
            name_map[tk] = nm

    X, M, S, Y = [], [], [], []
    for i, as_of in enumerate(as_of_dates, 1):
        try:
            picks = ens.select_top_n(as_of=as_of, top_n=pool_n, use_ttm_per=True,
                                     market_cap_min=market_cap_min)
        except Exception as e:  # noqa: BLE001
            logger.debug("%s 풀 생성 실패: %s", as_of, e)
            continue
        if picks is None or picks.empty:
            continue

        fwd = _forward_pct(db_path, picks["ticker"].tolist(), as_of, horizon)
        if not fwd:
            continue
        macro_sig = analyzer._macro(as_of)

        n_added = 0
        for _, r in picks.iterrows():
            tk = r["ticker"]
            if tk not in fwd:
                continue
            signals = {
                "chart": ModalitySignal(name="chart", score=float(r.get("ai_score", 50.0)),
                                        confidence=0.6, label="AI"),
                "fundamental": ModalitySignal(name="fundamental",
                                              score=float(r.get("rule_score", 50.0)),
                                              confidence=0.7, label="Rule"),
                "macro": macro_sig,
                "valuation": analyzer._valuation(tk, name_map.get(tk, tk), as_of),
            }
            feat, mask, score = signals_to_features(signals)
            X.append(feat); M.append(mask); S.append(score); Y.append(fwd[tk])
            n_added += 1
        if progress and i % 10 == 0:
            logger.info("[%d/%d] %s — 누적 샘플 %d", i, len(as_of_dates), as_of, len(Y))

    if not Y:
        return (np.empty((0, 12), np.float32), np.empty((0, 4), np.float32),
                np.empty((0, 4), np.float32), np.empty((0,), np.float32))
    return (np.array(X, np.float32), np.array(M, np.float32),
            np.array(S, np.float32), np.array(Y, np.float32))
