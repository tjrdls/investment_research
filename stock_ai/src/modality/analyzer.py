"""멀티모달 종목 분석 오케스트레이터.

단일 종목을 5개 모달리티(차트·펀더멘털·밸류에이션·뉴스·매크로)로 해석하고
`LateFusion` 으로 결합한 뒤 `llm.narrate` 로 설명을 붙인다.

구버전 `pipeline/prediction_pipeline.py` 의 Late Fusion 흐름을 stock_ai 의
데이터 계층(DB·DART) 위에 재구성한 것.

설계
----
- `fuse_signals()` : 이미 만들어진 신호 dict → 결과. **순수에 가까움**(LLM 키 없으면
  결정적 템플릿) → 테스트 가능.
- `MultimodalAnalyzer.analyze()` : DB/DART/뉴스에서 신호를 직접 빌드(I/O).
  각 모달리티는 try/except 로 감싸 데이터 없으면 `unavailable` 로 degrade.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from src.config import CFG, DB_PATH
from src.modality import llm as llm_mod
from src.modality.base import ModalitySignal
from src.modality.fusion import FusionResult, LateFusion

logger = logging.getLogger(__name__)

MODALITIES = ("fundamental", "chart", "macro", "valuation", "news")


def _make_fusion_engine(mode: str, weights: Optional[dict]):
    """융합 엔진 선택.

    mode:
      "weighted" → 고정 가중 평균 LateFusion (기본·항상 가능)
      "learned"  → 학습된 게이팅 LearnedFusion (모델 없으면 weighted 로 폴백)
      "auto"     → 학습 모델 있으면 learned, 없으면 weighted
    반환: (engine, 실제 사용 모드 라벨)
    """
    if mode in ("learned", "auto"):
        try:
            from src.modality.learned_fusion import LearnedFusion
            lf = LearnedFusion()
            if lf.available:
                return lf, "learned"
        except Exception as e:  # noqa: BLE001
            logger.warning("LearnedFusion 사용 불가 → weighted 폴백: %s", e)
        if mode == "learned":
            logger.info("학습된 fusion 모델 없음 → 가중 평균(weighted)으로 폴백")
    return LateFusion(weights or CFG.fusion.as_dict()), "weighted"


def fuse_signals(
    stock_name: str,
    signals: Dict[str, ModalitySignal],
    weights: Optional[dict] = None,
    with_narrative: bool = True,
    fusion_mode: str = "auto",
) -> dict:
    """신호 dict → {fusion, narrative, report, fusion_mode}.

    DB 없이도 동작(신호를 직접 넣어주면 됨) → 테스트에서 사용.
    fusion_mode: "weighted"(기본 가중평균) / "learned"(게이팅 신경망) / "auto".
    """
    engine, used_mode = _make_fusion_engine(fusion_mode, weights)
    fusion: FusionResult = engine.fuse(signals)
    narrative = llm_mod.narrate(stock_name, fusion) if with_narrative else {}
    report = llm_mod.format_report(stock_name, "", fusion, narrative) if with_narrative else ""
    return {"fusion": fusion, "narrative": narrative, "report": report,
            "fusion_mode": used_mode}


class MultimodalAnalyzer:
    """DB·DART·뉴스에서 모달리티 신호를 빌드해 융합한다."""

    def __init__(self, db_path: Path = DB_PATH, ai_weight: float = 0.2):
        self.db_path = Path(db_path)
        self.ai_weight = ai_weight
        self._ensemble = None  # lazy

    # ── 개별 모달리티 빌더 (각자 graceful degrade) ────────────
    def _chart_and_fundamental(self, ticker: str, as_of: str) -> Dict[str, ModalitySignal]:
        """LGBM 앙상블에서 차트(AI) + 펀더멘털(Rule) 신호 추출."""
        out: Dict[str, ModalitySignal] = {}
        try:
            if self._ensemble is None:
                from src.recommend.lgbm_ensemble import LGBMEnsembleScreener
                self._ensemble = LGBMEnsembleScreener(ai_weight=self.ai_weight)
            picks = self._ensemble.select_top_n(as_of=as_of, top_n=300, use_ttm_per=True)
            row = picks[picks["ticker"] == ticker]
            if row.empty:
                out["chart"] = ModalitySignal.unavailable("chart", "후보 풀 밖(필터 탈락)")
                out["fundamental"] = ModalitySignal.unavailable("fundamental", "후보 풀 밖(필터 탈락)")
                return out
            r = row.iloc[0]
            out["chart"] = ModalitySignal(
                name="chart", score=float(r.get("ai_score", 50.0)), confidence=0.6,
                label="AI 차트랭킹", detail={"ai_v3": r.get("ai_v3_score"), "ai_v4": r.get("ai_v4_score")},
            )
            out["fundamental"] = ModalitySignal(
                name="fundamental", score=float(r.get("rule_score", 50.0)), confidence=0.7,
                label="Rule 4팩터", detail={"roe": r.get("roe"), "per": r.get("per")},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("차트/펀더 신호 빌드 실패: %s", e)
            out.setdefault("chart", ModalitySignal.unavailable("chart", str(e)))
            out.setdefault("fundamental", ModalitySignal.unavailable("fundamental", str(e)))
        return out

    def _macro(self, as_of: str) -> ModalitySignal:
        """KOSPI200(069500) 종가의 60일선 괴리 → 시장 레짐 신호.

        (정식 Ichimoku/ADX 분할 스위칭은 backtest 엔진에서 운용되며, 여기서는
        단일 종목 분석용으로 동일 방향성을 가볍게 요약한다.)
        """
        try:
            import sqlite3
            import pandas as pd
        except Exception as e:  # noqa: BLE001
            return ModalitySignal.unavailable("macro", str(e))
        try:
            start = (pd.Timestamp(as_of) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
            with sqlite3.connect(self.db_path) as c:
                idx = pd.read_sql_query(
                    "SELECT date, close FROM ohlcv WHERE ticker='069500' "
                    "AND date BETWEEN ? AND ? ORDER BY date",
                    c, params=[start, as_of], parse_dates=["date"])
            if len(idx) < 60:
                return ModalitySignal.unavailable("macro", "지수 데이터 부족")
            ma60 = idx["close"].rolling(60).mean().iloc[-1]
            close = idx["close"].iloc[-1]
            score = 50 + (close / ma60 - 1.0) * 200  # 60일선 괴리율 → 점수
            return ModalitySignal(name="macro", score=score, confidence=0.5,
                                  label="시장 레짐", detail={"close": float(close), "ma60": float(ma60)})
        except Exception as e:  # noqa: BLE001
            return ModalitySignal.unavailable("macro", str(e))

    def _valuation(self, ticker: str, stock_name: str, as_of: str) -> ModalitySignal:
        try:
            from src.modality.valuation import calculate_ttm_metrics, valuation_to_signal
        except Exception as e:  # noqa: BLE001
            return ModalitySignal.unavailable("valuation", str(e))
        try:
            import sqlite3
            import pandas as pd
            with sqlite3.connect(self.db_path) as c:
                fin = pd.read_sql_query(
                    "SELECT * FROM financials WHERE ticker=? ORDER BY period_end DESC LIMIT 4",
                    c, params=[ticker])
                price = pd.read_sql_query(
                    "SELECT close FROM ohlcv WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT 1",
                    c, params=[ticker, as_of])
            if fin.empty or price.empty:
                return ModalitySignal.unavailable("valuation", "재무/가격 데이터 없음")
            metrics = calculate_ttm_metrics(fin.to_dict("records"), float(price["close"].iloc[0]))
            return valuation_to_signal(metrics)
        except Exception as e:  # noqa: BLE001
            return ModalitySignal.unavailable("valuation", str(e))

    def _news(self, stock_name: str, ticker: str) -> ModalitySignal:
        try:
            from src.modality.news import news_signal
            return news_signal(stock_name, ticker)
        except Exception as e:  # noqa: BLE001
            return ModalitySignal.unavailable("news", str(e))

    # ── 메인 API ─────────────────────────────────────────────
    def build_signals(self, ticker: str, stock_name: str, as_of: str,
                      include_news: bool = True) -> Dict[str, ModalitySignal]:
        """모달리티 신호 dict 빌드 (융합 전). 학습 데이터 빌더가 재사용.

        include_news=False 면 뉴스(현재 시점 헤드라인, 과거 재구성 불가)를 제외 →
        과거 시점 학습 샘플 생성에 사용.
        """
        signals: Dict[str, ModalitySignal] = {}
        signals.update(self._chart_and_fundamental(ticker, as_of))
        signals["macro"] = self._macro(as_of)
        signals["valuation"] = self._valuation(ticker, stock_name, as_of)
        if include_news:
            signals["news"] = self._news(stock_name, ticker)
        return signals

    def analyze(self, ticker: str, stock_name: str, as_of: Optional[str] = None,
                fusion_mode: str = "auto") -> dict:
        """단일 종목 멀티모달 분석. as_of 미지정 시 오늘.

        fusion_mode: "weighted" / "learned" / "auto" (학습 모델 있으면 게이팅).
        """
        import pandas as pd
        as_of = as_of or pd.Timestamp.today().strftime("%Y-%m-%d")
        logger.info("멀티모달 분석: %s (%s) as_of=%s", stock_name, ticker, as_of)

        signals = self.build_signals(ticker, stock_name, as_of, include_news=True)
        result = fuse_signals(stock_name, signals, fusion_mode=fusion_mode)
        result["ticker"] = ticker
        result["as_of"] = as_of
        return result
