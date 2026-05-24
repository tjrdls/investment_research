"""
규칙 기반 스크리너 (★ 사용자의 핵심 전략)
==========================================
사용자 전략:
  - PER이 낮을수록 좋음 (선택적 — 데이터 있을 때만)
  - ROE가 높을수록 좋음
  - 매출 성장률 높을수록 좋음
  - 순이익 성장률 높을수록 좋음

각 지표를 같은 시점의 다른 종목들과 비교해 백분위 점수 매김 → 가중 합산.
6개월마다 호출하면 리밸런싱.

PER 데이터(per_history 테이블)가 DB에 없으면 PER 점수는 비활성화하고
ROE/매출/순이익 성장률만으로 점수 매김 (가중치 자동 재정규화).

사용 예:
  from src.screener.rule_based import RuleBasedScreener
  s = RuleBasedScreener()
  picks = s.select_top_n(as_of='2024-12-30', top_n=10)
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from src.config import CFG, DB_PATH

logger = logging.getLogger(__name__)


@dataclass
class ScreenerWeights:
    """각 지표의 점수 가중치. PER 비활성화 시 자동 재정규화됨."""
    per: float = 0.30          # PER 매력도 (낮을수록 ↑) - 선택적
    roe: float = 0.30          # ROE (높을수록 ↑)
    revenue_growth: float = 0.20   # 매출 성장률 (YoY)
    profit_growth: float = 0.20    # 순이익 성장률 (YoY)


class RuleBasedScreener:
    """
    펀더멘털 지표만으로 종목을 점수화 → 상위 N개 반환.

    하드 필터 통과한 종목 중에서:
      각 지표 백분위(0~1) → 가중평균 → 0~100점
    """

    def __init__(self, db_path: Path = DB_PATH, weights: Optional[ScreenerWeights] = None):
        self.db_path = Path(db_path)
        self.weights = weights or ScreenerWeights()
        # PER 테이블이 있는지 한 번만 체크해서 캐싱 (try/except 없이)
        self._has_per_table = self._check_per_table()
        if not self._has_per_table:
            logger.info("per_history 테이블 없음 → PER 점수 비활성화 (ROE/성장률로만 평가)")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield conn
        finally:
            conn.close()

    def _check_per_table(self) -> bool:
        """per_history 테이블이 DB에 있는지 sqlite_master로 확인."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='per_history'"
            ).fetchone()
        return row is not None

    def get_ma_status(self, tickers: list[str], as_of: str) -> pd.DataFrame:
        """
        as_of 시점에 각 종목의 이동평균선 상태 계산.

        반환 컬럼:
          ticker, close, ma5, ma20, ma60, above_ma60(bool),
          trend_strength(0~1):  60일선 위(0.3) + 20>60(0.3) + 5>20(0.4) 가중합
        """
        if not tickers:
            return pd.DataFrame(columns=["ticker", "close", "ma5", "ma20", "ma60",
                                          "above_ma60", "trend_strength"])

        from datetime import datetime, timedelta
        start = (datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=130)).strftime("%Y-%m-%d")

        placeholders = ",".join("?" * len(tickers))
        with self._conn() as conn:
            df = pd.read_sql_query(f"""
                SELECT ticker, date, close FROM ohlcv
                WHERE ticker IN ({placeholders})
                  AND date BETWEEN ? AND ?
                ORDER BY ticker, date
            """, conn, params=[*tickers, start, as_of])

        if df.empty:
            return pd.DataFrame(columns=["ticker", "close", "ma5", "ma20", "ma60",
                                          "above_ma60", "trend_strength"])

        rows = []
        for tk, g in df.groupby("ticker"):
            g = g.sort_values("date")
            if len(g) < 60:
                continue
            close = g["close"].iloc[-1]
            ma5 = g["close"].rolling(5).mean().iloc[-1]
            ma20 = g["close"].rolling(20).mean().iloc[-1]
            ma60 = g["close"].rolling(60).mean().iloc[-1]
            if pd.isna(ma60) or ma60 <= 0:
                continue

            # 추세 강도 (사용자 제안)
            ts = 0.0
            if close > ma60:  ts += 0.3
            if ma20 > ma60:   ts += 0.3
            if ma5 > ma20:    ts += 0.4
            # 결과: 정배열 = 1.0, 역배열 = 0.0

            rows.append({
                "ticker": tk,
                "close": float(close),
                "ma5": float(ma5),
                "ma20": float(ma20),
                "ma60": float(ma60),
                "above_ma60": bool(close > ma60),
                "trend_strength": float(ts),
            })
        return pd.DataFrame(rows)

    # 하위 호환 (기존 get_ma60_status 유지)
    def get_ma60_status(self, tickers: list[str], as_of: str) -> pd.DataFrame:
        full = self.get_ma_status(tickers, as_of)
        if full.empty:
            return pd.DataFrame(columns=["ticker", "close", "ma60", "above_ma60", "trend_score"])
        full["trend_score"] = (full["close"] - full["ma60"]) / full["ma60"]
        return full[["ticker", "close", "ma60", "above_ma60", "trend_score"]]

    # ------------------------------------------------------------------
    # 유니버스 조회
    # ------------------------------------------------------------------
    def _fetch_universe(self, as_of: str) -> pd.DataFrame:
        """as_of 시점에 데이터가 있는 모든 종목의 펀더멘털을 한꺼번에 조회."""
        with self._conn() as conn:
            if self._has_per_table:
                df = pd.read_sql_query(self._sql_with_per(), conn,
                                        params=[as_of, as_of, as_of])
            else:
                df = pd.read_sql_query(self._sql_without_per(), conn,
                                        params=[as_of, as_of])
        return df

    @staticmethod
    def _sql_with_per() -> str:
        return """
            WITH latest_cap AS (
                SELECT ticker, market_cap FROM market_cap m
                WHERE date = (
                    SELECT MAX(date) FROM market_cap
                    WHERE ticker = m.ticker AND date <= ?
                )
            ),
            latest_fund AS (
                SELECT ticker, roe, debt_ratio,
                       revenue_growth_yoy, profit_growth_yoy
                FROM fundamentals f
                WHERE period_end = (
                    SELECT MAX(period_end) FROM fundamentals
                    WHERE ticker = f.ticker AND period_end <= ?
                )
            ),
            latest_per AS (
                SELECT ticker, per FROM per_history p
                WHERE date = (
                    SELECT MAX(date) FROM per_history
                    WHERE ticker = p.ticker AND date <= ?
                )
            )
            SELECT t.ticker, t.name, t.market,
                   c.market_cap, f.roe, p.per,
                   f.revenue_growth_yoy, f.profit_growth_yoy,
                   f.revenue_growth_qoq, f.profit_growth_qoq,
                   f.operating_margin, f.profit_quality,
                   f.operating_income_growth_yoy,
                   f.debt_ratio
            FROM tickers t
            JOIN latest_cap c ON c.ticker = t.ticker
            LEFT JOIN latest_fund f ON f.ticker = t.ticker
            LEFT JOIN latest_per p ON p.ticker = t.ticker
        """

    @staticmethod
    def _sql_without_per() -> str:
        return """
            WITH latest_cap AS (
                SELECT ticker, market_cap FROM market_cap m
                WHERE date = (
                    SELECT MAX(date) FROM market_cap
                    WHERE ticker = m.ticker AND date <= ?
                )
            ),
            latest_fund AS (
                SELECT ticker, roe, debt_ratio,
                       revenue_growth_yoy, profit_growth_yoy,
                       revenue_growth_qoq, profit_growth_qoq,
                       operating_margin, profit_quality,
                       operating_income_growth_yoy
                FROM fundamentals f
                WHERE period_end = (
                    SELECT MAX(period_end) FROM fundamentals
                    WHERE ticker = f.ticker AND period_end <= ?
                )
            )
            SELECT t.ticker, t.name, t.market,
                   c.market_cap, f.roe, NULL AS per,
                   f.revenue_growth_yoy, f.profit_growth_yoy,
                   f.revenue_growth_qoq, f.profit_growth_qoq,
                   f.operating_margin, f.profit_quality,
                   f.operating_income_growth_yoy,
                   f.debt_ratio
            FROM tickers t
            JOIN latest_cap c ON c.ticker = t.ticker
            LEFT JOIN latest_fund f ON f.ticker = t.ticker
        """

    # ------------------------------------------------------------------
    # 자체 TTM PER 배치 계산
    # ------------------------------------------------------------------
    def _batch_ttm_per(self, dart, tickers: list, as_of: str) -> dict:
        """배치로 TTM PER 계산 (한번에 financials + market_cap 로드).
        반환: {ticker: ttm_per} (계산 불가 종목은 빠짐).
        """
        import sqlite3
        from src.config import DB_PATH
        if not tickers:
            return {}
        as_of_ts = pd.Timestamp(as_of)
        ph = ",".join("?" * len(tickers))
        with sqlite3.connect(DB_PATH) as c:
            fin = pd.read_sql_query(
                f"SELECT ticker, year, quarter, period_end, net_income "
                f"FROM financials WHERE ticker IN ({ph}) "
                f"AND net_income IS NOT NULL AND net_income != 0",
                c, params=tickers, parse_dates=["period_end"])
            mc = pd.read_sql_query(
                f"SELECT ticker, date, market_cap FROM market_cap "
                f"WHERE ticker IN ({ph}) AND date <= ?",
                c, params=[*tickers, as_of], parse_dates=["date"])
        if fin.empty or mc.empty:
            return {}
        # 공시일 추정 (분기말 + 45일, Q4 +90일)
        lag = fin["quarter"].map(lambda q: 90 if q == "Q4" else 45)
        fin["publish_date"] = fin["period_end"] + pd.to_timedelta(lag, unit="D")
        fin = fin[fin["publish_date"] <= as_of_ts]
        if fin.empty:
            return {}
        # ticker별 시총 (가장 최근)
        mc = mc.sort_values("date").drop_duplicates("ticker", keep="last")
        mc_map = dict(zip(mc["ticker"], mc["market_cap"]))
        out = {}
        for tk, grp in fin.sort_values("period_end").groupby("ticker"):
            grp = grp.copy()
            grp["ni_std"] = grp["net_income"].astype(float)
            # Q4 = 연 누계 → standalone 변환
            for year in grp["year"].unique():
                yr = grp[grp["year"] == year]
                if "Q4" not in yr["quarter"].values:
                    continue
                q123 = yr[yr["quarter"].isin(["Q1", "Q2", "Q3"])]
                if q123.empty:
                    continue
                q4_idx = yr[yr["quarter"] == "Q4"].index[0]
                grp.loc[q4_idx, "ni_std"] = (
                    float(yr[yr["quarter"] == "Q4"]["net_income"].iloc[0])
                    - float(q123["net_income"].sum())
                )
            last4 = grp.tail(4)
            if len(last4) < 4:
                continue
            ttm = float(last4["ni_std"].sum())
            if ttm <= 0:
                continue
            mcap = mc_map.get(tk)
            if not mcap:
                continue
            out[tk] = float(mcap) / ttm
        return out

    # ------------------------------------------------------------------
    # 하드 필터
    # ------------------------------------------------------------------
    def _apply_hard_filter(self, df: pd.DataFrame,
                            market_cap_min: Optional[float] = None,
                            operating_margin_min: float = 10.0,
                            profit_quality_max: float = 2.0,
                            operating_growth_required: bool = True) -> pd.DataFrame:
        """
        시총, ROE, PER, 성장률, 영업이익률, 일회성 차단 필터.
        
        operating_margin_min: TTM 영업이익률 하한 (%). 기본 10%.
            본업 부진 회사 차단 (에코프로 케이스).
        profit_quality_max: TTM 순익/영업이익 비율 상한. 기본 2.0.
            일회성 이익 큰 비중 차단 (자회사 매각 등).
        operating_growth_required: 영업이익 성장률 양수 요구. 기본 True.
        """
        hf = CFG.hard_filter
        cap_min = market_cap_min if market_cap_min is not None else hf.market_cap_min_krw
        before = len(df)

        # 시총
        df = df[df["market_cap"].fillna(0) >= cap_min]
        # ROE
        df = df[df["roe"].fillna(-999) >= hf.roe_min]
        # PER (PER 데이터가 실제로 있는 경우만)
        if self._has_per_table and df["per"].notna().any():
            df = df[(df["per"].fillna(999) <= hf.per_max)
                    & (df["per"].fillna(-1) > hf.per_min)]
        # 매출 성장
        if hf.revenue_growth_required:
            df = df[df["revenue_growth_yoy"].fillna(-999) > 0]
        # 순이익 성장
        if hf.profit_growth_required:
            df = df[df["profit_growth_yoy"].fillna(-999) > 0]

        # ★ 새로 추가: 영업이익률 (본업 수익성)
        if "operating_margin" in df.columns and operating_margin_min > 0:
            # NaN이면 보수적으로 탈락 (모르는 회사 안 사기)
            df = df[df["operating_margin"].fillna(-999) >= operating_margin_min]

        # ★ 새로 추가: 일회성 이익 차단
        # profit_quality = TTM 순익 / TTM 영업이익
        # 1.0 ~ 1.5: 정상
        # 2.0+ : 일회성 이익 큰 비중 (에코프로 케이스)
        # 음수: 영업적자
        if "profit_quality" in df.columns:
            df = df[
                (df["profit_quality"].notna()) &
                (df["profit_quality"] >= 0.3) &  # 너무 작아도 의심 (적자 직전)
                (df["profit_quality"] <= profit_quality_max)
            ]

        # ★ 새로 추가: 영업이익 성장률 (본업 성장 추세)
        if operating_growth_required and "operating_income_growth_yoy" in df.columns:
            df = df[df["operating_income_growth_yoy"].fillna(-999) > 0]

        logger.info("하드 필터: %d → %d개 종목 통과 (영업이익률≥%.0f%%, 일회성 차단)",
                    before, len(df), operating_margin_min)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 점수 계산
    # ------------------------------------------------------------------
    def _compute_scores(self, df: pd.DataFrame,
                        cap_bonus: bool = True,
                        momentum_penalty: bool = True) -> pd.DataFrame:
        """
        각 지표 → 백분위(0~1) → 가중평균 → 0~100점
        PER 데이터 없으면 PER 가중치를 빼고 나머지 가중치를 자동 재정규화.
        
        cap_bonus: 시총 가산점 (대형주 우선)
        momentum_penalty: 실적 둔화 페널티 (성장 약화 감점)
        """
        if df.empty:
            return df

        w = self.weights
        scores = pd.DataFrame(index=df.index)

        # PER (낮을수록 좋음) — 데이터 있을 때만
        per_active = self._has_per_table and df["per"].notna().any()
        if per_active:
            scores["per_score"] = 1 - df["per"].rank(pct=True, na_option="bottom")

        # ROE (높을수록 좋음)
        scores["roe_score"] = df["roe"].rank(pct=True, na_option="bottom")
        # 매출 성장률
        scores["revenue_score"] = df["revenue_growth_yoy"].rank(pct=True, na_option="bottom")
        # 순이익 성장률
        scores["profit_score"] = df["profit_growth_yoy"].rank(pct=True, na_option="bottom")

        # 가중 합산 (PER 활성/비활성에 따라 자동 정규화)
        if per_active:
            total_w = w.per + w.roe + w.revenue_growth + w.profit_growth
            weighted = (
                w.per * scores["per_score"]
                + w.roe * scores["roe_score"]
                + w.revenue_growth * scores["revenue_score"]
                + w.profit_growth * scores["profit_score"]
            ) / total_w
        else:
            total_w = w.roe + w.revenue_growth + w.profit_growth
            weighted = (
                w.roe * scores["roe_score"]
                + w.revenue_growth * scores["revenue_score"]
                + w.profit_growth * scores["profit_score"]
            ) / total_w

        df["rule_score"] = (weighted * 100).round(2)

        # ★ 시총 가산점 (대형주 우선)
        if cap_bonus and "market_cap" in df.columns:
            def _cap_bonus(cap):
                if pd.isna(cap):
                    return 1.0
                if cap >= 1e13:    return 1.20  # 10조+ : +20%
                if cap >= 5e12:    return 1.15  # 5~10조: +15%
                if cap >= 2e12:    return 1.08  # 2~5조: +8%
                return 1.0  # 1~2조: 변화 없음
            df["cap_multiplier"] = df["market_cap"].apply(_cap_bonus)
            df["rule_score"] = (df["rule_score"] * df["cap_multiplier"]).round(2)
            logger.info("시총 가산점 적용 (평균 배수: %.3f)", df["cap_multiplier"].mean())

        # ★ 실적 둔화 페널티 (직전 분기 대비 성장률 ↓ 시 감점)
        if momentum_penalty and "revenue_growth_qoq" in df.columns:
            # QoQ가 음수면 둔화 신호
            def _momentum_factor(row):
                qoq_rev = row.get("revenue_growth_qoq")
                qoq_prof = row.get("profit_growth_qoq")
                if pd.isna(qoq_rev) and pd.isna(qoq_prof):
                    return 1.0
                qoq_rev = qoq_rev if pd.notna(qoq_rev) else 0
                qoq_prof = qoq_prof if pd.notna(qoq_prof) else 0
                # 매출+순익 모두 음의 QoQ면 -15% 감점
                if qoq_rev < -10 and qoq_prof < -10:
                    return 0.85
                # 둘 중 하나만 둔화면 -8%
                if qoq_rev < 0 or qoq_prof < 0:
                    return 0.92
                return 1.0
            df["momentum_factor"] = df.apply(_momentum_factor, axis=1)
            df["rule_score"] = (df["rule_score"] * df["momentum_factor"]).round(2)
            logger.info("둔화 페널티 적용 (평균 배수: %.3f)", df["momentum_factor"].mean())

        # ★ Earnings Acceleration (영업이익 가속화 가산점) — 학계 검증된 알파 신호
        # "분기마다 영업이익 성장 ↑ → 주도주 가능성" (사용자 직관)
        # 영업이익 YoY 성장률이 높을수록 + 양수일수록 가산
        if "operating_income_growth_yoy" in df.columns:
            def _acceleration_bonus(oi_yoy):
                if pd.isna(oi_yoy):
                    return 1.0
                if oi_yoy >= 100:  # 영업이익 2배 이상 (폭발 성장)
                    return 1.30  # +30%
                if oi_yoy >= 50:   # 영업이익 1.5배 이상 (강한 성장)
                    return 1.20  # +20%
                if oi_yoy >= 20:   # 영업이익 +20% 이상 (좋은 성장)
                    return 1.10  # +10%
                if oi_yoy > 0:     # 양수 (정상)
                    return 1.0
                return 0.90  # 영업이익 감소 시 -10%
            df["accel_factor"] = df["operating_income_growth_yoy"].apply(_acceleration_bonus)
            df["rule_score"] = (df["rule_score"] * df["accel_factor"]).round(2)
            logger.info("영업이익 가속화 가산점 적용 (평균 배수: %.3f)", df["accel_factor"].mean())

        # 분해 점수도 저장 (해석용)
        for c in ("roe_score", "revenue_score", "profit_score"):
            df[c] = (scores[c] * 100).round(1)
        if per_active:
            df["per_score"] = (scores["per_score"] * 100).round(1)
        else:
            df["per_score"] = None

        return df

    # ------------------------------------------------------------------
    # 메인 API
    # ------------------------------------------------------------------
    def select_top_n(
        self,
        as_of: str,
        top_n: int = 10,
        apply_hard_filter: bool = True,
        market_split: bool = False,
        trend_filter: bool = False,
        trend_bonus: float = 0.0,
        market_cap_min: Optional[float] = None,
        market_cap_percentile: Optional[float] = None,
        market_ratio: Optional[str] = None,
        use_ttm_per: bool = False,
    ) -> pd.DataFrame:
        """
        as_of 시점에 점수 상위 N개 종목 반환.

        market_split=True 면 KOSPI N//2 + KOSDAQ N//2 고정.
        market_ratio (예: "6:4") 면 코스피:코스닥 = 6:4 비율로 분배.
            market_split보다 우선 적용.
        use_ttm_per=True 면 KRX 공식 PER 대신 우리가 DART 분기 데이터로
            계산한 TTM PER을 사용 (KRX의 연 1회 EPS 갱신 지연 우회).
        """
        df = self._fetch_universe(as_of)
        if use_ttm_per and not df.empty:
            # 자체 TTM PER로 덮어쓰기
            from src.data.dart_loader import DartLoader
            dart = DartLoader()
            ttm_map = self._batch_ttm_per(dart, df["ticker"].tolist(), as_of)
            df["per"] = df["ticker"].map(ttm_map)
        if df.empty:
            logger.warning("[%s] 유니버스가 비어있음", as_of)
            return df

        # ★ 시간 가변 시총 필터 (시총 백분위)
        if market_cap_percentile is not None:
            before = len(df)
            cap_threshold = df["market_cap"].quantile(1.0 - market_cap_percentile)
            df = df[df["market_cap"] >= cap_threshold].reset_index(drop=True)
            logger.info(
                "[%s] 시총 상위 %.0f%% 필터: 임계값 %.0f억, %d → %d개",
                as_of, market_cap_percentile * 100, cap_threshold / 1e8,
                before, len(df),
            )

        if apply_hard_filter:
            # market_cap_percentile 쓸 때는 추가 시총 필터 안 함 (이미 백분위에서 처리)
            cap_min_for_filter = None if market_cap_percentile else market_cap_min
            df = self._apply_hard_filter(df, market_cap_min=cap_min_for_filter)
        if df.empty:
            logger.warning("[%s] 하드 필터 통과 종목 없음", as_of)
            return df

        # ★ 추세 필터: 60일선 위 종목만 (사용자 의도)
        if trend_filter:
            before = len(df)
            ma60_df = self.get_ma60_status(df["ticker"].tolist(), as_of)
            if not ma60_df.empty:
                above = ma60_df[ma60_df["above_ma60"]]["ticker"].tolist()
                df = df[df["ticker"].isin(above)].reset_index(drop=True)
                # 추세 점수도 컬럼으로 추가
                df = df.merge(ma60_df[["ticker", "trend_score"]], on="ticker", how="left")
                logger.info("[%s] 추세 필터(60일선↑): %d → %d개", as_of, before, len(df))
            if df.empty:
                logger.warning("[%s] 추세 필터 통과 종목 없음 — 시장 전체 하락중일 가능성", as_of)
                return df

        df = self._compute_scores(df)

        # ★ 추세 강도 가산점 (사용자 새 아이디어: 5/20/60 정배열 가산)
        # trend_filter 와 별개. trend_bonus > 0 이면 점수 보너스 적용.
        if trend_bonus > 0:
            ma_df = self.get_ma_status(df["ticker"].tolist(), as_of)
            if not ma_df.empty:
                df = df.merge(ma_df[["ticker", "trend_strength"]], on="ticker", how="left")
                df["trend_strength"] = df["trend_strength"].fillna(0.0)
                # 가산: rule_score × (1 + trend_bonus × trend_strength)
                df["rule_score"] = (df["rule_score"] * (1 + trend_bonus * df["trend_strength"])).round(2)
                logger.info("[%s] 추세 가산점 적용 (bonus=%.2f, 평균 강도=%.2f)",
                            as_of, trend_bonus, df["trend_strength"].mean())

        # ★ market_ratio 우선 적용 (예: "6:4" → 코스피 6 + 코스닥 4)
        if market_ratio is not None:
            try:
                ratio_parts = market_ratio.split(":")
                ratio_kospi = int(ratio_parts[0])
                ratio_kosdaq = int(ratio_parts[1])
                total_ratio = ratio_kospi + ratio_kosdaq
                n_kospi = round(top_n * ratio_kospi / total_ratio)
                n_kosdaq = top_n - n_kospi
            except Exception as e:
                logger.warning("market_ratio 파싱 실패: %s, market_split로 폴백", e)
                n_kospi = n_kosdaq = top_n // 2
            kospi = df[df["market"] == "KOSPI"].sort_values("rule_score", ascending=False).head(n_kospi)
            kosdaq = df[df["market"] == "KOSDAQ"].sort_values("rule_score", ascending=False).head(n_kosdaq)
            logger.info("[%s] 시장 비율 %s: KOSPI %d + KOSDAQ %d",
                        as_of, market_ratio, len(kospi), len(kosdaq))
            merged = pd.concat([kospi, kosdaq], ignore_index=True)
            return merged.sort_values("rule_score", ascending=False).reset_index(drop=True)

        if not market_split:
            return (df.sort_values("rule_score", ascending=False)
                      .head(top_n).reset_index(drop=True))

        # 시장별로 N//2씩 (KOSPI 5 + KOSDAQ 5)
        n_per_market = top_n // 2
        kospi = df[df["market"] == "KOSPI"].sort_values("rule_score", ascending=False).head(n_per_market)
        kosdaq = df[df["market"] == "KOSDAQ"].sort_values("rule_score", ascending=False).head(n_per_market)

        if len(kospi) < n_per_market:
            logger.warning("[%s] KOSPI 후보 부족: %d/%d", as_of, len(kospi), n_per_market)
        if len(kosdaq) < n_per_market:
            logger.warning("[%s] KOSDAQ 후보 부족: %d/%d", as_of, len(kosdaq), n_per_market)

        merged = pd.concat([kospi, kosdaq], ignore_index=True)
        return merged.sort_values("rule_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI 실행 — 단독 테스트용
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    s = RuleBasedScreener()
    picks = s.select_top_n(as_of="2024-12-30", top_n=10)
    if picks.empty:
        print("결과 없음 (데이터 수집이 먼저 필요할 수 있습니다)")
    else:
        cols = ["ticker", "name", "rule_score", "roe", "per",
                "revenue_growth_yoy", "profit_growth_yoy", "market_cap"]
        cols = [c for c in cols if c in picks.columns]
        print(picks[cols].to_string(index=False))
