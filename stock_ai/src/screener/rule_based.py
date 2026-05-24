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
                       revenue_growth_yoy, profit_growth_yoy,
                       revenue_growth_qoq, profit_growth_qoq,
                       operating_margin, profit_quality,
                       operating_income_growth_yoy
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
    # 자체 TTM PER 배치 계산 (KRX 공식 PER 대체용)
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
    # 자체 TTM ROE / 영업이익률 / 성장률 배치 계산
    # (KRX/DART 갱신 지연 + 누계 처리 버그 우회)
    # ------------------------------------------------------------------
    def _batch_ttm_fundamentals(self, tickers: list, as_of: str) -> dict:
        """배치로 TTM 기반 ROE/영업이익률/성장률/일회성 비율 계산.
        반환: {ticker: dict(roe, operating_margin, profit_quality,
                            revenue_growth_yoy, profit_growth_yoy,
                            operating_income_growth_yoy)}
        룩어헤드 안전: publish_date(=분기말+45/90일) <= as_of 만 사용.
        Q4 만 연 누계 가정으로 standalone 변환 (Q1/Q2/Q3 는 단독값).
        """
        import sqlite3
        from src.config import DB_PATH
        if not tickers:
            return {}
        as_of_ts = pd.Timestamp(as_of)
        ph = ",".join("?" * len(tickers))
        with sqlite3.connect(DB_PATH) as c:
            fin = pd.read_sql_query(
                f"SELECT ticker, year, quarter, period_end, "
                f"  revenue, operating_income, net_income, total_equity "
                f"FROM financials WHERE ticker IN ({ph})",
                c, params=tickers, parse_dates=["period_end"])
        if fin.empty:
            return {}
        lag = fin["quarter"].map(lambda q: 90 if q == "Q4" else 45)
        fin["publish_date"] = fin["period_end"] + pd.to_timedelta(lag, unit="D")
        fin = fin[fin["publish_date"] <= as_of_ts]
        if fin.empty:
            return {}

        def _standalone(grp: pd.DataFrame, col: str) -> pd.Series:
            """Q4 를 연 누계 → 단독으로 변환한 컬럼."""
            s = grp[col].astype(float).copy()
            for year in grp["year"].unique():
                yr = grp[grp["year"] == year]
                if "Q4" not in yr["quarter"].values:
                    continue
                q123 = yr[yr["quarter"].isin(["Q1", "Q2", "Q3"])]
                if len(q123) < 3 or q123[col].isna().any():
                    continue
                q4_idx = yr[yr["quarter"] == "Q4"].index[0]
                q4_val = yr[yr["quarter"] == "Q4"][col].iloc[0]
                if pd.isna(q4_val):
                    continue
                s.loc[q4_idx] = float(q4_val) - float(q123[col].sum())
            return s

        out = {}
        for tk, grp in fin.sort_values("period_end").groupby("ticker"):
            grp = grp.copy()
            grp["q_rev"] = _standalone(grp, "revenue")
            grp["q_ni"] = _standalone(grp, "net_income")
            grp["q_oi"] = _standalone(grp, "operating_income")

            if len(grp) < 4:
                continue

            last4 = grp.tail(4)
            ttm_rev = float(last4["q_rev"].sum()) if last4["q_rev"].notna().all() else None
            ttm_ni = float(last4["q_ni"].sum()) if last4["q_ni"].notna().all() else None
            ttm_oi = float(last4["q_oi"].sum()) if last4["q_oi"].notna().all() else None

            # 가장 최근 자본총계 (분기 보고서 BS 항목)
            equity = None
            eq_series = grp["total_equity"].dropna()
            if len(eq_series) > 0:
                equity = float(eq_series.iloc[-1])

            entry = {}

            # TTM 절대값 (실험용 — 영업이익 절대금액 랭킹 등)
            if ttm_oi is not None:
                entry["ttm_operating_income"] = ttm_oi
            if ttm_rev is not None:
                entry["ttm_revenue"] = ttm_rev
            if ttm_ni is not None:
                entry["ttm_net_income"] = ttm_ni

            # ROE
            if ttm_ni is not None and equity and equity > 0:
                roe = ttm_ni / equity * 100
                if -200 < roe < 200:
                    entry["roe"] = roe

            # 영업이익률
            if ttm_oi is not None and ttm_rev and ttm_rev > 0:
                entry["operating_margin"] = ttm_oi / ttm_rev * 100

            # 일회성 이익 비율
            if ttm_oi and ttm_oi > 0 and ttm_ni is not None:
                entry["profit_quality"] = ttm_ni / ttm_oi

            # YoY 성장률: 현 TTM vs 4분기 전 TTM
            # 분모 음수 (적자 → 흑자 턴어라운드) 도 abs 로 처리
            if len(grp) >= 8:
                prev4 = grp.iloc[-8:-4]
                if prev4["q_rev"].notna().all() and ttm_rev is not None:
                    prev_rev = float(prev4["q_rev"].sum())
                    if prev_rev != 0:
                        entry["revenue_growth_yoy"] = (ttm_rev - prev_rev) / abs(prev_rev) * 100
                if prev4["q_ni"].notna().all() and ttm_ni is not None:
                    prev_ni = float(prev4["q_ni"].sum())
                    if prev_ni != 0:
                        entry["profit_growth_yoy"] = (ttm_ni - prev_ni) / abs(prev_ni) * 100
                if prev4["q_oi"].notna().all() and ttm_oi is not None:
                    prev_oi = float(prev4["q_oi"].sum())
                    if prev_oi != 0:
                        entry["operating_income_growth_yoy"] = (ttm_oi - prev_oi) / abs(prev_oi) * 100

            if entry:
                out[tk] = entry

        return out

    # ------------------------------------------------------------------
    # 사업보고서 Q4 기준 연간 영업이익 성장률 (4-B 용)
    # ------------------------------------------------------------------
    def _batch_annual_op_growth(self, tickers: list, as_of: str) -> dict:
        """가장 최근 publish된 Q4 vs 그 이전 Q4 의 연간 영업이익(누계) 성장률.
        룩어헤드 안전 — publish_date = Q4말 + 90일 <= as_of 만 사용.
        반환: {ticker: growth_pct} (계산 불가 종목은 빠짐).
        """
        import sqlite3
        from src.config import DB_PATH
        if not tickers:
            return {}
        as_of_ts = pd.Timestamp(as_of)
        ph = ",".join("?" * len(tickers))
        with sqlite3.connect(DB_PATH) as c:
            fin = pd.read_sql_query(
                f"SELECT ticker, year, period_end, operating_income "
                f"FROM financials WHERE ticker IN ({ph}) AND quarter='Q4' "
                f"AND operating_income IS NOT NULL",
                c, params=tickers, parse_dates=["period_end"])
        if fin.empty:
            return {}
        # 사업보고서 공시 추정: Q4말 + 90일
        fin["publish_date"] = fin["period_end"] + pd.to_timedelta(90, unit="D")
        fin = fin[fin["publish_date"] <= as_of_ts]
        if fin.empty:
            return {}
        out = {}
        for tk, grp in fin.sort_values("year").groupby("ticker"):
            if len(grp) < 2:
                continue
            curr = float(grp["operating_income"].iloc[-1])
            prev = float(grp["operating_income"].iloc[-2])
            if prev == 0:
                continue
            out[tk] = (curr - prev) / abs(prev) * 100
        return out

    # ------------------------------------------------------------------
    # 하드 필터
    # ------------------------------------------------------------------
    def _apply_hard_filter(self, df: pd.DataFrame,
                            market_cap_min: Optional[float] = None,
                            operating_margin_min: float = 10.0,
                            profit_quality_min: float = 0.3,
                            profit_quality_max: float = 2.0,
                            operating_growth_required: bool = True,
                            revenue_growth_required: Optional[bool] = None,
                            profit_growth_required: Optional[bool] = None) -> pd.DataFrame:
        """
        시총, ROE, PER, 성장률, 영업이익률, 일회성 차단 필터.

        operating_margin_min: TTM 영업이익률 하한 (%). 기본 10%.
        profit_quality_min/max: TTM 순익/영업이익 비율 범위. 기본 0.3~2.0.
        operating_growth_required: 영업이익 성장률 양수 요구. 기본 True.
        revenue_growth_required/profit_growth_required: None 이면 CFG 기본값 사용.
            False 로 명시하면 해당 컷오프 건너뜀.
        """
        hf = CFG.hard_filter
        cap_min = market_cap_min if market_cap_min is not None else hf.market_cap_min_krw
        rev_req = hf.revenue_growth_required if revenue_growth_required is None else revenue_growth_required
        prof_req = hf.profit_growth_required if profit_growth_required is None else profit_growth_required
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
        if rev_req:
            df = df[df["revenue_growth_yoy"].fillna(-999) > 0]
        # 순이익 성장
        if prof_req:
            df = df[df["profit_growth_yoy"].fillna(-999) > 0]

        # 영업이익률
        if "operating_margin" in df.columns and operating_margin_min > 0:
            df = df[df["operating_margin"].fillna(-999) >= operating_margin_min]

        # 일회성 이익 차단
        if "profit_quality" in df.columns:
            df = df[
                (df["profit_quality"].notna()) &
                (df["profit_quality"] >= profit_quality_min) &
                (df["profit_quality"] <= profit_quality_max)
            ]

        # 영업이익 성장률
        if operating_growth_required and "operating_income_growth_yoy" in df.columns:
            df = df[df["operating_income_growth_yoy"].fillna(-999) > 0]

        logger.info("하드 필터: %d → %d개 (OPM≥%.0f%%, PQ %.1f~%.1f, rev_req=%s, prof_req=%s, op_g_req=%s)",
                    before, len(df), operating_margin_min, profit_quality_min, profit_quality_max,
                    rev_req, prof_req, operating_growth_required)
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
        use_ttm_fundamentals: bool = False,
        # ---- 실험: 1단계 매출성장 상위 필터 ----
        revenue_growth_top_pct: Optional[float] = None,    # 0.10 = 상위 10%
        revenue_growth_hard_cutoff: Optional[float] = None,  # 0.15 = +15% 이상
        revenue_growth_min_pool: int = 20,                   # 풀이 작으면 hard_cutoff 폴백
        # ---- 실험: 2단계 영업이익 결합 필터 ----
        secondary_filter_metric: Optional[str] = None,
        #   'op_amount' = 영업이익 절대금액 (2-A)
        #   'op_margin' = 영업이익률 (2-B)
        #   'op_growth' = 영업이익 성장률 YoY (2-B')
        secondary_top_n: Optional[int] = None,    # 2-A 용
        secondary_top_pct: Optional[float] = None,  # 2-B 용
        # ---- 실험: 3단계 멀티팩터 스코어링 ----
        scoring_mode: str = "default",            # 'default' | 'multifactor'
        multifactor_weights: tuple = (0.4, 0.4, 0.2),  # (rev_growth, op, per)
        multifactor_op_metric: str = "op_margin",      # 'op_margin' | 'op_growth'
        # ---- 실험 4: 하드필터 완화 + 연간 OP 성장 + 최종 정렬 기준 ----
        hard_filter_overrides: Optional[dict] = None,
        annual_op_growth_required: bool = False,  # 사업보고서 Q4 기준 연간 영업이익 성장 > 0
        annual_growth_threshold: Optional[float] = None,  # 13단계: 매출 OR 영익 TTM YoY ≥ X% (둘 중 하나)
        pick_by: str = "score",                   # 'score' | 'lowest_per'
        # ---- 실험 7: 고성장주 PER 프리미엄 ----
        # 매출 또는 영업이익 YoY 가 임계치 이상이면 'per_adj' 컬럼에서 PER 에 multiplier 곱해
        # pick_by='lowest_per' 정렬에서 우선순위 점프. 멀티팩터 PER 점수에도 per_adj 사용.
        growth_per_premium: Optional[dict] = None,  # 예: dict(threshold_pct=20.0, multiplier=0.5)
    ) -> pd.DataFrame:
        """
        as_of 시점에 점수 상위 N개 종목 반환.

        market_split=True 면 KOSPI N//2 + KOSDAQ N//2 고정.
        market_ratio (예: "6:4") 면 코스피:코스닥 = 6:4 비율로 분배.
            market_split보다 우선 적용.
        use_ttm_per=True 면 KRX 공식 PER 대신 우리가 DART 분기 데이터로
            계산한 TTM PER을 사용 (KRX의 연 1회 EPS 갱신 지연 우회).
        use_ttm_fundamentals=True 면 fundamentals 테이블의 사전계산 값 대신
            ROE/영업이익률/성장률을 그 자리에서 TTM 기반으로 다시 계산.
            (DART 누계 처리 + 공시지연 룩어헤드 안전 동시 해결.)

        실험 옵션 (2026-05-16):
          - revenue_growth_top_pct: 추세필터 후 매출성장 상위 X% 만 점수 계산.
            풀이 revenue_growth_min_pool 미만이면 hard_cutoff 로 폴백.
          - revenue_growth_hard_cutoff: 매출성장 YoY > 이 값(%) 만 통과.
          - secondary_filter_metric/top_n/top_pct: 1단계 후 영업이익 기반 추가 필터.
          - scoring_mode='multifactor': 기존 4팩터 점수 대신 (매출성장, 영업이익, PER) 가중평균.
        """
        df = self._fetch_universe(as_of)
        if use_ttm_per and not df.empty:
            # 자체 TTM PER로 덮어쓰기 (KRX EPS 갱신 지연 우회)
            from src.data.dart_loader import DartLoader
            dart = DartLoader()
            ttm_map = self._batch_ttm_per(dart, df["ticker"].tolist(), as_of)
            df["per"] = df["ticker"].map(ttm_map)
        if use_ttm_fundamentals and not df.empty:
            # 자체 TTM ROE/영업이익률/성장률로 덮어쓰기
            fmap = self._batch_ttm_fundamentals(df["ticker"].tolist(), as_of)
            for col in ("roe", "operating_margin", "profit_quality",
                        "revenue_growth_yoy", "profit_growth_yoy",
                        "operating_income_growth_yoy"):
                df[col] = df["ticker"].map(
                    {tk: e.get(col) for tk, e in fmap.items() if col in e}
                ).where(lambda s: s.notna(), df[col])
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
            hf_kwargs = dict(market_cap_min=cap_min_for_filter)
            if hard_filter_overrides:
                hf_kwargs.update(hard_filter_overrides)
            df = self._apply_hard_filter(df, **hf_kwargs)
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

        # ============================================================
        # ★ 4-B 전처리: 연간(Q4 사업보고서) 영업이익 성장률 > 0
        # ============================================================
        if annual_op_growth_required and not df.empty:
            before = len(df)
            ann_map = self._batch_annual_op_growth(df["ticker"].tolist(), as_of)
            df["_annual_op_growth"] = df["ticker"].map(ann_map)
            df = df[df["_annual_op_growth"].fillna(-999) > 0].reset_index(drop=True)
            logger.info("[%s] 연간 OP성장>0 필터: %d → %d", as_of, before, len(df))
            if df.empty:
                logger.warning("[%s] 연간 OP성장 통과 없음", as_of)
                return df

        # ============================================================
        # ★ 13단계: 연간 성장률 ≥ X% 하드 필터 (매출 OR 영업이익 둘 중 하나)
        # ============================================================
        if annual_growth_threshold is not None and not df.empty:
            before = len(df)
            rev_ok = df["revenue_growth_yoy"].fillna(-999) >= annual_growth_threshold
            op_ok = df["operating_income_growth_yoy"].fillna(-999) >= annual_growth_threshold
            df = df[rev_ok | op_ok].reset_index(drop=True)
            logger.info("[%s] 연간 성장률 ≥ %.0f%% (매출 또는 영익): %d → %d",
                        as_of, annual_growth_threshold, before, len(df))
            if df.empty:
                logger.warning("[%s] 연간 성장률 필터 통과 없음", as_of)
                return df

        # ============================================================
        # ★ 1단계 실험: 매출성장 상위 X% 또는 하드 컷오프
        # ============================================================
        if revenue_growth_top_pct is not None or revenue_growth_hard_cutoff is not None:
            before = len(df)
            rg = df["revenue_growth_yoy"].dropna()
            use_pct = (
                revenue_growth_top_pct is not None
                and len(rg) >= revenue_growth_min_pool
            )
            if use_pct:
                thresh = rg.quantile(1.0 - revenue_growth_top_pct)
                df = df[df["revenue_growth_yoy"].fillna(-999) >= thresh].reset_index(drop=True)
                logger.info("[%s] 1단계 매출성장 상위 %.0f%% (임계 %.1f%%): %d → %d",
                            as_of, revenue_growth_top_pct*100, thresh, before, len(df))
            elif revenue_growth_hard_cutoff is not None:
                df = df[df["revenue_growth_yoy"].fillna(-999) >= revenue_growth_hard_cutoff * 100].reset_index(drop=True)
                logger.info("[%s] 1단계 매출성장 하드컷 ≥%.1f%%: %d → %d",
                            as_of, revenue_growth_hard_cutoff*100, before, len(df))
            if df.empty:
                logger.warning("[%s] 1단계 통과 없음", as_of)
                return df

        # ============================================================
        # ★ 2단계 실험: 영업이익 결합 필터
        # ============================================================
        if secondary_filter_metric is not None:
            # 영업이익 절대금액(op_amount)은 fundamentals에 없으니 TTM 다시 조회
            if secondary_filter_metric == "op_amount":
                fmap = self._batch_ttm_fundamentals(df["ticker"].tolist(), as_of)
                df["_secondary"] = df["ticker"].map(
                    {tk: e.get("ttm_operating_income") for tk, e in fmap.items()}
                )
            elif secondary_filter_metric == "op_margin":
                df["_secondary"] = df["operating_margin"]
            elif secondary_filter_metric == "op_growth":
                df["_secondary"] = df["operating_income_growth_yoy"]
            else:
                raise ValueError(f"Unknown secondary_filter_metric: {secondary_filter_metric}")
            before = len(df)
            df = df[df["_secondary"].notna()].reset_index(drop=True)
            if secondary_top_n is not None:
                df = df.sort_values("_secondary", ascending=False).head(secondary_top_n).reset_index(drop=True)
                logger.info("[%s] 2단계 %s 상위 %d개: %d → %d",
                            as_of, secondary_filter_metric, secondary_top_n, before, len(df))
            elif secondary_top_pct is not None:
                thresh = df["_secondary"].quantile(1.0 - secondary_top_pct)
                df = df[df["_secondary"] >= thresh].reset_index(drop=True)
                logger.info("[%s] 2단계 %s 상위 %.0f%% (임계 %.2f): %d → %d",
                            as_of, secondary_filter_metric, secondary_top_pct*100, thresh, before, len(df))
            if df.empty:
                logger.warning("[%s] 2단계 통과 없음", as_of)
                return df

        # ============================================================
        # ★ 7단계: 고성장주 PER 프리미엄 — per_adj 컬럼 생성
        #   default scoring / multifactor scoring / pick_by 모두에 영향
        # ============================================================
        if growth_per_premium is not None and "per" in df.columns:
            thr = growth_per_premium.get("threshold_pct", 20.0)
            mul = growth_per_premium.get("multiplier", 0.5)
            rev_ok = df["revenue_growth_yoy"].fillna(-999) >= thr
            op_ok = df["operating_income_growth_yoy"].fillna(-999) >= thr
            is_growth = rev_ok | op_ok
            df["per_original"] = df["per"]
            df["per_adj"] = df["per"]
            df.loc[is_growth & df["per"].notna(), "per_adj"] = df.loc[is_growth & df["per"].notna(), "per"] * mul
            df["is_high_growth"] = is_growth
            # 점수 계산에서 per_adj 가 쓰이도록 per 자체를 임시 갈아끼움
            df["per"] = df["per_adj"]
            n_growth = int(is_growth.sum())
            if n_growth > 0:
                logger.info("[%s] PER 프리미엄: %d개 고성장주(≥%.0f%%) PER×%.2f → default/multifactor 점수에 반영",
                            as_of, n_growth, thr, mul)

        # ============================================================
        # ★ 3단계 실험: 멀티팩터 스코어링 (default 가 아닐 때만)
        # ============================================================
        if scoring_mode == "multifactor":
            w_rev, w_op, w_per = multifactor_weights
            # 매출성장률 점수 (높을수록 좋음)
            s_rev = df["revenue_growth_yoy"].rank(pct=True, na_option="bottom")
            # 영업이익 점수
            if multifactor_op_metric == "op_margin":
                op_series = df["operating_margin"]
            elif multifactor_op_metric == "op_growth":
                op_series = df["operating_income_growth_yoy"]
            else:
                raise ValueError(f"Unknown multifactor_op_metric: {multifactor_op_metric}")
            s_op = op_series.rank(pct=True, na_option="bottom")
            # PER 점수 (낮을수록 좋음)
            s_per = 1 - df["per"].rank(pct=True, na_option="bottom")
            total = w_rev + w_op + w_per
            weighted = (w_rev * s_rev + w_op * s_op + w_per * s_per) / total
            df["rule_score"] = (weighted * 100).round(2)
            # 분해 점수도 컬럼으로 저장 (디버깅용)
            df["revenue_score"] = (s_rev * 100).round(1)
            df["op_score"] = (s_op * 100).round(1)
            df["per_score"] = (s_per * 100).round(1)
            df["roe_score"] = None
            df["profit_score"] = None
            logger.info("[%s] 3단계 멀티팩터 (w_rev=%.2f, w_op=%.2f, w_per=%.2f, op=%s)",
                        as_of, w_rev, w_op, w_per, multifactor_op_metric)
        else:
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

        # ★ pick_by='lowest_per' (4-B 워크플로우): 최종 정렬 기준을 PER 오름차순으로
        # (per_adj 가 있으면 위에서 per 가 이미 per_adj 로 갈아끼워졌으므로 그대로 per 사용)
        if pick_by == "lowest_per":
            per_col = "per_adj" if "per_adj" in df.columns else "per"
            valid = df[df[per_col].notna() & (df[per_col] > 0)]
            return (valid.sort_values(per_col, ascending=True)
                         .head(top_n).reset_index(drop=True))

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
