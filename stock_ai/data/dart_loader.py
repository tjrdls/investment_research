"""
DART 재무제표 수집기
======================
당신의 하드 필터(ROE ≥ 30%, 매출/순이익 성장)를 위해 분기별 재무제표 수집.
DART API 키: https://opendart.fss.or.kr → .env 에 DART_API_KEY 설정
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import requests

from src.config import DB_PATH

logger = logging.getLogger(__name__)

DART_BASE = "https://opendart.fss.or.kr/api"

ACCOUNT_MAP = {
    "매출액": "revenue", "수익(매출액)": "revenue",
    "영업이익": "operating_income", "영업이익(손실)": "operating_income",
    "당기순이익": "net_income", "당기순이익(손실)": "net_income",
    "자본총계": "total_equity", "부채총계": "total_debt",
}
REPORT_CODES = {"Q1": "11013", "Q2": "11012", "Q3": "11014", "Q4": "11011"}


class DartLoader:
    def __init__(self, api_key: Optional[str] = None, db_path: Path = DB_PATH):
        self.api_key = api_key or os.environ.get("DART_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DART API 키 없음. .env에 DART_API_KEY 설정 필요\n"
                "발급: https://opendart.fss.or.kr"
            )
        self.db_path = Path(db_path)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS corp_codes (
                    ticker TEXT PRIMARY KEY,
                    corp_code TEXT NOT NULL, corp_name TEXT
                );
                CREATE TABLE IF NOT EXISTS financials (
                    ticker TEXT NOT NULL, year INTEGER NOT NULL,
                    quarter TEXT NOT NULL, period_end TEXT NOT NULL,
                    revenue REAL, operating_income REAL, net_income REAL,
                    total_equity REAL, total_debt REAL,
                    PRIMARY KEY (ticker, year, quarter)
                );
                CREATE INDEX IF NOT EXISTS idx_fin_period ON financials(period_end);
                CREATE TABLE IF NOT EXISTS fundamentals (
                    ticker TEXT NOT NULL, period_end TEXT NOT NULL,
                    roe REAL, debt_ratio REAL,
                    revenue_growth_yoy REAL, profit_growth_yoy REAL,
                    revenue_growth_qoq REAL, profit_growth_qoq REAL,
                    PRIMARY KEY (ticker, period_end)
                );
            """)

    def update_corp_codes(self) -> int:
        import io, zipfile
        import xml.etree.ElementTree as ET

        url = f"{DART_BASE}/corpCode.xml"
        r = requests.get(url, params={"crtfc_key": self.api_key}, timeout=30)
        r.raise_for_status()

        zf = zipfile.ZipFile(io.BytesIO(r.content))
        xml_data = zf.read(zf.namelist()[0])
        root = ET.fromstring(xml_data)

        rows = []
        for item in root.findall(".//list"):
            stock_code = (item.findtext("stock_code") or "").strip()
            if stock_code and len(stock_code) == 6:
                rows.append((
                    stock_code,
                    item.findtext("corp_code").strip(),
                    item.findtext("corp_name").strip(),
                ))

        with self._conn() as conn:
            conn.executemany("""
                INSERT INTO corp_codes (ticker, corp_code, corp_name)
                VALUES (?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    corp_code=excluded.corp_code, corp_name=excluded.corp_name
            """, rows)
        logger.info("corp_codes: %d개", len(rows))
        return len(rows)

    def _get_corp_code(self, ticker: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT corp_code FROM corp_codes WHERE ticker=?", (ticker,)
            ).fetchone()
        return row[0] if row else None

    def fetch_quarter(self, ticker: str, year: int, quarter: str) -> Optional[dict]:
        corp_code = self._get_corp_code(ticker)
        if not corp_code:
            return None

        params = {
            "crtfc_key": self.api_key, "corp_code": corp_code,
            "bsns_year": str(year), "reprt_code": REPORT_CODES[quarter],
            "fs_div": "OFS",
        }
        try:
            r = requests.get(f"{DART_BASE}/fnlttSinglAcntAll.json",
                             params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.debug("%s %d%s 실패: %s", ticker, year, quarter, e)
            return None

        if data.get("status") != "000":
            return None

        result = {}
        for item in data.get("list", []):
            account = item.get("account_nm", "").strip()
            our_col = ACCOUNT_MAP.get(account)
            if not our_col or our_col in result:
                continue
            try:
                val_str = item.get("thstrm_amount", "").replace(",", "")
                if val_str and val_str != "-":
                    result[our_col] = float(val_str)
            except (ValueError, AttributeError):
                continue
        return result if result else None

    def collect_ticker(self, ticker: str, start_year: int = 2015) -> int:
        from datetime import datetime
        current_year = datetime.now().year

        rows = []
        for year in range(start_year, current_year + 1):
            for q in ("Q1", "Q2", "Q3", "Q4"):
                with self._conn() as conn:
                    if conn.execute(
                        "SELECT 1 FROM financials WHERE ticker=? AND year=? AND quarter=?",
                        (ticker, year, q),
                    ).fetchone():
                        continue

                data = self.fetch_quarter(ticker, year, q)
                time.sleep(0.05)
                if not data:
                    continue

                rows.append((
                    ticker, year, q, self._quarter_end(year, q),
                    data.get("revenue"), data.get("operating_income"),
                    data.get("net_income"), data.get("total_equity"),
                    data.get("total_debt"),
                ))

        if not rows:
            return 0

        with self._conn() as conn:
            conn.executemany("""
                INSERT INTO financials (ticker, year, quarter, period_end,
                    revenue, operating_income, net_income, total_equity, total_debt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, year, quarter) DO NOTHING
            """, rows)
        return len(rows)

    @staticmethod
    def _quarter_end(year: int, quarter: str) -> str:
        return {
            "Q1": f"{year}-03-31", "Q2": f"{year}-06-30",
            "Q3": f"{year}-09-30", "Q4": f"{year}-12-31",
        }[quarter]

    def compute_fundamentals(self) -> int:
        """
        DART 분기 보고서의 누적값을 풀어서 진짜 분기 값을 만든 뒤,
        TTM (Trailing 12 Months) 기준으로 ROE / 성장률 계산.

        DART 분기 보고서의 함정:
          - Q1 = 1~3월 (1분기)
          - Q2 = 1~6월 누적 (반기)
          - Q3 = 1~9월 누적
          - Q4 = 1~12월 누적 (사업보고서)

        그래서 단순히 ×4 하거나 분기끼리 비교하면 결과가 부풀려짐.
        진짜 분기 값 = 누적 - 직전 누적 (Q1은 그대로).
        TTM = 최근 4분기 진짜 값의 합 = 진짜 연간 값.
        """
        with self._conn() as conn:
            df = pd.read_sql_query("""
                SELECT ticker, year, quarter, period_end,
                       revenue, operating_income, net_income,
                       total_equity, total_debt
                FROM financials ORDER BY ticker, period_end
            """, conn)

        if df.empty:
            logger.warning("financials 비어있음")
            return 0

        result_rows = []

        for tk, group in df.groupby("ticker"):
            g = group.sort_values("period_end").reset_index(drop=True)

            # 1) 누적 → 진짜 분기 값 변환 (Q2/Q3/Q4)
            #    Q1 또는 같은 연도 직전 분기가 있을 때만 빼기
            g["q_revenue"] = g["revenue"].astype(float)
            g["q_net_income"] = g["net_income"].astype(float)
            g["q_operating_income"] = g["operating_income"].astype(float)

            for i in range(len(g)):
                q = g.loc[i, "quarter"]
                if q == "Q1":
                    continue
                # 같은 연도의 직전 분기 찾기
                year = g.loc[i, "year"]
                prev_q = {"Q2": "Q1", "Q3": "Q2", "Q4": "Q3"}[q]
                prev_rows = g[(g["year"] == year) & (g["quarter"] == prev_q)]
                if len(prev_rows) == 0:
                    # 직전 분기 데이터 없으면 누적값 그대로 두지 말고 NaN으로
                    # (단순 누적값을 분기값처럼 쓰면 부풀려짐)
                    g.loc[i, "q_revenue"] = float("nan")
                    g.loc[i, "q_net_income"] = float("nan")
                    g.loc[i, "q_operating_income"] = float("nan")
                else:
                    prev = prev_rows.iloc[0]
                    if pd.notna(g.loc[i, "revenue"]) and pd.notna(prev["revenue"]):
                        g.loc[i, "q_revenue"] = g.loc[i, "revenue"] - prev["revenue"]
                    if pd.notna(g.loc[i, "net_income"]) and pd.notna(prev["net_income"]):
                        g.loc[i, "q_net_income"] = g.loc[i, "net_income"] - prev["net_income"]
                    if pd.notna(g.loc[i, "operating_income"]) and pd.notna(prev["operating_income"]):
                        g.loc[i, "q_operating_income"] = g.loc[i, "operating_income"] - prev["operating_income"]

            # 2) TTM (최근 4분기 합) 계산
            for i in range(len(g)):
                # 현재 분기 + 직전 3분기 (총 4분기) 의 진짜 분기값 합
                start = max(0, i - 3)
                window = g.iloc[start:i + 1]
                if len(window) < 4 or window["q_revenue"].isna().any():
                    ttm_revenue = None
                    ttm_net_income = None
                    ttm_operating_income = None
                else:
                    ttm_revenue = float(window["q_revenue"].sum())
                    ttm_net_income = float(window["q_net_income"].sum())
                    ttm_operating_income = float(window["q_operating_income"].sum()) \
                        if not window["q_operating_income"].isna().any() else None

                row = g.iloc[i]
                equity = row["total_equity"]
                debt = row["total_debt"]

                # ROE = TTM 순이익 / 자본 × 100  (×4 안 함, 이미 연간)
                roe = None
                if ttm_net_income is not None and equity and equity > 0:
                    roe = ttm_net_income / equity * 100
                    # 비현실적 값 클리핑 (자본잠식, 일회성 등)
                    if roe < -200 or roe > 200:
                        roe = None

                debt_ratio = None
                if equity and equity > 0 and debt is not None:
                    debt_ratio = debt / equity * 100

                # 영업이익률 = TTM 영업이익 / TTM 매출 (본업 수익성)
                operating_margin = None
                if ttm_operating_income is not None and ttm_revenue and ttm_revenue > 0:
                    operating_margin = ttm_operating_income / ttm_revenue * 100

                # 일회성 이익 비율 = TTM 순익 / TTM 영업이익
                # 정상: 0.5~1.0 (세금/금융비용 차감 후)
                # 1.5+ : 일회성 큰 비영업이익 의심
                # 음수: 영업적자
                profit_quality = None
                if ttm_operating_income and ttm_operating_income > 0 and ttm_net_income is not None:
                    profit_quality = ttm_net_income / ttm_operating_income

                # YoY 성장률: 다단계 fallback 로직
                # 1차: TTM YoY (가장 정확) — 현재 4분기 vs 1년 전 4분기
                # 2차: 같은 분기 YoY — 예: 2025 Q4 vs 2024 Q4 (TTM 불가 시)
                # 3차: 직전 가용 데이터 평균 — 일부 분기 누락 시
                rev_yoy = profit_yoy = oi_yoy = None

                # 1차: 표준 TTM YoY
                if i >= 4:
                    prev_start = max(0, (i - 4) - 3)
                    prev_window = g.iloc[prev_start:(i - 4) + 1]
                    if (len(prev_window) >= 4 
                        and not prev_window["q_revenue"].isna().any()
                        and ttm_revenue is not None):
                        prev_ttm_rev = float(prev_window["q_revenue"].sum())
                        prev_ttm_profit = float(prev_window["q_net_income"].sum())
                        rev_yoy = _growth(ttm_revenue, prev_ttm_rev)
                        profit_yoy = _growth(ttm_net_income, prev_ttm_profit)
                        if (not prev_window["q_operating_income"].isna().any()
                            and ttm_operating_income is not None):
                            prev_ttm_oi = float(prev_window["q_operating_income"].sum())
                            oi_yoy = _growth(ttm_operating_income, prev_ttm_oi)

                # 2차 fallback: 같은 분기 YoY (TTM 실패 시)
                current_q = row["quarter"]
                current_year = int(row["year"])
                same_q_prev = g[(g["year"] == current_year - 1) & (g["quarter"] == current_q)]
                if len(same_q_prev) > 0:
                    prev_row = same_q_prev.iloc[0]
                    if rev_yoy is None and pd.notna(row["revenue"]) and pd.notna(prev_row["revenue"]):
                        rev_yoy = _growth(row["revenue"], prev_row["revenue"])
                    if profit_yoy is None and pd.notna(row["net_income"]) and pd.notna(prev_row["net_income"]):
                        profit_yoy = _growth(row["net_income"], prev_row["net_income"])
                    if oi_yoy is None and pd.notna(row["operating_income"]) and pd.notna(prev_row["operating_income"]):
                        oi_yoy = _growth(row["operating_income"], prev_row["operating_income"])

                # 3차 fallback: 진짜 분기값(q_*) YoY (가용 데이터로)
                if i >= 4:
                    same_q_idx = i - 4
                    if same_q_idx >= 0:
                        prev_q_row = g.iloc[same_q_idx]
                        if rev_yoy is None and pd.notna(row["q_revenue"]) and pd.notna(prev_q_row["q_revenue"]):
                            rev_yoy = _growth(row["q_revenue"], prev_q_row["q_revenue"])
                        if profit_yoy is None and pd.notna(row["q_net_income"]) and pd.notna(prev_q_row["q_net_income"]):
                            profit_yoy = _growth(row["q_net_income"], prev_q_row["q_net_income"])
                        if oi_yoy is None and pd.notna(row["q_operating_income"]) and pd.notna(prev_q_row["q_operating_income"]):
                            oi_yoy = _growth(row["q_operating_income"], prev_q_row["q_operating_income"])

                # QoQ 성장률: 진짜 분기값 비교 (전분기 대비)
                rev_qoq = profit_qoq = None
                if i >= 1:
                    prev = g.iloc[i - 1]
                    rev_qoq = _growth(row["q_revenue"], prev["q_revenue"])
                    profit_qoq = _growth(row["q_net_income"], prev["q_net_income"])

                result_rows.append((
                    tk, row["period_end"], roe, debt_ratio,
                    rev_yoy, profit_yoy, rev_qoq, profit_qoq,
                    operating_margin, profit_quality, oi_yoy,
                ))

        with self._conn() as conn:
            # 기존 컬럼에 추가 컬럼 (없으면 추가)
            for col, typ in [
                ("operating_margin", "REAL"),
                ("profit_quality", "REAL"),
                ("operating_income_growth_yoy", "REAL"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE fundamentals ADD COLUMN {col} {typ}")
                except Exception:
                    pass  # 이미 존재
            conn.execute("DELETE FROM fundamentals")
            conn.executemany("""
                INSERT INTO fundamentals
                (ticker, period_end, roe, debt_ratio,
                 revenue_growth_yoy, profit_growth_yoy,
                 revenue_growth_qoq, profit_growth_qoq,
                 operating_margin, profit_quality, operating_income_growth_yoy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, result_rows)

        logger.info("fundamentals: %d행 (TTM ROE/YoY + 영업이익률 + 일회성 차단)", len(result_rows))
        return len(result_rows)


def _growth(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    """
    성장률 계산 (개선 버전).
    
    - 둘 다 양수: 정상 계산
    - prev가 음수 (적자→흑자 또는 적자 심화): abs(prev) 사용
    - prev가 0에 가까움: None 반환 (분모 폭주 방지)
    - 결과가 ±300% 넘으면 클리핑 (비현실적 값)
    """
    if curr is None or prev is None:
        return None
    # 분모가 0에 너무 가까우면 의미없음
    if abs(prev) < 1e-6:
        return None
    # 적자→흑자 같은 경우도 측정 (abs 사용)
    result = (curr - prev) / abs(prev) * 100
    # 비현실적 값 클리핑 (-300% ~ +500%)
    if result < -300:
        return -300.0
    if result > 500:
        return 500.0
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    dl = DartLoader()
    dl.update_corp_codes()
    dl.compute_fundamentals()
