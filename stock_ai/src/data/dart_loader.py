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
    # 매출 — SK하이닉스/카카오/LG화학 등은 "영업수익" / "매출" 로 보고
    "매출액": "revenue", "수익(매출액)": "revenue", "영업수익": "revenue",
    "매출": "revenue",  # LG화학 케이스
    # 영업이익 — 호텔신라는 "영업손익"
    "영업이익": "operating_income", "영업이익(손실)": "operating_income",
    "영업손익": "operating_income",
    # 순이익 변형 — 호텔신라 "분기순손익", 코오롱인더 "당기순이익(손실)"
    "당기순이익": "net_income", "당기순이익(손실)": "net_income",
    "분기순이익": "net_income", "분기순이익(손실)": "net_income",
    "반기순이익": "net_income", "반기순이익(손실)": "net_income",
    "분기순손익": "net_income", "당기순손익": "net_income", "반기순손익": "net_income",
    "자본총계": "total_equity", "부채총계": "total_debt",
}
# 로마자/숫자 접두 strip (예: "V.영업손익" → "영업손익", "I. 매출액" → "매출액", "Ⅴ.영업이익" → "영업이익")
import re as _re
_PREFIX_RE = _re.compile(r"^[\sIVXLCDMⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ\d]+\.?\s*")
def _normalize_account(nm: str) -> str:
    if not nm:
        return ""
    stripped = _PREFIX_RE.sub("", nm).strip()
    return stripped if stripped else nm.strip()
# 손익계산서 (Income Statement)만 채택 — 같은 계정명이 CF/SCE에 중복될 수 있음
IS_SJ_DIVS = {"IS", "CIS"}
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
            raw_account = item.get("account_nm", "").strip()
            # 로마자/숫자 접두 정규화 (호텔신라/카카오뱅크/고려아연 등)
            account = _normalize_account(raw_account)
            our_col = ACCOUNT_MAP.get(account) or ACCOUNT_MAP.get(raw_account)
            if not our_col or our_col in result:
                continue
            # 손익계산서 항목만 — BS/CF/SCE의 중복 항목 회피 (분기순이익이 SCE에선 누계라 위험)
            if our_col in ("revenue", "operating_income", "net_income"):
                if item.get("sj_div", "") not in IS_SJ_DIVS:
                    continue
            try:
                val_str = item.get("thstrm_amount", "").replace(",", "")
                if val_str and val_str != "-":
                    result[our_col] = float(val_str)
            except (ValueError, AttributeError):
                continue
        return result if result else None

    def compute_ttm_per(
        self, ticker: str, as_of: str,
        publish_lag_q123: int = 45, publish_lag_q4: int = 90,
    ) -> Optional[float]:
        """ticker의 as_of 시점 자체 TTM PER (시총 / TTM 순이익).
        공시일 추정: 분기말 + 45일 (Q1-Q3), 연말 + 90일 (Q4).
        룩어헤드 안전 — as_of 시점 알 수 있는 가장 최근 4분기만 사용.
        반환: float (PER) 또는 None (데이터 부족).
        """
        import pandas as pd
        with self._conn() as conn:
            fin = pd.read_sql_query(
                "SELECT year, quarter, period_end, net_income FROM financials "
                "WHERE ticker=? AND net_income IS NOT NULL AND net_income != 0 "
                "ORDER BY period_end",
                conn, params=[ticker], parse_dates=["period_end"])
        if fin.empty:
            return None
        # 공시 추정일 계산
        lag = fin["quarter"].map(lambda q: publish_lag_q4 if q == "Q4" else publish_lag_q123)
        fin["publish_date"] = fin["period_end"] + pd.to_timedelta(lag, unit="D")
        as_of_ts = pd.Timestamp(as_of)
        avail = fin[fin["publish_date"] <= as_of_ts].copy()
        if len(avail) < 4:
            return None
        # Q4는 연간 누계로 저장되어 있음 → standalone 변환
        avail["ni_std"] = avail["net_income"].astype(float)
        for year in avail["year"].unique():
            year_rows = avail[avail["year"] == year]
            if "Q4" not in year_rows["quarter"].values:
                continue
            q123 = year_rows[year_rows["quarter"].isin(["Q1", "Q2", "Q3"])]
            if q123.empty:
                continue                            # Q4만 있으면 그대로 사용 (구식 데이터)
            q4_idx = year_rows[year_rows["quarter"] == "Q4"].index[0]
            avail.loc[q4_idx, "ni_std"] = (
                float(year_rows[year_rows["quarter"] == "Q4"]["net_income"].iloc[0])
                - float(q123["net_income"].sum())
            )
        # 가장 최근 4분기
        last4 = avail.sort_values("publish_date").tail(4)
        if len(last4) < 4:
            return None
        ttm_ni = float(last4["ni_std"].sum())
        if ttm_ni <= 0:
            return None                              # 적자면 PER 의미 없음
        # 시가총액
        with self._conn() as conn:
            mc = conn.execute(
                "SELECT market_cap FROM market_cap "
                "WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT 1",
                (ticker, as_of)).fetchone()
        if not mc or not mc[0]:
            return None
        return float(mc[0]) / ttm_ni

    def refetch_missing_net_income(self, tickers: list[str]) -> dict:
        """net_income 결손 행만 UPDATE 모드로 재수집.
        반환: {ok, fail, updated} 통계.
        """
        ok = fail = updated = 0
        for ticker in tickers:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT year, quarter FROM financials "
                    "WHERE ticker=? AND quarter IN ('Q1','Q2','Q3') "
                    "AND (net_income IS NULL OR net_income=0) "
                    "ORDER BY year, quarter",
                    (ticker,)).fetchall()
            for year, q in rows:
                data = self.fetch_quarter(ticker, year, q)
                time.sleep(0.05)
                if not data:
                    fail += 1
                    continue
                ni = data.get("net_income")
                if ni is None or ni == 0:
                    fail += 1
                    continue
                with self._conn() as conn:
                    conn.execute(
                        "UPDATE financials SET net_income=? "
                        "WHERE ticker=? AND year=? AND quarter=?",
                        (float(ni), ticker, year, q))
                updated += 1
            ok += 1
        return {"tickers_scanned": ok, "updated_rows": updated, "fetch_fail": fail}

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
        DART 분기 보고서를 단독 분기값으로 정규화한 뒤,
        TTM (Trailing 12 Months) 기준으로 ROE / 성장률 계산.

        실측 패턴 (Samsung·SK하이닉스·현대차·셀트리온 검증):
          - Q1/Q2/Q3 → thstrm_amount 가 이미 단독 분기값으로 들어옴
          - Q4 (사업보고서)만 연간 누계 → standalone = Q4 - (Q1+Q2+Q3)

        TTM = 최근 4개 단독 분기 합 = 진짜 연간 값.
        compute_ttm_per 과 동일한 정규화 로직.
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

            # 1) Q4 만 standalone 변환 (Q1/Q2/Q3는 이미 단독)
            g["q_revenue"] = g["revenue"].astype(float)
            g["q_net_income"] = g["net_income"].astype(float)
            g["q_operating_income"] = g["operating_income"].astype(float)

            for i in range(len(g)):
                if g.loc[i, "quarter"] != "Q4":
                    continue
                year = g.loc[i, "year"]
                q123 = g[(g["year"] == year) & (g["quarter"].isin(["Q1", "Q2", "Q3"]))]
                if len(q123) < 3:
                    # 같은 연도 Q1~Q3 일부 결손 → Q4 누계를 단독으로 쓰면 폭증
                    # 안전하게 NaN 처리
                    g.loc[i, "q_revenue"] = float("nan")
                    g.loc[i, "q_net_income"] = float("nan")
                    g.loc[i, "q_operating_income"] = float("nan")
                    continue
                if pd.notna(g.loc[i, "revenue"]) and q123["revenue"].notna().all():
                    g.loc[i, "q_revenue"] = g.loc[i, "revenue"] - q123["revenue"].sum()
                else:
                    g.loc[i, "q_revenue"] = float("nan")
                if pd.notna(g.loc[i, "net_income"]) and q123["net_income"].notna().all():
                    g.loc[i, "q_net_income"] = g.loc[i, "net_income"] - q123["net_income"].sum()
                else:
                    g.loc[i, "q_net_income"] = float("nan")
                if pd.notna(g.loc[i, "operating_income"]) and q123["operating_income"].notna().all():
                    g.loc[i, "q_operating_income"] = g.loc[i, "operating_income"] - q123["operating_income"].sum()
                else:
                    g.loc[i, "q_operating_income"] = float("nan")

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

                # YoY 성장률: 현재 TTM vs 4분기 전 TTM
                rev_yoy = profit_yoy = oi_yoy = None
                if i >= 4:
                    prev_start = max(0, (i - 4) - 3)
                    prev_window = g.iloc[prev_start:(i - 4) + 1]
                    if len(prev_window) >= 4 and not prev_window["q_revenue"].isna().any():
                        prev_ttm_rev = float(prev_window["q_revenue"].sum())
                        prev_ttm_profit = float(prev_window["q_net_income"].sum())
                        rev_yoy = _growth(ttm_revenue, prev_ttm_rev)
                        profit_yoy = _growth(ttm_net_income, prev_ttm_profit)
                        if not prev_window["q_operating_income"].isna().any():
                            prev_ttm_oi = float(prev_window["q_operating_income"].sum())
                            if ttm_operating_income is not None:
                                oi_yoy = _growth(ttm_operating_income, prev_ttm_oi)

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
    """YoY 성장률. 분모 음수도 처리 (적자→흑자 같은 턴어라운드 캡처).
    prev=0 일 때만 None (0 으로 나누기 불가).
    """
    if curr is None or prev is None or prev == 0:
        return None
    return (curr - prev) / abs(prev) * 100


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    dl = DartLoader()
    dl.update_corp_codes()
    dl.compute_fundamentals()
