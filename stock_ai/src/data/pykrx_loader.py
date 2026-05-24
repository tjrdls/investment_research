"""
PYKRX OHLCV + 시가총액 수집기
==============================
- KOSPI + KOSDAQ 전 종목 일봉 (2000~현재)
- 시가총액 (당신의 1조원 필터를 위해)
- 증분 수집 + 재시도 + 요청 간격 조절
"""
from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

try:
    from pykrx import stock
except ImportError:
    raise ImportError("pykrx 미설치. `pip install pykrx`")

from src.config import CFG, DB_PATH

# pykrx 라이브러리가 root logger로 직접 호출하면서 args 포맷팅 버그를 일으키므로
# pykrx 관련 root 로거의 INFO를 WARNING 이상으로만 보이게 한다.
logging.getLogger("pykrx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class PyKrxLoader:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
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
                CREATE TABLE IF NOT EXISTS tickers (
                    ticker TEXT PRIMARY KEY, name TEXT, market TEXT
                );
                CREATE TABLE IF NOT EXISTS ohlcv (
                    ticker TEXT NOT NULL, date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
                    PRIMARY KEY (ticker, date)
                );
                CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv(date);
                CREATE TABLE IF NOT EXISTS market_cap (
                    ticker TEXT NOT NULL, date TEXT NOT NULL,
                    market_cap REAL, shares INTEGER,
                    PRIMARY KEY (ticker, date)
                );
                CREATE INDEX IF NOT EXISTS idx_mcap_date ON market_cap(date);
            """)

    def _find_recent_trading_day(
        self, max_lookback: int = 14, anchor: Optional[str] = None
    ) -> str:
        """
        anchor(기준일)부터 거꾸로 최근 거래일을 찾는다.
        anchor가 None이면 시스템 오늘 날짜를 사용. 주말 자동 스킵.
        anchor 형식: 'YYYY-MM-DD' 또는 'YYYYMMDD'.
        """
        if anchor:
            anchor = anchor.replace("-", "")
            base = datetime.strptime(anchor, "%Y%m%d")
        else:
            base = datetime.now()

        for d in range(max_lookback):
            cand_dt = base - timedelta(days=d)
            # 토(5) 일(6) 즉시 스킵
            if cand_dt.weekday() >= 5:
                continue
            cand = cand_dt.strftime("%Y%m%d")
            try:
                tickers = stock.get_market_ticker_list(cand, market="KOSPI")
                if tickers and len(tickers) > 100:  # 평일이면 KOSPI 800+ 종목
                    return cand
            except Exception:
                continue
        # 폴백: 직전 평일
        d = base
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d.strftime("%Y%m%d")

    def update_universe(self, anchor: Optional[str] = None) -> int:
        """현재 상장된 KOSPI + KOSDAQ 종목 갱신.
        anchor='2024-12-30' 처럼 명시하면 시스템 시계와 무관하게 그 날짜 기준으로 조회.
        """
        ref_date = self._find_recent_trading_day(anchor=anchor)
        logger.info("거래일 기준일: %s", ref_date)

        rows = []
        for market in ("KOSPI", "KOSDAQ"):
            tickers = []
            try:
                tickers = stock.get_market_ticker_list(ref_date, market=market)
            except Exception as e:
                logger.warning("%s 조회 실패: %s", market, e)
                continue
            if not tickers:
                logger.warning("%s 종목 0개 (날짜: %s) — 비거래일일 수 있음", market, ref_date)
                continue

            logger.info("%s: %d개 종목 발견 — 종목명 조회 중...", market, len(tickers))
            for tk in tickers:
                try:
                    name = stock.get_market_ticker_name(tk)
                except Exception:
                    name = None
                rows.append((tk, name, market))

        if not rows:
            logger.error(
                "종목 0개 — KRX 서버 일시 장애일 수 있습니다. 잠시 후 다시 시도하세요."
            )
            return 0

        with self._conn() as conn:
            conn.executemany("""
                INSERT INTO tickers (ticker, name, market) VALUES (?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    name=excluded.name, market=excluded.market
            """, rows)
        logger.info("✓ 종목 유니버스: %d개 등록", len(rows))
        return len(rows)

    def list_tickers(self) -> list[tuple[str, str, str]]:
        with self._conn() as conn:
            return list(conn.execute(
                "SELECT ticker, name, market FROM tickers ORDER BY ticker"
            ))

    def _last_ohlcv_date(self, ticker: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM ohlcv WHERE ticker=?", (ticker,)
            ).fetchone()
        return row[0].replace("-", "") if row and row[0] else None

    def _fetch_with_retry(self, fn, *args, **kwargs) -> Optional[pd.DataFrame]:
        for attempt in range(CFG.data.pykrx_retry):
            try:
                df = fn(*args, **kwargs)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.debug("재시도 %d: %s", attempt, e)
            time.sleep(0.5 * (attempt + 1))
        return None

    def collect_one(self, ticker: str, start: str, end: str) -> int:
        """단일 종목 OHLCV + 시총 증분 수집."""
        last = self._last_ohlcv_date(ticker)
        eff_start = start
        if last and last >= start:
            eff_start = (datetime.strptime(last, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        if eff_start > end:
            return 0

        df_ohlcv = self._fetch_with_retry(stock.get_market_ohlcv, eff_start, end, ticker)
        if df_ohlcv is None or df_ohlcv.empty:
            return 0
        df_cap = self._fetch_with_retry(stock.get_market_cap, eff_start, end, ticker)

        ohlcv_rows = [
            (ticker, idx.strftime("%Y-%m-%d"),
             _f(r.get("시가")), _f(r.get("고가")), _f(r.get("저가")),
             _f(r.get("종가")), _i(r.get("거래량")))
            for idx, r in df_ohlcv.iterrows()
        ]
        cap_rows = []
        if df_cap is not None and not df_cap.empty:
            cap_rows = [
                (ticker, idx.strftime("%Y-%m-%d"),
                 _f(r.get("시가총액")), _i(r.get("상장주식수")))
                for idx, r in df_cap.iterrows()
            ]

        with self._conn() as conn:
            conn.executemany("""
                INSERT INTO ohlcv (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date) DO NOTHING
            """, ohlcv_rows)
            if cap_rows:
                conn.executemany("""
                    INSERT INTO market_cap (ticker, date, market_cap, shares)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(ticker, date) DO NOTHING
                """, cap_rows)
        return len(ohlcv_rows)

    def collect_all(self, start: str = None, end: str = None) -> dict:
        start = (start or CFG.data.ohlcv_start).replace("-", "")
        # end가 미래일 수 있으니 가장 최근 거래일로 클램프
        if end is None:
            end = self._find_recent_trading_day()
        else:
            end = end.replace("-", "")
            today = datetime.now().strftime("%Y%m%d")
            if end > today:
                end = self._find_recent_trading_day()
                logger.info("end 날짜를 최근 거래일로 보정: %s", end)

        tickers = self.list_tickers()
        if not tickers:
            raise RuntimeError("먼저 update_universe() 호출 필요")

        stats = {"total": len(tickers), "ok": 0, "fail": 0, "rows": 0}
        t0 = time.time()

        for i, (tk, name, _) in enumerate(tickers, 1):
            try:
                n = self.collect_one(tk, start, end)
                stats["ok"] += 1
                stats["rows"] += n
            except Exception as e:
                logger.warning("%s 실패: %s", tk, e)
                stats["fail"] += 1

            if i % 50 == 0:
                el = time.time() - t0
                eta = (len(tickers) - i) * (el / i) / 60
                logger.info("[%d/%d] %s — 실패 %d, ETA %.1f분",
                           i, len(tickers), name or tk, stats["fail"], eta)

            time.sleep(CFG.data.pykrx_request_sleep)

        stats["elapsed_sec"] = time.time() - t0
        logger.info("수집 완료: %s", stats)
        return stats


def _f(v) -> Optional[float]:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v) -> Optional[int]:
    try:
        if v is None or pd.isna(v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    loader = PyKrxLoader()
    print(">>> 종목 유니버스 갱신")
    loader.update_universe()
    print(">>> OHLCV + 시총 수집")
    print(loader.collect_all())
