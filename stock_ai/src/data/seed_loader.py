"""시드 로더 — KRX OpenAPI 승인 전 임시 데이터 적재
====================================================
KRX OpenAPI '활용신청 승인'을 기다리는 동안, **하드코딩한 소수 우량주**만으로
cache.db 를 채워 전체 파이프라인(필터·점수·앙상블·백테스트·대시보드)을 돌려볼 수 있게 한다.

데이터 소스 (모두 로그인/승인 불필요):
  - OHLCV     : pykrx `get_market_ohlcv(start, end, ticker)`  (종목별 — 게이트 안 걸림)
  - 상장주식수 : yfinance `Ticker("{code}.KS|.KQ").fast_info.shares`  (없으면 하드코딩 폴백)
  - 시가총액   : 일별 종가 × 상장주식수 (근사)
  - 벤치마크/금 : 069500·132030 ETF OHLCV (daily_update._backfill_etfs)

승인되면 `python main.py collect` (KRX OpenAPI 전종목) 로 대체하면 된다.
이 모듈은 cache.db 스키마(pykrx_loader 와 동일)를 그대로 채우므로 이후 collect 와 충돌 없음.
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

from src.config import DB_PATH

logger = logging.getLogger(__name__)

# ── 하드코딩 유니버스 (ticker, 종목명, 시장) ──────────────────
# 대형 + 우량 위주로 KOSPI/KOSDAQ 를 섞어, 코스피5+코스닥5 선정이 가능하도록 구성.
SEED_UNIVERSE: list[tuple[str, str, str]] = [
    # ── KOSPI ──
    ("005930", "삼성전자", "KOSPI"),
    ("000660", "SK하이닉스", "KOSPI"),
    ("005380", "현대차", "KOSPI"),
    ("000270", "기아", "KOSPI"),
    ("005490", "POSCO홀딩스", "KOSPI"),
    ("035420", "NAVER", "KOSPI"),
    ("035720", "카카오", "KOSPI"),
    ("051910", "LG화학", "KOSPI"),
    ("006400", "삼성SDI", "KOSPI"),
    ("207940", "삼성바이오로직스", "KOSPI"),
    ("373220", "LG에너지솔루션", "KOSPI"),
    ("068270", "셀트리온", "KOSPI"),
    ("012330", "현대모비스", "KOSPI"),
    ("066570", "LG전자", "KOSPI"),
    ("105560", "KB금융", "KOSPI"),
    ("028260", "삼성물산", "KOSPI"),
    # ── KOSDAQ ──
    ("247540", "에코프로비엠", "KOSDAQ"),
    ("086520", "에코프로", "KOSDAQ"),
    ("196170", "알테오젠", "KOSDAQ"),
    ("058470", "리노공업", "KOSDAQ"),
    ("357780", "솔브레인", "KOSDAQ"),
    ("240810", "원익IPS", "KOSDAQ"),
    ("263750", "펄어비스", "KOSDAQ"),
    ("293490", "카카오게임즈", "KOSDAQ"),
    ("145020", "휴젤", "KOSDAQ"),
    ("042700", "한미반도체", "KOSDAQ"),
]

# yfinance 가 실패할 때만 쓰는 상장주식수 폴백 (근사값, 시총 추정용)
SHARES_FALLBACK: dict[str, int] = {
    "005930": 5_969_782_550,
    "000660": 728_002_365,
    "005380": 209_416_191,
    "000270": 405_363_347,
    "005490": 84_571_230,
    "035420": 146_000_000,
    "035720": 445_000_000,
    "051910": 70_592_343,
    "006400": 68_764_530,
    "207940": 71_174_000,
    "373220": 234_000_000,
    "068270": 217_021_190,
    "247540": 97_801_344,
    "086520": 26_645_445,
}


class SeedLoader:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._yf_warned = False

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.db_path, timeout=30)
        c.execute("PRAGMA journal_mode=WAL")
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
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

    # ── 상장주식수 ────────────────────────────────────────
    def _shares(self, ticker: str, market: str) -> Optional[int]:
        suffix = "KS" if market == "KOSPI" else "KQ"
        try:
            import yfinance as yf
            fi = yf.Ticker(f"{ticker}.{suffix}").fast_info
            s = getattr(fi, "shares", None)
            if s:
                return int(s)
        except Exception as e:  # noqa: BLE001
            if not self._yf_warned:
                logger.warning("yfinance 상장주식수 조회 실패 (%s) — 폴백 사용: %s", ticker, e)
                self._yf_warned = True
        return SHARES_FALLBACK.get(ticker)

    # ── 한 종목 적재 ──────────────────────────────────────
    def collect_one(self, ticker: str, name: str, market: str,
                    start: str, end: str) -> dict:
        ymd_s, ymd_e = start.replace("-", ""), end.replace("-", "")
        df = _fetch_with_retry(stock.get_market_ohlcv, ymd_s, ymd_e, ticker)
        if df is None or df.empty:
            return {"ticker": ticker, "ohlcv_rows": 0, "cap_rows": 0, "ok": False}

        shares = self._shares(ticker, market)
        ohlcv_rows, cap_rows = _rows_from_ohlcv(ticker, df, shares)

        with self._conn() as c:
            c.execute(
                "INSERT INTO tickers (ticker, name, market) VALUES (?, ?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET name=excluded.name, market=excluded.market",
                (ticker, name, market),
            )
            c.executemany(
                "INSERT INTO ohlcv (ticker, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(ticker, date) DO NOTHING",
                ohlcv_rows,
            )
            if cap_rows:
                c.executemany(
                    "INSERT INTO market_cap (ticker, date, market_cap, shares) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(ticker, date) DO NOTHING",
                    cap_rows,
                )
        return {"ticker": ticker, "name": name, "ohlcv_rows": len(ohlcv_rows),
                "cap_rows": len(cap_rows), "shares": shares, "ok": True}

    # ── 전체 시드 적재 ────────────────────────────────────
    def collect(self, start: str = "2015-01-01", end: Optional[str] = None,
                universe: Optional[list] = None, progress_cb=None) -> dict:
        end = end or _yesterday().strftime("%Y-%m-%d")
        uni = universe or SEED_UNIVERSE
        total = len(uni)
        stats = {"start": start, "end": end, "tickers": 0, "ohlcv_rows": 0,
                 "cap_rows": 0, "no_shares": [], "failed": []}
        t0 = time.time()
        logger.info("시드 적재: %d종목 · %s ~ %s", total, start, end)

        for i, (tk, name, market) in enumerate(uni, 1):
            if progress_cb:
                progress_cb(i - 1, total, f"{name}({tk})")
            try:
                r = self.collect_one(tk, name, market, start, end)
            except Exception as e:  # noqa: BLE001
                logger.warning("%s(%s) 실패: %s", name, tk, e)
                stats["failed"].append(tk)
                continue
            if not r["ok"]:
                stats["failed"].append(tk)
                continue
            stats["tickers"] += 1
            stats["ohlcv_rows"] += r["ohlcv_rows"]
            stats["cap_rows"] += r["cap_rows"]
            if not r["cap_rows"]:
                stats["no_shares"].append(tk)
            logger.info("[%d/%d] %s: OHLCV %d행%s", i, total, name, r["ohlcv_rows"],
                        "" if r["cap_rows"] else " (시총 없음 — 주식수 미확보)")
            time.sleep(0.1)

        if progress_cb:
            progress_cb(total, total, "완료")
        stats["elapsed_sec"] = time.time() - t0
        return stats


# ── 모듈 헬퍼 (순수 — 단위 테스트 가능) ─────────────────────
def _rows_from_ohlcv(ticker: str, df: pd.DataFrame, shares: Optional[int]):
    """pykrx OHLCV DataFrame → (ohlcv_rows, cap_rows). 시총 = 종가 × shares."""
    ohlcv_rows, cap_rows = [], []
    for idx, r in df.iterrows():
        iso = idx.strftime("%Y-%m-%d")
        close = _f(r.get("종가"))
        ohlcv_rows.append((
            ticker, iso,
            _f(r.get("시가")), _f(r.get("고가")), _f(r.get("저가")),
            close, _i(r.get("거래량")),
        ))
        if shares and close is not None:
            cap_rows.append((ticker, iso, float(close) * float(shares), int(shares)))
    return ohlcv_rows, cap_rows


def _fetch_with_retry(fn, *args, retries: int = 3, sleep: float = 0.5, **kwargs):
    for i in range(retries):
        try:
            df = fn(*args, **kwargs)
            if df is not None and not df.empty:
                return df
        except Exception as e:  # noqa: BLE001
            logger.debug("retry %d: %s", i, e)
        time.sleep(sleep * (i + 1))
    return None


def _f(v) -> Optional[float]:
    try:
        if v is None or pd.isna(v):
            return None
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _i(v) -> Optional[int]:
    f = _f(v)
    return int(f) if f is not None else None


def _yesterday():
    d = datetime.now().date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    except ImportError:
        pass
    print(SeedLoader().collect(start="2024-01-01"))
