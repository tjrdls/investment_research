"""일별 증분 데이터 업데이트 (대시보드용)
====================================
DB 의 마지막 날짜부터 어제(또는 지정일)까지 OHLCV + 시가총액을 빠르게 채워넣는다.

데이터 소스 (2026 KRX 로그인 게이트 대응):
  - 주식(KOSPI/KOSDAQ): **KRX OpenAPI** (`KRX_AUTH_KEY`) — 날짜-bulk, 시총·상장주식수 포함
  - ETF/지수(069500 벤치마크 · 132030 금헤지): **pykrx 종목별 조회** (로그인 없이 동작)

종목별 수집(`PyKrxLoader.collect_all`)보다 빠르고, 대시보드 "데이터 업데이트" 버튼이
호출하는 `update_to` / `status_summary` 의 시그니처·반환 계약은 그대로 유지한다.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

try:
    from pykrx import stock
except ImportError:
    raise ImportError("pykrx 미설치. `pip install pykrx`")

from src.config import DB_PATH
from src.data.krx_openapi_loader import KrxApiError, KrxOpenApiLoader

logger = logging.getLogger(__name__)

# 백테스트 정합성에 필수인 ETF/지수 (벤치마크·금헤지). pykrx 종목별 조회로 받는다.
ESSENTIAL_ETFS = ["069500", "132030"]


@contextmanager
def _conn(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(db_path, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def get_db_max_date(db_path: Path = DB_PATH) -> Optional[str]:
    """OHLCV 테이블의 최대 날짜 (YYYY-MM-DD). 없으면 None."""
    if not Path(db_path).exists():
        return None
    try:
        with _conn(db_path) as c:
            row = c.execute("SELECT MAX(date) FROM ohlcv").fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def get_db_max_cap_date(db_path: Path = DB_PATH) -> Optional[str]:
    if not Path(db_path).exists():
        return None
    try:
        with _conn(db_path) as c:
            row = c.execute("SELECT MAX(date) FROM market_cap").fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def yesterday(today: Optional[date] = None) -> date:
    """오늘 기준 어제. 주말이면 직전 평일로 보정 (한국 시장 기준)."""
    today = today or date.today()
    d = today - timedelta(days=1)
    while d.weekday() >= 5:  # 5=토, 6=일
        d -= timedelta(days=1)
    return d


def _has_credentials() -> bool:
    """KRX OpenAPI 인증키 우선, 레거시 KRX_ID/PW 도 허용."""
    if os.environ.get("KRX_AUTH_KEY"):
        return True
    return bool(os.environ.get("KRX_ID")) and bool(os.environ.get("KRX_PW"))


def _fetch_with_retry(fn, *args, retries: int = 3, sleep: float = 0.5, **kwargs):
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.debug("retry %d failed: %s", i, e)
            time.sleep(sleep * (i + 1))
    return None


def _f(v):
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        if v is None or pd.isna(v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _backfill_etfs(start_iso: str, end_iso: str, db_path: Path = DB_PATH) -> int:
    """ESSENTIAL_ETFS 의 OHLCV 를 pykrx 종목별(기간) 조회로 적재. 적재 행수 반환.

    pykrx `get_market_ohlcv(start, end, ticker)` 는 로그인 없이도 동작하므로
    벤치마크·금헤지 ETF 는 이 경로로 받는다 (KRX OpenAPI ETF 엔드포인트 의존 회피).
    """
    if not start_iso or not end_iso:
        return 0
    ymd_s, ymd_e = start_iso.replace("-", ""), end_iso.replace("-", "")
    rows: list[tuple] = []
    for tk in ESSENTIAL_ETFS:
        df = _fetch_with_retry(stock.get_market_ohlcv, ymd_s, ymd_e, tk)
        if df is None or df.empty:
            continue
        for idx, r in df.iterrows():
            rows.append((
                tk, idx.strftime("%Y-%m-%d"),
                _f(r.get("시가")), _f(r.get("고가")), _f(r.get("저가")),
                _f(r.get("종가")), _i(r.get("거래량")),
            ))
    if rows:
        with _conn(db_path) as c:
            c.executemany(
                "INSERT INTO ohlcv (ticker, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(ticker, date) DO NOTHING",
                rows,
            )
    return len(rows)


def update_to(target_date: Optional[date] = None,
              db_path: Path = DB_PATH,
              progress_cb=None) -> dict:
    """DB 의 마지막 OHLCV 날짜 다음 날부터 target_date(기본=어제)까지 증분 적재.

    주식은 KRX OpenAPI, ETF/지수는 pykrx 종목별 조회. 반환 dict 는 기존 계약 유지:
        {from, to, days_processed, days_added, ohlcv_rows, cap_rows, etf_rows,
         skipped, elapsed_sec, ok, error}
    """
    target = target_date or yesterday()
    last = get_db_max_date(db_path)
    if last is None:
        return _err(None, target, "OHLCV 테이블 비어있음 — 먼저 `python main.py collect` 로 초기 적재 필요")

    if not _has_credentials():
        return _err(last, target, "KRX_AUTH_KEY 미설정 (.env 파일 확인)")

    last_dt = datetime.strptime(last, "%Y-%m-%d").date()
    if last_dt >= target:
        return {"from": last, "to": target.strftime("%Y-%m-%d"),
                "days_processed": 0, "days_added": 0,
                "ohlcv_rows": 0, "cap_rows": 0, "etf_rows": 0, "skipped": [],
                "elapsed_sec": 0.0, "ok": True, "error": None}

    t0 = time.time()
    start_iso = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    target_iso = target.strftime("%Y-%m-%d")

    # 1) 주식 — KRX OpenAPI
    loader = KrxOpenApiLoader(db_path=db_path)
    try:
        res = loader.collect_range(start_iso, target_iso, progress_cb=progress_cb)
    except KrxApiError as e:
        return _err(last, target, str(e))
    if not res.get("ok"):
        # 호출제한/인증 등 → 부분 결과라도 ETF 백필 시도 후 반환
        res.setdefault("from", start_iso)
        res.setdefault("to", target_iso)

    # 2) ETF/지수 — pykrx 종목별
    etf_rows = _backfill_etfs(res.get("from", start_iso), res.get("to", target_iso), db_path)

    return {
        "from": last,
        "to": target_iso,
        "days_processed": res.get("days_processed", 0),
        "days_added": res.get("days_added", 0),
        "ohlcv_rows": res.get("ohlcv_rows", 0),
        "cap_rows": res.get("cap_rows", 0),
        "etf_rows": etf_rows,
        "skipped": res.get("skipped", []),
        "api_calls": res.get("api_calls"),
        "elapsed_sec": time.time() - t0,
        "ok": bool(res.get("ok")),
        "error": res.get("error"),
    }


def _err(last, target, msg: str) -> dict:
    return {"from": last, "to": target.strftime("%Y-%m-%d"),
            "days_processed": 0, "days_added": 0,
            "ohlcv_rows": 0, "cap_rows": 0, "etf_rows": 0, "skipped": [],
            "elapsed_sec": 0.0, "ok": False, "error": msg}


def status_summary(db_path: Path = DB_PATH, today: Optional[date] = None) -> dict:
    """대시보드용 상태 요약."""
    today = today or date.today()
    ohlcv_max = get_db_max_date(db_path)
    cap_max = get_db_max_cap_date(db_path)
    target = yesterday(today)
    is_stale = ohlcv_max is None or (
        datetime.strptime(ohlcv_max, "%Y-%m-%d").date() < target
    )
    days_behind = None
    if ohlcv_max:
        days_behind = (target - datetime.strptime(ohlcv_max, "%Y-%m-%d").date()).days
    return {
        "ohlcv_max": ohlcv_max,
        "cap_max": cap_max,
        "target": target.strftime("%Y-%m-%d"),
        "today": today.strftime("%Y-%m-%d"),
        "is_stale": is_stale,
        "days_behind": days_behind,
        "has_credentials": _has_credentials(),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    except ImportError:
        pass

    s = status_summary()
    print(f"DB 마지막 OHLCV: {s['ohlcv_max']}  (목표 {s['target']}, {s['days_behind']}일 뒤처짐)")
    if s["is_stale"]:
        print("→ 증분 업데이트 시작")
        r = update_to()
        print(r)
    else:
        print("최신 상태 — 업데이트 불필요")
