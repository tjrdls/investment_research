"""
일별 증분 데이터 업데이트 (대시보드용)
====================================
DB 의 마지막 날짜부터 어제(또는 지정일)까지 OHLCV + 시가총액을
일별 일괄 조회 API 로 빠르게 채워넣는다.

종목별 수집 (`PyKrxLoader.collect_all`) 은 종목당 한 번씩 API 호출하므로 수 시간이 걸리지만,
이 모듈은 일자별 bulk API 한 번에 모든 종목을 가져오므로 며칠 분량을 분 단위에 마무리한다.
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

logger = logging.getLogger(__name__)


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
    return bool(os.environ.get("KRX_ID")) and bool(os.environ.get("KRX_PW"))


def _trading_days_between(start: date, end: date) -> list[date]:
    """start(불포함) ~ end(포함) 사이의 평일 리스트."""
    out = []
    d = start + timedelta(days=1)
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _fetch_with_retry(fn, *args, retries: int = 3, sleep: float = 0.5, **kwargs):
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.debug("retry %d failed: %s", i, e)
            time.sleep(sleep * (i + 1))
    return None


def _ingest_day(target: date, db_path: Path = DB_PATH) -> dict:
    """단일 거래일의 KOSPI+KOSDAQ 전종목 OHLCV+시총 + ETF OHLCV 를 일괄 조회 후 적재.

    ETF 는 일반 주식 bulk API 에 안 잡혀 별도 호출 필요 (벤치마크 069500 KODEX 200,
    금 헤지 132030 KODEX 골드선물 등이 ETF 라 백테스트 정합성에 필수).

    Returns: {"date": ..., "ohlcv_rows": int, "cap_rows": int,
              "etf_rows": int, "trading": bool}
    """
    ymd = target.strftime("%Y%m%d")
    iso = target.strftime("%Y-%m-%d")

    ohlcv_rows: list[tuple] = []
    cap_rows: list[tuple] = []

    for market in ("KOSPI", "KOSDAQ"):
        df_o = _fetch_with_retry(stock.get_market_ohlcv_by_ticker, ymd, market=market)
        if df_o is None or df_o.empty:
            continue
        for tk, row in df_o.iterrows():
            ohlcv_rows.append((
                str(tk), iso,
                _f(row.get("시가")), _f(row.get("고가")), _f(row.get("저가")),
                _f(row.get("종가")), _i(row.get("거래량")),
            ))

        df_c = _fetch_with_retry(stock.get_market_cap_by_ticker, ymd, market=market)
        if df_c is not None and not df_c.empty:
            for tk, row in df_c.iterrows():
                cap_rows.append((
                    str(tk), iso,
                    _f(row.get("시가총액")), _i(row.get("상장주식수")),
                ))

    # ETF — 별도 bulk API. (벤치마크/금헤지 ETF 가 여기 들어있음)
    etf_rows: list[tuple] = []
    df_e = _fetch_with_retry(stock.get_etf_ohlcv_by_ticker, ymd)
    if df_e is not None and not df_e.empty:
        for tk, row in df_e.iterrows():
            etf_rows.append((
                str(tk), iso,
                _f(row.get("시가")), _f(row.get("고가")), _f(row.get("저가")),
                _f(row.get("종가")), _i(row.get("거래량")),
            ))

    trading = bool(ohlcv_rows) or bool(etf_rows)
    if not trading:
        return {"date": iso, "ohlcv_rows": 0, "cap_rows": 0,
                "etf_rows": 0, "trading": False}

    with _conn(db_path) as c:
        if ohlcv_rows:
            c.executemany(
                "INSERT INTO ohlcv (ticker, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(ticker, date) DO NOTHING",
                ohlcv_rows,
            )
        if etf_rows:
            c.executemany(
                "INSERT INTO ohlcv (ticker, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(ticker, date) DO NOTHING",
                etf_rows,
            )
        if cap_rows:
            c.executemany(
                "INSERT INTO market_cap (ticker, date, market_cap, shares) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(ticker, date) DO NOTHING",
                cap_rows,
            )

    return {"date": iso, "ohlcv_rows": len(ohlcv_rows), "cap_rows": len(cap_rows),
            "etf_rows": len(etf_rows), "trading": True}


def update_to(target_date: Optional[date] = None,
              db_path: Path = DB_PATH,
              progress_cb=None) -> dict:
    """DB 의 마지막 OHLCV 날짜 다음 날부터 target_date(기본=어제)까지 일별 적재.
    progress_cb(current_idx, total, label) — 선택, Streamlit 진행률 표시용.

    Returns: {
        "from": "YYYY-MM-DD" | None,        # 시작 시 DB 최대일
        "to":   "YYYY-MM-DD",               # 목표일
        "days_processed": int,              # 시도한 거래일 수
        "days_added": int,                  # 데이터 적재된 거래일 수
        "ohlcv_rows": int, "cap_rows": int,
        "skipped": ["YYYY-MM-DD", ...],     # 비거래일/휴장일
        "elapsed_sec": float,
        "ok": bool, "error": str|None,
    }
    """
    target = target_date or yesterday()
    last = get_db_max_date(db_path)
    if last is None:
        return {"from": None, "to": target.strftime("%Y-%m-%d"),
                "days_processed": 0, "days_added": 0,
                "ohlcv_rows": 0, "cap_rows": 0, "skipped": [],
                "elapsed_sec": 0.0, "ok": False,
                "error": "OHLCV 테이블 비어있음 — 먼저 `python main.py collect` 로 초기 적재 필요"}

    if not _has_credentials():
        return {"from": last, "to": target.strftime("%Y-%m-%d"),
                "days_processed": 0, "days_added": 0,
                "ohlcv_rows": 0, "cap_rows": 0, "skipped": [],
                "elapsed_sec": 0.0, "ok": False,
                "error": "KRX_ID / KRX_PW 미설정 (.env 파일 확인)"}

    last_dt = datetime.strptime(last, "%Y-%m-%d").date()
    if last_dt >= target:
        return {"from": last, "to": target.strftime("%Y-%m-%d"),
                "days_processed": 0, "days_added": 0,
                "ohlcv_rows": 0, "cap_rows": 0, "skipped": [],
                "elapsed_sec": 0.0, "ok": True, "error": None}

    days = _trading_days_between(last_dt, target)
    total = len(days)
    added, ohlcv_rows, cap_rows, etf_rows, skipped = 0, 0, 0, 0, []
    t0 = time.time()

    for i, d in enumerate(days, 1):
        if progress_cb:
            progress_cb(i - 1, total, d.strftime("%Y-%m-%d"))
        try:
            res = _ingest_day(d, db_path=db_path)
        except Exception as e:
            logger.warning("%s 적재 실패: %s", d, e)
            continue
        if res["trading"]:
            added += 1
            ohlcv_rows += res["ohlcv_rows"]
            cap_rows += res["cap_rows"]
            etf_rows += res.get("etf_rows", 0)
            logger.info("✓ %s: OHLCV %d행, ETF %d행, 시총 %d행",
                        res["date"], res["ohlcv_rows"], res.get("etf_rows", 0),
                        res["cap_rows"])
        else:
            skipped.append(res["date"])
            logger.info("∙ %s: 휴장", res["date"])

    if progress_cb:
        progress_cb(total, total, "완료")

    return {"from": last, "to": target.strftime("%Y-%m-%d"),
            "days_processed": total, "days_added": added,
            "ohlcv_rows": ohlcv_rows, "cap_rows": cap_rows, "etf_rows": etf_rows,
            "skipped": skipped,
            "elapsed_sec": time.time() - t0, "ok": True, "error": None}


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
