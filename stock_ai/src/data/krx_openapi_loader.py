"""KRX OpenAPI 로더 (data-dbg.krx.co.kr)
=========================================
pykrx 가 긁던 `data.krx.co.kr` 마켓플레이스가 로그인 게이트로 막히면서
유니버스·시가총액·일자별 bulk 조회가 불능이 됐다. 이 로더는 KRX 가 공식 제공하는
**OpenAPI REST 서비스**를 `KRX_AUTH_KEY` 인증키로 직접 호출해 같은 cache.db 를 채운다.

호출 규격
---------
- GET  {base}/{group}/{endpoint}?basDd=YYYYMMDD
- 헤더 AUTH_KEY: <인증키>   (공백/개행 strip)
- 응답 JSON: 데이터는 "OutBlock_1" 배열 (날짜 1개 = 그날 전 종목)
- 하루 10,000 호출 제한 · ~2010년+ 과거 제공

지수/ETF(069500·132030·QQQ) OHLCV 는 pykrx 종목별 조회가 아직 동작하므로
이 로더의 범위 밖이다 (daily_update 가 그쪽을 담당).

⚠️ 인증키 발급만으로는 부족하고, openapi.krx.co.kr MyPage 에서 각 API 를
"활용신청"해 승인받아야 한다. 미승인 시 모든 호출이 401 을 반환한다.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    raise ImportError("requests 미설치. `pip install requests`")

from src.config import CFG, DB_PATH

logger = logging.getLogger(__name__)

# ── 엔드포인트 (group, endpoint, market 라벨) ──────────────
STOCK_ENDPOINTS = [
    ("sto", "stk_bydd_trd", "KOSPI"),    # 유가증권 일별매매정보
    ("sto", "ksq_bydd_trd", "KOSDAQ"),   # 코스닥 일별매매정보
]

# ── 응답 필드 매핑 (표준 KRX 필드명; 실제 응답과 다르면 여기만 수정) ──
F_TICKER = "ISU_CD"        # 종목코드 (단축코드일 수도, 표준코드일 수도 → _norm_ticker 로 보정)
F_NAME = "ISU_NM"          # 종목명
F_OPEN = "TDD_OPNPRC"      # 시가
F_HIGH = "TDD_HGPRC"       # 고가
F_LOW = "TDD_LWPRC"        # 저가
F_CLOSE = "TDD_CLSPRC"     # 종가
F_VOLUME = "ACC_TRDVOL"    # 누적거래량
F_MKTCAP = "MKTCAP"        # 시가총액
F_SHARES = "LIST_SHRS"     # 상장주식수


class KrxApiError(RuntimeError):
    """KRX OpenAPI 호출 실패 (상태코드·메시지 포함)."""


class KrxOpenApiLoader:
    """KRX OpenAPI 로 OHLCV + 시가총액 + 유니버스를 cache.db 에 적재."""

    def __init__(self, db_path: Path = DB_PATH, auth_key: Optional[str] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.auth_key = (auth_key or os.getenv("KRX_AUTH_KEY", "")).strip()
        self.base = CFG.data.krx_api_base.rstrip("/")
        self._call_count = 0
        self._session = requests.Session()
        self._init_schema()

    # ── DB ────────────────────────────────────────────────
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
        """pykrx_loader 와 동일한 스키마 (이미 있으면 no-op)."""
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

    # ── HTTP ──────────────────────────────────────────────
    def _get(self, group: str, endpoint: str, bas_dd: str) -> list[dict]:
        """단일 (group/endpoint, basDd) 호출 → OutBlock_1 리스트.

        실패 분류:
          - 401 → KrxApiError("활용신청 승인 필요")
          - 404 → KrxApiError("엔드포인트 경로 오류")
          - 기타/빈응답 → 빈 리스트 (휴장일 등은 정상적으로 빈 OutBlock_1)
        """
        if not self.auth_key:
            raise KrxApiError("KRX_AUTH_KEY 미설정 (.env 확인)")
        url = f"{self.base}/{group}/{endpoint}"
        headers = {"AUTH_KEY": self.auth_key, "Accept": "application/json"}
        last_err = None
        for attempt in range(CFG.data.krx_api_retry):
            try:
                self._call_count += 1
                r = self._session.get(url, headers=headers,
                                      params={"basDd": bas_dd}, timeout=30)
                if r.status_code == 401:
                    raise KrxApiError(
                        f"401 Unauthorized — [{group}/{endpoint}] 활용신청 승인 필요. "
                        "openapi.krx.co.kr MyPage 에서 해당 API 활용신청 후 승인 대기."
                    )
                if r.status_code == 404:
                    raise KrxApiError(f"404 — 엔드포인트 경로 오류: {group}/{endpoint}")
                if r.status_code != 200:
                    raise KrxApiError(f"HTTP {r.status_code}: {r.text[:200]}")
                data = r.json()
                return data.get("OutBlock_1") or data.get("outBlock_1") or []
            except KrxApiError:
                raise  # 인증/경로 오류는 재시도 무의미 → 즉시 전파
            except Exception as e:  # noqa: BLE001 — 네트워크/JSON 일시 오류만 재시도
                last_err = e
                logger.debug("재시도 %d (%s): %s", attempt, endpoint, e)
                time.sleep(0.5 * (attempt + 1))
        raise KrxApiError(f"호출 실패 ({group}/{endpoint}, {bas_dd}): {last_err}")

    # ── 한 거래일 적재 ─────────────────────────────────────
    def _ingest_day(self, bas_dd: str) -> dict:
        """단일 거래일 KOSPI+KOSDAQ 전종목 OHLCV+시총 적재.

        bas_dd: 'YYYYMMDD'.
        Returns: {"date": iso, "ohlcv_rows", "cap_rows", "ticker_rows", "trading": bool}
        """
        iso = f"{bas_dd[:4]}-{bas_dd[4:6]}-{bas_dd[6:8]}"
        ohlcv_rows: list[tuple] = []
        cap_rows: list[tuple] = []
        ticker_rows: list[tuple] = []

        for group, endpoint, market in STOCK_ENDPOINTS:
            block = self._get(group, endpoint, bas_dd)
            for row in block:
                tk = _norm_ticker(row.get(F_TICKER))
                if not tk:
                    continue
                ohlcv_rows.append((
                    tk, iso,
                    _f(row.get(F_OPEN)), _f(row.get(F_HIGH)), _f(row.get(F_LOW)),
                    _f(row.get(F_CLOSE)), _i(row.get(F_VOLUME)),
                ))
                cap_rows.append((tk, iso, _f(row.get(F_MKTCAP)), _i(row.get(F_SHARES))))
                name = (row.get(F_NAME) or "").strip()
                ticker_rows.append((tk, name, market))

        trading = bool(ohlcv_rows)
        if not trading:
            return {"date": iso, "ohlcv_rows": 0, "cap_rows": 0,
                    "ticker_rows": 0, "trading": False}

        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO ohlcv (ticker, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(ticker, date) DO NOTHING",
                ohlcv_rows,
            )
            conn.executemany(
                "INSERT INTO market_cap (ticker, date, market_cap, shares) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(ticker, date) DO NOTHING",
                cap_rows,
            )
            conn.executemany(
                "INSERT INTO tickers (ticker, name, market) VALUES (?, ?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET name=excluded.name, market=excluded.market",
                ticker_rows,
            )
        return {"date": iso, "ohlcv_rows": len(ohlcv_rows), "cap_rows": len(cap_rows),
                "ticker_rows": len(ticker_rows), "trading": True}

    # ── 기간 수집 ──────────────────────────────────────────
    def collect_range(self, start: str, end: Optional[str] = None,
                      progress_cb=None) -> dict:
        """start ~ end(기본=어제) 거래일 전체 적재.

        start/end: 'YYYY-MM-DD' 또는 'YYYYMMDD'. 주말 스킵, 빈 OutBlock=휴장 스킵.
        10k/day 제한에 근접하면 중단하고 안내.
        """
        start_dt = _parse_day(start)
        end_dt = _parse_day(end) if end else _yesterday()
        if start_dt > end_dt:
            return {"ok": False, "error": "start > end", "days_added": 0}

        days = _weekdays(start_dt, end_dt)
        total = len(days)
        added, ohlcv_rows, cap_rows, skipped = 0, 0, 0, []
        t0 = time.time()
        logger.info("KRX OpenAPI 수집: %s ~ %s (%d 평일)",
                    start_dt, end_dt, total)

        for i, d in enumerate(days, 1):
            if progress_cb:
                progress_cb(i - 1, total, d.strftime("%Y-%m-%d"))
            # 호출 제한 보호 (시장 2개 × 안전마진)
            if self._call_count + len(STOCK_ENDPOINTS) > CFG.data.krx_api_daily_limit:
                logger.warning("일일 호출 제한 근접 — 중단 (%d/%d일 완료)",
                               added, total)
                return {"ok": False, "error": "daily_limit_reached",
                        "days_processed": i - 1, "days_added": added,
                        "ohlcv_rows": ohlcv_rows, "cap_rows": cap_rows,
                        "skipped": skipped, "elapsed_sec": time.time() - t0}
            try:
                res = self._ingest_day(d.strftime("%Y%m%d"))
            except KrxApiError as e:
                logger.error("적재 중단: %s", e)
                return {"ok": False, "error": str(e),
                        "days_processed": i - 1, "days_added": added,
                        "ohlcv_rows": ohlcv_rows, "cap_rows": cap_rows,
                        "skipped": skipped, "elapsed_sec": time.time() - t0}
            if res["trading"]:
                added += 1
                ohlcv_rows += res["ohlcv_rows"]
                cap_rows += res["cap_rows"]
                if added % 50 == 0:
                    logger.info("[%d/%d] %s — 누적 OHLCV %d행",
                                i, total, res["date"], ohlcv_rows)
            else:
                skipped.append(res["date"])
            time.sleep(CFG.data.krx_api_sleep)

        if progress_cb:
            progress_cb(total, total, "완료")
        return {"ok": True, "error": None,
                "from": start_dt.strftime("%Y-%m-%d"), "to": end_dt.strftime("%Y-%m-%d"),
                "days_processed": total, "days_added": added,
                "ohlcv_rows": ohlcv_rows, "cap_rows": cap_rows,
                "skipped": skipped, "api_calls": self._call_count,
                "elapsed_sec": time.time() - t0}

    def update_to(self, target_date: Optional[str] = None, progress_cb=None) -> dict:
        """DB 의 마지막 OHLCV 날짜 다음날부터 target(기본=어제)까지 증분 적재."""
        last = self._db_max_date()
        if last is None:
            return {"ok": False, "error": "OHLCV 비어있음 — 먼저 collect_range 로 초기 적재 필요",
                    "days_added": 0}
        next_day = (_parse_day(last) + timedelta(days=1)).strftime("%Y-%m-%d")
        return self.collect_range(next_day, target_date, progress_cb=progress_cb)

    def _db_max_date(self) -> Optional[str]:
        try:
            with self._conn() as conn:
                row = conn.execute("SELECT MAX(date) FROM ohlcv").fetchone()
            return row[0] if row and row[0] else None
        except Exception:
            return None


# ── 모듈 헬퍼 ──────────────────────────────────────────────
def _norm_ticker(v) -> Optional[str]:
    """KRX 표준코드(KR7005930003) 또는 단축코드(005930) → 6자리 단축코드."""
    if not v:
        return None
    s = str(v).strip()
    # 표준코드 KR7xxxxxx0yy → 가운데 6자리 추출
    if len(s) == 12 and s.upper().startswith("KR"):
        return s[3:9]
    # 이미 6자리 숫자
    if len(s) >= 6:
        return s[-6:] if s[-6:].isdigit() else s
    return s.zfill(6)


def _f(v) -> Optional[float]:
    try:
        if v is None or v == "" or pd.isna(v):
            return None
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _i(v) -> Optional[int]:
    f = _f(v)
    return int(f) if f is not None else None


def _parse_day(s: str) -> "datetime.date":
    s = str(s).replace("-", "")
    return datetime.strptime(s, "%Y%m%d").date()


def _yesterday():
    d = datetime.now().date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _weekdays(start, end) -> list:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    except ImportError:
        pass
    loader = KrxOpenApiLoader()
    # 최근 5거래일만 시범 적재
    import sys
    end = _yesterday()
    start = end - timedelta(days=7)
    print(loader.collect_range(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
