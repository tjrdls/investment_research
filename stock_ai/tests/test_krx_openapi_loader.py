"""KRX OpenAPI 로더 단위 테스트 — 실 API/네트워크 불필요 (_get 모킹 + 임시 DB).

검증 대상:
  - 응답 필드 → cache.db 스키마 매핑
  - 종목코드 정규화 (표준코드 KR7…003 → 단축 6자리)
  - 숫자 파싱 (콤마 포함 문자열)
  - 날짜 YYYYMMDD → YYYY-MM-DD
  - 휴장(빈 OutBlock) → trading=False, DB 무변경
"""
import sqlite3

import pytest

from src.data import krx_openapi_loader as K
from src.data.krx_openapi_loader import KrxOpenApiLoader, _f, _i, _norm_ticker


# ── 순수 헬퍼 ─────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ("KR7005930003", "005930"),   # 표준코드 → 단축
    ("005930", "005930"),          # 이미 단축
    ("5930", "005930"),            # zero-fill
    (None, None),
    ("", None),
])
def test_norm_ticker(raw, expected):
    assert _norm_ticker(raw) == expected


def test_f_parses_commas_and_blanks():
    assert _f("1,234,567") == 1234567.0
    assert _f("") is None
    assert _f(None) is None
    assert _f("12.5") == 12.5


def test_i_from_comma_string():
    assert _i("1,000,000") == 1000000
    assert _i("") is None


# ── _ingest_day (모킹) ────────────────────────────────────
def _fake_row(code, name, o, h, l, c, vol, cap, shares):
    return {
        K.F_TICKER: code, K.F_NAME: name,
        K.F_OPEN: o, K.F_HIGH: h, K.F_LOW: l, K.F_CLOSE: c,
        K.F_VOLUME: vol, K.F_MKTCAP: cap, K.F_SHARES: shares,
    }


@pytest.fixture
def loader(tmp_path):
    db = tmp_path / "test_cache.db"
    return KrxOpenApiLoader(db_path=db, auth_key="test-key")


def test_ingest_day_maps_fields_and_normalizes(loader, monkeypatch):
    # KOSPI 1종목(표준코드), KOSDAQ 1종목(단축코드)
    def fake_get(group, endpoint, bas_dd):
        if endpoint == "stk_bydd_trd":
            return [_fake_row("KR7005930003", "삼성전자",
                              "70,000", "71,000", "69,500", "70,500",
                              "12,345,678", "420,000,000,000,000", "5,969,782,550")]
        if endpoint == "ksq_bydd_trd":
            return [_fake_row("247540", "에코프로비엠",
                              "100,000", "102,000", "99,000", "101,000",
                              "1,000,000", "9,000,000,000,000", "97,000,000")]
        return []
    monkeypatch.setattr(loader, "_get", fake_get)

    res = loader._ingest_day("20260528")
    assert res["trading"] is True
    assert res["ohlcv_rows"] == 2
    assert res["cap_rows"] == 2

    with sqlite3.connect(loader.db_path) as c:
        # 날짜 변환
        o = c.execute("SELECT ticker, date, open, high, low, close, volume FROM ohlcv "
                      "WHERE ticker='005930'").fetchone()
        assert o == ("005930", "2026-05-28", 70000.0, 71000.0, 69500.0, 70500.0, 12345678)
        # 시총/상장주식수 콤마 파싱
        cap = c.execute("SELECT market_cap, shares FROM market_cap WHERE ticker='005930'").fetchone()
        assert cap == (420_000_000_000_000.0, 5_969_782_550)
        # 유니버스 + market 라벨
        tk = c.execute("SELECT name, market FROM tickers WHERE ticker='005930'").fetchone()
        assert tk == ("삼성전자", "KOSPI")
        tk2 = c.execute("SELECT name, market FROM tickers WHERE ticker='247540'").fetchone()
        assert tk2 == ("에코프로비엠", "KOSDAQ")


def test_ingest_day_holiday_empty_no_write(loader, monkeypatch):
    monkeypatch.setattr(loader, "_get", lambda g, e, d: [])  # 모든 시장 빈 응답
    res = loader._ingest_day("20260101")  # 신정 가정
    assert res["trading"] is False
    assert res["ohlcv_rows"] == 0
    with sqlite3.connect(loader.db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0] == 0


def test_ingest_day_skips_rows_without_ticker(loader, monkeypatch):
    def fake_get(group, endpoint, bas_dd):
        if endpoint == "stk_bydd_trd":
            return [
                _fake_row("", "이름없음코드", "1", "1", "1", "1", "1", "1", "1"),  # 코드 없음 → 스킵
                _fake_row("005930", "삼성전자", "70000", "70000", "70000", "70000", "1", "1", "1"),
            ]
        return []
    monkeypatch.setattr(loader, "_get", fake_get)
    res = loader._ingest_day("20260528")
    assert res["ohlcv_rows"] == 1   # 코드 없는 행 제외


def test_get_requires_auth_key(tmp_path):
    ld = KrxOpenApiLoader(db_path=tmp_path / "x.db", auth_key="")
    with pytest.raises(K.KrxApiError):
        ld._get("sto", "stk_bydd_trd", "20260528")
