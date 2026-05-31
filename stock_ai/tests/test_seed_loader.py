"""시드 로더 순수 헬퍼 단위 테스트 (네트워크 불필요)."""
import pandas as pd

from src.data.seed_loader import _f, _i, _rows_from_ohlcv


def _df():
    idx = pd.to_datetime(["2026-05-27", "2026-05-28"])
    return pd.DataFrame(
        {"시가": [100, 110], "고가": [120, 115], "저가": [95, 108],
         "종가": [110, 112], "거래량": [1000, 2000]},
        index=idx,
    )


def test_rows_from_ohlcv_with_shares():
    ohlcv, cap = _rows_from_ohlcv("005930", _df(), shares=1_000_000)
    assert ohlcv[0] == ("005930", "2026-05-27", 100.0, 120.0, 95.0, 110.0, 1000)
    # 시총 = 종가 × 주식수
    assert cap[0] == ("005930", "2026-05-27", 110.0 * 1_000_000, 1_000_000)
    assert cap[1] == ("005930", "2026-05-28", 112.0 * 1_000_000, 1_000_000)


def test_rows_from_ohlcv_no_shares_skips_cap():
    ohlcv, cap = _rows_from_ohlcv("005930", _df(), shares=None)
    assert len(ohlcv) == 2
    assert cap == []   # 주식수 없으면 시총 미생성 (OHLCV 는 유지)


def test_f_i_helpers():
    assert _f("1,234") == 1234.0
    assert _i("2,000") == 2000
    assert _f(None) is None
