# -*- coding: utf-8 -*-
"""
역할: pykrx를 사용하여 한국 주식 데이터를 수집하는 모듈.
OHLCV (시가, 고가, 저가, 종가, 거래량) 데이터를 가져와 CSV 파일로 저장한다.
"""

import logging
from pykrx import stock as pykrx_stock
import pandas as pd
import os
from datetime import datetime, timedelta
import glob

from config import DEFAULT_STOCKS, PERIOD_DAYS, DATA_DIR

logger = logging.getLogger(__name__)

# 한국어 → 영어 컬럼 표준화 맵
_KR_RENAME_MAP = {
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "종가": "close",
    "거래량": "volume",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명을 소문자 영어로 표준화한다."""
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]
    df = df.rename(columns=lambda c: _KR_RENAME_MAP.get(c, c))
    return df


def collect_price_data(stock_code: str, start_date: str = None, end_date: str = None, period: str = "3y") -> pd.DataFrame:
    """
    한국 주식 데이터를 pykrx로 수집한다.

    :param stock_code: 주식 코드 (예: "005930")
    :param start_date: 시작 날짜 (YYYYMMDD 형식, 없으면 자동 계산)
    :param end_date: 종료 날짜 (YYYYMMDD 형식, 없으면 오늘)
    :param period: 기간 (기본 3년)
    :return: DataFrame with OHLCV
    """
    logger.info("📈 주가 데이터 수집 중 (%s)...", stock_code)

    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    if start_date is None:
        days = PERIOD_DAYS.get(period, PERIOD_DAYS["1y"])
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    filename = "{}/{}.csv".format(DATA_DIR, stock_code)

    existing_df = _load_existing(stock_code, filename)

    req_start = pd.to_datetime(start_date)
    req_end = pd.to_datetime(end_date)

    if existing_df is not None and len(existing_df) > 0:
        existing_start = existing_df.index.min()
        existing_end = existing_df.index.max()
        if existing_start <= req_start and existing_end >= req_end:
            logger.info("✅ 기존 데이터가 요청 기간을 커버함 (%s ~ %s)",
                        existing_start.strftime("%Y-%m-%d"), existing_end.strftime("%Y-%m-%d"))
            return existing_df

    missing_periods = _compute_missing_periods(existing_df, start_date, end_date, req_start, req_end)

    all_data = existing_df.copy() if existing_df is not None else pd.DataFrame()

    for miss_start, miss_end in missing_periods:
        try:
            logger.info("📥 누락 데이터 다운로드: %s ~ %s", miss_start, miss_end)
            df = pykrx_stock.get_market_ohlcv(miss_start, miss_end, stock_code)

            if df is not None and len(df) > 0:
                df = _normalize_columns(df)
                df.index = pd.to_datetime(df.index)
                all_data = df if all_data.empty else pd.concat([all_data, df])
                logger.info("➕ 데이터 병합 완료 (%d 거래일 추가)", len(df))
            else:
                logger.warning("누락 데이터 없음: %s ~ %s", miss_start, miss_end)

        except Exception as e:
            logger.error("❌ 누락 데이터 다운로드 실패: %s", e)

    if all_data.empty:
        logger.warning("데이터 없음: %s", stock_code)
        return pd.DataFrame()

    all_data = all_data[~all_data.index.duplicated(keep="last")].sort_index()
    all_data = all_data[(all_data.index >= req_start) & (all_data.index <= req_end)]

    os.makedirs(DATA_DIR, exist_ok=True)
    all_data.to_csv(filename)
    logger.info("💾 주가 데이터 저장 완료 (%d 거래일)", len(all_data))
    return all_data


def _load_existing(stock_code: str, filename: str) -> pd.DataFrame | None:
    """기존 캐시 파일을 로드한다. 없으면 None 반환."""
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            df = _normalize_columns(df)
            logger.info("📂 기존 데이터 로드 완료 (%d 거래일)", len(df))
            return df
        except Exception as e:
            logger.warning("기존 데이터 로드 실패: %s", e)
            return None

    # 구형 파일명 패턴 지원
    pattern = "{}/{}_*.csv".format(DATA_DIR, stock_code)
    data_files = [f for f in glob.glob(pattern) if "financial" not in f]
    if data_files:
        latest = max(data_files, key=os.path.getmtime)
        try:
            df = pd.read_csv(latest, index_col=0, parse_dates=True)
            df = _normalize_columns(df)
            logger.info("📂 기존 데이터 로드 완료 (%d 거래일, 파일: %s)", len(df), os.path.basename(latest))
            return df
        except Exception as e:
            logger.warning("기존 데이터 로드 실패: %s", e)

    return None


def _compute_missing_periods(existing_df, start_date, end_date, req_start, req_end):
    """기존 데이터와 요청 범위를 비교해 누락 구간 목록을 반환한다."""
    if existing_df is None or len(existing_df) == 0:
        logger.info("🔄 전체 기간 다운로드 필요")
        return [(start_date, end_date)]

    missing = []
    existing_start = existing_df.index.min()
    existing_end = existing_df.index.max()

    if req_start < existing_start:
        ms = start_date
        me = min(req_end, existing_start - pd.Timedelta(days=1)).strftime("%Y%m%d")
        if pd.to_datetime(ms) <= pd.to_datetime(me):
            missing.append((ms, me))
            logger.info("🔄 시작 부분 누락: %s ~ %s", ms, me)

    if req_end > existing_end:
        ms = max(req_start, existing_end + pd.Timedelta(days=1)).strftime("%Y%m%d")
        me = end_date
        if pd.to_datetime(ms) <= pd.to_datetime(me):
            missing.append((ms, me))
            logger.info("🔄 끝 부분 누락: %s ~ %s", ms, me)

    return missing


def get_top_stocks(market: str = "KOSPI", top_n: int = 5) -> list:
    """
    시총 상위 종목 가져오기

    :param market: "KOSPI" 또는 "KOSDAQ"
    :param top_n: 상위 N개
    :return: list of (stock_code, stock_name)
    """
    logger.info("🏆 %s 시총 상위 %d 종목 조회 중...", market, top_n)

    try:
        today = datetime.now().strftime("%Y%m%d")
        market_cap = pykrx_stock.get_market_cap_by_ticker(today, market=market)
        top_stocks = market_cap.head(top_n)
        result = [(code, row.get("name", code)) for code, row in top_stocks.iterrows()]
        logger.info("✅ 조회 완료: %s", [name for _, name in result])
        return result

    except Exception as e:
        logger.warning("시총 조회 실패: %s — 기본 종목 사용", e)
        return list(DEFAULT_STOCKS[:top_n])


if __name__ == "__main__":
    data = collect_price_data("005930")
    print(data.head())
