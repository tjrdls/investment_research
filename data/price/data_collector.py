# -*- coding: utf-8 -*-
"""
역할: pykrx를 사용하여 한국 주식 데이터를 수집하는 모듈.
OHLCV (시가, 고가, 저가, 종가, 거래량) 데이터를 가져와 CSV 파일로 저장한다.
"""

from pykrx import stock as pykrx_stock
import pandas as pd
import os
from datetime import datetime, timedelta


def collect_price_data(stock_code, start_date=None, end_date=None, period="3y"):
    """
    한국 주식 데이터를 pykrx로 수집한다.
    
    :param stock_code: 주식 코드 (예: "005930" - 삼성전자)
    :param start_date: 시작 날짜 (YYYYMMDD 형식, 없으면 자동 계산)
    :param end_date: 종료 날짜 (YYYYMMDD 형식, 없으면 오늘)
    :param period: 기간 (기본 3년)
    :return: DataFrame with OHLCV
    """
    print("   📈 주가 데이터 수집 중 ({})...".format(stock_code))
    
    # 날짜 설정
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    
    if start_date is None:
        # period에 따라 자동 계산
        if period == "3y":
            days = 1095
        elif period == "1y":
            days = 365
        elif period == "6m":
            days = 180
        else:
            days = 365
        
        start_dt = datetime.now() - timedelta(days=days)
        start_date = start_dt.strftime("%Y%m%d")
    
    try:
        # pykrx에서 OHLCV 데이터 수집
        df = pykrx_stock.get_market_ohlcv(start_date, end_date, stock_code)
        
        if df is None or len(df) == 0:
            print("   ⚠️  데이터 없음: {}".format(stock_code))
            return pd.DataFrame()
        
        # 컬럼명 정규화 (소문자)
        df.columns = [col.lower() for col in df.columns]
        
        # 인덱스를 datetime으로 변환
        df.index = pd.to_datetime(df.index)
        
        # CSV로 저장
        os.makedirs("data/downloads", exist_ok=True)
        filename = "data/downloads/{}_{}.csv".format(stock_code, end_date)
        df.to_csv(filename)
        
        print("   💾 주가 데이터 저장 완료 ({} 거래일)".format(len(df)))
        return df
    
    except Exception as e:
        print("   ❌ 데이터 수집 실패: {}".format(str(e)))
        return pd.DataFrame()


def get_top_stocks(market="KOSPI", top_n=5):
    """
    시총 상위 종목 가져오기
    
    :param market: "KOSPI" 또는 "KOSDAQ"
    :param top_n: 상위 N개
    :return: list of (stock_code, stock_name)
    """
    print("   🏆 {} 시총 상위 {} 종목 조회 중...".format(market, top_n))
    
    try:
        today = datetime.now().strftime("%Y%m%d")
        
        # 시총 기반으로 정렬된 데이터 가져오기
        market_cap = pykrx_stock.get_market_cap_by_ticker(today, market=market)
        
        # 상위 N개 추출
        top_stocks = market_cap.head(top_n)
        
        result = []
        for code, row in top_stocks.iterrows():
            result.append((code, row.get('name', code)))
        
        print("   ✅ 조회 완료: {}".format([name for _, name in result]))
        return result
    
    except Exception as e:
        print("   ⚠️  시총 조회 실패: {}".format(str(e)))
        # 폴백: 하드코딩된 상위 종목
        return [
            ("005930", "삼성전자"),
            ("000660", "SK하이닉스"),
            ("373220", "LG에너지솔루션"),
            ("207940", "삼성바이오로직스"),
            ("005380", "현대차")
        ]


if __name__ == "__main__":
    # 테스트: 삼성전자 데이터 수집
    data = collect_price_data("005930")
    print(data.head())