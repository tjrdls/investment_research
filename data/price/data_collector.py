# 역할: yfinance 라이브러리를 사용하여 주가 데이터를 수집하는 모듈.
# OHLCV (시가, 고가, 저가, 종가, 거래량) 데이터를 가져와 CSV 파일로 저장한다.

import yfinance as yf
import pandas as pd
import os

def collect_price_data(stock_symbol, period="1y"):
    """
    주가 데이터를 수집한다.
    :param stock_symbol: 주식 심볼 (예: "005930.KS")
    :param period: 데이터 기간 (기본 1년)
    :return: DataFrame
    """
    print("   📈 주가 데이터 다운로드 중...")
    data = yf.download(stock_symbol, period=period)
    # 저장
    os.makedirs("data/downloads", exist_ok=True)
    data.to_csv(f"data/downloads/{stock_symbol}_price.csv")
    print("   💾 주가 데이터 저장 완료")
    return data

if __name__ == "__main__":
    # 테스트
    data = collect_price_data("005930.KS")
    print(data.head())