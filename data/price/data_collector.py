# -*- coding: utf-8 -*-
"""
역할: pykrx를 사용하여 한국 주식 데이터를 수집하는 모듈.
OHLCV (시가, 고가, 저가, 종가, 거래량) 데이터를 가져와 CSV 파일로 저장한다.
"""

from pykrx import stock as pykrx_stock
import pandas as pd
import os
from datetime import datetime, timedelta
import glob


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
    
    # 파일명 설정
    filename = "data/downloads/{}.csv".format(stock_code)
    
    # 기존 데이터 확인
    existing_df = None
    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename, index_col=0, parse_dates=True)
            existing_df.columns = [col.lower() for col in existing_df.columns]
            print("   📂 기존 데이터 로드 완료 ({} 거래일)".format(len(existing_df)))
        except Exception as e:
            print("   ⚠️  기존 데이터 로드 실패: {}".format(str(e)))
            existing_df = None
    else:
        # 기존 파일명 패턴으로 찾기 (호환성)
        pattern = "data/downloads/{}_*.csv".format(stock_code)
        existing_files = glob.glob(pattern)
        data_files = [f for f in existing_files if 'financial' not in f]
        if data_files:
            try:
                # 가장 최근 파일 사용
                latest_file = max(data_files, key=os.path.getmtime)
                existing_df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
                existing_df.columns = [col.lower() for col in existing_df.columns]
                print("   📂 기존 데이터 로드 완료 ({} 거래일, 파일: {})".format(len(existing_df), os.path.basename(latest_file)))
            except Exception as e:
                print("   ⚠️  기존 데이터 로드 실패: {}".format(str(e)))
                existing_df = None
    
    # 요청된 기간 계산
    req_start = pd.to_datetime(start_date)
    req_end = pd.to_datetime(end_date)
    
    # 기존 데이터가 요청된 기간을 완전히 커버하는지 확인
    if existing_df is not None and len(existing_df) > 0:
        existing_start = existing_df.index.min()
        existing_end = existing_df.index.max()
        
        if existing_start <= req_start and existing_end >= req_end:
            print("   ✅ 기존 데이터가 요청 기간을 커버함 ({} ~ {})".format(
                existing_start.strftime("%Y-%m-%d"), existing_end.strftime("%Y-%m-%d")))
            return existing_df
    
    # 누락된 기간 계산
    missing_periods = []
    
    if existing_df is None or len(existing_df) == 0:
        # 전체 기간 다운로드
        missing_periods.append((start_date, end_date))
        print("   🔄 전체 기간 다운로드 필요")
    else:
        existing_start = existing_df.index.min()
        existing_end = existing_df.index.max()
        
        # 시작 부분 누락
        if req_start < existing_start:
            missing_start = start_date
            missing_end = min(req_end, existing_start - pd.Timedelta(days=1)).strftime("%Y%m%d")
            if pd.to_datetime(missing_start) <= pd.to_datetime(missing_end):
                missing_periods.append((missing_start, missing_end))
                print("   🔄 시작 부분 누락: {} ~ {}".format(missing_start, missing_end))
        
        # 끝 부분 누락
        if req_end > existing_end:
            missing_start = max(req_start, existing_end + pd.Timedelta(days=1)).strftime("%Y%m%d")
            missing_end = end_date
            if pd.to_datetime(missing_start) <= pd.to_datetime(missing_end):
                missing_periods.append((missing_start, missing_end))
                print("   🔄 끝 부분 누락: {} ~ {}".format(missing_start, missing_end))
    
    # 누락된 데이터 다운로드 및 병합
    all_data = existing_df.copy() if existing_df is not None else pd.DataFrame()
    
    for miss_start, miss_end in missing_periods:
        try:
            print("   📥 누락 데이터 다운로드: {} ~ {}".format(miss_start, miss_end))
            df = pykrx_stock.get_market_ohlcv(miss_start, miss_end, stock_code)
            
            if df is not None and len(df) > 0:
                # 컬럼명 정규화
                df.columns = [col.lower() for col in df.columns]
                
                # 한국어 컬럼명을 영어 표준명으로 변환
                rename_map = {
                    "시가": "open",
                    "고가": "high",
                    "저가": "low",
                    "종가": "close",
                    "거래량": "volume",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume"
                }
                df = df.rename(columns=lambda col: rename_map.get(col, col))
                
                # 인덱스를 datetime으로 변환
                df.index = pd.to_datetime(df.index)
                
                # 병합
                if all_data.empty:
                    all_data = df
                else:
                    all_data = pd.concat([all_data, df])
                    
                print("   ➕ 데이터 병합 완료 ({} 거래일 추가)".format(len(df)))
            else:
                print("   ⚠️  누락 데이터 없음")
                
        except Exception as e:
            print("   ❌ 누락 데이터 다운로드 실패: {}".format(str(e)))
    
    if all_data.empty:
        print("   ⚠️  데이터 없음: {}".format(stock_code))
        return pd.DataFrame()
    
    # 중복 제거 및 정렬
    all_data = all_data[~all_data.index.duplicated(keep='last')]
    all_data = all_data.sort_index()
    
    # 요청된 기간으로 필터링
    all_data = all_data[(all_data.index >= req_start) & (all_data.index <= req_end)]
    
    # CSV로 저장
    os.makedirs("data/downloads", exist_ok=True)
    all_data.to_csv(filename)
    
    print("   💾 주가 데이터 저장 완료 ({} 거래일)".format(len(all_data)))
    return all_data


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