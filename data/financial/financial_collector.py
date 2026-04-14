# 역할: openDART API를 사용하여 재무제표 데이터를 수집하는 모듈.
# 매출, 영업이익, 순이익, 부채비율 등의 주요 재무 지표를 가져와 저장한다.

import dart_fss as dart
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def collect_financial_data(stock_symbol):
    """
    재무제표 데이터를 수집한다.
    :param stock_symbol: 주식 심볼
    :return: dict of DataFrames
    """
    print("   📊 재무제표 데이터 수집 중...")
    # dart_fss 초기화 (API 키 필요)
    dart.set_api_key(os.getenv("DART_API_KEY"))

    # 기업 코드 찾기 (간단하게 가정)
    corp_code = "00126380"  # 삼성전자 예시

    # 재무제표 가져오기
    reports = dart.api.filings.get_corp_info(corp_code)
    # 간단하게 주요 지표 추출 (실제로는 파싱 필요)
    financial_data = {
        "매출": 100000,  # 예시 값
        "영업이익": 20000,
        "순이익": 15000,
        "부채비율": 30.5
    }

    # 저장
    os.makedirs("data/downloads", exist_ok=True)
    pd.DataFrame([financial_data]).to_csv(f"data/downloads/{stock_symbol}_financial.csv", index=False)
    print("   💾 재무제표 데이터 저장 완료")
    return financial_data

if __name__ == "__main__":
    # 테스트
    data = collect_financial_data("005930.KS")
    print(data)