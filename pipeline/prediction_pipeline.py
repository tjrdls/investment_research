# 역할: 투자 분석 파이프라인의 핵심 모듈. 데이터 수집부터 LLM 분석까지의 전체 흐름을 조율한다.
# 이 파일은 각 모듈을 순차적으로 호출하여 결과를 종합하고 최종 출력을 생성한다.

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.price.data_collector import collect_price_data
from data.financial.financial_collector import collect_financial_data
from analysis.indicators.technical_indicators import calculate_indicators
from models.lstm.lstm_model import predict_with_lstm, build_llm_input, llm_analysis
from rag.financial_rag.rag_system import search_financial_rag
from models.llm.llm_analyzer import analyze_with_llm

def run_analysis(stock_symbol):
    print("🚀 투자 분석 시작: {}".format(stock_symbol))

    # 1. 데이터 수집
    print("📊 1. 데이터 수집 시작...")
    price_data = collect_price_data(stock_symbol)
    financial_data = collect_financial_data(stock_symbol)
    print("✅ 1. 데이터 수집 완료")

    # 2. 데이터 분석 (기술적 지표 계산)
    print("📈 2. 데이터 분석 시작...")
    indicators = calculate_indicators(price_data)
    print("✅ 2. 데이터 분석 완료")

    # 3. LSTM 분석 feature 생성
    print("🤖 3. LSTM 분석 시작...")
    lstm_feature = predict_with_lstm(indicators)
    print("✅ 3. LSTM 분석 완료")

    # 4. LLM 입력용 feature 생성
    print("🔧 4. LLM 입력 feature 생성...")
    llm_input = build_llm_input(lstm_feature, indicators)
    print("✅ 4. LLM 입력 feature 생성 완료")

    # 5. LLM 최종 분석
    print("🧠 5. LLM 최종 분석 시작...")
    final_analysis = llm_analysis(llm_input)
    print("✅ 5. LLM 최종 분석 완료")

    # 6. 결과 저장 및 출력
    print("💾 6. 결과 저장 및 출력")
    print("📋 최종 분석 결과:")
    print(final_analysis)
    return final_analysis

if __name__ == "__main__":
    # 테스트용
    result = run_analysis("005930.KS")
    print(result)