# 역할: LSTM 결과, 기술적 분석 결과, RAG 검색 결과를 입력으로 받아 OpenAI API를 통해 최종 투자 분석을 수행하는 모듈.
# 모든 정보를 종합하여 자연어로 된 분석 결과를 생성한다.

import openai

def analyze_with_llm(lstm_result, indicators, rag_result):
    """
    LLM으로 최종 분석.
    :param lstm_result: dict from LSTM
    :param indicators: DataFrame
    :param rag_result: str from RAG
    :return: analysis text
    """
    print("   🧠 LLM 분석 중...")
    openai.api_key = "your_openai_api_key"  # 실제 키로 교체

    prompt = f"""
    다음 정보를 바탕으로 투자 분석을 수행하시오:
    LSTM 예측: 상승 확률 {lstm_result['상승 확률']*100}%, 하락 확률 {lstm_result['하락 확률']*100}%
    기술적 지표: 최근 RSI {indicators['RSI'].iloc[-1] if not indicators.empty else 'N/A'}
    재무제표: {rag_result}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    print("   ✅ 분석 완료")
    return response.choices[0].message.content

if __name__ == "__main__":
    # 테스트
    lstm = {"상승 확률": 0.62, "하락 확률": 0.38}
    indicators = pd.DataFrame({'RSI': [70]})
    rag = "매출 증가, 영업이익 증가"
    result = analyze_with_llm(lstm, indicators, rag)
    print(result)