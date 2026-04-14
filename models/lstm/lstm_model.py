# 역할: 기술적 지표와 주가 데이터를 입력으로 받아 LSTM 모델을 사용하여 분석용 feature를 출력하는 모듈.
# 단기 상승/하락 예측 대신, LLM에 전달할 수 있는 feature 벡터를 생성한다.

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=32):  # output_size 변경
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 분석 feature 출력으로 변경
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)  # 32차원 feature
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def predict_with_lstm(indicators_df):
    """
    LSTM으로 분석 feature를 생성.
    :param indicators_df: DataFrame with indicators
    :return: feature vector
    """
    print("   🤖 LSTM 모델 분석 중...")
    # 간단한 모델 로드 (실제로는 학습된 모델 필요)
    model = LSTMModel(input_size=10, hidden_size=50, num_layers=2, output_size=32)
    # 더미 feature
    feature = np.random.randn(32).astype(np.float32)
    print("   ✅ 분석 완료")
    return feature

# LLM 입력용 feature 생성
def build_llm_input(lstm_output, indicators):
    """
    LSTM 출력과 기술적 지표를 LLM 입력용으로 조합.
    :param lstm_output: LSTM feature vector
    :param indicators: DataFrame with indicators
    :return: dict for LLM
    """
    return {
        "lstm_signal": lstm_output.tolist(),
        "rsi": indicators["rsi"].iloc[-1] if not indicators.empty else 0,
        "macd": indicators["macd"].iloc[-1] if not indicators.empty else 0,
        "trend": "상승" if indicators["close"].iloc[-1] > indicators["close"].iloc[-2] else "하락"
    }

# LLM 추론 단계
def llm_analysis(data):
    """
    LLM으로 최종 분석 수행.
    :param data: dict from build_llm_input
    :return: analysis text
    """
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
다음 주식 분석 데이터를 기반으로 분석하세요:

LSTM 신호: {data['lstm_signal']}
RSI: {data['rsi']}
MACD: {data['macd']}
추세: {data['trend']}

분석:
1. 현재 추세
2. 위험 요인
3. 상승 가능성
"""

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # 테스트
    result = predict_with_lstm(pd.DataFrame())
    print("LSTM Feature:", result)
    indicators = pd.DataFrame({'rsi': [70], 'macd': [0.5], 'close': [100, 101]})
    llm_input = build_llm_input(result, indicators)
    print("LLM Input:", llm_input)
    analysis = llm_analysis(llm_input)
    print("LLM Analysis:", analysis)