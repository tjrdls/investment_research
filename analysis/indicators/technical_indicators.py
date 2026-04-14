# 역할: 수집된 주가 데이터를 기반으로 기술적 지표를 계산하는 모듈.
# 이동평균선, RSI, MACD, 변동성 등의 지표를 생성하여 분석에 활용한다.

import pandas as pd
import ta

def calculate_indicators(price_data):
    """
    기술적 지표를 계산한다.
    :param price_data: DataFrame with OHLCV
    :return: DataFrame with indicators
    """
    print("   📊 기술적 지표 계산 중...")
    df = price_data.copy()

    # 이동평균
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)

    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    # 변동성 (볼린저 밴드)
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_lower'] = bollinger.bollinger_lband()

    print("   ✅ 지표 계산 완료")
    return df

if __name__ == "__main__":
    # 테스트용 더미 데이터
    import numpy as np
    dates = pd.date_range('2023-01-01', periods=100)
    data = pd.DataFrame({
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100 + 10,
        'Low': np.random.rand(100) * 100 - 10,
        'Close': np.random.rand(100) * 100,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    indicators = calculate_indicators(data)
    print(indicators.tail())