# -*- coding: utf-8 -*-
"""
역할: 수집된 주가 데이터를 기반으로 기술적 지표를 계산하는 모듈.
이동평균선, RSI, MACD, 볼린저밴드, 일목균형표 등의 지표를 생성하여 분석에 활용한다.
"""

import pandas as pd
import numpy as np


def compute_rsi(series, period=14):
    """
    RSI (Relative Strength Index) 계산
    
    :param series: 종가 시리즈
    :param period: 기간 (기본 14)
    :return: RSI 값 배열
    """
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_bollinger(series, window=20, n_std=2):
    """
    볼린저 밴드 계산
    
    :param series: 종가 시리즈
    :param window: 기간 (기본 20)
    :param n_std: 표준편차 배수 (기본 2)
    :return: (upper, lower, pct_b, width)
    """
    mu = series.rolling(window).mean()
    sig = series.rolling(window).std()
    upper = mu + n_std * sig
    lower = mu - n_std * sig
    pct_b = (series - lower) / (upper - lower + 1e-8)
    width = (upper - lower) / (mu + 1e-8)
    return upper, lower, pct_b, width


def compute_macd(series, fast=12, slow=26, signal=9):
    """
    MACD (이동평균수렴발산) 계산
    
    :param series: 종가 시리즈
    :param fast: 빠른 EMA 기간 (기본 12)
    :param slow: 느린 EMA 기간 (기본 26)
    :param signal: 시그널 기간 (기본 9)
    :return: (macd, macd_signal, macd_hist)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_sig, macd - macd_sig


def compute_ichimoku(df, t=9, k=26, s=52):
    """
    일목균형표 (Ichimoku Cloud) 계산
    
    :param df: OHLCV DataFrame
    :param t: 전환선 기간 (기본 9)
    :param k: 기준선 기간 (기본 26)
    :param s: 후행선 기간 (기본 52)
    :return: (tenkan, kijun, span_a, span_b)
    """
    high, low = df["high"], df["low"]
    
    tenkan = (high.rolling(t).max() + low.rolling(t).min()) / 2
    kijun = (high.rolling(k).max() + low.rolling(k).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(k)
    span_b = ((high.rolling(s).max() + low.rolling(s).min()) / 2).shift(k)
    
    return tenkan, kijun, span_a, span_b


def calculate_indicators(price_data):
    """
    기술적 지표를 계산한다.
    
    :param price_data: DataFrame with OHLCV (open, high, low, close, volume)
    :return: DataFrame with indicators
    """
    print("   📊 기술적 지표 계산 중...")
    
    df = price_data.copy()
    
    # 컬럼명 정규화
    df.columns = [col.lower() for col in df.columns]
    
    if "close" not in df.columns:
        print("   ❌ 'close' 컬럼이 없습니다")
        return df
    
    # 이동평균 (데이터 길이에 맞게 조정)
    min_periods = min(5, len(df))
    df["sma_5"] = df["close"].rolling(window=min(5, len(df)), min_periods=min_periods).mean()
    df["sma_20"] = df["close"].rolling(window=min(20, len(df)), min_periods=min_periods).mean()
    df["sma_50"] = df["close"].rolling(window=min(50, len(df)), min_periods=min_periods).mean()
    df["sma_200"] = df["close"].rolling(window=min(200, len(df)), min_periods=min_periods).mean()
    
    # 지수이동평균
    df["ema_12"] = df["close"].ewm(span=min(12, len(df)), adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=min(26, len(df)), adjust=False).mean()
    
    # RSI
    df["rsi"] = compute_rsi(df["close"], period=min(14, len(df)-1))
    
    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
    
    # 볼린저 밴드
    df["bb_upper"], df["bb_lower"], df["bb_pct_b"], df["bb_width"] = compute_bollinger(df["close"])
    
    # 일목균형표
    df["ichi_tenkan"], df["ichi_kijun"], df["ichi_span_a"], df["ichi_span_b"] = compute_ichimoku(df)
    
    # 변동성 (5일 표준편차)
    df["volatility"] = df["close"].pct_change().rolling(window=min(5, len(df)), min_periods=1).std()
    
    # 거래량 이동평균
    if "volume" in df.columns:
        df["volume_ma"] = df["volume"].rolling(window=min(20, len(df)), min_periods=1).mean()
    
    # NaN 값은 0으로 채우기 (dropna 대신)
    df = df.fillna(0)
    
    print("   ✅ 지표 계산 완료 ({} 거래일)".format(len(df)))
    return df


def get_technical_signals(df):
    """
    기술적 지표 기반 매매 신호 추출
    
    :param df: 지표가 계산된 DataFrame
    :return: (signals, warnings, score)
    """
    signals = []
    warnings = []
    score = 0
    
    if len(df) < 2:
        return signals, warnings, score
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    close = latest["close"]
    
    # 볼린저 밴드
    pct_b = latest["bb_pct_b"]
    if pct_b > 1.0:
        warnings.append("⛔ 볼린저 상단 돌파 → 과매수")
        score -= 2
    elif pct_b < 0.0:
        signals.append("✅ 볼린저 하단 이탈 → 반등 가능")
        score += 1
    
    # MACD
    if latest["macd_hist"] > 0 and prev["macd_hist"] <= 0:
        signals.append("✅ MACD 골든크로스 → 상승 전환")
        score += 2
    elif latest["macd_hist"] < 0 and prev["macd_hist"] >= 0:
        warnings.append("⛔ MACD 데드크로스 → 하락 전환")
        score -= 2
    
    # RSI
    rsi = latest["rsi"]
    if rsi > 70:
        warnings.append("⛔ RSI {:.0f} → 과매수".format(rsi))
        score -= 1
    elif rsi < 30:
        signals.append("✅ RSI {:.0f} → 과매도 반등 가능".format(rsi))
        score += 1
    
    # 이동평균 정렬
    if latest["sma_20"] > latest["sma_50"]:
        signals.append("✅ 20일 > 50일 → 단기 상승")
        score += 1
    else:
        warnings.append("⚠️  20일 < 50일 → 단기 하락")
        score -= 1
    
    return signals, warnings, score


if __name__ == "__main__":
    # 테스트용 더미 데이터
    dates = pd.date_range("2023-01-01", periods=100)
    data = pd.DataFrame({
        "open": np.random.rand(100) * 100 + 50,
        "high": np.random.rand(100) * 100 + 60,
        "low": np.random.rand(100) * 100 + 40,
        "close": np.random.rand(100) * 100 + 50,
        "volume": np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    indicators = calculate_indicators(data)
    print(indicators.tail())