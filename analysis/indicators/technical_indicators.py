# -*- coding: utf-8 -*-
"""
역할: 수집된 주가 데이터를 기반으로 기술적 지표를 계산하는 모듈.
이동평균선, RSI, MACD, 볼린저밴드, 일목균형표 등의 지표를 생성하여 분석에 활용한다.
"""

import logging
from typing import List, TypedDict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TechnicalSignals(TypedDict):
    signals: List[str]
    warnings: List[str]
    score: int


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_bollinger(series: pd.Series, window: int = 20, n_std: int = 2) -> tuple:
    mu = series.rolling(window).mean()
    sig = series.rolling(window).std()
    upper = mu + n_std * sig
    lower = mu - n_std * sig
    pct_b = (series - lower) / (upper - lower + 1e-8)
    width = (upper - lower) / (mu + 1e-8)
    return upper, lower, pct_b, width


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_sig, macd - macd_sig


def compute_ichimoku(df: pd.DataFrame, t: int = 9, k: int = 26, s: int = 52) -> tuple:
    high, low = df["high"], df["low"]
    tenkan = (high.rolling(t).max() + low.rolling(t).min()) / 2
    kijun = (high.rolling(k).max() + low.rolling(k).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(k)
    span_b = ((high.rolling(s).max() + low.rolling(s).min()) / 2).shift(k)
    return tenkan, kijun, span_a, span_b


def calculate_indicators(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    기술적 지표를 계산한다.

    :param price_data: DataFrame with OHLCV
    :return: DataFrame with indicators
    """
    logger.info("📊 기술적 지표 계산 중...")

    df = price_data.copy()
    df.columns = [col.lower() for col in df.columns]

    if "close" not in df.columns:
        logger.error("'close' 컬럼이 없습니다")
        return df

    n = len(df)
    mp = min(5, n)

    df["sma_5"] = df["close"].rolling(window=min(5, n), min_periods=mp).mean()
    df["sma_20"] = df["close"].rolling(window=min(20, n), min_periods=mp).mean()
    df["sma_50"] = df["close"].rolling(window=min(50, n), min_periods=mp).mean()
    df["sma_200"] = df["close"].rolling(window=min(200, n), min_periods=mp).mean()

    df["ema_12"] = df["close"].ewm(span=min(12, n), adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=min(26, n), adjust=False).mean()

    df["rsi"] = compute_rsi(df["close"], period=min(14, n - 1))
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
    df["bb_upper"], df["bb_lower"], df["bb_pct_b"], df["bb_width"] = compute_bollinger(df["close"])
    df["ichi_tenkan"], df["ichi_kijun"], df["ichi_span_a"], df["ichi_span_b"] = compute_ichimoku(df)
    df["volatility"] = df["close"].pct_change().rolling(window=min(5, n), min_periods=1).std()

    if "volume" in df.columns:
        df["volume_ma"] = df["volume"].rolling(window=min(20, n), min_periods=1).mean()

    df = df.fillna(0)
    logger.info("✅ 지표 계산 완료 (%d 거래일)", n)
    return df


def get_technical_signals(df: pd.DataFrame) -> TechnicalSignals:
    """
    기술적 지표 기반 매매 신호 추출.

    :param df: 지표가 계산된 DataFrame
    :return: TechnicalSignals dict (signals, warnings, score)
    """
    signals: List[str] = []
    warnings: List[str] = []
    score: int = 0

    if len(df) < 2:
        return {"signals": signals, "warnings": warnings, "score": score}

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    pct_b = latest["bb_pct_b"]
    if pct_b > 1.0:
        warnings.append("⛔ 볼린저 상단 돌파 → 과매수")
        score -= 2
    elif pct_b < 0.0:
        signals.append("✅ 볼린저 하단 이탈 → 반등 가능")
        score += 1

    if latest["macd_hist"] > 0 and prev["macd_hist"] <= 0:
        signals.append("✅ MACD 골든크로스 → 상승 전환")
        score += 2
    elif latest["macd_hist"] < 0 and prev["macd_hist"] >= 0:
        warnings.append("⛔ MACD 데드크로스 → 하락 전환")
        score -= 2

    rsi = latest["rsi"]
    if rsi > 70:
        warnings.append("⛔ RSI {:.0f} → 과매수".format(rsi))
        score -= 1
    elif rsi < 30:
        signals.append("✅ RSI {:.0f} → 과매도 반등 가능".format(rsi))
        score += 1

    if latest["sma_20"] > latest["sma_50"]:
        signals.append("✅ 20일 > 50일 → 단기 상승")
        score += 1
    else:
        warnings.append("⚠️  20일 < 50일 → 단기 하락")
        score -= 1

    return {"signals": signals, "warnings": warnings, "score": score}


if __name__ == "__main__":
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
