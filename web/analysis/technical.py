# -*- coding: utf-8 -*-
"""
역할: 기술 분석 UI 컴포넌트
기술 지표 계산 및 표시
"""

import streamlit as st
import pandas as pd


def format_value(value, digits=2, suffix=""):
    """값 포맷팅"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{suffix}"
    return f"{value}{suffix}"


def build_technical_summary(indicators_df: pd.DataFrame):
    """기술 분석 요약 생성"""
    latest = indicators_df.iloc[-1]
    prev = indicators_df.iloc[-2] if len(indicators_df) > 1 else latest

    # RSI 분석
    rsi_value = latest.get("rsi", None)
    if rsi_value is not None:
        if rsi_value > 70:
            rsi_state = "과매수"
        elif rsi_value < 30:
            rsi_state = "과매도"
        else:
            rsi_state = "중립"
    else:
        rsi_state = "N/A"

    # MACD 분석
    macd_hist = latest.get("macd_hist", None)
    prev_macd_hist = prev.get("macd_hist", None)
    if macd_hist is not None and prev_macd_hist is not None:
        if macd_hist > 0 and prev_macd_hist <= 0:
            macd_state = "골든크로스"
        elif macd_hist < 0 and prev_macd_hist >= 0:
            macd_state = "데드크로스"
        else:
            macd_state = "중립"
    else:
        macd_state = "N/A"

    # 볼린저 밴드 분석
    boll_pct = latest.get("bb_pct_b", None)
    if boll_pct is not None:
        if boll_pct < 0:
            boll_state = "하단 (반등)"
        elif boll_pct > 1:
            boll_state = "상단 (과매수)"
        else:
            boll_state = "중립"
    else:
        boll_state = "N/A"

    table = pd.DataFrame(
        {
            "지표": ["RSI", "MACD", "볼린저"],
            "값": [
                format_value(rsi_value, digits=0),
                macd_state,
                format_value(boll_pct, digits=2),
            ],
            "상태": [rsi_state, macd_state, boll_state],
        }
    )
    return table


def render_technical_tab(result: dict, indicators_df: pd.DataFrame):
    """기술 분석 탭 렌더링"""
    st.markdown("#### 기술 분석")
    table = build_technical_summary(indicators_df)
    st.table(table)

    signals = result.get("technical", {}).get("signals", []) or []
    warnings = result.get("technical", {}).get("warnings", []) or []
    summary = pd.DataFrame(
        {
            "신호 유형": ["상승 신호", "하락 신호"],
            "개수": [len(signals), len(warnings)],
        }
    )
    st.table(summary)

    if signals:
        st.markdown("**상승 신호**")
        for item in signals:
            st.write(f"- {item}")
    if warnings:
        st.markdown("**하락 신호**")
        for item in warnings:
            st.write(f"- {item}")