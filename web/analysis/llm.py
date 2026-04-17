# -*- coding: utf-8 -*-
"""
역할: LLM 분석 결과 UI 컴포넌트
종합 분석 및 투자 의견 표시
"""

import streamlit as st
from .technical import format_value


def render_summary_tab(stock_name: str, result: dict):
    """종합 분석 탭 렌더링"""
    if result is None:
        st.warning("⚠️ 분석 결과가 없습니다.")
        return
        
    final_analysis = result.get("final_analysis", {}) or {}
    lstm_pred = result.get("lstm_prediction", {}) or {}
    news_analysis = result.get("news", {}).get("analysis", {}) or {}
    valuation = result.get("valuation", {}) or {}

    col1, col2 = st.columns(2)
    with col1:
        st.metric("투자 의견", final_analysis.get("recommendation", "N/A"))
        st.metric("신뢰도", f"{final_analysis.get('confidence', 'N/A')}%")
    with col2:
        st.metric("상승 확률", f"{int(lstm_pred.get('probabilities', {}).get('상승', 0))}%")
        st.metric("하락 확률", f"{int(lstm_pred.get('probabilities', {}).get('하락', 0))}%")

    st.markdown("#### LLM 요약")
    st.write(final_analysis.get("summary", "분석 결과 없음"))

    st.markdown("#### 주요 리스크")
    risks = final_analysis.get("risks", [])
    if risks:
        for item in risks:
            st.write(f"- {item}")
    else:
        st.write("- 없음")

    st.markdown("#### 주요 기회")
    opps = final_analysis.get("opportunities", [])
    if opps:
        for item in opps:
            st.write(f"- {item}")
    else:
        st.write("- 없음")

    st.markdown("#### 전략")
    st.write(final_analysis.get("strategy", "전략 정보 없음"))

    st.markdown("---")
    st.markdown("#### 종합 정보")
    st.write(
        "**현재가:** {}원  \
        **PER:** {}  \
        **PBR:** {}  \
        **ROE:** {}%".format(
            format_value(result.get('current_price'), digits=0),
            format_value(valuation.get('PER')),
            format_value(valuation.get('PBR')),
            format_value(valuation.get('ROE'))
        )
    )


def render_lstm_tab(result: dict):
    """LSTM 예측 탭 렌더링"""
    if result is None:
        st.warning("⚠️ LSTM 예측 결과가 없습니다.")
        return
        
    lstm_pred = result.get("lstm_prediction", {}) or {}
    if lstm_pred:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "상승 확률",
                f"{int(lstm_pred.get('probabilities', {}).get('상승', 0))}%",
                delta=f"신뢰도: {int(lstm_pred.get('confidence', 0))}%"
            )
        with col2:
            st.metric(
                "하락 확률",
                f"{int(lstm_pred.get('probabilities', {}).get('하락', 0))}%"
            )
        with col3:
            st.metric(
                "횡보 확률",
                f"{int(lstm_pred.get('probabilities', {}).get('횡보', 0))}%"
            )

        if lstm_pred.get("prediction"):
            st.info(f"**모델 예측:** {lstm_pred.get('prediction')}")
    else:
        st.warning("LSTM 예측 결과를 불러올 수 없습니다.")