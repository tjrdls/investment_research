# -*- coding: utf-8 -*-
"""
역할: 사이드바 UI 컴포넌트
종목 선택, 기간 선택, 모델 설정 등
"""

import streamlit as st
from data_loader.price.data_collector import get_top_stocks
from web.core.state import get_default_stocks, get_period_options


@st.cache_data(show_spinner=False)
def get_stock_options():
    """종목 옵션 목록 가져오기"""
    try:
        top_stocks = get_top_stocks("KOSPI", 10)
        if top_stocks:
            return top_stocks
    except Exception:
        pass
    return get_default_stocks()


def render_sidebar():
    """사이드바 렌더링"""
    sidebar = st.sidebar
    sidebar.header("⚙️ 분석 설정")

    # 종목 선택
    stock_options = get_stock_options()
    stock_labels = [f"{code} {name}" for code, name in stock_options]
    selected_option = sidebar.selectbox("📌 종목 선택", stock_labels, index=0)
    selected_code, selected_name = selected_option.split(" ", 1)

    # 기간 선택
    period_options = get_period_options()
    selected_period_label = sidebar.selectbox("📅 분석 기간", list(period_options.keys()), index=0)
    selected_period = period_options[selected_period_label]

    # LSTM 모델 설정
    sidebar.markdown("---")
    sidebar.markdown("### 🧠 LSTM 모델")
    lstm_options = sidebar.radio("모델 선택", ["기본 제공 (빠름)", "커스텀 학습 (정확)"], index=0)
    use_custom_lstm = lstm_options == "커스텀 학습 (정확)"

    if use_custom_lstm:
        st.session_state.lstm_mode = "custom"
        sidebar.info("💡 커스텀 모델을 사용합니다. (시간이 더 소요될 수 있습니다)")
    else:
        st.session_state.lstm_mode = "default"

    # GPT 모델 설정
    sidebar.markdown("---")
    sidebar.markdown("### 🤖 AI 모델 (GPT)")
    gpt_models = ["gpt-5.4-nano", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    selected_gpt_model = sidebar.selectbox("모델 선택", gpt_models, index=0)
    st.session_state.gpt_model = selected_gpt_model
    sidebar.caption(f"선택된 모델: {selected_gpt_model}")

    # 분석 시작 버튼
    start_analysis = sidebar.button("🚀 분석 시작", use_container_width=True, type="primary")

    return selected_code, selected_name, selected_period, start_analysis