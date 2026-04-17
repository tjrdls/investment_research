# -*- coding: utf-8 -*-
"""
역할: 분석 진행 상황 UI (탭 기반)
실시간 진행 상태 표시 및 업데이트
"""

import streamlit as st
from web.core.state import get_analysis_stages, get_tab_display_name, is_tab_accessible, set_tab_status, TAB_STATUS

# UI 모듈 임포트 (탭 내용 표시용)
from web.ui.chart_view import render_chart_tab
from web.analysis.technical import render_technical_tab
from web.analysis.financial import render_finance_tab
from web.analysis.news import render_news_tab
from web.analysis.llm import render_summary_tab, render_lstm_tab


def render_pipeline_progress(selected_name: str, result: dict, price_df, indicators_df, stock_news, macro_news):
    """분석 진행 상황을 탭으로 표시"""
    stages = get_analysis_stages()

    # 탭 생성 (상태에 따른 이름 표시)
    tab_names = [get_tab_display_name(stage[0], stage[1]) for stage in stages]
    tabs = st.tabs(tab_names)

    # 각 탭 내용 렌더링
    for i, (stage_name, stage_emoji) in enumerate(stages):
        with tabs[i]:
            render_tab_content(stage_name, selected_name, result, price_df, indicators_df, stock_news, macro_news)


def render_tab_content(stage_name: str, selected_name: str, result: dict, price_df, indicators_df, stock_news, macro_news):
    """각 탭의 내용 렌더링"""
    if not is_tab_accessible(stage_name):
        # 접근 불가능한 탭
        status = st.session_state.tab_states.get(stage_name, "waiting")
        status_info = TAB_STATUS[status]

        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; color: {status_info['color']};">
            <h3>{status_info['emoji']} {stage_name}</h3>
            <p>{status_info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # 접근 가능한 탭 내용 표시
    if stage_name == "데이터 수집":
        render_data_collection_tab(price_df)
    elif stage_name == "기술 분석":
        render_technical_analysis_tab(result, indicators_df)
    elif stage_name == "재무 분석":
        render_financial_analysis_tab(result)
    elif stage_name == "뉴스 수집":
        render_news_collection_tab(selected_name, result, stock_news, macro_news)
    elif stage_name == "AI 분석":
        render_ai_analysis_tab(selected_name, result)


def render_data_collection_tab(price_df):
    """데이터 수집 탭 내용"""
    st.markdown("### 💾 데이터 수집 결과")

    if price_df is not None and not price_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 거래일", f"{len(price_df)}일")
        with col2:
            st.metric("현재가", f"{price_df['close'].iloc[-1]:,.0f}원")
        with col3:
            st.metric("데이터 기간", f"{price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")

        # 가격 차트 표시
        render_chart_tab(price_df, None)
    else:
        st.warning("⚠️ 가격 데이터를 불러올 수 없습니다.")


def render_technical_analysis_tab(result: dict, indicators_df):
    """기술 분석 탭 내용"""
    st.markdown("### 📊 기술 분석 결과")

    if indicators_df is not None and not indicators_df.empty:
        render_technical_tab(result, indicators_df)
    else:
        st.warning("⚠️ 기술 분석 데이터를 불러올 수 없습니다.")


def render_financial_analysis_tab(result: dict):
    """재무 분석 탭 내용"""
    st.markdown("### 💰 재무 분석 결과")
    render_finance_tab(result)


def render_news_collection_tab(selected_name: str, result: dict, stock_news, macro_news):
    """뉴스 수집 탭 내용"""
    st.markdown("### 📰 뉴스 수집 결과")
    render_news_tab(selected_name, result, stock_news or [], macro_news or [])


def render_ai_analysis_tab(selected_name: str, result: dict):
    """AI 분석 탭 내용"""
    st.markdown("### 🤖 AI 분석 결과")

    # 종합 분석
    render_summary_tab(selected_name, result)

    # LSTM 예측 (있는 경우)
    lstm_pred = result.get("lstm_prediction", {}) or {}
    if lstm_pred:
        st.markdown("---")
        render_lstm_tab(result)


def update_progress(stage_name: str, message: str, is_complete: bool = False):
    """진행 상황 업데이트 (탭 상태 변경)"""
    if is_complete:
        set_tab_status(stage_name, "completed")
        st.rerun()  # 탭 상태 변경을 즉시 반영
    else:
        set_tab_status(stage_name, "running")
        st.rerun()


def show_completion_message():
    """분석 완료 메시지 표시"""
    st.balloons()
    st.success("✨ 분석이 완료되었습니다! 탭에서 결과를 확인하세요.")