# -*- coding: utf-8 -*-
"""
역할: 분석 진행 상황 UI (탭 기반 - Chrome/Edge 스타일)
실시간 진행 상태 표시 및 업데이트
"""

import streamlit as st
from web.core.state import get_analysis_stages, set_tab_status, TAB_STATUS

# UI 모듈 임포트 (탭 내용 표시용)
from web.ui.chart_view import render_chart_tab
from web.analysis.technical import render_technical_tab
from web.analysis.financial import render_finance_tab
from web.analysis.news import render_news_tab
from web.analysis.llm import render_summary_tab, render_lstm_tab


def render_tab_buttons(stages):
    """Chrome 스타일의 탭 버튼 렌더링"""
    st.markdown("### 📑 분석 진행 단계")
    
    # 탭 버튼 행 생성
    cols = st.columns(len(stages) + 1)
    
    for idx, (stage_name, stage_emoji) in enumerate(stages):
        with cols[idx]:
            status = st.session_state.tab_states.get(stage_name, "waiting")
            status_info = TAB_STATUS[status]
            
            # 탭 버튼 스타일
            button_label = f"{status_info['emoji']} {stage_name}"
            
            # 버튼 클릭 시 해당 탭으로 이동
            if st.button(
                button_label,
                key=f"tab_btn_{stage_name}",
                use_container_width=True
            ):
                st.session_state.active_tab = stage_name
                st.rerun()
    
    # "모두 진행" 버튼
    with cols[-1]:
        if st.button("▶️ 모두 진행", use_container_width=True, key="run_all_btn"):
            st.session_state.run_all_analysis = True
            st.rerun()


def render_tab_header(stage_name: str, stage_emoji: str):
    """탭 헤더 렌더링"""
    status = st.session_state.tab_states.get(stage_name, "waiting")
    status_info = TAB_STATUS[status]
    
    # 헤더 색상 박스
    st.markdown(f"""
    <div style="
        background-color: {status_info['bg_color']};
        border-left: 4px solid {status_info['color']};
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    ">
        <h3 style="color: {status_info['color']}; margin: 0;">
            {status_info['emoji']} {stage_name}
        </h3>
        <p style="color: {status_info['color']}; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            상태: {status_info['desc']}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_progress(selected_name: str, result: dict, price_df, indicators_df, stock_news, macro_news):
    """분석 진행 상황을 탭으로 표시 (Chrome 스타일)"""
    stages = get_analysis_stages()
    
    # 초기화
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "데이터 수집"
    
    # 탭 버튼 렌더링
    render_tab_buttons(stages)
    
    st.markdown("---")
    
    # 활성 탭 내용 렌더링
    active_tab = st.session_state.active_tab
    
    for stage_name, stage_emoji in stages:
        if stage_name == active_tab:
            render_tab_header(stage_name, stage_emoji)
            render_tab_content(stage_name, selected_name, result, price_df, indicators_df, stock_news, macro_news)
            break
    else:
        # 활성 탭이 없으면 첫 번째 탭 표시
        stage_name, stage_emoji = stages[0]
        render_tab_header(stage_name, stage_emoji)
        render_tab_content(stage_name, selected_name, result, price_df, indicators_df, stock_news, macro_news)


def render_tab_content(stage_name: str, selected_name: str, result: dict, price_df, indicators_df, stock_news, macro_news):
    """각 탭의 내용 렌더링"""
    status = st.session_state.tab_states.get(stage_name, "waiting")
    
    if status == "waiting":
        st.info(f"⏳ {stage_name}을(를) 기다리는 중입니다...")
        return
    elif status == "error":
        st.error(f"❌ {stage_name} 중에 오류가 발생했습니다.")
        return
    
    # 진행 중이거나 완료된 탭의 내용 표시
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
    if indicators_df is not None and not indicators_df.empty:
        render_technical_tab(result, indicators_df)
    else:
        st.warning("⚠️ 기술 분석 데이터를 불러올 수 없습니다.")


def render_financial_analysis_tab(result: dict):
    """재무 분석 탭 내용"""
    if result is not None:
        render_finance_tab(result)
    else:
        st.warning("⚠️ 재무 분석 데이터를 불러올 수 없습니다.")


def render_news_collection_tab(selected_name: str, result: dict, stock_news, macro_news):
    """뉴스 수집 탭 내용"""
    render_news_tab(selected_name, result, stock_news or [], macro_news or [])


def render_ai_analysis_tab(selected_name: str, result: dict):
    """AI 분석 탭 내용"""
    if result is None:
        st.warning("⚠️ AI 분석 데이터를 불러올 수 없습니다.")
        return

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
    else:
        set_tab_status(stage_name, "running")


def show_completion_message():
    """분석 완료 메시지 표시"""
    st.balloons()
    st.success("✨ 분석이 완료되었습니다! 모든 탭을 확인할 수 있습니다.")
