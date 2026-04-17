# -*- coding: utf-8 -*-
"""
역할: 애플리케이션 상태 관리
세션 상태 초기화 및 관리
"""

import streamlit as st

# 분석 진행 상태
ANALYSIS_STAGES = [
    ("데이터 수집", "💾"),
    ("재무 데이터 수집", "📈"),
    ("기술 분석", "📊"),
    ("재무 분석", "💰"),
    ("뉴스 수집", "📰"),
    ("뉴스 분석", "🔍"),
    ("LSTM 예측", "🧠"),
    ("AI 종합 분석", "🤖"),
]

# 탭 상태 정의
TAB_STATUS = {
    "waiting": {"emoji": "⏳", "color": "#999999", "desc": "대기중", "bg_color": "#f0f0f0"},
    "running": {"emoji": "🔄", "color": "#ffa500", "desc": "진행중", "bg_color": "#fff8e1"},
    "completed": {"emoji": "✅", "color": "#4caf50", "desc": "완료", "bg_color": "#e8f5e9"},
    "error": {"emoji": "❌", "color": "#f44336", "desc": "오류", "bg_color": "#ffebee"},
}

DEFAULT_STOCKS = [
    ("005930", "삼성전자"),
    ("000660", "SK하이닉스"),
    ("373220", "LG에너지솔루션"),
    ("207940", "삼성바이오로직스"),
    ("005380", "현대차"),
]

PERIOD_OPTIONS = {
    "3년": "3y",
    "1년": "1y",
    "6개월": "6m",
}


def init_session_state():
    """세션 상태 초기화"""
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
        st.session_state.price_df = None
        st.session_state.indicators_df = None
        st.session_state.financial_data = None
        st.session_state.stock_news = None
        st.session_state.macro_news = None
        st.session_state.analysis_stock = None
        st.session_state.analysis_in_progress = False
        st.session_state.current_stage = None
        st.session_state.analysis_stages = {stage[0]: False for stage in ANALYSIS_STAGES}
        st.session_state.lstm_mode = "default"
        st.session_state.gpt_model = "gpt-5.4-nano"
        # 탭 상태 초기화
        st.session_state.tab_states = {stage[0]: "waiting" for stage in ANALYSIS_STAGES}
        # UI 표시 여부
        st.session_state.analysis_ready = False
        st.session_state.active_tab = "데이터 수집"


def get_analysis_stages():
    """분석 단계 목록 반환"""
    return ANALYSIS_STAGES


def get_tab_status(stage_name: str):
    """특정 단계의 탭 상태 반환"""
    return st.session_state.tab_states.get(stage_name, "waiting")


def set_tab_status(stage_name: str, status: str):
    """특정 단계의 탭 상태 설정"""
    if status in TAB_STATUS:
        st.session_state.tab_states[stage_name] = status


def get_tab_display_name(stage_name: str, stage_emoji: str):
    """탭 표시 이름 생성 (상태 이모지 포함)"""
    status = get_tab_status(stage_name)
    status_emoji = TAB_STATUS[status]["emoji"]
    return f"{status_emoji} {stage_name}"


def is_tab_accessible(stage_name: str):
    """탭 접근 가능 여부 확인 (완료된 탭만 접근 가능)"""
    return get_tab_status(stage_name) == "completed"


def get_default_stocks():
    """기본 종목 목록 반환"""
    return DEFAULT_STOCKS


def get_period_options():
    """기간 옵션 반환"""
    return PERIOD_OPTIONS


def reset_analysis_state():
    """분석 상태 초기화"""
    st.session_state.analysis_result = None
    st.session_state.price_df = None
    st.session_state.indicators_df = None
    st.session_state.stock_news = None
    st.session_state.macro_news = None
    st.session_state.analysis_in_progress = False
    st.session_state.current_stage = None
    st.session_state.tab_states = {stage[0]: "waiting" for stage in ANALYSIS_STAGES}
    st.session_state.analysis_stock = None
    st.session_state.analysis_stages = {stage[0]: False for stage in ANALYSIS_STAGES}