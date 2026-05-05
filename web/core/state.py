# -*- coding: utf-8 -*-
"""
역할: 애플리케이션 상태 관리 — 세션 상태 초기화 및 관리
"""

import streamlit as st
from config import DEFAULT_STOCKS, PERIOD_OPTIONS, GPT_MODEL

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

TAB_STATUS = {
    "waiting":   {"emoji": "⏳", "color": "#999999", "desc": "대기중",  "bg_color": "#f0f0f0"},
    "running":   {"emoji": "🔄", "color": "#ffa500", "desc": "진행중",  "bg_color": "#fff8e1"},
    "completed": {"emoji": "✅", "color": "#4caf50", "desc": "완료",    "bg_color": "#e8f5e9"},
    "error":     {"emoji": "❌", "color": "#f44336", "desc": "오류",    "bg_color": "#ffebee"},
}


def init_session_state() -> None:
    """세션 상태 초기화 (앱 시작 시 한 번만 실행)."""
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
        st.session_state.lstm_mode = "default"
        st.session_state.gpt_model = GPT_MODEL
        # tab_states 하나로 단계 완료 여부와 UI 상태를 모두 추적
        st.session_state.tab_states = {s[0]: "waiting" for s in ANALYSIS_STAGES}
        st.session_state.analysis_ready = False
        st.session_state.active_tab = ANALYSIS_STAGES[0][0]


def get_analysis_stages() -> list:
    return ANALYSIS_STAGES


def get_tab_status(stage_name: str) -> str:
    return st.session_state.tab_states.get(stage_name, "waiting")


def set_tab_status(stage_name: str, status: str) -> None:
    if status in TAB_STATUS:
        st.session_state.tab_states[stage_name] = status


def get_tab_display_name(stage_name: str, stage_emoji: str) -> str:
    status = get_tab_status(stage_name)
    return f"{TAB_STATUS[status]['emoji']} {stage_name}"


def is_tab_accessible(stage_name: str) -> bool:
    return get_tab_status(stage_name) == "completed"


def get_default_stocks() -> list:
    return list(DEFAULT_STOCKS)


def get_period_options() -> dict:
    return dict(PERIOD_OPTIONS)


def reset_analysis_state() -> None:
    """분석 상태 초기화."""
    st.session_state.analysis_result = None
    st.session_state.price_df = None
    st.session_state.indicators_df = None
    st.session_state.financial_data = None
    st.session_state.stock_news = None
    st.session_state.macro_news = None
    st.session_state.analysis_in_progress = False
    st.session_state.current_stage = None
    st.session_state.tab_states = {s[0]: "waiting" for s in ANALYSIS_STAGES}
    st.session_state.analysis_stock = None
