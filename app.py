# -*- coding: utf-8 -*-
"""
역할: 메인 애플리케이션 엔트리포인트
Streamlit 웹 대시보드 실행
"""

import logging

from utils.logger import configure_root_logger
configure_root_logger()

import streamlit as st

# 필요한 모듈들을 미리 임포트하여 메모리에 로드
try:
    import analysis.indicators.technical_indicators
    import analysis.news_analyzer
    import pipeline.prediction_pipeline
    import data_loader.price.data_collector
except ImportError as e:
    st.error(f"모듈 임포트 실패: {e}")
    st.stop()

# 코어 모듈
from web.core.state import init_session_state, get_analysis_stages
from web.core.pipeline import AnalysisPipeline

# UI 모듈
from web.ui.sidebar import render_sidebar
from web.ui.pipeline_view import render_pipeline_progress, update_progress


def main():
    """메인 애플리케이션 함수"""
    st.set_page_config(page_title="투자 분석 웹 대시보드", layout="wide")

    st.title("📈 AI 투자 분석 플랫폼")
    st.markdown("종목을 선택하고 분석을 시작하면 데이터 흐름에 따라 결과가 표시됩니다.")

    # ==================== 세션 상태 초기화 ====================
    init_session_state()

    # ==================== 사이드바 ====================
    selected_code, selected_name, selected_period, start_analysis = render_sidebar()

    # ==================== 분석 UI 준비 ====================
    # "분석 시작" 버튼을 누르면 UI만 표시 (실제 분석은 탭 클릭 시 진행)
    if start_analysis and st.session_state.analysis_stock != selected_code:
        st.session_state.analysis_stock = selected_code
        st.session_state.analysis_ready = True
        st.session_state.active_tab = "데이터 수집"
        # 탭 상태를 모두 "waiting"으로 초기화
        st.session_state.tab_states = {stage[0]: "waiting" for stage in get_analysis_stages()}
        st.rerun()

    # ==================== 분석 결과 표시 ====================
    if not st.session_state.analysis_ready:
        st.info("👈 왼쪽 사이드바에서 종목을 선택하고 '분석 시작' 버튼을 눌러주세요.")
        return

    st.markdown("---")
    st.markdown(f"## 📊 {st.session_state.analysis_stock} 분석 준비")
    st.markdown("---")

    # ==================== 탭 기반 분석 UI ====================
    render_pipeline_progress(
        selected_name,
        selected_code,
        selected_period
    )


if __name__ == "__main__":
    main()
