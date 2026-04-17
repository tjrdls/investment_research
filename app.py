# -*- coding: utf-8 -*-
"""
역할: 메인 애플리케이션 엔트리포인트
Streamlit 웹 대시보드 실행
"""

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
from web.core.state import init_session_state
from web.core.pipeline import AnalysisPipeline

# UI 모듈
from web.ui.sidebar import render_sidebar
from web.ui.pipeline_view import render_pipeline_progress, update_progress, show_completion_message


def main():
    """메인 애플리케이션 함수"""
    st.set_page_config(page_title="투자 분석 웹 대시보드", layout="wide")

    st.title("📈 AI 투자 분석 플랫폼")
    st.markdown("종목을 선택하고 분석을 시작하면 데이터 흐름에 따라 결과가 표시됩니다.")

    # ==================== 세션 상태 초기화 ====================
    init_session_state()

    # ==================== 사이드바 ====================
    selected_code, selected_name, selected_period, start_analysis = render_sidebar()

    # ==================== 분석 실행 ====================
    if start_analysis and st.session_state.analysis_stock != selected_code:
        st.session_state.analysis_stock = selected_code

        try:
            # 파이프라인 실행
            pipeline = AnalysisPipeline()
            for stage_name, message in pipeline.run_pipeline(selected_code, selected_period):
                update_progress(stage_name, message, message.startswith("✅"))

            # 완료 메시지
            show_completion_message()

        except Exception as e:
            st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
            return

    # ==================== 분석 결과 표시 ====================
    result = st.session_state.analysis_result

    # 분석 중 상태 확인
    if st.session_state.analysis_stock is not None and st.session_state.analysis_result is None:
        st.info("🔄 분석을 수행 중입니다. 잠시만 기다려주세요...")
        st.stop()

    if result is None:
        st.info("👈 왼쪽 사이드바에서 종목을 선택하고 '분석 시작' 버튼을 눌러주세요.")
        return

    st.markdown("---")
    st.markdown(f"## 📊 {selected_name} ({selected_code}) 분석 결과")
    st.markdown("---")

    # ==================== 탭 기반 분석 결과 ====================
    render_pipeline_progress(
        selected_name,
        result,
        st.session_state.price_df,
        st.session_state.indicators_df,
        st.session_state.stock_news,
        st.session_state.macro_news
    )


if __name__ == "__main__":
    main()
