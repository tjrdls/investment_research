# -*- coding: utf-8 -*-
"""
역할: 분석 진행 상황 UI (탭 기반 - Chrome/Edge 스타일)
실시간 진행 상태 표시 및 업데이트
"""

import streamlit as st
from web.core.state import get_analysis_stages, set_tab_status, TAB_STATUS
from web.core.pipeline import AnalysisPipeline

# UI 모듈 임포트 (탭 내용 표시용)
from web.ui.chart_view import render_chart_tab
from web.analysis.technical import render_technical_tab
from web.analysis.financial import render_finance_tab
from web.analysis.news import render_news_tab
from web.analysis.llm import render_summary_tab, render_lstm_tab


def run_single_stage_analysis(stage_name: str, selected_code: str, selected_period: str):
    """단일 단계 분석 실행"""
    try:
        set_tab_status(stage_name, "running")
        
        if stage_name == "데이터 수집":
            from data_loader.price.data_collector import collect_price_data
            from analysis.indicators.technical_indicators import calculate_indicators
            
            st.session_state.price_df = collect_price_data(selected_code, period=selected_period)
            st.session_state.indicators_df = calculate_indicators(st.session_state.price_df)
            
        elif stage_name == "재무 데이터 수집":
            from data_loader.financial.financial_collector import collect_financial_data, get_corp_code_map
            
            # 기업 코드 매핑 가져오기
            corp_code_map = get_corp_code_map()
            corp_info = corp_code_map.get(selected_code)
            
            if corp_info:
                corp_code = corp_info["corp_code"]
                st.session_state.financial_data = collect_financial_data(selected_code, corp_code)
            else:
                st.error(f"❌ {selected_code}에 대한 기업 코드를 찾을 수 없습니다.")
            
        elif stage_name == "기술 분석":
            # 이미 데이터 수집 시 함께 수행됨
            pass
            
        elif stage_name == "재무 분석":
            from pipeline.prediction_pipeline import run_financial_analysis
            
            if hasattr(st.session_state, 'financial_data') and st.session_state.financial_data is not None and not st.session_state.financial_data.empty:
                current_price = st.session_state.price_df['close'].iloc[-1] if st.session_state.price_df is not None else None
                if current_price:
                    valuation_result = run_financial_analysis(selected_code, st.session_state.financial_data, current_price)
                    if st.session_state.analysis_result is None:
                        st.session_state.analysis_result = {}
                    st.session_state.analysis_result["valuation"] = valuation_result
            
        elif stage_name == "뉴스 수집":
            from analysis.news_analyzer import collect_stock_news, collect_macro_news
            
            # 종목명 추출
            stock_name_map = {
                "005930": "삼성전자",
                "000660": "SK하이닉스",
                "373220": "LG에너지솔루션",
                "207940": "삼성바이오로직스",
                "005380": "현대차"
            }
            display_name = stock_name_map.get(selected_code, selected_code)
            
            st.session_state.stock_news, st.session_state.macro_news = collect_stock_news(display_name, [display_name]), collect_macro_news()
            
        elif stage_name == "뉴스 분석":
            from analysis.news_analyzer import analyze_news_with_gpt
            
            if st.session_state.stock_news and st.session_state.macro_news:
                # 종목명 추출
                stock_name_map = {
                    "005930": "삼성전자",
                    "000660": "SK하이닉스",
                    "373220": "LG에너지솔루션",
                    "207940": "삼성바이오로직스",
                    "005380": "현대차"
                }
                display_name = stock_name_map.get(selected_code, selected_code)
                
                news_analysis = analyze_news_with_gpt(display_name, st.session_state.stock_news, st.session_state.macro_news)
                if st.session_state.analysis_result is None:
                    st.session_state.analysis_result = {}
                st.session_state.analysis_result["news"] = {"analysis": news_analysis}
            
        elif stage_name == "LSTM 예측":
            from pipeline.prediction_pipeline import run_lstm_prediction
            
            if st.session_state.price_df is not None and st.session_state.indicators_df is not None:
                lstm_result = run_lstm_prediction(
                    st.session_state.price_df, 
                    st.session_state.indicators_df
                )
                if st.session_state.analysis_result is None:
                    st.session_state.analysis_result = {}
                st.session_state.analysis_result["lstm_prediction"] = lstm_result
            
        elif stage_name == "AI 종합 분석":
            from pipeline.prediction_pipeline import run_final_analysis
            
            if st.session_state.analysis_result:
                # 종목명 추출
                stock_name_map = {
                    "005930": "삼성전자",
                    "000660": "SK하이닉스",
                    "373220": "LG에너지솔루션",
                    "207940": "삼성바이오로직스",
                    "005380": "현대차"
                }
                display_name = stock_name_map.get(selected_code, selected_code)
                
                final_analysis = run_final_analysis(selected_code, display_name, st.session_state.analysis_result)
                st.session_state.analysis_result["final_analysis"] = final_analysis
        
        set_tab_status(stage_name, "completed")
        st.success(f"✅ {stage_name} 완료!")
        return True
        
    except Exception as e:
        set_tab_status(stage_name, "error")
        st.error(f"❌ {stage_name} 중 오류 발생: {str(e)}")
        return False


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
            
            # 버튼 클릭 시 해당 탭으로 이동하고 단계 실행
            if st.button(
                button_label,
                key=f"tab_btn_{stage_name}",
                use_container_width=True
            ):
                st.session_state.active_tab = stage_name
                # 해당 단계가 아직 실행되지 않았으면 실행
                if status == "waiting":
                    run_single_stage_analysis(stage_name, st.session_state.analysis_stock, "3y")  # 기본 기간 사용
                st.rerun()
    
    # "모두 진행" 버튼
    with cols[-1]:
        if st.button("▶️ 모두 진행", use_container_width=True, key="run_all_btn"):
            st.session_state.run_all_analysis = True
            st.rerun()


def render_tab_header(stage_name: str):
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


def render_pipeline_progress(selected_name: str, selected_code: str, selected_period: str):
    """분석 진행 상황을 탭으로 표시 (Chrome 스타일)"""
    stages = get_analysis_stages()
    
    # "모두 진행" 버튼 처리 - 전체 단계를 순차적으로 실행
    if st.session_state.get("run_all_analysis", False):
        st.session_state.run_all_analysis = False
        
        # 진행 상황 표시를 위한 placeholder
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.markdown("### 🔄 전체 분석 진행 중...")
            
            for stage_name, _ in stages:
                if st.session_state.tab_states.get(stage_name, "waiting") == "waiting":
                    st.info(f"🔄 {stage_name} 시작...")
                    success = run_single_stage_analysis(stage_name, selected_code, selected_period)
                    if not success:
                        break
        
        progress_placeholder.empty()
        st.balloons()
        st.success("✨ 모든 분석이 완료되었습니다!")
        st.rerun()
    
    # 탭 버튼 렌더링
    render_tab_buttons(stages)
    
    st.markdown("---")
    
    # 활성 탭 내용 렌더링
    active_tab = st.session_state.active_tab
    
    for stage_name, stage_emoji in stages:
        if stage_name == active_tab:
            render_tab_header(stage_name)
            
            status = st.session_state.tab_states.get(stage_name, "waiting")
            
            if status == "waiting":
                st.info(f"⏳ {stage_name}을(를) 시작하려면 위의 탭을 클릭하거나 '모두 진행' 버튼을 누르세요.")
            elif status == "error":
                st.error(f"❌ {stage_name} 중에 오류가 발생했습니다.")
            else:
                # 진행 중 또는 완료된 탭: 결과 표시
                result = st.session_state.analysis_result
                price_df = st.session_state.price_df
                indicators_df = st.session_state.indicators_df
                stock_news = st.session_state.stock_news
                macro_news = st.session_state.macro_news
                
                if stage_name == "데이터 수집":
                    if price_df is not None and not price_df.empty:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("총 거래일", f"{len(price_df)}일")
                        with col2:
                            st.metric("현재가", f"{price_df['close'].iloc[-1]:,.0f}원")
                        with col3:
                            st.metric("데이터 기간", f"{price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")
                        render_chart_tab(price_df, None)
                    else:
                        st.warning("⚠️ 가격 데이터를 불러올 수 없습니다.")
                
                elif stage_name == "재무 데이터 수집":
                    if hasattr(st.session_state, 'financial_data') and st.session_state.financial_data is not None and not st.session_state.financial_data.empty:
                        st.success("✅ 재무 데이터 수집 완료")
                        st.json(st.session_state.financial_data.to_dict())
                    else:
                        st.warning("⚠️ 재무 데이터를 불러올 수 없습니다.")
                
                elif stage_name == "기술 분석":
                    if indicators_df is not None and not indicators_df.empty:
                        render_technical_tab(result, indicators_df)
                    else:
                        st.warning("⚠️ 기술 분석 데이터를 불러올 수 없습니다.")
                
                elif stage_name == "재무 분석":
                    if result is not None:
                        render_finance_tab(result)
                    else:
                        st.warning("⚠️ 재무 분석 데이터를 불러올 수 없습니다.")
                
                elif stage_name == "뉴스 수집":
                    render_news_tab(selected_name, result or {}, stock_news or [], macro_news or [])
                
                elif stage_name == "뉴스 분석":
                    if result and result.get("news", {}).get("analysis"):
                        render_news_tab(selected_name, result, stock_news or [], macro_news or [])
                    else:
                        st.warning("⚠️ 뉴스 분석 데이터를 불러올 수 없습니다.")
                
                elif stage_name == "LSTM 예측":
                    if result and result.get("lstm_prediction"):
                        render_lstm_tab(result)
                    else:
                        st.warning("⚠️ LSTM 예측 데이터를 불러올 수 없습니다.")
                
                elif stage_name == "AI 종합 분석":
                    if result and result.get("final_analysis"):
                        render_summary_tab(selected_name, result)
                    else:
                        st.warning("⚠️ AI 종합 분석 데이터를 불러올 수 없습니다.")
            break


def update_progress(stage_name: str, message: str, is_complete: bool = False):
    """진행 상황 업데이트 (탭 상태 변경)"""
    if is_complete:
        set_tab_status(stage_name, "completed")
    else:
        set_tab_status(stage_name, "running")


def show_completion_message():
    """분석 완료 메시지 표시"""
    pass  # run_full_analysis에서 처리
