# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from pipeline.prediction_pipeline import run_analysis
from data_loader.price.data_collector import collect_price_data, get_top_stocks
from analysis.indicators.technical_indicators import calculate_indicators
from analysis.news_analyzer import collect_stock_news, collect_macro_news

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

# 분석 진행 상태
ANALYSIS_STAGES = [
    ("데이터 수집", "💾"),
    ("기술 분석", "📊"),
    ("재무 분석", "💰"),
    ("뉴스 수집", "📰"),
    ("AI 분석", "🤖"),
]


def format_value(value, digits=2, suffix=""):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{suffix}"
    return f"{value}{suffix}"


@st.cache_data(show_spinner=False)
def get_stock_options():
    try:
        top_stocks = get_top_stocks("KOSPI", 10)
        if top_stocks:
            return top_stocks
    except Exception:
        pass
    return DEFAULT_STOCKS


@st.cache_data(show_spinner=False)
def load_price_data(stock_code: str, period: str):
    return collect_price_data(stock_code, period=period)


@st.cache_data(show_spinner=False)
def load_news(stock_name: str):
    return collect_stock_news(stock_name, [stock_name]), collect_macro_news()


def build_price_chart(price_df: pd.DataFrame, indicators_df: pd.DataFrame):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "xy"}], [{"type": "bar"}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df["open"],
            high=price_df["high"],
            low=price_df["low"],
            close=price_df["close"],
            name="캔들 차트",
            increasing_line_color="#0f9d58",
            decreasing_line_color="#db4437",
        ),
        row=1,
        col=1,
    )

    if "sma_20" in indicators_df.columns:
        fig.add_trace(
            go.Scatter(
                x=indicators_df.index,
                y=indicators_df["sma_20"],
                mode="lines",
                name="20일 이동평균",
                line=dict(color="#f4b400", width=1.5),
            ),
            row=1,
            col=1,
        )
    if "sma_50" in indicators_df.columns:
        fig.add_trace(
            go.Scatter(
                x=indicators_df.index,
                y=indicators_df["sma_50"],
                mode="lines",
                name="50일 이동평균",
                line=dict(color="#4285f4", width=1.5),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=price_df.index,
            y=price_df["volume"],
            name="거래량",
            marker_color="#6b6b6b",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )

    fig.update_yaxes(title_text="가격 (원)", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    return fig


def build_technical_summary(indicators_df: pd.DataFrame):
    latest = indicators_df.iloc[-1]
    prev = indicators_df.iloc[-2] if len(indicators_df) > 1 else latest

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


def build_value_judgement(valuation: dict):
    per = valuation.get("PER")
    debt = valuation.get("debt_ratio")
    if per is None:
        valuation_status = "판단불가"
    elif per < 8:
        valuation_status = "저평가"
    elif per <= 18:
        valuation_status = "적정"
    else:
        valuation_status = "고평가"

    if debt is None:
        finance_status = "판단불가"
    elif debt < 50:
        finance_status = "양호"
    elif debt < 80:
        finance_status = "주의"
    else:
        finance_status = "취약"

    return valuation_status, finance_status


def render_summary_tab(stock_name: str, result: dict):
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


def render_chart_tab(price_df: pd.DataFrame, indicators_df: pd.DataFrame):
    st.markdown("#### 차트")
    fig = build_price_chart(price_df, indicators_df)
    st.plotly_chart(fig, use_container_width=True)


def render_technical_tab(result: dict, indicators_df: pd.DataFrame):
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


def render_finance_tab(result: dict):
    st.markdown("#### 재무 분석")
    valuation = result.get("valuation", {}) or {}
    valuation_status, finance_status = build_value_judgement(valuation)

    metrics = pd.DataFrame(
        {
            "지표": ["PER", "PBR", "ROE", "부채비율"],
            "값": [
                format_value(valuation.get("PER")),
                format_value(valuation.get("PBR")),
                format_value(valuation.get("ROE"), digits=2, suffix="%"),
                format_value(valuation.get("debt_ratio"), digits=2, suffix="%"),
            ],
        }
    )
    st.table(metrics)

    judgement = pd.DataFrame(
        {
            "판정": ["밸류에이션", "재무 안정성"],
            "결과": [valuation_status, finance_status],
        }
    )
    st.table(judgement)


def render_news_tab(stock_name: str, result: dict, stock_news: list, macro_news: list):
    st.markdown("#### 뉴스 리스트")
    if stock_news:
        st.markdown("**종목 뉴스**")
        for news in stock_news[:5]:
            st.write(f"- {news.get('title')} ({news.get('source')})")
            if news.get("description"):
                st.write(f"  - {news.get('description')}")
    else:
        st.write("관련 뉴스 없음")

    if macro_news:
        st.markdown("**매크로 뉴스**")
        for news in macro_news[:5]:
            st.write(f"- {news.get('title')} ({news.get('source')})")
    st.markdown("---")

    st.markdown("#### 뉴스 분석 결과")
    news_analysis = result.get("news", {}).get("analysis", {}) or {}
    st.write(f"- 뉴스 판정: {news_analysis.get('verdict', 'N/A')}")
    st.write(f"- 점수: {format_value(news_analysis.get('score'), digits=2)}")
    st.write(f"- Bullish: {format_value(news_analysis.get('bullish_prob'), digits=0)}%")
    st.write(f"- Bearish: {format_value(news_analysis.get('bearish_prob'), digits=0)}%")
    st.write(f"- Caution: {format_value(news_analysis.get('caution_prob'), digits=0)}%")


def main():
    st.set_page_config(page_title="투자 분석 웹 대시보드", layout="wide")

    st.title("📈 AI 투자 분석 플랫폼")
    st.markdown("종목을 선택하고 분석을 시작하면 데이터 흐름에 따라 결과가 표시됩니다.")

    # ==================== 세션 상태 초기화 (먼저 실행) ====================
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
        st.session_state.price_df = None
        st.session_state.indicators_df = None
        st.session_state.stock_news = None
        st.session_state.macro_news = None
        st.session_state.analysis_stock = None
        st.session_state.analysis_stages = {stage[0]: False for stage in ANALYSIS_STAGES}
        st.session_state.lstm_mode = "default"
        st.session_state.gpt_model = "gpt-5.4-nano"

    # ==================== 사이드바 설정 ====================
    sidebar = st.sidebar
    sidebar.header("⚙️ 분석 설정")

    stock_options = get_stock_options()
    stock_labels = [f"{code} {name}" for code, name in stock_options]
    selected_option = sidebar.selectbox("📌 종목 선택", stock_labels, index=0)
    selected_code, selected_name = selected_option.split(" ", 1)

    selected_period_label = sidebar.selectbox("📅 분석 기간", list(PERIOD_OPTIONS.keys()), index=0)
    selected_period = PERIOD_OPTIONS[selected_period_label]

    # LSTM 모델 설정 섹션
    sidebar.markdown("---")
    sidebar.markdown("### 🧠 LSTM 모델")
    lstm_options = sidebar.radio("모델 선택", ["기본 제공 (빠름)", "커스텀 학습 (정확)"], index=0)
    use_custom_lstm = lstm_options == "커스텀 학습 (정확)"
    
    if use_custom_lstm:
        st.session_state.lstm_mode = "custom"
        sidebar.info("💡 커스텀 모델을 사용합니다. (시간이 더 소요될 수 있습니다)")
    else:
        st.session_state.lstm_mode = "default"

    # GPT 모델 설정 섹션
    sidebar.markdown("---")
    sidebar.markdown("### 🤖 AI 모델 (GPT)")
    gpt_models = ["gpt-5.4-nano", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    selected_gpt_model = sidebar.selectbox("모델 선택", gpt_models, index=0)
    st.session_state.gpt_model = selected_gpt_model
    sidebar.caption(f"선택된 모델: {selected_gpt_model}")

    # 분석 시작 버튼
    start_analysis = sidebar.button("🚀 분석 시작", use_container_width=True, type="primary")

    # ==================== 분석 실행 ====================
    if start_analysis and st.session_state.analysis_stock != selected_code:
        st.session_state.analysis_stock = selected_code
        
        # 진행 상황 해더 및 플레이스홀더
        progress_container = st.container(border=True)
        progress_header = progress_container.empty()
        progress_bars = {stage[0]: progress_container.empty() for stage in ANALYSIS_STAGES}
        
        with progress_container:
            st.markdown("### 📊 분석 진행 상황")
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"🧠 LSTM: {st.session_state.lstm_mode}")
            with col2:
                st.caption(f"🤖 GPT: {st.session_state.gpt_model}")
        
        # 분석 수행
        try:
            # 1. 데이터 수집
            with progress_bars[ANALYSIS_STAGES[0][0]]:
                st.progress(0.2)
                st.text("데이터를 수집 중입니다...")
            
            st.session_state.price_df = load_price_data(selected_code, selected_period)
            st.session_state.indicators_df = calculate_indicators(st.session_state.price_df)
            st.session_state.analysis_stages[ANALYSIS_STAGES[0][0]] = True
            progress_bars[ANALYSIS_STAGES[0][0]].success("✅ " + ANALYSIS_STAGES[0][0] + " 완료")
            
            # 2-4. 분석 수행
            with progress_bars[ANALYSIS_STAGES[1][0]]:
                st.progress(0.4)
                st.text("AI 분석을 수행 중입니다...")
            
            # 전체 분석 실행
            st.session_state.analysis_result = run_analysis(selected_code)
            
            for i in range(1, 4):
                st.session_state.analysis_stages[ANALYSIS_STAGES[i][0]] = True
                progress_bars[ANALYSIS_STAGES[i][0]].success("✅ " + ANALYSIS_STAGES[i][0] + " 완료")
            
            # 5. 뉴스 수집
            with progress_bars[ANALYSIS_STAGES[4][0]]:
                st.progress(0.9)
                st.text("AI 분석을 완료했습니다...")
            
            st.session_state.stock_news, st.session_state.macro_news = load_news(selected_name)
            st.session_state.analysis_stages[ANALYSIS_STAGES[4][0]] = True
            progress_bars[ANALYSIS_STAGES[4][0]].success("✅ " + ANALYSIS_STAGES[4][0] + " 완료")
            
            # 완료 메시지
            st.balloons()
            st.success("✨ 분석이 완료되었습니다! 아래에서 결과를 확인하세요.")
            
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

    # ==================== 종합 분석 ====================
    with st.expander("🎯 **종합 분석 요약**", expanded=True):
        render_summary_tab(selected_name, result)

    # ==================== 차트 분석 ====================
    with st.expander("📈 **가격 차트 및 거래량**", expanded=True):
        if st.session_state.price_df is not None and not st.session_state.price_df.empty:
            render_chart_tab(st.session_state.price_df, st.session_state.indicators_df)
        else:
            st.warning("⚠️ 차트 데이터를 불러올 수 없습니다.")

    # ==================== 기술 분석 ====================
    with st.expander("📊 **기술 분석**", expanded=False):
        if st.session_state.indicators_df is not None and not st.session_state.indicators_df.empty:
            render_technical_tab(result, st.session_state.indicators_df)
        else:
            st.warning("⚠️ 기술 분석 데이터를 불러올 수 없습니다.")

    # ==================== 재무 분석 ====================
    with st.expander("💰 **재무 분석**", expanded=False):
        render_finance_tab(result)

    # ==================== 뉴스 분석 ====================
    with st.expander("📰 **뉴스 및 시장 분석**", expanded=False):
        render_news_tab(selected_name, result, st.session_state.stock_news or [], st.session_state.macro_news or [])

    # ==================== LSTM 예측 결과 ====================
    lstm_pred = result.get("lstm_prediction", {}) or {}
    if lstm_pred:
        st.markdown("---")
        with st.expander("🤖 **LSTM 모델 예측**", expanded=False):
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



if __name__ == "__main__":
    main()
