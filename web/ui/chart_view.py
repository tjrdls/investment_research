# -*- coding: utf-8 -*-
"""
역할: 차트 및 시각화 UI 컴포넌트
가격 차트, 기술 지표 차트 등
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_price_chart(price_df: pd.DataFrame, indicators_df: pd.DataFrame):
    """가격 차트 생성"""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "xy"}], [{"type": "bar"}]],
    )

    # 캔들 차트
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

    # 이동평균선 (indicators_df가 있을 때만)
    if indicators_df is not None and not indicators_df.empty:
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

    # 거래량
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

    # 레이아웃 설정
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )

    fig.update_yaxes(title_text="가격 (원)", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    return fig


def render_chart_tab(price_df: pd.DataFrame, indicators_df: pd.DataFrame):
    """차트 탭 렌더링"""
    st.markdown("#### 차트")
    if price_df is not None and not price_df.empty:
        fig = build_price_chart(price_df, indicators_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 차트를 표시할 데이터가 없습니다.")