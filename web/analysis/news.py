# -*- coding: utf-8 -*-
"""
역할: 뉴스 분석 UI 컴포넌트
종목 뉴스 및 매크로 뉴스 표시
"""

import streamlit as st
from .technical import format_value


def render_news_tab(stock_name: str, result: dict, stock_news: list, macro_news: list):
    """뉴스 분석 탭 렌더링"""
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
    if result is None:
        st.warning("⚠️ 뉴스 분석 결과가 없습니다.")
        return
        
    news_analysis = result.get("news", {}).get("analysis", {}) or {}
    st.write(f"- 뉴스 판정: {news_analysis.get('verdict', 'N/A')}")
    st.write(f"- 점수: {format_value(news_analysis.get('score'), digits=2)}")
    st.write(f"- Bullish: {format_value(news_analysis.get('bullish_prob'), digits=0)}%")
    st.write(f"- Bearish: {format_value(news_analysis.get('bearish_prob'), digits=0)}%")
    st.write(f"- Caution: {format_value(news_analysis.get('caution_prob'), digits=0)}%")