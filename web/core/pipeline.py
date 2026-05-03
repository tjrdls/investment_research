# -*- coding: utf-8 -*-
"""
역할: 분석 파이프라인 관리 — 데이터 수집부터 최종 분석까지의 전체 흐름 제어
"""

import sys
import os
import logging

import streamlit as st

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import STOCK_CODE_TO_NAME
from pipeline.prediction_pipeline import run_analysis
from data_loader.price.data_collector import collect_price_data
from analysis.indicators.technical_indicators import calculate_indicators
from analysis.news_analyzer import collect_stock_news, collect_macro_news

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """분석 파이프라인 클래스."""

    def run_pipeline(self, stock_code: str, period: str):
        """전체 파이프라인 실행 (제너레이터)."""
        try:
            yield "데이터 수집", "데이터를 수집 중입니다..."
            self._collect_data(stock_code, period)
            yield "데이터 수집", "✅ 데이터 수집 완료"

            yield "기술 분석", "기술적 지표를 계산 중입니다..."
            yield "기술 분석", "✅ 기술 분석 완료"

            yield "재무 분석", "재무 데이터를 분석 중입니다..."
            yield "AI 분석", "AI 모델로 분석 중입니다..."
            result = self._run_full_analysis(stock_code)
            yield "재무 분석", "✅ 재무 분석 완료"
            yield "AI 분석", "✅ AI 분석 완료"

            yield "뉴스 수집", "최신 뉴스를 수집 중입니다..."
            self._collect_news(stock_code)
            yield "뉴스 수집", "✅ 뉴스 수집 완료"

            return result

        except Exception as e:
            raise RuntimeError("파이프라인 실행 중 오류: {}".format(e)) from e

    def _collect_data(self, stock_code: str, period: str) -> None:
        st.session_state.price_df = collect_price_data(stock_code, period=period)
        st.session_state.indicators_df = calculate_indicators(st.session_state.price_df)

    def _run_full_analysis(self, stock_code: str):
        result = run_analysis(stock_code)
        st.session_state.analysis_result = result
        return result

    def _collect_news(self, stock_code: str) -> None:
        name = STOCK_CODE_TO_NAME.get(stock_code, stock_code)
        st.session_state.stock_news, st.session_state.macro_news = load_news(name)


def load_news(stock_name: str) -> tuple:
    return collect_stock_news(stock_name, [stock_name]), collect_macro_news()
