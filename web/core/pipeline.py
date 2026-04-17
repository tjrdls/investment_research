# -*- coding: utf-8 -*-
"""
역할: 분석 파이프라인 관리
데이터 수집부터 최종 분석까지의 전체 흐름 제어
"""

import sys
import os
import streamlit as st

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipeline.prediction_pipeline import run_analysis
from data_loader.price.data_collector import collect_price_data
from analysis.indicators.technical_indicators import calculate_indicators
from analysis.news_analyzer import collect_stock_news, collect_macro_news


class AnalysisPipeline:
    """분석 파이프라인 클래스"""

    def __init__(self):
        self.stages = [
            self._collect_data,
            self._calculate_indicators,
            self._run_full_analysis,
            self._collect_news
        ]

    def run_pipeline(self, stock_code: str, period: str):
        """전체 파이프라인 실행"""
        try:
            # 1. 데이터 수집
            yield "데이터 수집", "데이터를 수집 중입니다..."
            self._collect_data(stock_code, period)
            yield "데이터 수집", "✅ 데이터 수집 완료"

            # 2. 기술 지표 계산
            yield "기술 분석", "기술적 지표를 계산 중입니다..."
            self._calculate_indicators()
            yield "기술 분석", "✅ 기술 분석 완료"

            # 3. 전체 분석 실행
            yield "재무 분석", "재무 데이터를 분석 중입니다..."
            yield "AI 분석", "AI 모델로 분석 중입니다..."
            result = self._run_full_analysis(stock_code)
            yield "재무 분석", "✅ 재무 분석 완료"
            yield "AI 분석", "✅ AI 분석 완료"

            # 4. 뉴스 수집
            yield "뉴스 수집", "최신 뉴스를 수집 중입니다..."
            self._collect_news(stock_code)
            yield "뉴스 수집", "✅ 뉴스 수집 완료"

            return result

        except Exception as e:
            raise Exception(f"파이프라인 실행 중 오류: {str(e)}")

    def _collect_data(self, stock_code: str, period: str):
        """데이터 수집"""
        st.session_state.price_df = collect_price_data(stock_code, period=period)
        st.session_state.indicators_df = calculate_indicators(st.session_state.price_df)

    def _calculate_indicators(self):
        """기술 지표 계산 (이미 데이터 수집 시 함께 수행)"""
        pass

    def _run_full_analysis(self, stock_code: str):
        """전체 분석 실행"""
        result = run_analysis(stock_code)
        st.session_state.analysis_result = result
        return result

    def _collect_news(self, stock_name: str):
        """뉴스 수집"""
        # 종목명 추출 (간단한 방식)
        stock_name_map = {
            "005930": "삼성전자",
            "000660": "SK하이닉스",
            "373220": "LG에너지솔루션",
            "207940": "삼성바이오로직스",
            "005380": "현대차"
        }
        display_name = stock_name_map.get(stock_name, stock_name)

        st.session_state.stock_news, st.session_state.macro_news = load_news(display_name)


def load_news(stock_name: str):
    """뉴스 로드 (캐싱)"""
    return collect_stock_news(stock_name, [stock_name]), collect_macro_news()