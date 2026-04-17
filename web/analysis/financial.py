# -*- coding: utf-8 -*-
"""
역할: 재무 분석 UI 컴포넌트
재무 지표 계산 및 표시
"""

import streamlit as st
import pandas as pd
from .technical import format_value


def build_value_judgement(valuation: dict):
    """밸류에이션 판단"""
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


def render_finance_tab(result: dict):
    """재무 분석 탭 렌더링"""
    st.markdown("#### 재무 분석")
    valuation = result.get("valuation", {}) if result else {}
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