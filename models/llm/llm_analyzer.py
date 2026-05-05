# -*- coding: utf-8 -*-
"""
역할: LSTM 결과, 기술적 분석, 뉴스 분석, 밸류에이션을 종합하여
OpenAI API로 최종 투자 분석을 수행하는 모듈.
"""

import json
import logging
import os
from typing import Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv
from datetime import date

from config import GPT_MODEL, GPT_TEMPERATURE, GPT_MAX_TOKENS_ANALYSIS, GPT_MAX_TOKENS_VALUATION

load_dotenv()

logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


def analyze_with_llm(
    stock_name: str,
    lstm_result: dict,
    tech_signals: dict,
    news_analysis: Optional[dict],
    valuation_metrics: dict,
    model: Optional[str] = None,
) -> Optional[dict]:
    """
    모든 분석 결과를 종합하여 최종 투자 의견 생성.

    :param tech_signals: TechnicalSignals dict (signals, warnings, score)
    :return: 최종 의견 dict 또는 None
    """
    logger.info("[LLM] 최종 분석 중...")

    model_choice = model or GPT_MODEL

    signals_list: List[str] = tech_signals.get("signals") or []
    warnings_list: List[str] = tech_signals.get("warnings") or []
    tech_score = tech_signals.get("score", 0)

    signals_str = "\n".join(signals_list) if signals_list else "신호 없음"
    warnings_str = "\n".join(warnings_list) if warnings_list else "경고 없음"

    lstm_prob_up = lstm_result.get("probabilities", {}).get("상승", 0)
    lstm_prob_down = lstm_result.get("probabilities", {}).get("하락", 0)

    news_verdict = (news_analysis or {}).get("verdict", "N/A")
    news_score = (news_analysis or {}).get("score", 0)

    per = valuation_metrics.get("PER", "N/A")
    pbr = valuation_metrics.get("PBR", "N/A")
    roe = valuation_metrics.get("ROE", "N/A")
    debt_ratio = valuation_metrics.get("debt_ratio", "N/A")

    today_str = date.today().strftime("%Y년 %m월 %d일")

    prompt = """당신은 한국 주식시장 전문 애널리스트입니다.
{}

다음 분석 결과를 종합하여 {} 종목에 대한 투자 의견을 제시하세요.

▸ LSTM 기계학습 예측:
  - 상승 확률 {:.1%}  하락 확률 {:.1%}

▸ 기술적 지표:
  매수 신호: {}
  주의 신호: {}
  기술 신호 스코어: {}

▸ 뉴스 분석:
  판정: {}
  점수: {:.2f}

▸ 밸류에이션 (TTM):
  PER: {}  PBR: {}  ROE: {}%
  부채비율: {}%

위 모든 정보를 종합하여:
1. 종목의 현재 상태를 1문장으로 요약
2. 매수/매도/관망 투자 의견 제시
3. 주요 리스크 및 기회 각 2‐3개
4. 목표 시점 및 매매 전략

반드시 아래 JSON만 출력하세요:
{{
  "summary": "종목 현재 상태 1문장",
  "recommendation": "매수 강력 추천 / 매수 / 관망 / 매도 / 매도 강력 추천",
  "target_upside": "목표가 상승률 예: +15%",
  "target_downside": "하방 위험 예: -10%",
  "risks": ["리스크1", "리스크2", "리스크3"],
  "opportunities": ["기회1", "기회2"],
  "strategy": "투자 전략 1‐2문장",
  "confidence": "신뢰도 0~100(%)",
  "key_watch_points": ["주시사항1", "주시사항2"]
}}""".format(
        today_str, stock_name,
        lstm_prob_up, lstm_prob_down,
        signals_str, warnings_str, tech_score,
        news_verdict, news_score,
        per, pbr,
        roe if roe != "N/A" else "N/A",
        debt_ratio if debt_ratio != "N/A" else "N/A",
    )

    try:
        resp = openai_client.chat.completions.create(
            model=model_choice,
            messages=[{"role": "user", "content": prompt}],
            temperature=GPT_TEMPERATURE,
            max_tokens=GPT_MAX_TOKENS_ANALYSIS,
        )
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        analysis = json.loads(raw)
        logger.info("✅ 분석 완료")
        return analysis

    except json.JSONDecodeError as e:
        logger.warning("⚠️ JSON 파싱 실패: %s", e)
        return None
    except Exception as e:
        logger.warning("⚠️ LLM 오류: %s", e)
        return None


def format_final_report(stock_name: str, stock_code: str, analysis_result: dict, current_price: float = None) -> str:
    if not analysis_result:
        return "분석 실패: {}({})".format(stock_name, stock_code)

    lines: List[str] = [
        "╔" + "═" * 58 + "╗",
        "  📊 {} ({})".format(stock_name, stock_code),
        "╚" + "═" * 58 + "╝",
        "",
        "▸ 현재 상태:",
        "  {}".format(analysis_result.get("summary", "N/A")),
        "",
        "▸ 투자 의견: {} (신뢰도 {}%)".format(
            analysis_result.get("recommendation", "N/A"),
            analysis_result.get("confidence", "N/A"),
        ),
    ]

    upside = analysis_result.get("target_upside", "N/A")
    downside = analysis_result.get("target_downside", "N/A")
    if current_price:
        lines.append("  목표: {} / 위험: {}".format(upside, downside))

    lines.append("")

    risks = analysis_result.get("risks", [])
    if risks:
        lines.append("▸ 주요 리스크:")
        lines.extend("  ⛔ {}".format(r) for r in risks)

    opps = analysis_result.get("opportunities", [])
    if opps:
        lines.append("▸ 주요 기회:")
        lines.extend("  ✅ {}".format(o) for o in opps)

    lines.append("")

    strategy = analysis_result.get("strategy", "")
    if strategy:
        lines += ["▸ 투자 전략:", "  {}".format(strategy)]

    watch = analysis_result.get("key_watch_points", [])
    if watch:
        lines.append("▸ 주시사항:")
        lines.extend("  📌 {}".format(w) for w in watch)

    lines += ["", "⚠️  참고용 분석이며 투자 책임은 투자자에게 있습니다."]
    return "\n".join(lines)


if __name__ == "__main__":
    lstm_result = {"probabilities": {"상승": 0.6, "하락": 0.2, "횡보": 0.2}, "confidence": 0.6}
    tech_signals = {"signals": ["✅ MACD 상향"], "warnings": ["⚠️  RSI 고위험"], "score": 1}
    news_analysis = {"verdict": "호재", "score": 0.5}
    valuation = {"PER": 15.5, "PBR": 1.2, "ROE": 18.5, "debt_ratio": 40.0}

    result = analyze_with_llm("삼성전자", lstm_result, tech_signals, news_analysis, valuation)
    if result:
        print(format_final_report("삼성전자", "005930", result, 70000))
