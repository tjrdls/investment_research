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

    lstm_confidence = lstm_result.get("confidence", 0)
    lstm_pred_label = lstm_result.get("prediction", "")
    if lstm_confidence >= 0.70:
        lstm_weight_note = "고신뢰 신호 (confidence {:.0%}) — LSTM 예측을 핵심 근거로 활용하세요.".format(lstm_confidence)
    elif lstm_confidence >= 0.50:
        lstm_weight_note = "중간 신뢰 신호 (confidence {:.0%}) — 다른 신호와 교차 확인하세요.".format(lstm_confidence)
    else:
        lstm_weight_note = "저신뢰 신호 (confidence {:.0%}) — 기술·뉴스 신호를 우선하세요.".format(lstm_confidence)

    # 신호 충돌 감지
    lstm_direction = "상승" if lstm_prob_up > lstm_prob_down + 0.1 else ("하락" if lstm_prob_down > lstm_prob_up + 0.1 else "중립")
    tech_direction = "긍정" if tech_score > 0 else ("부정" if tech_score < 0 else "중립")
    news_direction = "호재" if "호재" in str(news_verdict) else ("악재" if "악재" in str(news_verdict) else "중립")
    conflict = (lstm_direction == "상승" and (tech_direction == "부정" or news_direction == "악재")) or \
               (lstm_direction == "하락" and (tech_direction == "긍정" or news_direction == "호재"))
    conflict_note = "신호 간 충돌 감지됨 — 보수적 판단을 권장합니다." if conflict else "신호 간 충돌 없음."

    prompt = """당신은 한국 주식시장 전문 애널리스트입니다.
{}

다음 분석 결과를 종합하여 {} 종목에 대한 투자 의견을 제시하세요.

▸ LSTM 기계학습 예측 [{}]:
  - 상승 확률 {:.1%}  하락 확률 {:.1%}

▸ 기술적 지표 [기술 방향: {}]:
  매수 신호: {}
  주의 신호: {}
  기술 신호 스코어: {}

▸ 뉴스 분석 [뉴스 방향: {}]:
  판정: {}
  점수: {:.2f}

▸ 밸류에이션 (TTM):
  PER: {}  PBR: {}  ROE: {}%
  부채비율: {}%

▸ 신호 일관성: {}

위 모든 정보를 종합하여 투자 의견을 제시하세요.
신뢰도가 높은 신호에 더 높은 비중을 두고, 신호 간 충돌이 있으면 반드시 해석 근거를 명시하세요.

반드시 아래 JSON만 출력하세요:
{{
  "summary": "종목 현재 상태 1문장",
  "recommendation": "매수 강력 추천 / 매수 / 관망 / 매도 / 매도 강력 추천",
  "signal_interpretation": "각 신호(LSTM·기술·뉴스)를 어떻게 해석하여 결론에 반영했는지 1문장",
  "conflict_resolution": "신호 간 충돌이 있다면 어떻게 판단했는지, 없으면 null",
  "target_upside": "목표가 상승률 예: +15%",
  "target_downside": "하방 위험 예: -10%",
  "risks": ["리스크1", "리스크2", "리스크3"],
  "opportunities": ["기회1", "기회2"],
  "strategy": "투자 전략 1‐2문장",
  "confidence": "신뢰도 0~100",
  "key_watch_points": ["주시사항1", "주시사항2"]
}}""".format(
        today_str, stock_name,
        lstm_weight_note,
        lstm_prob_up, lstm_prob_down,
        tech_direction, signals_str, warnings_str, tech_score,
        news_direction, news_verdict, news_score,
        per, pbr,
        roe if roe != "N/A" else "N/A",
        debt_ratio if debt_ratio != "N/A" else "N/A",
        conflict_note,
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

    sig_interp = analysis_result.get("signal_interpretation")
    conflict_res = analysis_result.get("conflict_resolution")
    if sig_interp:
        lines += ["▸ 신호 해석:", "  {}".format(sig_interp)]
    if conflict_res:
        lines += ["▸ 충돌 해소:", "  {}".format(conflict_res)]

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
