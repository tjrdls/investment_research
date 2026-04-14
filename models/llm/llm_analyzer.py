# -*- coding: utf-8 -*-
"""
역할: LSTM 결과, 기술적 분석, 뉴스 분석, 밸류에이션을 종합하여
OpenAI API로 최종 투자 분석을 수행하는 모듈.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from datetime import date

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


def analyze_with_llm(stock_name, lstm_result, tech_signals, news_analysis, valuation_metrics):
    """
    모든 분석 결과를 종합하여 최종 투자 의견 생성
    
    :param stock_name: 종목명
    :param lstm_result: LSTM 예측 dict
    :param tech_signals: 기술적 신호 (signals, warnings, score)
    :param news_analysis: 뉴스 분석 dict
    :param valuation_metrics: 밸류에이션 dict
    :return: 최종 의견 string
    """
    
    print("   🧠 LLM 최종 분석 중...")
    
    tech_signals_text, tech_warnings_text, tech_score = tech_signals
    
    signals_str = "\n".join(tech_signals_text) if tech_signals_text else "신호 없음"
    warnings_str = "\n".join(tech_warnings_text) if tech_warnings_text else "경고 없음"
    
    lstm_prob_up = lstm_result.get("probabilities", {}).get("상승", 0)
    lstm_prob_down = lstm_result.get("probabilities", {}).get("하락", 0)
    
    news_verdict = news_analysis.get("verdict", "중립") if news_analysis else "N/A"
    news_score = news_analysis.get("score", 0) if news_analysis else 0
    
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
4. 목표 시점 및 매매 전략 (예: "단기 상승 기대 시 매수, 저항선 돌파 시 익절")

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
        today_str,
        stock_name,
        lstm_prob_up,
        lstm_prob_down,
        signals_str,
        warnings_str,
        tech_score,
        news_verdict,
        news_score,
        per,
        pbr,
        roe if roe != "N/A" else "N/A",
        debt_ratio if debt_ratio != "N/A" else "N/A"
    )
    
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )
        
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        
        analysis = json.loads(raw)
        print("   ✅ 분석 완료")
        return analysis
    
    except json.JSONDecodeError as e:
        print("    ⚠️  JSON 파싱 실패: {}".format(str(e)))
        return None
    except Exception as e:
        print("    ⚠️  LLM 오류: {}".format(str(e)))
        return None


def format_final_report(stock_name, stock_code, analysis_result, current_price=None):
    """
    최종 분석 리포트 포맷팅
    
    :param stock_name: 종목명
    :param stock_code: 종목 코드
    :param analysis_result: LLM 분석 결과 dict
    :param current_price: 현재 주가
    :return: formatted report string
    """
    
    if not analysis_result:
        return "분석 실패: {}({})".format(stock_name, stock_code)
    
    report = []
    report.append("╔" + "═" * 58 + "╗")
    report.append("  📊 {} ({})".format(stock_name, stock_code))
    report.append("╚" + "═" * 58 + "╝")
    report.append("")
    
    # 요약
    report.append("▸ 현재 상태:")
    report.append("  {}".format(analysis_result.get("summary", "N/A")))
    report.append("")
    
    # 투자 의견
    rec = analysis_result.get("recommendation", "N/A")
    report.append("▸ 투자 의견: {} (신뢰도 {}%)".format(
        rec,
        analysis_result.get("confidence", "N/A")
    ))
    
    # 목표가
    upside = analysis_result.get("target_upside", "N/A")
    downside = analysis_result.get("target_downside", "N/A")
    if current_price:
        report.append("  목표: {} / 위험: {}".format(upside, downside))
    
    report.append("")
    
    # 리스크/기회
    risks = analysis_result.get("risks", [])
    if risks:
        report.append("▸ 주요 리스크:")
        for r in risks:
            report.append("  ⛔ {}".format(r))
    
    opps = analysis_result.get("opportunities", [])
    if opps:
        report.append("▸ 주요 기회:")
        for o in opps:
            report.append("  ✅ {}".format(o))
    
    report.append("")
    
    # 투자 전략
    strategy = analysis_result.get("strategy", "")
    if strategy:
        report.append("▸ 투자 전략:")
        report.append("  {}".format(strategy))
    
    # 주시사항
    watch = analysis_result.get("key_watch_points", [])
    if watch:
        report.append("▸ 주시사항:")
        for w in watch:
            report.append("  📌 {}".format(w))
    
    report.append("")
    report.append("⚠️  참고용 분석이며 투자 책임은 투자자에게 있습니다.")
    
    return "\n".join(report)


if __name__ == "__main__":
    # 테스트
    lstm_result = {
        "probabilities": {"상승": 0.6, "하락": 0.2, "횡보": 0.2},
        "confidence": 0.6
    }
    
    tech_signals = (
        ["✅ MACD 상향"],
        ["⚠️  RSI 고위험"],
        1
    )
    
    news_analysis = {
        "verdict": "호재",
        "score": 0.5
    }
    
    valuation = {
        "PER": 15.5,
        "PBR": 1.2,
        "ROE": 18.5,
        "debt_ratio": 40.0
    }
    
    result = analyze_with_llm(
        "삼성전자",
        lstm_result,
        tech_signals,
        news_analysis,
        valuation
    )
    
    if result:
        report = format_final_report("삼성전자", "005930", result, 70000)
        print(report)
