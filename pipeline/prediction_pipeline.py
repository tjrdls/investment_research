# -*- coding: utf-8 -*-
"""
역할: 투자 분석 파이프라인의 핵심 모듈. 데이터 수집부터 LLM 분석까지의 전체 흐름을 조율한다.
각 모듈을 순차적으로 호출하여 최종 분석을 생성한다.
"""

import sys
import os
import time
import pickle
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.price.data_collector import collect_price_data, get_top_stocks
from data.financial.financial_collector import collect_financial_data, get_corp_code_map
from analysis.indicators.technical_indicators import calculate_indicators, get_technical_signals
from analysis.news_analyzer import collect_stock_news, collect_macro_news, analyze_news_with_gpt
from analysis.valuation_analyzer import calculate_ttm_metrics, get_industry_valuation_from_gpt
from models.lstm.lstm_model import prepare_dataset, predict_next_trend
from models.llm.llm_analyzer import analyze_with_llm, format_final_report
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# 설정값
CACHE_PATH = "text_embeddings_cache.pkl"
MODEL_PATH = "best_multimodal_stock_model.pth"
SEQ_LEN = 20
PRED_DAYS = 5
THRESHOLD = 0.02


def get_text_embedding(text):
    """
    OpenAI Embedding API로 텍스트를 벡터로 변환
    
    :param text: 텍스트
    :return: 1536-d numpy array
    """
    try:
        resp = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text[:8000]
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print("    ⚠️  Embedding 실패: {}".format(str(e)))
        # 폴백: 랜덤 벡터
        return np.random.randn(1536).astype(np.float32)


def build_context_text(stock_name, financial_df):
    """
    DART 재무 정보로부터 컨텍스트 텍스트 구성 (Embedding용)
    
    :param stock_name: 회사명
    :param financial_df: 재무 데이터 DataFrame
    :return: 텍스트
    """
    if financial_df.empty:
        return "{} 코스피 상장 주식".format(stock_name)
    
    parts = [stock_name]
    
    for _, row in financial_df.iterrows():
        rev = row.get("revenue", 0) / 1e8 if row.get("revenue") else 0
        net = row.get("net_income", 0) / 1e8 if row.get("net_income") else 0
        roe = row.get("roe", 0)
        
        year = int(row.get("year", 2025))
        report = row.get("report_type", "")
        
        parts.append("[{}년 {}] 매출 {:.0f}억 순이익 {:.0f}억 ROE {:.1f}%".format(
            year, report, rev, net, roe if roe else 0
        ))
    
    return " | ".join(parts)[:4000]


def run_single_stock_analysis(stock_code, stock_name):
    """
    단일 종목 분석 실행
    
    :param stock_code: 종목 코드 (예: "005930")
    :param stock_name: 종목명 (예: "삼성전자")
    :return: dict with analysis results
    """
    
    print("\n" + "━" * 60)
    print("  📊 {} ({}) 분석".format(stock_name, stock_code))
    print("━" * 60)
    
    result = {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "timestamp": datetime.now().isoformat(),
        "status": "진행중"
    }
    
    try:
        # 1. 주가 데이터 수집
        print("\n  [1/6] 주가 데이터 수집...")
        price_df = collect_price_data(stock_code)
        
        if price_df.empty:
            print("     ⚠️  주가 데이터 수집 실패")
            result["status"] = "실패"
            return result
        
        current_price = price_df["close"].iloc[-1]
        result["current_price"] = float(current_price)
        print("     ✅ 완료 ({} 거래일, 현재가 {:,.0f}원)".format(len(price_df), current_price))
        
        # 2. 기술적 지표 계산
        print("\n  [2/6] 기술적 지표 계산...")
        indicators_df = calculate_indicators(price_df)
        
        if indicators_df.empty:
            print("     ⚠️  지표 계산 실패")
            result["status"] = "실패"
            return result
        
        tech_signals, tech_warnings, tech_score = get_technical_signals(indicators_df)
        result["technical"] = {
            "signals": tech_signals,
            "warnings": tech_warnings,
            "score": tech_score
        }
        print("     ✅ 완료 (신호 {}, 경고 {})".format(len(tech_signals), len(tech_warnings)))
        
        # 3. 재무 데이터 수집
        print("\n  [3/6] 재무 데이터 수집 (DART)...")
        corp_map = get_corp_code_map()
        corp_code = corp_map.get(stock_code, {}).get("corp_code")
        
        if corp_code:
            financial_df = collect_financial_data(stock_code, corp_code)
            print("     ✅ 완료 ({} 기간)".format(len(financial_df)))
        else:
            print("     ⚠️  기업코드 없음 - 스킵")
            financial_df = None
        
        result["financial"] = financial_df.to_dict() if financial_df is not None else {}
        
        # 4. 텍스트 임베딩 생성
        print("\n  [4/6] 텍스트 임베딩 생성...")
        context_text = build_context_text(stock_name, financial_df) if financial_df is not None else stock_name
        
        text_emb = get_text_embedding(context_text)
        result["text_embedding"] = text_emb.tolist()
        print("     ✅ 완료 ({:.0f}차원)".format(len(text_emb)))
        
        # 5. LSTM 예측
        print("\n  [5/6] LSTM 예측...")
        try:
            lstm_pred = predict_next_trend(
                None,  # 모델 없이 진행 가능
                indicators_df,
                text_emb,
                seq_len=SEQ_LEN
            )
            result["lstm_prediction"] = lstm_pred
            print("     ✅ {}: 신뢰도 {:.1%}".format(
                lstm_pred["prediction"],
                lstm_pred["confidence"]
            ))
        except Exception as e:
            print("     ⚠️  LSTM 예측 실패: {}".format(str(e)))
            lstm_pred = {
                "prediction": "기술 오류",
                "probabilities": {"상승": 0.33, "하락": 0.33, "횡보": 0.34},
                "confidence": 0.0
            }
            result["lstm_prediction"] = lstm_pred
        
        time.sleep(0.5)
        
        # 6. 뉴스 분석
        print("\n  [6/6] 뉴스 분석 (NewsAPI + GPT)...")
        stock_news = collect_stock_news(stock_name, [stock_name, "{}({})".format(stock_name, stock_code)])
        macro_news = collect_macro_news()
        
        news_analysis = analyze_news_with_gpt(stock_name, stock_news, macro_news)
        result["news"] = {
            "stock_count": len(stock_news),
            "macro_count": len(macro_news),
            "analysis": news_analysis
        }
        
        if news_analysis:
            print("     ✅ {}: {}".format(
                news_analysis.get("verdict", "중립"),
                news_analysis.get("recommendation", "관망")
            ))
        else:
            print("     ⚠️  뉴스 분석 실패")
        
        time.sleep(1)
        
        # 7. 밸류에이션 분석
        print("\n  [7/6] 밸류에이션 분석...")
        if financial_df is not None and not financial_df.empty:
            valuation_metrics = calculate_ttm_metrics(
                financial_df.to_dict('records'),
                current_price
            )
            
            # 업종 정보는 스킵 (비용 절감)
            result["valuation"] = valuation_metrics
            print("     ✅ PER: {} PBR: {} ROE: {}%".format(
                valuation_metrics.get("PER"),
                valuation_metrics.get("PBR"),
                valuation_metrics.get("ROE")
            ))
        else:
            result["valuation"] = {}
            print("     ⚠️  재무 데이터 없음")
        
        # 8. 최종 LLM 분석
        print("\n  [8/6] 최종 LLM 분석...")
        final_analysis = analyze_with_llm(
            stock_name,
            lstm_pred,
            tech_signals,
            news_analysis,
            result.get("valuation", {})
        )
        
        result["final_analysis"] = final_analysis
        result["status"] = "완료"
        
        if final_analysis:
            print("     ✅ {}".format(final_analysis.get("recommendation", "관망")))
            
            # 최종 리포트 출력
            report = format_final_report(stock_name, stock_code, final_analysis, current_price)
            print("\n" + report)
        else:
            print("     ⚠️  LLM 분석 실패")
        
        return result
    
    except Exception as e:
        print("\n   ❌ 분석 중 오류: {}".format(str(e)))
        result["status"] = "오류"
        result["error"] = str(e)
        return result


def run_analysis(stock_code_or_name):
    """
    투자 분석 실행 (단일 종목)
    
    :param stock_code_or_name: 종목 코드 또는 명칭
    :return: analysis results dict
    """
    
    # 종목 코드 정리 (006930.KS → 005930)
    if "." in str(stock_code_or_name):
        stock_code = stock_code_or_name.split(".")[0]
    else:
        stock_code = str(stock_code_or_name)
    
    # 종목명 매핑 (간단한 케이스)
    name_map = {
        "005930": "삼성전자",
        "000660": "SK하이닉스",
        "373220": "LG에너지솔루션",
        "207940": "삼성바이오로직스",
        "005380": "현대차"
    }
    
    stock_name = name_map.get(stock_code, "주식")
    
    return run_single_stock_analysis(stock_code, stock_name)


def run_batch_analysis(top_n=5):
    """
    여러 종목 일괄 분석
    
    :param top_n: 분석할 상위 N개 종목
    :return: list of analysis results
    """
    
    print("\n╔" + "═" * 58 + "╗")
    print("  🚀 코스피 시총 상위 {} 종목 분석".format(top_n))
    print("╚" + "═" * 58 + "╝")
    
    # 상위 종목 조회
    top_stocks = get_top_stocks("KOSPI", top_n)
    
    results = []
    for stock_code, stock_name in top_stocks:
        result = run_single_stock_analysis(stock_code, stock_name)
        results.append(result)
        time.sleep(2)  # API 레이트 제한 고려
    
    # 결과 요약
    print("\n" + "╔" + "═" * 58 + "╗")
    print("  📊 분석 결과 요약")
    print("╚" + "═" * 58 + "╝")
    
    for r in results:
        if r["status"] == "완료":
            rec = r.get("final_analysis", {}).get("recommendation", "N/A")
            print("  {} ({}) → {}".format(
                r["stock_name"],
                r["stock_code"],
                rec
            ))
    
    return results


if __name__ == "__main__":
    # 테스트: 삼성전자
    result = run_analysis("005930")
    print("\n분석 완료!")
    # 테스트용
    result = run_analysis("005930.KS")
    print(result)