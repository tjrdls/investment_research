# -*- coding: utf-8 -*-
"""
역할: 투자 분석 파이프라인의 핵심 모듈. 데이터 수집부터 LLM 분석까지의 전체 흐름을 조율한다.
"""

import logging
import sys
import os
import time
import numpy as np
import torch
from datetime import datetime
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import (
    MODEL_PATH, LSTM_SEQ_LEN,
    EMBEDDING_MODEL, EMBEDDING_DIM, STOCK_CODE_TO_NAME,
)
from data_loader.price.data_collector import collect_price_data, get_top_stocks
from data_loader.financial.financial_collector import collect_financial_data, get_corp_code_map
from analysis.indicators.technical_indicators import calculate_indicators, get_technical_signals
from analysis.news_analyzer import collect_stock_news, collect_macro_news, analyze_news_with_gpt
from analysis.valuation_analyzer import calculate_ttm_metrics
from models.lstm.lstm_model import predict_next_trend, MultimodalStockPredictor
from models.llm.llm_analyzer import analyze_with_llm, format_final_report
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

SEQ_LEN = LSTM_SEQ_LEN


def get_text_embedding(text: str) -> np.ndarray:
    """
    OpenAI Embedding API로 텍스트를 벡터로 변환.

    실패 시 random 벡터가 아닌 zero 벡터를 반환하여 downstream에서
    일관된 폴백 동작을 보장한다.
    """
    try:
        resp = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:8000]
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)
    except Exception as e:
        logger.warning("⚠️ Embedding 실패 — zero 벡터 사용: %s", e)
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)


def load_lstm_model():
    """훈련된 LSTM 모델 로드. 없거나 로드 실패 시 None 반환."""
    if not os.path.exists(MODEL_PATH):
        logger.warning("⚠️ 모델 파일 없음: %s  (학습: python train_lstm.py)", MODEL_PATH)
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultimodalStockPredictor().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        logger.info("✅ LSTM 모델 로드 완료")
        return model
    except Exception as e:
        logger.error("❌ 모델 로드 실패: %s", e)
        return None


def build_context_text(stock_name: str, financial_df) -> str:
    """DART 재무 정보로 Embedding용 컨텍스트 텍스트 구성."""
    if financial_df is None or financial_df.empty:
        return "{} 코스피 상장 주식".format(stock_name)

    parts = [stock_name]
    for _, row in financial_df.iterrows():
        rev = (row.get("revenue") or 0) / 1e8
        net = (row.get("net_income") or 0) / 1e8
        roe = row.get("roe") or 0
        parts.append("[{}년 {}] 매출 {:.0f}억 순이익 {:.0f}억 ROE {:.1f}%".format(
            int(row.get("year", 2025)), row.get("report_type", ""), rev, net, roe
        ))
    return " | ".join(parts)[:4000]


# ---------------------------------------------------------------------------
# 단계별 분석 헬퍼 (run_single_stock_analysis에서 분리)
# ---------------------------------------------------------------------------

def _step_price(stock_code: str) -> tuple:
    """[1] 주가 + 기술적 지표 수집. (price_df, indicators_df, current_price) 반환."""
    logger.info("[1/6] 주가 데이터 수집...")
    price_df = collect_price_data(stock_code)
    if price_df.empty:
        raise ValueError("주가 데이터 수집 실패: {}".format(stock_code))
    current_price = float(price_df["close"].iloc[-1])
    logger.info("✅ 완료 (%d 거래일, 현재가 %,.0f원)", len(price_df), current_price)

    logger.info("[2/6] 기술적 지표 계산...")
    indicators_df = calculate_indicators(price_df)
    if indicators_df.empty:
        raise ValueError("지표 계산 실패: {}".format(stock_code))
    tech = get_technical_signals(indicators_df)
    logger.info("✅ 완료 (신호 %d, 경고 %d)", len(tech["signals"]), len(tech["warnings"]))
    return price_df, indicators_df, current_price, tech


def _step_financial(stock_code: str):
    """[3] DART 재무 데이터 수집. financial_df(또는 None) 반환."""
    logger.info("[3/6] 재무 데이터 수집 (DART)...")
    corp_map = get_corp_code_map()
    corp_code = corp_map.get(stock_code, {}).get("corp_code")
    if not corp_code:
        logger.warning("기업코드 없음 — 재무 단계 스킵")
        return None
    financial_df = collect_financial_data(stock_code, corp_code)
    logger.info("✅ 완료 (%d 기간)", len(financial_df))
    return financial_df


def _step_embedding(stock_name: str, financial_df) -> np.ndarray:
    """[4] 텍스트 임베딩 생성."""
    logger.info("[4/6] 텍스트 임베딩 생성...")
    context = build_context_text(stock_name, financial_df)
    emb = get_text_embedding(context)
    logger.info("✅ 완료 (%d 차원)", len(emb))
    return emb


def _step_lstm(indicators_df, text_emb: np.ndarray) -> dict:
    """[5] LSTM 예측."""
    logger.info("[5/6] LSTM 예측...")
    try:
        model = load_lstm_model()
        result = predict_next_trend(model, indicators_df, text_emb, seq_len=SEQ_LEN)
        logger.info("✅ %s: 신뢰도 %.1f%%", result["prediction"], result["confidence"] * 100)
        return result
    except Exception as e:
        logger.warning("⚠️ LSTM 예측 실패: %s", e)
        return {"prediction": "기술 오류", "probabilities": {"상승": 0.33, "하락": 0.33, "횡보": 0.34}, "confidence": 0.0}


def _step_news(stock_name: str, stock_code: str, model: Optional[str] = None) -> tuple:
    """[6] 뉴스 수집 + GPT 감성 분석. (news_meta, news_analysis) 반환."""
    logger.info("[6/6] 뉴스 분석...")
    stock_news = collect_stock_news(stock_name, [stock_name, "{}({})".format(stock_name, stock_code)])
    macro_news = collect_macro_news()
    analysis = analyze_news_with_gpt(stock_name, stock_news, macro_news, model=model)
    if analysis:
        logger.info("✅ %s: %s", analysis.get("verdict", "중립"), analysis.get("recommendation", "관망"))
    else:
        logger.warning("뉴스 분석 실패")
    return {"stock_count": len(stock_news), "macro_count": len(macro_news), "analysis": analysis}, analysis


def _step_valuation(financial_df, current_price: float) -> dict:
    """[7] 밸류에이션 계산."""
    if financial_df is None or financial_df.empty:
        logger.warning("재무 데이터 없음 — 밸류에이션 스킵")
        return {}
    metrics = calculate_ttm_metrics(financial_df.to_dict("records"), current_price)
    logger.info("✅ PER: %s  PBR: %s  ROE: %s%%", metrics.get("PER"), metrics.get("PBR"), metrics.get("ROE"))
    return metrics


def _step_llm(
    stock_name: str,
    stock_code: str,
    lstm_pred: dict,
    tech: dict,
    news_analysis,
    valuation: dict,
    current_price: float,
    model: Optional[str] = None,
) -> dict | None:
    """[8] 최종 LLM 종합 분석."""
    logger.info("[8/6] 최종 LLM 분석...")
    final = analyze_with_llm(stock_name, lstm_pred, tech, news_analysis, valuation, model=model)
    if final:
        logger.info("✅ %s", final.get("recommendation", "관망"))
        report = format_final_report(stock_name, stock_code, final, current_price)
        logger.info("\n%s", report)
    else:
        logger.warning("LLM 분석 실패")
    return final


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def run_single_stock_analysis(stock_code: str, stock_name: str) -> dict:
    """단일 종목 전체 분석 파이프라인 실행."""
    logger.info("\n%s\n  📊 %s (%s) 분석\n%s", "━" * 60, stock_name, stock_code, "━" * 60)

    result: dict = {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "timestamp": datetime.now().isoformat(),
        "status": "진행중",
    }

    try:
        price_df, indicators_df, current_price, tech = _step_price(stock_code)
        result["current_price"] = current_price
        result["technical"] = tech

        financial_df = _step_financial(stock_code)
        result["financial"] = financial_df.to_dict() if financial_df is not None else {}

        text_emb = _step_embedding(stock_name, financial_df)
        result["text_embedding"] = text_emb.tolist()

        lstm_pred = _step_lstm(indicators_df, text_emb)
        result["lstm_prediction"] = lstm_pred
        time.sleep(0.5)

        news_meta, news_analysis = _step_news(
            stock_name,
            stock_code,
            model=getattr(getattr(sys.modules.get("streamlit"), "session_state", {}), "gpt_model", None) if "streamlit" in sys.modules else None,
        )
        result["news"] = news_meta
        time.sleep(1)

        valuation = _step_valuation(financial_df, current_price)
        result["valuation"] = valuation

        final = _step_llm(
            stock_name,
            stock_code,
            lstm_pred,
            tech,
            news_analysis,
            valuation,
            current_price,
            model=getattr(getattr(sys.modules.get("streamlit"), "session_state", {}), "gpt_model", None) if "streamlit" in sys.modules else None,
        )
        result["final_analysis"] = final
        result["status"] = "완료"

    except Exception as e:
        logger.error("❌ 분석 중 오류: %s", e)
        result["status"] = "오류"
        result["error"] = str(e)

    return result


def run_analysis(stock_code_or_name: str) -> dict:
    """단일 종목 분석 (코드 또는 명칭)."""
    stock_code = str(stock_code_or_name).split(".")[0]
    stock_name = STOCK_CODE_TO_NAME.get(stock_code, "주식")
    return run_single_stock_analysis(stock_code, stock_name)


def run_batch_analysis(top_n: int = 5) -> list:
    """여러 종목 일괄 분석."""
    logger.info("\n🚀 코스피 시총 상위 %d 종목 분석", top_n)
    top_stocks = get_top_stocks("KOSPI", top_n)
    results = []
    for stock_code, stock_name in top_stocks:
        results.append(run_single_stock_analysis(stock_code, stock_name))
        time.sleep(2)

    logger.info("\n📊 분석 결과 요약")
    for r in results:
        if r["status"] == "완료":
            rec = r.get("final_analysis", {}).get("recommendation", "N/A")
            logger.info("  %s (%s) → %s", r["stock_name"], r["stock_code"], rec)
    return results


def run_lstm_prediction(price_df, indicators_df, text_embedding=None) -> dict:
    """LSTM 예측만 실행."""
    try:
        model = load_lstm_model()
        emb = text_embedding if text_embedding is not None else np.zeros(EMBEDDING_DIM)
        return predict_next_trend(model, indicators_df, emb, seq_len=SEQ_LEN)
    except Exception as e:
        logger.error("LSTM 예측 실패: %s", e)
        return {"prediction": "기술 오류", "probabilities": {"상승": 0.33, "하락": 0.33, "횡보": 0.34}, "confidence": 0.0}


def run_financial_analysis(stock_code: str, financial_data, current_price: float) -> dict:
    """재무 분석만 실행."""
    try:
        if financial_data is None or financial_data.empty:
            return {}
        return calculate_ttm_metrics(financial_data.to_dict("records"), current_price)
    except Exception as e:
        logger.error("재무 분석 실패: %s", e)
        return {}


def run_final_analysis(stock_code: str, stock_name: str, analysis_result: dict, model: Optional[str] = None) -> dict:
    """최종 AI 종합 분석만 실행."""
    try:
        logger.info("🔄 AI 종합 분석 시작...")
        lstm_pred = analysis_result.get("lstm_prediction", {})
        tech = analysis_result.get("technical", {})
        news_analysis = analysis_result.get("news", {}).get("analysis", {})
        valuation = analysis_result.get("valuation", {})

        current_price = None
        try:
            import streamlit as st
            if hasattr(st.session_state, "price_df") and st.session_state.price_df is not None:
                current_price = float(st.session_state.price_df["close"].iloc[-1])
            if model is None:
                model = getattr(st.session_state, "gpt_model", None)
        except Exception:
            pass
        if current_price is None:
            current_price = 50000

        final = analyze_with_llm(stock_name, lstm_pred, tech, news_analysis, valuation, model=model)
        logger.info("분석 완료: %s", final.get("recommendation", "N/A") if final else "None")
        return final

    except Exception as e:
        logger.error("❌ AI 종합 분석 실패: %s", e)
        return {"recommendation": "분석 실패", "confidence": 0, "summary": "기술적 오류로 분석을 완료할 수 없습니다."}


if __name__ == "__main__":
    result = run_analysis("005930")
    print("\n분석 완료!")
