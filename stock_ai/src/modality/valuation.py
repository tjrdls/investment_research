"""밸류에이션 모달리티 — TTM 지표 + 업종 대비 평가 → ModalitySignal.

구버전 `analysis/valuation_analyzer.py` 이식.
  - `calculate_ttm_metrics` : 순수 함수 (DB/API 불필요) → 테스트 가능
  - `valuation_to_signal`    : 순수 함수 — PER/ROE/PBR 로 0~100 점수 산출
  - `industry_valuation_gpt` : OpenAI 호출 (I/O, 키 없으면 graceful skip)
"""
from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

from src.config import CFG
from src.modality.base import ModalitySignal, clamp_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 순수 로직 — DB/API 불필요
# ---------------------------------------------------------------------------
def calculate_ttm_metrics(financial_records: List[dict], current_price: float) -> dict:
    """분기별 재무 레코드(최근순) + 현재가 → TTM 밸류에이션 지표.

    financial_records[i] keys: net_income, revenue, operating_income,
    total_equity, total_debt, shares.
    """
    if not financial_records:
        return {}

    recent4 = financial_records[:4]
    ttm_net = sum(r.get("net_income", 0) or 0 for r in recent4)
    ttm_rev = sum(r.get("revenue", 0) or 0 for r in recent4)
    ttm_op = sum(r.get("operating_income", 0) or 0 for r in recent4)

    latest = financial_records[0]
    latest_equity = latest.get("total_equity", 0) or 0
    latest_debt = latest.get("total_debt", 0) or 0
    shares = latest.get("shares", 0) or 0

    metrics: dict = {}

    if shares > 0:
        metrics["EPS"] = round(ttm_net / shares, 2)
        metrics["BPS"] = round(latest_equity / shares, 2)
    else:
        metrics["EPS"] = 0
        metrics["BPS"] = 0

    metrics["PER"] = round(current_price / metrics["EPS"], 2) if metrics["EPS"] > 0 and current_price > 0 else None
    metrics["PBR"] = round(current_price / metrics["BPS"], 2) if metrics["BPS"] > 0 and current_price > 0 else None
    metrics["ROE"] = round(ttm_net / latest_equity * 100, 2) if latest_equity > 0 else None

    market_cap = current_price * shares if (current_price and shares) else 0
    metrics["PSR"] = round(market_cap / ttm_rev, 2) if ttm_rev > 0 and market_cap > 0 else None

    metrics["TTM_revenue"] = ttm_rev
    metrics["TTM_op_income"] = ttm_op
    metrics["TTM_net_income"] = ttm_net
    metrics["equity"] = latest_equity
    metrics["debt"] = latest_debt
    metrics["debt_ratio"] = round(latest_debt / latest_equity * 100, 2) if latest_equity > 0 else None
    metrics["op_margin"] = round(ttm_op / ttm_rev * 100, 2) if ttm_rev > 0 else None

    return metrics


def valuation_to_signal(metrics: dict, hard_filter=None) -> ModalitySignal:
    """TTM 지표 → 밸류에이션 신호 (순수 함수).

    하드필터 기준(ROE_min, PER_max)을 앵커로:
      - PER 가 낮을수록(저평가) ↑, ROE 가 높을수록 ↑
    점수는 PER 매력도와 ROE 우수성의 평균을 0~100 으로.
    """
    if not metrics:
        return ModalitySignal.unavailable("valuation", "재무 데이터 없음")

    hf = hard_filter or CFG.hard_filter
    per = metrics.get("PER")
    roe = metrics.get("ROE")
    pbr = metrics.get("PBR")

    sub_scores: List[float] = []
    detail = {"PER": per, "PBR": pbr, "ROE": roe, "op_margin": metrics.get("op_margin")}

    # PER 매력도: per_max 에서 0, per_max/4 이하면 만점 (선형)
    if per is not None and per > 0:
        per_max = max(hf.per_max, 1.0)
        # per 가 per_max 이상 → 25점, per 가 per_max/4 이하 → 90점
        attractiveness = (per_max - per) / per_max  # 1(아주쌈) ~ 음수(아주비쌈)
        per_score = clamp_score(50 + attractiveness * 45)
        sub_scores.append(per_score)
        detail["per_score"] = round(per_score, 1)

    # ROE 우수성: roe_min 에서 중립, 2×roe_min 이상이면 강세
    if roe is not None:
        roe_min = max(hf.roe_min, 1.0)
        roe_score = clamp_score(50 + (roe - roe_min) / roe_min * 40)
        sub_scores.append(roe_score)
        detail["roe_score"] = round(roe_score, 1)

    # PBR 보조: 1 이하 강세, 3 이상 약세 (가중 작게)
    if pbr is not None and pbr > 0:
        pbr_score = clamp_score(50 + (1.5 - pbr) * 20)
        sub_scores.append(pbr_score)
        detail["pbr_score"] = round(pbr_score, 1)

    if not sub_scores:
        return ModalitySignal.unavailable("valuation", "유효 지표 없음")

    score = sum(sub_scores) / len(sub_scores)
    # 지표가 많을수록 신뢰 ↑ (1개 0.4 ~ 3개 0.8)
    confidence = min(0.8, 0.3 + 0.17 * len(sub_scores))
    status = "저평가" if score >= 60 else ("고평가" if score <= 40 else "적정")
    return ModalitySignal(
        name="valuation", score=score, confidence=confidence,
        label=status, detail=detail,
    )


# ---------------------------------------------------------------------------
# I/O — OpenAI 업종 대비 평가 (키 없으면 graceful skip)
# ---------------------------------------------------------------------------
def industry_valuation_gpt(
    stock_name: str, industry: str, metrics: dict, current_price: float,
    model: Optional[str] = None,
) -> dict:
    """OpenAI 로 업종 평균 PER/PBR/ROE 대비 평가. 키 없으면 빈 dict."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or not CFG.llm.enabled:
        logger.info("OPENAI_API_KEY 없음 또는 LLM 비활성 — 업종 밸류에이션 스킵")
        return {}
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai 패키지 미설치 — 업종 밸류에이션 스킵")
        return {}

    client = OpenAI(api_key=api_key)
    prompt = (
        f"한국 주식시장 전문 애널리스트입니다.\n\n"
        f"{stock_name} ({industry} 업종) TTM 기준 지표:\n"
        f"- PER : {metrics.get('PER')}\n- PBR : {metrics.get('PBR')}\n"
        f"- ROE : {metrics.get('ROE')}%\n- 현재가 : {current_price:,.0f}원\n\n"
        "업종 평균 PER/PBR/ROE 와 고/저평가 여부를 평가해 아래 JSON 만 출력:\n"
        '{"industry_avg_per": 0, "industry_avg_pbr": 0, "industry_avg_roe": 0,'
        ' "valuation_status": "고평가/적정/저평가", "comment": "1~2문장"}'
    )
    try:
        resp = client.chat.completions.create(
            model=model or CFG.llm.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=CFG.llm.temperature,
            max_completion_tokens=CFG.llm.max_tokens_valuation,
        )
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:  # noqa: BLE001 — 외부 API 폴백
        logger.warning("업종 밸류에이션 GPT 오류: %s", e)
        return {}
