# -*- coding: utf-8 -*-
"""
역할: 재무 데이터를 기반으로 TTM (Trailing Twelve Months) 밸류에이션 지표를 계산하는 모듈.
PER, PBR, ROE, PSR 등의 지표를 생성한다.
"""

import json
import logging
import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from config import GPT_MODEL, GPT_TEMPERATURE, GPT_MAX_TOKENS_VALUATION

load_dotenv()

logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


def calculate_ttm_metrics(financial_records: list, current_price: float) -> dict:
    """
    TTM 기반 밸류에이션 지표 계산.

    :param financial_records: 분기별 재무 데이터 list (최근순)
    :param current_price: 현재 주가
    :return: dict with valuation metrics
    """
    if not financial_records:
        return {}

    ttm_net = sum(r.get("net_income", 0) for r in financial_records[:4])
    ttm_rev = sum(r.get("revenue", 0) for r in financial_records[:4])
    ttm_op = sum(r.get("operating_income", 0) for r in financial_records[:4])

    latest = financial_records[0]
    latest_equity = latest.get("total_equity", 0)
    latest_debt = latest.get("total_debt", 0)
    shares = latest.get("shares", 0)

    metrics: dict = {}

    if shares and shares > 0:
        metrics["EPS"] = round(ttm_net / shares, 0)
        metrics["BPS"] = round(latest_equity / shares, 0)
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

    return metrics


def get_industry_valuation_from_gpt(stock_name: str, industry: str, metrics: dict, current_price: float) -> dict:
    """OpenAI GPT를 통해 업종 대비 밸류에이션 평가."""
    prompt = """한국 주식시장 전문 애널리스트입니다.

{} ({} 업종) TTM 기준 지표:
- PER : {}
- PBR : {}
- ROE : {}%
- EPS : {}원
- 현재 주가 : {:,.0f}원

다음을 분석해주세요:
1. {} 업종의 평균 PER/PBR/ROE
2. 현재 고평가/저평가 여부
3. 조정 임계값

아래 JSON만 출력하세요:
{{
  "industry_avg_per": 업종평균 PER,
  "industry_avg_pbr": 업종평균 PBR,
  "industry_avg_roe": 업종평균 ROE,
  "valuation_status": "고평가/적정/저평가",
  "correction_per": 조정 임계 PER,
  "comment": "분석 1~2문장"
}}""".format(
        stock_name, industry,
        metrics.get("PER"), metrics.get("PBR"), metrics.get("ROE"), metrics.get("EPS"),
        current_price, industry,
    )

    try:
        resp = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=GPT_TEMPERATURE,
            max_tokens=GPT_MAX_TOKENS_VALUATION,
        )
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except Exception as e:
        logger.warning("업종 밸류에이션 GPT 오류: %s", e)
        return {
            "industry_avg_per": 0,
            "industry_avg_pbr": 0,
            "industry_avg_roe": 0,
            "valuation_status": "판단불가",
            "correction_per": 0,
            "comment": "오류: {}".format(e),
        }


def format_valuation_report(stock_name: str, metrics: dict, industry_info: Optional[dict]) -> str:
    if not metrics:
        return "{}: 밸류에이션 데이터 없음".format(stock_name)

    lines = [
        "📊 {} 밸류에이션 분석".format(stock_name), "",
        "▸ TTM 기반 지표:",
        "  EPS: {:,.0f}원  PER: {}".format(metrics.get("EPS", 0), metrics.get("PER")),
        "  BPS: {:,.0f}원  PBR: {}".format(metrics.get("BPS", 0), metrics.get("PBR")),
        "  ROE: {}%  PSR: {}".format(metrics.get("ROE"), metrics.get("PSR")),
    ]

    if industry_info:
        lines += [
            "",
            "▸ 업종 대비:",
            "  상태: {} (업종평균 PER {})".format(
                industry_info.get("valuation_status", "N/A"),
                industry_info.get("industry_avg_per"),
            ),
            "  의견: {}".format(industry_info.get("comment", "")),
        ]

    return "\n".join(lines)


if __name__ == "__main__":
    records = [
        {"net_income": 15000e8, "revenue": 100000e8, "operating_income": 20000e8,
         "total_equity": 200000e8, "total_debt": 50000e8, "shares": 100e6},
    ]
    print(json.dumps(calculate_ttm_metrics(records, 70000), ensure_ascii=False, indent=2))
