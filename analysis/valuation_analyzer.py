# -*- coding: utf-8 -*-
"""
역할: 재무 데이터를 기반으로 TTM (Trailing Twelve Months) 밸류에이션 지표를 계산하는 모듈.
PER, PBR, ROE, PSR 등의 지표를 생성한다.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


def calculate_ttm_metrics(financial_records, current_price):
    """
    TTM 기반 밸류에이션 지표 계산
    
    :param financial_records: 분기별 재무 데이터 list (최근순)
    :param current_price: 현재 주가
    :return: dict with valuation metrics
    """
    if not financial_records or len(financial_records) == 0:
        return {}
    
    # 최근 4개 보고서의 순이익/매출 합산 (TTM)
    ttm_net = sum(r.get("net_income", 0) for r in financial_records[:4])
    ttm_rev = sum(r.get("revenue", 0) for r in financial_records[:4])
    ttm_op = sum(r.get("operating_income", 0) for r in financial_records[:4])
    
    # 최근 보고서의 자본/부채
    latest = financial_records[0]
    latest_equity = latest.get("total_equity", 0)
    latest_debt = latest.get("total_debt", 0)
    shares = latest.get("shares", 0)
    
    # 지표 계산
    metrics = {}
    
    if shares and shares > 0:
        metrics["EPS"] = round(ttm_net / shares, 0)
        metrics["BPS"] = round(latest_equity / shares, 0)
    else:
        metrics["EPS"] = 0
        metrics["BPS"] = 0
    
    if metrics["EPS"] > 0 and current_price and current_price > 0:
        metrics["PER"] = round(current_price / metrics["EPS"], 2)
    else:
        metrics["PER"] = None
    
    if metrics["BPS"] > 0 and current_price and current_price > 0:
        metrics["PBR"] = round(current_price / metrics["BPS"], 2)
    else:
        metrics["PBR"] = None
    
    if latest_equity > 0:
        metrics["ROE"] = round(ttm_net / latest_equity * 100, 2)
    else:
        metrics["ROE"] = None
    
    market_cap = current_price * shares if (current_price and shares) else 0
    if ttm_rev > 0 and market_cap > 0:
        metrics["PSR"] = round(market_cap / ttm_rev, 2)
    else:
        metrics["PSR"] = None
    
    # 원본 값
    metrics["TTM_revenue"] = ttm_rev
    metrics["TTM_op_income"] = ttm_op
    metrics["TTM_net_income"] = ttm_net
    metrics["equity"] = latest_equity
    metrics["debt"] = latest_debt
    
    if latest_equity > 0:
        metrics["debt_ratio"] = round(latest_debt / latest_equity * 100, 2)
    else:
        metrics["debt_ratio"] = None
    
    return metrics


def get_industry_valuation_from_gpt(stock_name, industry, metrics, current_price):
    """
    OpenAI GPT를 통해 업종 대비 밸류에이션 평가 받기
    
    :param stock_name: 회사명
    :param industry: 업종
    :param metrics: 밸류에이션 지표 dict
    :param current_price: 현재 주가
    :return: dict with industry comparison
    """
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
        stock_name,
        industry,
        metrics.get("PER"),
        metrics.get("PBR"),
        metrics.get("ROE"),
        metrics.get("EPS"),
        current_price,
        industry
    )
    
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        
        return json.loads(raw)
    
    except Exception as e:
        return {
            "industry_avg_per": 0,
            "industry_avg_pbr": 0,
            "industry_avg_roe": 0,
            "valuation_status": "판단불가",
            "correction_per": 0,
            "comment": "오류: {}".format(str(e))
        }


def format_valuation_report(stock_name, metrics, industry_info):
    """
    밸류에이션 정보를 리포트 형식으로 포맷팅
    
    :param stock_name: 회사명
    :param metrics: 지표 dict
    :param industry_info: 업종 정보 dict
    :return: formatted report string
    """
    if not metrics:
        return "{}: 밸류에이션 데이터 없음".format(stock_name)
    
    report = []
    report.append("📊 {} 밸류에이션 분석".format(stock_name))
    report.append("")
    
    # TTM 지표
    report.append("▸ TTM 기반 지표:")
    report.append("  EPS: {:,.0f}원  PER: {}".format(metrics.get("EPS", 0), metrics.get("PER")))
    report.append("  BPS: {:,.0f}원  PBR: {}".format(metrics.get("BPS", 0), metrics.get("PBR")))
    report.append("  ROE: {}%  PSR: {}".format(metrics.get("ROE"), metrics.get("PSR")))
    
    # 업종 대비
    if industry_info:
        status = industry_info.get("valuation_status", "N/A")
        report.append("")
        report.append("▸ 업종 대비:")
        report.append("  상태: {} (업종평균 PER {})".format(
            status,
            industry_info.get("industry_avg_per")
        ))
        report.append("  의견: {}".format(industry_info.get("comment", "")))
    
    return "\n".join(report)


if __name__ == "__main__":
    # 테스트용 더미 데이터
    financial_records = [
        {
            "net_income": 15000e8,
            "revenue": 100000e8,
            "operating_income": 20000e8,
            "total_equity": 200000e8,
            "total_debt": 50000e8,
            "shares": 100e6
        },
        {
            "net_income": 14000e8,
            "revenue": 95000e8,
            "operating_income": 19000e8,
            "total_equity": 200000e8,
            "total_debt": 50000e8,
            "shares": 100e6
        }
    ]
    
    current_price = 70000  # 주가
    
    metrics = calculate_ttm_metrics(financial_records, current_price)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
