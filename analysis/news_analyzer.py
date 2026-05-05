# -*- coding: utf-8 -*-
"""
역할: NewsAPI와 OpenAI를 사용하여 종목과 시장 뉴스를 수집하고 호재/악재를 분석하는 모듈.
"""

import json
import logging
import os
from datetime import date, timedelta
from typing import List, Optional

import requests
from openai import OpenAI
from dotenv import load_dotenv

from config import GPT_MODEL, GPT_TEMPERATURE, GPT_MAX_TOKENS_NEWS

load_dotenv()

logger = logging.getLogger(__name__)

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


def fetch_news(query: str, max_items: int = 5, language: str = "ko") -> List[dict]:
    """
    NewsAPI로 최신 뉴스 수집.

    :param query: 검색 쿼리
    :param max_items: 최대 아이템 수
    :param language: "ko" 또는 "en"
    :return: list of news articles
    """
    try:
        today = date.today().strftime("%Y-%m-%d")
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "from": yesterday,
                "to": today,
                "language": language,
                "sortBy": "publishedAt",
                "pageSize": max_items,
                "apiKey": NEWSAPI_KEY,
            },
            timeout=10,
        )
        data = r.json()

        if data.get("status") != "ok":
            return []

        result = []
        for a in data.get("articles", []):
            title = a.get("title", "")
            if title and "[Removed]" not in title:
                result.append({
                    "title": title,
                    "source": a.get("source", {}).get("name", ""),
                    "published": a.get("publishedAt", "")[:16],
                    "description": (a.get("description") or "")[:100],
                })
        return result

    except requests.RequestException as e:
        logger.warning("⚠️ NewsAPI 요청 오류: %s", e)
        return []
    except Exception as e:
        logger.warning("⚠️ NewsAPI 오류: %s", e)
        return []


def _deduplicate(news_list: List[dict]) -> List[dict]:
    seen: set = set()
    unique = []
    for item in news_list:
        if item["title"] not in seen:
            seen.add(item["title"])
            unique.append(item)
    return unique


def collect_stock_news(stock_name: str, queries: List[str] = None) -> List[dict]:
    """특정 주식 종목의 뉴스 수집."""
    if queries is None:
        queries = [stock_name]

    news: List[dict] = []
    for q in queries:
        lang = "ko" if any(ord(c) > 127 for c in q) else "en"
        news.extend(fetch_news(q, max_items=4, language=lang))

    return _deduplicate(news)


def collect_macro_news() -> List[dict]:
    """글로벌 매크로 뉴스 수집."""
    queries_en = ["stock market today", "US Federal Reserve interest rate", "oil price today", "semiconductor market today"]
    queries_ko = ["코스피", "미국 증시", "환율"]

    news: List[dict] = []
    for q in queries_en:
        news.extend(fetch_news(q, max_items=2, language="en"))
    for q in queries_ko:
        news.extend(fetch_news(q, max_items=2, language="ko"))

    return _deduplicate(news)[:10]


def format_news_for_gpt(news_list: List[dict]) -> str:
    if not news_list:
        return "관련 뉴스 없음"
    lines = []
    for n in news_list:
        lines.append("[{}][{}] {}".format(n["published"], n["source"], n["title"]))
        if n["description"]:
            lines.append("  └ {}".format(n["description"]))
    return "\n".join(lines)


def analyze_news_with_gpt(
    stock_name: str,
    stock_news: List[dict],
    macro_news: List[dict],
    model: Optional[str] = None,
) -> Optional[dict]:
    """
    뉴스를 종합하여 GPT로 호재/악재 분석.

    빈 뉴스 리스트는 GPT를 호출하지 않고 None을 반환한다.
    """
    if not stock_news and not macro_news:
        logger.warning("뉴스 없음 — GPT 분석 스킵")
        return None

    model_choice = model or GPT_MODEL
    today_str = date.today().strftime("%Y년 %m월 %d일")
    stock_text = format_news_for_gpt(stock_news[:8])
    macro_text = format_news_for_gpt(macro_news[:10])

    prompt = """당신은 한국 주식시장 전문 애널리스트입니다.
오늘은 {}입니다.

━━━ [{}] 종목 관련 뉴스 ━━━
{}

━━━ 오늘의 글로벌/매크로 뉴스 ━━━
{}

위 뉴스 전체를 분석하여 {} 주가에 미칠 영향을 평가하세요.

반드시 아래 JSON만 출력하세요:
{{
  "verdict": "호재 또는 악재 또는 중립",
  "score": -1.0~1.0 범위의 숫자,
  "bullish_prob": 0~100,
  "bearish_prob": 0~100,
  "caution_prob": 0~100,
  "stock_summary": "종목 관련 핵심 내용 1~2문장",
  "macro_summary": "글로벌 매크로 핵심 1~2문장",
  "top_risks": ["리스크1", "리스크2"],
  "top_opportunities": ["기회1", "기회2"],
  "recommendation": "매수적극/매수신중/관망/매도고려/매도추천 중 하나"
}}""".format(today_str, stock_name, stock_text, macro_text, stock_name)

    try:
        resp = openai_client.chat.completions.create(
            model=model_choice,
            messages=[{"role": "user", "content": prompt}],
            temperature=GPT_TEMPERATURE,
            max_tokens=GPT_MAX_TOKENS_NEWS,
        )
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except json.JSONDecodeError:
        logger.warning("⚠️ JSON 파싱 실패")
        return None
    except Exception as e:
        logger.warning("⚠️ GPT 오류: %s", e)
        return None


if __name__ == "__main__":
    sn = collect_stock_news("삼성전자", ["삼성전자", "Samsung Electronics"])
    mn = collect_macro_news()
    print("종목 뉴스: {}건".format(len(sn)))
    print("매크로 뉴스: {}건".format(len(mn)))
    result = analyze_news_with_gpt("삼성전자", sn, mn)
    print(json.dumps(result, ensure_ascii=False, indent=2))
