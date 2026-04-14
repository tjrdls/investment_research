# -*- coding: utf-8 -*-
"""
역할: NewsAPI와 OpenAI를 사용하여 종목과 시장 뉴스를 수집하고 호재/악재를 분석하는 모듈.
"""

import requests
import json
import os
from datetime import date, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


def fetch_news(query, max_items=5, language="ko"):
    """
    NewsAPI로 최신 뉴스 수집
    
    :param query: 검색 쿼리
    :param max_items: 최대 아이템 수
    :param language: 언어 ("ko" 또는 "en")
    :return: list of news articles
    """
    try:
        today = date.today().strftime("%Y-%m-%d")
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": yesterday,
            "to": today,
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": max_items,
            "apiKey": NEWSAPI_KEY
        }
        
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        if data.get("status") != "ok":
            return []
        
        articles = data.get("articles", [])
        result = []
        
        for a in articles:
            title = a.get("title", "")
            source = a.get("source", {}).get("name", "")
            pub_at = a.get("publishedAt", "")[:16]
            desc = a.get("description") or ""
            
            if title and "[Removed]" not in title:
                result.append({
                    "title": title,
                    "source": source,
                    "published": pub_at,
                    "description": desc[:100]
                })
        
        return result
    
    except Exception as e:
        print("    ⚠️  NewsAPI 오류: {}".format(str(e)))
        return []


def collect_stock_news(stock_name, queries=None):
    """
    특정 주식 종목의 뉴스 수집
    
    :param stock_name: 회사명 (예: "삼성전자")
    :param queries: 검색 쿼리 리스트 (없으면 회사명만 사용)
    :return: list of news
    """
    if queries is None:
        queries = [stock_name]
    
    news = []
    
    for q in queries:
        # 한글/영문 판단
        lang = "ko" if any(ord(c) > 127 for c in q) else "en"
        items = fetch_news(q, max_items=4, language=lang)
        news.extend(items)
    
    # 중복 제거
    seen = set()
    unique = []
    for n in news:
        if n["title"] not in seen:
            seen.add(n["title"])
            unique.append(n)
    
    return unique


def collect_macro_news():
    """
    글로벌 매크로 뉴스 수집
    
    :return: list of macro news
    """
    queries_en = [
        "stock market today",
        "US Federal Reserve interest rate",
        "oil price today",
        "semiconductor market today"
    ]
    
    queries_ko = [
        "코스피",
        "미국 증시",
        "환율",
    ]
    
    news = []
    
    for q in queries_en:
        items = fetch_news(q, max_items=2, language="en")
        news.extend(items)
    
    for q in queries_ko:
        items = fetch_news(q, max_items=2, language="ko")
        news.extend(items)
    
    # 중복 제거
    seen = set()
    unique = []
    for n in news:
        if n["title"] not in seen:
            seen.add(n["title"])
            unique.append(n)
    
    return unique[:10]  # 최상위 10개만


def format_news_for_gpt(news_list):
    """
    뉴스 리스트를 GPT 프롬프트용으로 포맷팅
    
    :param news_list: news articles list
    :return: formatted text
    """
    if not news_list:
        return "관련 뉴스 없음"
    
    lines = []
    for n in news_list:
        lines.append("[{}][{}] {}".format(
            n["published"],
            n["source"],
            n["title"]
        ))
        if n["description"]:
            lines.append("  └ {}".format(n["description"]))
    
    return "\n".join(lines)


def analyze_news_with_gpt(stock_name, stock_news, macro_news):
    """
    뉴스를 종합하여 GPT로 호재/악재 분석
    
    :param stock_name: 종목명
    :param stock_news: 종목 관련 뉴스 리스트
    :param macro_news: 매크로 뉴스 리스트
    :return: dict with analysis result
    """
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
}}""".format(
        today_str,
        stock_name,
        stock_text,
        macro_text,
        stock_name
    )
    
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        
        return json.loads(raw)
    
    except json.JSONDecodeError:
        print("    ⚠️  JSON 파싱 실패")
        return None
    except Exception as e:
        print("    ⚠️  GPT 오류: {}".format(str(e)))
        return None


if __name__ == "__main__":
    # 테스트
    stock_news = collect_stock_news("삼성전자", ["삼성전자", "Samsung Electronics"])
    macro_news = collect_macro_news()
    
    print("종목 뉴스: {}건".format(len(stock_news)))
    print("매크로 뉴스: {}건".format(len(macro_news)))
    
    analysis = analyze_news_with_gpt("삼성전자", stock_news, macro_news)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
