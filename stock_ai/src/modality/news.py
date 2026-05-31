"""뉴스 감성 모달리티 — NewsAPI 수집 + GPT 감성 분석 → ModalitySignal.

구버전 `analysis/news_analyzer.py` 이식.
  - `news_result_to_signal` : 순수 함수 (GPT 결과 dict → 신호) → 테스트 가능
  - `fetch_news` / `analyze_news_with_gpt` : I/O (키 없으면 graceful skip)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from typing import List, Optional

from src.config import CFG
from src.modality.base import ModalitySignal, clamp_conf, from_bipolar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 순수 로직 — GPT 결과 dict → 신호 (DB/API 불필요)
# ---------------------------------------------------------------------------
def news_result_to_signal(analysis: Optional[dict]) -> ModalitySignal:
    """GPT 뉴스 분석 결과 → ModalitySignal.

    기대 키: score(-1~1), bullish_prob/bearish_prob(0~100), verdict, recommendation.
    """
    if not analysis:
        return ModalitySignal.unavailable("news", "뉴스 분석 결과 없음")

    score_pm1 = analysis.get("score")
    if score_pm1 is None:
        # score 없으면 확률 차이로 추정
        bull = analysis.get("bullish_prob", 0) or 0
        bear = analysis.get("bearish_prob", 0) or 0
        score_pm1 = (bull - bear) / 100.0

    # 신뢰도: bullish/bearish 확률의 우세 폭 + caution 역가중
    bull = (analysis.get("bullish_prob", 0) or 0) / 100.0
    bear = (analysis.get("bearish_prob", 0) or 0) / 100.0
    caution = (analysis.get("caution_prob", 0) or 0) / 100.0
    confidence = clamp_conf(abs(bull - bear) * (1.0 - 0.5 * caution) + 0.2)

    return ModalitySignal(
        name="news",
        score=from_bipolar(score_pm1),
        confidence=confidence,
        label=str(analysis.get("verdict", "중립")),
        detail={
            "verdict": analysis.get("verdict"),
            "recommendation": analysis.get("recommendation"),
            "stock_summary": analysis.get("stock_summary"),
            "macro_summary": analysis.get("macro_summary"),
            "top_risks": analysis.get("top_risks"),
            "top_opportunities": analysis.get("top_opportunities"),
        },
    )


def _deduplicate(news_list: List[dict]) -> List[dict]:
    seen: set = set()
    unique = []
    for item in news_list:
        if item["title"] not in seen:
            seen.add(item["title"])
            unique.append(item)
    return unique


def _format_news_for_gpt(news_list: List[dict]) -> str:
    if not news_list:
        return "관련 뉴스 없음"
    lines = []
    for n in news_list:
        lines.append(f"[{n['published']}][{n['source']}] {n['title']}")
        if n.get("description"):
            lines.append(f"  └ {n['description']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O — NewsAPI 수집
# ---------------------------------------------------------------------------
def fetch_news(query: str, max_items: int = 5, language: str = "ko") -> List[dict]:
    """NewsAPI 최신 뉴스 수집. 키 없거나 실패 시 빈 리스트."""
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        return []
    try:
        import requests
    except ImportError:
        return []
    try:
        today = date.today().strftime("%Y-%m-%d")
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query, "from": yesterday, "to": today, "language": language,
                "sortBy": "publishedAt", "pageSize": max_items, "apiKey": api_key,
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
    except Exception as e:  # noqa: BLE001
        logger.warning("NewsAPI 오류: %s", e)
        return []


def collect_stock_news(stock_name: str, queries: Optional[List[str]] = None) -> List[dict]:
    """특정 종목 뉴스 수집."""
    queries = queries or [stock_name]
    news: List[dict] = []
    for q in queries:
        lang = "ko" if any(ord(c) > 127 for c in q) else "en"
        news.extend(fetch_news(q, max_items=4, language=lang))
    return _deduplicate(news)


def collect_macro_news() -> List[dict]:
    """글로벌 매크로 뉴스 수집."""
    queries = [
        ("stock market today", "en"), ("US Federal Reserve interest rate", "en"),
        ("semiconductor market today", "en"), ("코스피", "ko"), ("환율", "ko"),
    ]
    news: List[dict] = []
    for q, lang in queries:
        news.extend(fetch_news(q, max_items=2, language=lang))
    return _deduplicate(news)[:10]


# ---------------------------------------------------------------------------
# I/O — GPT 감성 분석
# ---------------------------------------------------------------------------
def analyze_news_with_gpt(
    stock_name: str, stock_news: List[dict], macro_news: List[dict],
    model: Optional[str] = None,
) -> Optional[dict]:
    """뉴스 → GPT 호재/악재 분석. 키/뉴스 없으면 None."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or not CFG.llm.enabled:
        return None
    if not stock_news and not macro_news:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)
    today_str = date.today().strftime("%Y년 %m월 %d일")
    prompt = (
        f"당신은 한국 주식시장 전문 애널리스트입니다. 오늘은 {today_str}입니다.\n\n"
        f"━━━ [{stock_name}] 종목 뉴스 ━━━\n{_format_news_for_gpt(stock_news[:8])}\n\n"
        f"━━━ 글로벌/매크로 뉴스 ━━━\n{_format_news_for_gpt(macro_news[:10])}\n\n"
        f"위 뉴스가 {stock_name} 주가에 미칠 영향을 평가해 아래 JSON 만 출력:\n"
        '{"verdict": "호재/악재/중립", "score": -1.0~1.0, "bullish_prob": 0~100,'
        ' "bearish_prob": 0~100, "caution_prob": 0~100, "stock_summary": "1~2문장",'
        ' "macro_summary": "1~2문장", "top_risks": ["r1","r2"],'
        ' "top_opportunities": ["o1","o2"], "recommendation": "매수적극/매수신중/관망/매도고려/매도추천"}'
    )
    try:
        resp = client.chat.completions.create(
            model=model or CFG.llm.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=CFG.llm.temperature,
            max_completion_tokens=CFG.llm.max_tokens_news,
        )
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:  # noqa: BLE001
        logger.warning("뉴스 GPT 오류: %s", e)
        return None


def news_signal(stock_name: str, stock_code: str = "") -> ModalitySignal:
    """종목명 → 뉴스 신호 (수집 + 분석 + 변환 일괄). 키 없으면 unavailable."""
    queries = [stock_name]
    if stock_code:
        queries.append(f"{stock_name}({stock_code})")
    stock_news = collect_stock_news(stock_name, queries)
    macro_news = collect_macro_news()
    analysis = analyze_news_with_gpt(stock_name, stock_news, macro_news)
    sig = news_result_to_signal(analysis)
    sig.detail["stock_news_count"] = len(stock_news)
    sig.detail["macro_news_count"] = len(macro_news)
    return sig
