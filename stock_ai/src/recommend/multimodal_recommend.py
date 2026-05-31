"""추천 GPT 종합 레이어 — 이미 계산된 값 + 뉴스 텍스트 → GPT (재평가 없음).

[1단] Rule+AI **숫자** 스크리닝(`select_top_n`)으로 Top N 선별 (이미 계산됨).
[2단] picks 의 **기존 값**(앙상블/Rule/AI 점수·ROE·PER 등)과 종목별 **뉴스 텍스트 원문**을
      **그대로 GPT 에 전달** → GPT 가 (정량 점수 + 텍스트)를 읽고 포트폴리오 종합.

핵심: **재평가·2차 정량 fusion 없음.** GPT 가 신호+텍스트를 종합(LLM=fusion 역할) — 원본 stockAI
"LLM 이 종합 판단" 철학. 멀티모달성 = 숫자 모달(점수) + 텍스트 모달(뉴스)을 GPT 가 결합.
라이브 전용(뉴스·GPT 는 과거 재구성 불가라 백테스트엔 미적용).
"""
from __future__ import annotations

import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from src.modality import llm as llm_mod

logger = logging.getLogger(__name__)


def fetch_news_text(stock_name: str, limit: int = 5, timeout: int = 8) -> list:
    """종목 관련 최근 뉴스 헤드라인(텍스트) 리스트. 실패 시 빈 리스트 (graceful)."""
    url = ("https://news.google.com/rss/search?q="
           + urllib.parse.quote(f"{stock_name} 주가") + "&hl=ko&gl=KR&ceid=KR:ko")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            root = ET.fromstring(r.read())
    except Exception as e:  # noqa: BLE001
        logger.debug("뉴스 조회 실패 (%s): %s", stock_name, e)
        return []
    out = []
    for item in list(root.iter("item"))[:limit]:
        title = (item.findtext("title") or "").strip()
        src = item.find("source")
        src_name = (src.text or "") if src is not None else ""
        if src_name and title.endswith(f" - {src_name}"):
            title = title[: -(len(src_name) + 3)].strip()
        if title:
            out.append(title)
    return out


_VALUE_COLS = ("name", "ticker", "market", "ensemble_score", "rule_score",
               "ai_score", "roe", "per", "operating_margin", "revenue_growth_yoy")


def _row_values(row) -> dict:
    """picks 의 한 행에서 이미 계산된 값만 추출 (재계산 없음)."""
    get = row.get if hasattr(row, "get") else (lambda k, d=None: d)
    v = {c: get(c) for c in _VALUE_COLS}
    if not v.get("name"):
        v["name"] = v.get("ticker")
    return v


def recommend_with_gpt(picks, as_of: str, news_limit: int = 5,
                       with_news: bool = True) -> dict:
    """picks(DataFrame) → {stocks:[값+뉴스], summary: GPT 종합}.

    재평가 없음 — picks 의 기존 점수 + 종목별 뉴스 텍스트만 모아 GPT 에 전달.
    """
    stocks = []
    for _, row in picks.iterrows():
        v = _row_values(row)
        v["news"] = fetch_news_text(v["name"], limit=news_limit) if with_news else []
        stocks.append(v)
    summary = llm_mod.recommend_summary(stocks, as_of)
    return {"stocks": stocks, "summary": summary}
