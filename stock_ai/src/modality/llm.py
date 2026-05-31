"""LLM 내러티브 레이어 — 융합 결과를 사람이 읽는 설명으로.

구버전 `models/llm/llm_analyzer.py` 이식·재구성.

핵심 차이: 구버전은 LLM 이 신호를 '종합 판단'까지 했지만(블랙박스),
여기서는 **정량 Late Fusion(`fusion.py`)이 점수·판정을 먼저 결정**하고,
LLM 은 그 결정의 **근거를 설명(narrative)** 하는 역할만 한다.
→ 점수는 재현 가능·검증 가능하게, 설명은 풍부하게.

키가 없으면 결정적 템플릿(`_template_narrative`)으로 fallback 하므로
LLM 없이도 항상 동작한다.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

from src.config import CFG
from src.modality.base import ModalitySignal
from src.modality.fusion import FusionResult

logger = logging.getLogger(__name__)


def _signals_block(signals: Dict[str, ModalitySignal]) -> str:
    lines = []
    label_map = {
        "chart": "차트(기술적)", "fundamental": "펀더멘털",
        "valuation": "밸류에이션", "news": "뉴스 감성", "macro": "시장 매크로",
    }
    for name, sig in signals.items():
        ko = label_map.get(name, name)
        if not sig.available:
            lines.append(f"- {ko}: 데이터 없음")
        else:
            lines.append(
                f"- {ko}: {sig.score:.0f}/100 ({sig.display_label}, 신뢰도 {sig.confidence*100:.0f}%)"
            )
    return "\n".join(lines)


def _template_narrative(stock_name: str, fusion: FusionResult) -> dict:
    """LLM 없이 만드는 결정적 설명 (fallback)."""
    used = ", ".join(fusion.used) if fusion.used else "없음"
    interp = f"{used} 모달리티를 가중 결합해 {fusion.score:.0f}점({fusion.direction})으로 평가."
    if fusion.conflict:
        interp += f" 단, {fusion.conflict_note}."
    return {
        "summary": f"{stock_name}: 멀티모달 융합 {fusion.score:.0f}/100 → {fusion.verdict}",
        "recommendation": fusion.verdict,
        "signal_interpretation": interp,
        "conflict_resolution": fusion.conflict_note if fusion.conflict else None,
        "confidence": round(fusion.confidence * 100),
        "source": "template",
    }


def narrate(stock_name: str, fusion: FusionResult, model: Optional[str] = None) -> dict:
    """융합 결과 → 투자 의견 내러티브.

    LLM 키가 있으면 GPT 로 풍부한 설명을, 없으면 템플릿으로 생성.
    **추천 라벨·점수는 항상 fusion 값을 따른다** (LLM 이 판정을 바꾸지 않음).
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or not CFG.llm.enabled:
        return _template_narrative(stock_name, fusion)
    try:
        from openai import OpenAI
    except ImportError:
        return _template_narrative(stock_name, fusion)

    client = OpenAI(api_key=api_key)
    prompt = (
        "당신은 한국 주식시장 전문 애널리스트입니다.\n"
        f"아래는 {stock_name} 에 대한 정량 멀티모달 분석 결과입니다.\n\n"
        f"▸ 모달리티별 신호:\n{_signals_block(fusion.signals)}\n\n"
        f"▸ 정량 융합 결과(확정):\n"
        f"  - 융합점수 {fusion.score:.0f}/100 ({fusion.direction})\n"
        f"  - 판정: {fusion.verdict}\n"
        f"  - 신호 일관성: {fusion.conflict_note}\n\n"
        "위 '확정된 판정'을 바꾸지 말고, 그 판정에 이른 근거를 설명하세요.\n"
        "아래 JSON 만 출력:\n"
        '{"summary": "현재 상태 1문장", "signal_interpretation": "각 신호를 어떻게 해석했는지 1~2문장",'
        ' "conflict_resolution": "충돌이 있으면 해석, 없으면 null", "risks": ["r1","r2"],'
        ' "opportunities": ["o1","o2"], "strategy": "전략 1~2문장", "key_watch_points": ["w1","w2"]}'
    )
    try:
        resp = client.chat.completions.create(
            model=model or CFG.llm.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=CFG.llm.temperature,
            max_completion_tokens=CFG.llm.max_tokens_analysis,
        )
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        # 판정·신뢰도는 fusion 이 결정 (LLM 출력으로 덮어쓰지 않음)
        result["recommendation"] = fusion.verdict
        result["confidence"] = round(fusion.confidence * 100)
        result["source"] = "llm"
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM narrate 오류 — 템플릿 fallback: %s", e)
        return _template_narrative(stock_name, fusion)


def _num(x, suf: str = "") -> str:
    try:
        if x is None or (isinstance(x, float) and x != x):
            return "–"
        return f"{float(x):.1f}{suf}"
    except (TypeError, ValueError):
        return "–"


def recommend_summary(stocks: list, as_of: str, model: Optional[str] = None) -> dict:
    """추천 종목의 [**이미 계산된 점수** + **뉴스 텍스트 원문**] → GPT 포트폴리오 종합.

    재평가·2차 fusion 없음 — GPT 가 (정량 점수 + 텍스트)를 읽고 종합(LLM=fusion 역할).
    stocks: [{name, ticker, market, ensemble_score, rule_score, ai_score, roe, per,
              operating_margin, revenue_growth_yoy, news:[헤드라인,...]}, ...]
    """
    def _template() -> dict:
        ranked = sorted(stocks, key=lambda s: -(s.get("ensemble_score")
                                                 or s.get("rule_score") or 0))
        names = ", ".join(s.get("name", s.get("ticker", "?")) for s in ranked)
        return {"overview": f"추천 {len(stocks)}종목(점수순): {names}. "
                            "상세 종합은 OPENAI_API_KEY 설정 시 GPT 가 점수+뉴스로 생성.",
                "source": "template"}

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not stocks or not api_key or not CFG.llm.enabled:
        return _template()
    try:
        from openai import OpenAI
    except ImportError:
        return _template()

    blocks = []
    for s in stocks:
        blk = (
            f"■ {s.get('name')}({s.get('ticker')}) [{s.get('market', '')}]\n"
            f"   점수(이미 계산됨): 앙상블 {_num(s.get('ensemble_score'))} "
            f"(Rule {_num(s.get('rule_score'))} / AI {_num(s.get('ai_score'))}) · "
            f"ROE {_num(s.get('roe'), '%')} · PER {_num(s.get('per'))} · "
            f"영업이익률 {_num(s.get('operating_margin'), '%')}"
        )
        news = s.get("news") or []
        if news:
            blk += "\n   최근 뉴스:\n" + "\n".join(f"     - {t}" for t in news[:5])
        else:
            blk += "\n   최근 뉴스: (없음)"
        blocks.append(blk)
    body = "\n\n".join(blocks)

    prompt = (
        "당신은 한국 주식 포트폴리오 애널리스트입니다.\n"
        f"기준일 {as_of}. 아래는 정량 스크리닝(Rule+AI)으로 선별된 종목들의 "
        "**이미 계산된 점수**와 **최근 뉴스 헤드라인(원문)** 입니다.\n"
        "점수는 바꾸지 말고, **점수와 뉴스를 함께 읽어** 포트폴리오 관점에서 종합하세요.\n\n"
        f"{body}\n\n"
        "아래 JSON 만 출력:\n"
        '{"overview":"전체 1~2문장","top_picks":["우선순위 2~3종목과 이유(점수+뉴스 근거)"],'
        '"cautions":["뉴스/지표상 주의점"],"theme":"공통 테마 1문장"}'
    )
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model or CFG.llm.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=CFG.llm.temperature,
            max_completion_tokens=CFG.llm.max_tokens_analysis,
        )
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["source"] = "llm"
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning("recommend_summary LLM 오류 — 템플릿 fallback: %s", e)
        return _template()


def format_report(stock_name: str, stock_code: str, fusion: FusionResult, narrative: dict) -> str:
    """콘솔/대시보드용 텍스트 리포트."""
    lines = [
        "╔" + "═" * 56 + "╗",
        f"  📊 {stock_name} ({stock_code}) — 멀티모달 분석",
        "╚" + "═" * 56 + "╝",
        "",
        f"▸ 융합점수: {fusion.score:.1f}/100 ({fusion.direction}) → {narrative.get('recommendation')}",
        f"  신뢰도 {narrative.get('confidence')}%   |   {narrative.get('summary','')}",
        "",
        "▸ 모달리티별 신호:",
        _signals_block(fusion.signals),
        "",
        "▸ 신호 해석:",
        f"  {narrative.get('signal_interpretation','')}",
    ]
    if fusion.conflict:
        lines += ["", f"⚠️ {fusion.conflict_note}"]
    for key, head in (("risks", "주요 리스크"), ("opportunities", "주요 기회"),
                      ("key_watch_points", "주시사항")):
        items = narrative.get(key)
        if items:
            lines.append(f"▸ {head}:")
            lines.extend(f"  · {it}" for it in items)
    if narrative.get("strategy"):
        lines += ["", f"▸ 전략: {narrative['strategy']}"]
    lines += ["", "⚠️ 참고용 분석이며 투자 책임은 투자자에게 있습니다."]
    return "\n".join(lines)
