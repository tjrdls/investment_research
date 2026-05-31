"""뉴스 신호 변환 + 오케스트레이터 fuse_signals 테스트 (DB/API 불필요).

OPENAI_API_KEY 가 없으면 llm.narrate 는 결정적 템플릿으로 fallback 하므로
이 테스트는 키 없이도 통과한다.
"""
import os

import pytest

from src.modality.base import NEUTRAL, ModalitySignal
from src.modality.news import news_result_to_signal
from src.modality.analyzer import fuse_signals


# ── 뉴스 결과 → 신호 ──────────────────────────────────────
def test_news_none_unavailable():
    assert news_result_to_signal(None).available is False


def test_news_bullish_score():
    analysis = {"score": 0.6, "verdict": "호재", "bullish_prob": 70, "bearish_prob": 10}
    sig = news_result_to_signal(analysis)
    assert sig.available is True
    assert sig.score > NEUTRAL
    assert sig.label == "호재"


def test_news_infers_score_from_probs_when_missing():
    analysis = {"verdict": "악재", "bullish_prob": 10, "bearish_prob": 80}
    sig = news_result_to_signal(analysis)
    assert sig.score < NEUTRAL          # 하락확률 우세 → 약세


def test_news_high_caution_lowers_confidence():
    calm = news_result_to_signal({"score": 0.4, "bullish_prob": 60, "bearish_prob": 20, "caution_prob": 0})
    nervous = news_result_to_signal({"score": 0.4, "bullish_prob": 60, "bearish_prob": 20, "caution_prob": 90})
    assert nervous.confidence < calm.confidence


# ── 오케스트레이터 (템플릿 fallback) ──────────────────────
def test_fuse_signals_template_fallback_no_api(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    signals = {
        "fundamental": ModalitySignal(name="fundamental", score=75, confidence=0.7),
        "chart": ModalitySignal(name="chart", score=65, confidence=0.6),
        "valuation": ModalitySignal(name="valuation", score=60, confidence=0.5, label="저평가"),
        "news": ModalitySignal.unavailable("news"),
        "macro": ModalitySignal(name="macro", score=55, confidence=0.5),
    }
    out = fuse_signals("테스트종목", signals)
    fusion = out["fusion"]
    assert fusion.score > NEUTRAL
    assert "news" in fusion.skipped
    # 키 없으면 template 소스
    assert out["narrative"]["source"] == "template"
    # 판정 라벨은 fusion 과 일치 (LLM 이 바꾸지 않음)
    assert out["narrative"]["recommendation"] == fusion.verdict
    assert "테스트종목" in out["report"]


def test_fuse_signals_conflict_surfaced_in_narrative(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    signals = {
        "chart": ModalitySignal(name="chart", score=90, confidence=0.8),
        "news": ModalitySignal(name="news", score=10, confidence=0.8),
    }
    out = fuse_signals("충돌종목", signals, weights={"chart": 0.5, "news": 0.5})
    assert out["fusion"].conflict is True
    assert out["narrative"]["conflict_resolution"] is not None
