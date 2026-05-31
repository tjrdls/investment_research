"""추천 GPT 종합 — 템플릿 fallback + 값/뉴스 전달 테스트 (네트워크 불필요)."""
from src.modality.llm import recommend_summary


def test_recommend_summary_template_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stocks = [
        {"name": "삼성전자", "ticker": "005930", "market": "KOSPI",
         "ensemble_score": 100.3, "rule_score": 100.3, "ai_score": 45.9,
         "roe": 23.6, "per": 31.0, "news": ["삼성전자 신고가", "반도체 업황 회복"]},
        {"name": "SK하이닉스", "ticker": "000660", "market": "KOSPI",
         "ensemble_score": 133.7, "rule_score": 133.7, "ai_score": 41.3,
         "roe": 46.2, "per": 22.7, "news": []},
    ]
    s = recommend_summary(stocks, "2026-05-29")
    assert s["source"] == "template"            # 키 없음 → 템플릿
    # 점수순 정렬 (SK 133.7 > 삼성 100.3)
    assert s["overview"].index("SK하이닉스") < s["overview"].index("삼성전자")


def test_recommend_summary_empty():
    s = recommend_summary([], "2026-05-29")
    assert s["source"] == "template"


def test_row_values_extracts_existing_only():
    import pandas as pd
    from src.recommend.multimodal_recommend import _row_values
    row = pd.Series({"ticker": "005930", "name": "삼성전자", "market": "KOSPI",
                     "ensemble_score": 100.3, "rule_score": 99.0, "roe": 23.6})
    v = _row_values(row)
    assert v["name"] == "삼성전자" and v["ensemble_score"] == 100.3
    assert v["per"] is None        # 없는 값은 None (재계산 안 함)
