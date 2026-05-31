"""멀티모달 분석 레이어.

stock_ai 의 정량 파이프라인(하드필터 → Rule → LGBM 앙상블 → 포트폴리오)
위에 **멀티모달 종목 분석**을 얹는 모듈.

구버전(단일 종목 LSTM+RAG+LLM)의 강점이던 뉴스 감성·밸류에이션·LLM Late Fusion 을
표준화된 `ModalitySignal` 단위로 재구성해, 개선본의 차트·펀더·매크로 신호와 함께
`LateFusion` 으로 결합한다.

설계 원칙
---------
- **순수 로직과 I/O 분리**: 신호 정규화·가중 결합·충돌 감지는 전부 순수 함수
  (`base`, `fusion`) → DB/API 없이 단위 테스트 가능.
- **우아한 degrade**: 키·DB·데이터가 없으면 해당 모달리티는 `unavailable` 신호를
  내고 가중치에서 자동 제외된다 (전체 분석은 멈추지 않음).
- **점수 규약 통일**: 모든 신호 score 는 0~100 (50=중립, 기존 rule_score/ai_score 와 동일 스케일).
"""
from __future__ import annotations

from src.modality.base import (
    NEUTRAL,
    ModalitySignal,
    clamp_conf,
    clamp_score,
    direction_of,
    from_bipolar,
    from_probability,
    to_bipolar,
)
from src.modality.fusion import FusionResult, LateFusion

__all__ = [
    "NEUTRAL",
    "ModalitySignal",
    "clamp_conf",
    "clamp_score",
    "direction_of",
    "from_bipolar",
    "from_probability",
    "to_bipolar",
    "FusionResult",
    "LateFusion",
]
