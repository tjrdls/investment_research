"""멀티모달 신호 표준 — 모든 모달리티가 공통으로 내는 신호 단위.

각 모달리티(차트·펀더멘털·밸류에이션·뉴스·매크로)는 `ModalitySignal` 을 만들고,
`fusion.LateFusion` 이 이를 가중 결합한다.

이 파일의 함수·메서드는 **전부 순수 함수**다 (외부 I/O 없음).
→ DB·API 키 없이 `tests/` 에서 완전히 단위 테스트할 수 있다.

점수 규약
---------
score 는 0~100 스케일.
  - 50  = 중립
  - >50 = 강세 / 호재 (높을수록 매수 우호)
  - <50 = 약세 / 악재
기존 `rule_score`, `ai_score` (0~100) 와 동일 스케일이라 그대로 섞을 수 있다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

NEUTRAL: float = 50.0


# ---------------------------------------------------------------------------
# 클램프 / 스케일 변환 헬퍼 (순수 함수)
# ---------------------------------------------------------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def clamp_score(x: Optional[float]) -> float:
    """임의 값 → 0~100 (None/NaN 은 중립 50)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return NEUTRAL
    return _clamp(float(x), 0.0, 100.0)


def clamp_conf(x: Optional[float]) -> float:
    """임의 값 → 0~1 (None/NaN 은 0)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return _clamp(float(x), 0.0, 1.0)


def from_bipolar(score: Optional[float]) -> float:
    """[-1, 1] 양극 점수 → [0, 100].  (GPT 뉴스 score 등에 사용)

    -1 → 0, 0 → 50, +1 → 100.
    """
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return NEUTRAL
    return clamp_score((float(score) + 1.0) * 50.0)


def to_bipolar(score100: Optional[float]) -> float:
    """[0, 100] → [-1, 1] (역변환)."""
    return clamp_score(score100) / 50.0 - 1.0


def from_probability(prob_up: Optional[float], prob_down: Optional[float] = None) -> float:
    """확률 → 0~100.

    - `prob_down` 이 주어지면 상승확률과 하락확률의 차이를 0~100 으로 매핑
      (LSTM 3-class 출력 등). 둘 다 0.5 면 50.
    - `prob_down` 이 없으면 `prob_up` (0~1) 을 그대로 0~100 으로.
    입력이 0~100(퍼센트)로 들어와도 자동 감지해 0~1 로 정규화한다.
    """
    def _norm(p: Optional[float]) -> float:
        if p is None or (isinstance(p, float) and math.isnan(p)):
            return float("nan")
        p = float(p)
        return p / 100.0 if p > 1.0 else p

    up = _norm(prob_up)
    if math.isnan(up):
        return NEUTRAL
    if prob_down is None:
        return clamp_score(up * 100.0)
    down = _norm(prob_down)
    if math.isnan(down):
        return clamp_score(up * 100.0)
    # 차이를 [-1, 1] 로 보고 변환: up=down → 0 → 50
    return from_bipolar(up - down)


def direction_of(score: float, deadband: float = 5.0) -> str:
    """score(0~100) → 방향 라벨. 50 ± deadband 안은 '중립'."""
    if score >= NEUTRAL + deadband:
        return "강세"
    if score <= NEUTRAL - deadband:
        return "약세"
    return "중립"


# ---------------------------------------------------------------------------
# ModalitySignal
# ---------------------------------------------------------------------------
@dataclass
class ModalitySignal:
    """단일 모달리티의 표준 출력.

    Attributes
    ----------
    name : str
        모달리티 이름 ("chart", "fundamental", "valuation", "news", "macro").
    score : float
        0~100. 50=중립, 높을수록 강세.
    confidence : float
        0~1. 이 신호를 얼마나 신뢰할지 (Late Fusion 에서 가중에 반영).
    available : bool
        데이터/키 부재로 신호를 못 낸 경우 False → 융합 시 가중치에서 제외.
    label : str
        사람이 읽는 방향/판정 ("호재", "저평가" 등). 비면 direction 으로 대체.
    detail : dict
        원시 근거 (PER, 뉴스 요약, feature 값 등).
    """

    name: str
    score: float = NEUTRAL
    confidence: float = 0.5
    available: bool = True
    label: str = ""
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = clamp_score(self.score)
        self.confidence = clamp_conf(self.confidence)
        if not self.available:
            # 사용 불가 신호는 중립·무신뢰로 고정
            self.score = NEUTRAL
            self.confidence = 0.0

    @property
    def direction(self) -> str:
        return direction_of(self.score)

    @property
    def display_label(self) -> str:
        return self.label or self.direction

    # ── 생성 헬퍼 ────────────────────────────────────────────
    @classmethod
    def from_bipolar(
        cls,
        name: str,
        score_pm1: Optional[float],
        confidence: float = 0.5,
        label: str = "",
        **detail,
    ) -> "ModalitySignal":
        """[-1,1] 점수로부터 생성 (GPT 뉴스 등)."""
        return cls(
            name=name,
            score=from_bipolar(score_pm1),
            confidence=confidence,
            label=label,
            detail=detail,
        )

    @classmethod
    def from_probability(
        cls,
        name: str,
        prob_up: Optional[float],
        prob_down: Optional[float] = None,
        confidence: Optional[float] = None,
        label: str = "",
        **detail,
    ) -> "ModalitySignal":
        """확률로부터 생성 (LSTM 등). confidence 미지정 시 |up-down| 로 추정."""
        score = from_probability(prob_up, prob_down)
        if confidence is None:
            # 방향이 뚜렷할수록 신뢰 ↑
            confidence = clamp_conf(abs(to_bipolar(score)))
        return cls(name=name, score=score, confidence=confidence, label=label, detail=detail)

    @classmethod
    def unavailable(cls, name: str, reason: str = "") -> "ModalitySignal":
        """데이터/키 부재 신호."""
        return cls(name=name, available=False, label="데이터없음", detail={"reason": reason})
