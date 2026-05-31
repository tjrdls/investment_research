"""Late Fusion 엔진 — 여러 모달리티 신호를 가중 결합.

구버전 `models/llm/llm_analyzer.py` 의 "신호 종합 + 충돌 감지" 철학을
정량 시스템에 맞게 순수 함수로 재구성했다.

Intermediate Fusion(모델 내부에서 섞기) 대신 **Late Fusion** 을 쓰는 이유는
구버전 ARCHITECTURE.md 의 결론과 동일하다: 각 모달리티가 독립적으로 해석 가능한
신호를 내야 (a) 신호 간 충돌을 감지하고 (b) 근거를 설명할 수 있기 때문.

이 파일은 외부 I/O 가 없다 → `tests/test_fusion.py` 로 완전 검증 가능.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.modality.base import NEUTRAL, ModalitySignal, direction_of

# 융합 점수 → 추천 라벨 임계값 (0~100)
VERDICT_THRESHOLDS: List[tuple] = [
    (70.0, "매수 강력추천"),
    (58.0, "매수"),
    (42.0, "관망"),
    (30.0, "매도"),
    (0.0, "매도 강력추천"),
]

# 기본 가중치 (정량 스크리닝 기준: 차트·펀더·매크로 중심, 뉴스/밸류는 보조)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "fundamental": 0.40,
    "chart": 0.25,
    "macro": 0.15,
    "valuation": 0.10,
    "news": 0.10,
}


def verdict_of(score: float) -> str:
    """융합 점수 → 추천 라벨."""
    for thr, label in VERDICT_THRESHOLDS:
        if score >= thr:
            return label
    return VERDICT_THRESHOLDS[-1][1]


@dataclass
class FusionResult:
    """Late Fusion 결과."""

    score: float                       # 0~100 최종 융합 점수
    confidence: float                  # 0~1 평균 신뢰도
    direction: str                     # 강세/중립/약세
    verdict: str                       # 추천 라벨
    conflict: bool                     # 신호 간 충돌 여부
    conflict_note: str                 # 충돌 설명
    contributions: Dict[str, float]    # 모달리티별 가중 기여도 (정규화, 합=1)
    used: List[str] = field(default_factory=list)        # 융합에 쓰인 모달리티
    skipped: List[str] = field(default_factory=list)     # 데이터 없어 제외된 모달리티
    signals: Dict[str, ModalitySignal] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"융합점수 {self.score:.1f}/100 ({self.direction}) → {self.verdict}",
            f"신뢰도 {self.confidence*100:.0f}%",
        ]
        if self.conflict:
            parts.append(f"⚠️ {self.conflict_note}")
        if self.skipped:
            parts.append(f"제외: {', '.join(self.skipped)}")
        return " | ".join(parts)


class LateFusion:
    """모달리티 신호 가중 결합기.

    Parameters
    ----------
    weights : dict[str, float], optional
        모달리티별 기본 가중치. 미지정 시 `DEFAULT_WEIGHTS`.
    confidence_weighted : bool
        True 면 가중치에 각 신호의 confidence 를 곱해 신뢰도 낮은 신호의
        영향을 줄인다.
    conflict_strong : float
        '강한 신호'로 볼 중립 이탈 폭 (기본 15 → score>65 또는 <35).
    """

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        confidence_weighted: bool = True,
        conflict_strong: float = 15.0,
    ):
        self.weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        self.confidence_weighted = confidence_weighted
        self.conflict_strong = conflict_strong

    def _detect_conflict(self, signals: Dict[str, ModalitySignal]) -> tuple[bool, str]:
        """강세 신호와 약세 신호가 동시에 존재하면 충돌."""
        s = self.conflict_strong
        bull = [sig.name for sig in signals.values()
                if sig.available and sig.score >= NEUTRAL + s]
        bear = [sig.name for sig in signals.values()
                if sig.available and sig.score <= NEUTRAL - s]
        if bull and bear:
            return True, f"강세({'/'.join(bull)}) vs 약세({'/'.join(bear)}) 충돌 — 보수적 해석 권장"
        return False, "신호 간 충돌 없음"

    def fuse(self, signals: Dict[str, ModalitySignal]) -> FusionResult:
        """신호 dict → FusionResult.

        사용 불가(available=False) 모달리티는 가중치에서 제외하고
        남은 모달리티로 가중치를 재정규화한다.
        """
        used, skipped = [], []
        eff_weights: Dict[str, float] = {}
        for name, w in self.weights.items():
            sig = signals.get(name)
            if sig is None or not sig.available:
                skipped.append(name)
                continue
            ew = w * (sig.confidence if self.confidence_weighted else 1.0)
            if ew <= 0:
                skipped.append(name)
                continue
            eff_weights[name] = ew
            used.append(name)

        total = sum(eff_weights.values())
        if total <= 0:
            # 쓸 신호가 하나도 없음 → 완전 중립
            return FusionResult(
                score=NEUTRAL, confidence=0.0, direction="중립",
                verdict=verdict_of(NEUTRAL), conflict=False,
                conflict_note="유효 신호 없음", contributions={},
                used=[], skipped=list(self.weights.keys()), signals=signals,
            )

        contributions = {n: w / total for n, w in eff_weights.items()}
        fused_score = sum(contributions[n] * signals[n].score for n in used)

        # 평균 신뢰도는 기본 가중치(신뢰도 제외) 기준
        base_total = sum(self.weights[n] for n in used)
        fused_conf = (
            sum(self.weights[n] * signals[n].confidence for n in used) / base_total
            if base_total > 0 else 0.0
        )

        conflict, note = self._detect_conflict(signals)

        return FusionResult(
            score=round(fused_score, 2),
            confidence=round(fused_conf, 3),
            direction=direction_of(fused_score),
            verdict=verdict_of(fused_score),
            conflict=conflict,
            conflict_note=note,
            contributions={n: round(c, 3) for n, c in contributions.items()},
            used=used,
            skipped=skipped,
            signals=signals,
        )
