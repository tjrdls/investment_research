"""학습된 Late Fusion — 게이팅 신경망 (옵션).

기본 `LateFusion`(고정 가중 평균)은 **그대로 유지**한다. 이 모듈은 모달리티
가중치를 고정하지 않고, 신호 패턴(점수·신뢰도·가용성)에 따라 **조건부로 학습**한
게이팅 가중치를 산출한다. 여전히 모달별 가중치를 출력하므로 해석 가능성은 유지
(late fusion 철학 존중) — "고정 평균"과 "신경망 게이팅"의 중간.

설계
----
- 게이트 입력: 학습 가능한 4개 모달(chart/fundamental/valuation/macro) × [score/100, confidence, available]
  = 12차원.  (뉴스는 과거 재구성이 불가 → 학습 제외, 추론 시 고정 비중으로 blend)
- 게이트 출력: 4개 logit → 가용 모달만 마스킹 softmax → 가중치(합=1).
- 융합 점수 = Σ 가중치 × score.
- 타깃: 42거래일 후 횡단면 수익률 백분위(0~1).

모델 파일(`data/models/fusion_gate.pt`)이 없으면 `available=False` → 호출측이
기본 `LateFusion` 으로 폴백한다 (graceful degrade).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config import MODEL_DIR
from src.modality.base import NEUTRAL, ModalitySignal, direction_of
from src.modality.fusion import FusionResult, verdict_of

logger = logging.getLogger(__name__)

# 게이트가 학습하는 모달 (순서 고정 — 입력/출력 차원 정의)
TRAIN_MODALITIES: List[str] = ["chart", "fundamental", "valuation", "macro"]
FEATS_PER_MODAL = 3            # [score/100, confidence, available]
GATE_INPUT_DIM = len(TRAIN_MODALITIES) * FEATS_PER_MODAL   # 12
DEFAULT_GATE_PATH = MODEL_DIR / "fusion_gate.pt"
NEWS_BLEND = 0.10              # 추론 시 뉴스 신호를 섞는 고정 비중


def signals_to_features(signals: Dict[str, ModalitySignal]):
    """signals → (feat[12], mask[4], score[4]). 학습/추론 공용."""
    feat, mask, score = [], [], []
    for m in TRAIN_MODALITIES:
        sig = signals.get(m)
        avail = bool(sig and sig.available)
        s = float(sig.score) if avail else NEUTRAL
        c = float(sig.confidence) if avail else 0.0
        feat.extend([s / 100.0, c, 1.0 if avail else 0.0])
        mask.append(1.0 if avail else 0.0)
        score.append(s)
    return (np.array(feat, dtype=np.float32),
            np.array(mask, dtype=np.float32),
            np.array(score, dtype=np.float32))


def _make_gate():
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(GATE_INPUT_DIM, 16), nn.ReLU(),
        nn.Linear(16, len(TRAIN_MODALITIES)),
    )


def _masked_softmax(logits, mask):
    """가용 모달만 softmax (불가 모달은 가중치 0). logits/mask: torch (N,4)."""
    import torch
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(mask > 0, logits, torch.full_like(logits, neg_inf))
    w = torch.softmax(masked, dim=-1)
    return torch.where(mask > 0, w, torch.zeros_like(w))


class LearnedFusion:
    """게이팅 신경망 기반 융합기 (LateFusion 과 drop-in 호환: .fuse(signals)->FusionResult)."""

    def __init__(self, model_path: Optional[Path] = None,
                 news_blend: float = NEWS_BLEND, conflict_strong: float = 15.0):
        # 모듈 전역을 호출 시점에 읽음 (테스트에서 monkeypatch 가능, 기본 인자 바인딩 회피)
        self.model_path = Path(model_path) if model_path else DEFAULT_GATE_PATH
        self.news_blend = news_blend
        self.conflict_strong = conflict_strong
        self._net = None
        self.available = False
        self._try_load()

    def _try_load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            import torch
            self._net = _make_gate()
            self._net.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            self._net.eval()
            self.available = True
            logger.info("LearnedFusion 로드: %s", self.model_path.name)
        except Exception as e:  # noqa: BLE001
            logger.warning("LearnedFusion 로드 실패 → 폴백: %s", e)
            self.available = False

    # ── 추론 ──────────────────────────────────────────────
    def gate_weights(self, signals: Dict[str, ModalitySignal]) -> Dict[str, float]:
        """학습 4개 모달에 대한 게이팅 가중치 (가용분 합=1)."""
        import torch
        feat, mask, _ = signals_to_features(signals)
        with torch.no_grad():
            logits = self._net(torch.from_numpy(feat).unsqueeze(0))
            w = _masked_softmax(logits, torch.from_numpy(mask).unsqueeze(0))
        return {m: float(w[0, i]) for i, m in enumerate(TRAIN_MODALITIES)}

    def fuse(self, signals: Dict[str, ModalitySignal]) -> FusionResult:
        if not self.available:
            raise RuntimeError("LearnedFusion 모델 미로드 — LateFusion 으로 폴백하세요")

        w = self.gate_weights(signals)
        used = [m for m in TRAIN_MODALITIES
                if signals.get(m) and signals[m].available and w[m] > 0]
        skipped = [m for m in TRAIN_MODALITIES if m not in used]

        if not used:
            return FusionResult(
                score=NEUTRAL, confidence=0.0, direction="중립",
                verdict=verdict_of(NEUTRAL), conflict=False,
                conflict_note="유효 신호 없음", contributions={},
                used=[], skipped=list(TRAIN_MODALITIES), signals=signals)

        core = sum(w[m] * signals[m].score for m in used)
        contributions = {m: w[m] for m in used}

        # 뉴스: 학습 제외 모달 → 가용 시 고정 비중으로 blend
        news = signals.get("news")
        if news is not None and news.available:
            nb = self.news_blend
            fused = (1 - nb) * core + nb * news.score
            contributions = {m: (1 - nb) * w[m] for m in used}
            contributions["news"] = nb
            used = used + ["news"]
        else:
            fused = core
            if news is not None:
                skipped.append("news")

        conf = (sum(contributions[m] * signals[m].confidence for m in used)
                / sum(contributions.values())) if used else 0.0
        conflict, note = self._detect_conflict(signals)

        return FusionResult(
            score=round(fused, 2), confidence=round(conf, 3),
            direction=direction_of(fused), verdict=verdict_of(fused),
            conflict=conflict, conflict_note=note,
            contributions={m: round(c, 3) for m, c in contributions.items()},
            used=used, skipped=skipped, signals=signals)

    def _detect_conflict(self, signals: Dict[str, ModalitySignal]):
        s = self.conflict_strong
        bull = [g.name for g in signals.values() if g.available and g.score >= NEUTRAL + s]
        bear = [g.name for g in signals.values() if g.available and g.score <= NEUTRAL - s]
        if bull and bear:
            return True, f"강세({'/'.join(bull)}) vs 약세({'/'.join(bear)}) 충돌 — 보수적 해석 권장"
        return False, "신호 간 충돌 없음"

    # ── 학습 ──────────────────────────────────────────────
    def fit(self, X: np.ndarray, mask: np.ndarray, score: np.ndarray, y: np.ndarray,
            epochs: int = 400, lr: float = 0.01, weight_decay: float = 1e-3,
            patience: int = 40, val_frac: float = 0.2, seed: int = 42) -> dict:
        """게이트 학습.  X(N,12) mask(N,4) score(N,4 0~100) y(N, 0~1).

        손실: MSE( Σ gate_w·score/100 , y ).  반환: {train_loss, val_loss, n}.
        """
        import torch
        rng = np.random.default_rng(seed)
        N = len(y)
        if N < 8:
            raise ValueError(f"샘플 부족({N}) — 최소 8개 필요 (데이터 더 수집 필요)")
        idx = rng.permutation(N)
        n_val = max(1, int(N * val_frac))
        va, tr = idx[:n_val], idx[n_val:]

        Xt = torch.from_numpy(X.astype(np.float32))
        Mt = torch.from_numpy(mask.astype(np.float32))
        St = torch.from_numpy(score.astype(np.float32) / 100.0)
        yt = torch.from_numpy(y.astype(np.float32))

        self._net = _make_gate()
        opt = torch.optim.Adam(self._net.parameters(), lr=lr, weight_decay=weight_decay)
        lossf = torch.nn.MSELoss()

        def _pred(ix):
            logits = self._net(Xt[ix])
            w = _masked_softmax(logits, Mt[ix])
            return (w * St[ix]).sum(dim=-1)

        best, best_state, bad = float("inf"), None, 0
        for ep in range(epochs):
            self._net.train(); opt.zero_grad()
            loss = lossf(_pred(tr), yt[tr]); loss.backward(); opt.step()
            self._net.eval()
            with torch.no_grad():
                vl = float(lossf(_pred(va), yt[va]))
            if vl < best - 1e-5:
                best, best_state, bad = vl, {k: v.clone() for k, v in self._net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state:
            self._net.load_state_dict(best_state)
        self._net.eval(); self.available = True
        with torch.no_grad():
            tl = float(lossf(_pred(tr), yt[tr]))
        return {"train_loss": round(tl, 5), "val_loss": round(best, 5),
                "n": N, "n_train": len(tr), "n_val": len(va), "epochs_ran": ep + 1}

    def save(self, path: Optional[Path] = None) -> Path:
        import torch
        path = Path(path or self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path)
        logger.info("LearnedFusion 저장: %s", path)
        return path
