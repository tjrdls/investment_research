"""학습된 Late Fusion 단위 테스트 (소규모 합성 데이터, 빠름)."""
import numpy as np
import pytest

from src.modality.base import ModalitySignal
from src.modality.learned_fusion import (
    GATE_INPUT_DIM,
    TRAIN_MODALITIES,
    LearnedFusion,
    signals_to_features,
)


def _sig(name, score, conf=0.6, avail=True):
    return (ModalitySignal(name=name, score=score, confidence=conf)
            if avail else ModalitySignal.unavailable(name, "no data"))


def test_signals_to_features_shapes_and_values():
    signals = {
        "chart": _sig("chart", 80),
        "fundamental": _sig("fundamental", 60),
        "valuation": _sig("valuation", 0, avail=False),   # 제외
        "macro": _sig("macro", 50),
    }
    feat, mask, score = signals_to_features(signals)
    assert feat.shape == (GATE_INPUT_DIM,)
    assert mask.tolist() == [1.0, 1.0, 0.0, 1.0]          # valuation 불가
    assert score.tolist() == [80.0, 60.0, 50.0, 50.0]      # 불가 → 중립 50
    # chart: score/100=0.8, conf=0.6, avail=1
    assert feat[0] == pytest.approx(0.8)
    assert feat[2] == 1.0


def test_fallback_when_no_model(tmp_path):
    lf = LearnedFusion(model_path=tmp_path / "none.pt")
    assert lf.available is False
    with pytest.raises(RuntimeError):
        lf.fuse({"chart": _sig("chart", 70)})


def test_fuse_signals_learned_falls_back_to_weighted(tmp_path, monkeypatch):
    # 학습 모델 없을 때 fusion_mode='learned' → weighted 폴백
    import src.modality.learned_fusion as LFM
    monkeypatch.setattr(LFM, "DEFAULT_GATE_PATH", tmp_path / "none.pt")
    from src.modality.analyzer import fuse_signals
    res = fuse_signals("X", {"chart": _sig("chart", 70),
                             "fundamental": _sig("fundamental", 65)},
                       with_narrative=False, fusion_mode="learned")
    assert res["fusion_mode"] == "weighted"
    assert 0 <= res["fusion"].score <= 100


def test_fit_and_predict(tmp_path):
    # 합성: 타깃이 chart 점수에 강하게 연동 → 게이트가 chart 에 가중 쏠림 기대
    rng = np.random.default_rng(0)
    N = 200
    chart = rng.uniform(0, 100, N)
    fund = rng.uniform(0, 100, N)
    val = rng.uniform(0, 100, N)
    mac = rng.uniform(0, 100, N)
    X = np.stack([
        chart / 100, np.full(N, 0.6), np.ones(N),
        fund / 100, np.full(N, 0.7), np.ones(N),
        val / 100, np.full(N, 0.5), np.ones(N),
        mac / 100, np.full(N, 0.5), np.ones(N),
    ], axis=1).astype(np.float32)
    M = np.ones((N, 4), np.float32)
    S = np.stack([chart, fund, val, mac], axis=1).astype(np.float32)
    y = (chart / 100).astype(np.float32)        # 타깃 = chart 백분위

    lf = LearnedFusion(model_path=tmp_path / "gate.pt")
    metrics = lf.fit(X, M, S, y, epochs=200)
    assert metrics["n"] == N
    assert metrics["val_loss"] < 0.08            # chart 만 보면 거의 완벽 → 낮은 손실
    lf.save()
    assert (tmp_path / "gate.pt").exists()

    # 재로드 후 추론
    lf2 = LearnedFusion(model_path=tmp_path / "gate.pt")
    assert lf2.available
    w = lf2.gate_weights({
        "chart": _sig("chart", 90), "fundamental": _sig("fundamental", 50),
        "valuation": _sig("valuation", 50), "macro": _sig("macro", 50),
    })
    assert abs(sum(w.values()) - 1.0) < 1e-4
    assert w["chart"] == max(w.values())         # chart 가중이 최대
    fr = lf2.fuse({"chart": _sig("chart", 90), "fundamental": _sig("fundamental", 50),
                   "valuation": _sig("valuation", 50), "macro": _sig("macro", 50)})
    assert 0 <= fr.score <= 100
