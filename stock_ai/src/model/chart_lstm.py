"""
차트 전용 LSTM (추세 패턴 인식 모델)
===========================================
펀더멘털 제외, 차트 피처만으로 6개월 추세 신뢰도(0~1) 출력.
한국 + 미국 모든 종목으로 학습 가능 (차트는 시장 공통 패턴).

입력: chart_dim 개 피처 (RSI, MACD, MA들, 볼린저, 거래량 등)
출력: trend_confidence (0~1) — 6개월 후 상승 확률
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """(B, T, H) → (B, H) — 시점 가중치 학습."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = (x * weights.unsqueeze(-1)).sum(dim=1)
        return context, weights


class ChartLSTM(nn.Module):
    """
    차트 패턴 전용 LSTM.
    - 입력: (B, T, chart_dim) 시계열
    - 출력: trend_confidence (0~1), pred_return (실수)
    파라미터 약 100~200K (이전 모델의 1/3).
    """

    def __init__(
        self,
        chart_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.chart_dim = chart_dim

        self.lstm = nn.LSTM(
            input_size=chart_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = TemporalAttention(out_dim)

        # 추세 신뢰도 헤드 (0~1)
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 회귀 헤드 (실제 수익률)
        self.regressor = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, chart: torch.Tensor) -> dict:
        """
        chart: (B, T, chart_dim)
        반환:
          trend_confidence: (B,) 0~1 (상승 신뢰도)
          pred_return:      (B,) 실수 (6개월 예측 수익률)
        """
        out, _ = self.lstm(chart)  # (B, T, out_dim)
        context, attn = self.attention(out)  # (B, out_dim)

        confidence = self.confidence_head(context).squeeze(-1)
        pred_return = self.regressor(context).squeeze(-1)

        return {
            "trend_confidence": confidence,
            "pred_return": pred_return,
            "attn": attn,
        }


class ChartLSTMLoss(nn.Module):
    """회귀 + 분류 + 정합 손실."""

    def __init__(self, cls_weight: float = 0.5):
        super().__init__()
        self.cls_weight = cls_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, outputs: dict, targets: torch.Tensor) -> dict:
        # 회귀: 수익률 예측
        reg_loss = self.mse(outputs["pred_return"], targets)

        # 분류: trend_confidence 가 상승 확률에 가깝게
        labels_up = (targets > 0).float()
        cls_loss = self.bce(outputs["trend_confidence"], labels_up)

        # 정합 손실: confidence ↔ pred_return 상관
        conf = outputs["trend_confidence"]
        if len(conf) > 1:
            conf_centered = conf - conf.mean()
            target_centered = targets - targets.mean()
            cov = (conf_centered * target_centered).mean()
            conf_std = conf_centered.std() + 1e-6
            target_std = target_centered.std() + 1e-6
            corr = cov / (conf_std * target_std)
            align_loss = -corr
        else:
            align_loss = torch.tensor(0.0, device=targets.device)

        total = reg_loss + self.cls_weight * cls_loss + 0.1 * align_loss

        return {
            "total": total,
            "reg": reg_loss.detach(),
            "cls": cls_loss.detach(),
            "align": align_loss.detach() if isinstance(align_loss, torch.Tensor) else torch.tensor(0.0),
        }


if __name__ == "__main__":
    B, T = 8, 60
    chart_dim = 8

    model = ChartLSTM(chart_dim=chart_dim)
    print(f"파라미터: {sum(p.numel() for p in model.parameters()):,}")

    chart = torch.randn(B, T, chart_dim)
    out = model(chart)

    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}")

    targets = torch.randn(B) * 0.3
    loss_fn = ChartLSTMLoss()
    losses = loss_fn(out, targets)
    print(f"손실: total={losses['total'].item():.4f}")
