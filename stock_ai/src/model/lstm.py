"""
3-인코더 LSTM 모델 (★ 시스템의 심장)
=========================================
6:2:2 가중치를 모델 구조에 정확히 반영.

  ┌─ 펀더멘털 인코더 (BiLSTM + Attention) ─→ 펀더멘털 점수 ─┐
  ├─ 차트 인코더    (BiLSTM + Attention) ─→ 차트 점수    ─┤→ 가중합
  └─ 시장 인코더    (BiLSTM + Attention) ─→ 시장 점수    ─┘    ↓
                                                           최종 점수 (0~100)
                                                                ↓
                                              회귀 헤드 → 6개월 예측 수익률
                                              분류 헤드 → 상승/하락 확률
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CFG


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


class Encoder(nn.Module):
    """피처 그룹 → context vector + 점수(0~1)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, bidirectional: bool):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = TemporalAttention(out_dim)
        self.score_head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        context, attn = self.attention(out)
        score = self.score_head(context).squeeze(-1)
        return context, score, attn


class MultiEncoderLSTM(nn.Module):
    """펀더멘털(60%) + 차트(20%) + 시장(20%) → 최종 추천 점수."""

    def __init__(
        self,
        fundamental_dim: int = None,
        chart_dim: int = None,
        market_dim: int = None,
        hidden_dim: int = None,
        num_layers: int = None,
        dropout: float = None,
        bidirectional: bool = None,
        fixed_weights: bool = True,
    ):
        super().__init__()
        m = CFG.model
        fundamental_dim = fundamental_dim or m.fundamental_dim
        chart_dim = chart_dim or m.chart_dim
        market_dim = market_dim or m.market_dim
        hidden_dim = hidden_dim or m.hidden_dim
        num_layers = num_layers or m.num_layers
        dropout = dropout if dropout is not None else m.dropout
        bidirectional = bidirectional if bidirectional is not None else m.bidirectional

        self.enc_fund = Encoder(fundamental_dim, hidden_dim, num_layers, dropout, bidirectional)
        self.enc_chart = Encoder(chart_dim, hidden_dim, num_layers, dropout, bidirectional)
        self.enc_market = Encoder(market_dim, hidden_dim, num_layers, dropout, bidirectional)

        self.fixed_weights = fixed_weights
        if fixed_weights:
            w = CFG.weights.normalize()
            self.register_buffer(
                "weights",
                torch.tensor([w.fundamental, w.chart, w.market], dtype=torch.float32),
            )
        else:
            init = CFG.weights.normalize()
            self.weight_logits = nn.Parameter(torch.tensor(
                [init.fundamental, init.chart, init.market], dtype=torch.float32,
            ))

        combined_dim = self.enc_fund.out_dim * 3
        self.regressor = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def get_weights(self) -> torch.Tensor:
        if self.fixed_weights:
            return self.weights
        return F.softmax(self.weight_logits, dim=0)

    def forward(self, fund: torch.Tensor, chart: torch.Tensor, market: torch.Tensor) -> dict:
        ctx_f, score_f, attn_f = self.enc_fund(fund)
        ctx_c, score_c, attn_c = self.enc_chart(chart)
        ctx_m, score_m, attn_m = self.enc_market(market)

        weights = self.get_weights()
        final_score = (
            weights[0] * score_f
            + weights[1] * score_c
            + weights[2] * score_m
        )

        combined = torch.cat([ctx_f, ctx_c, ctx_m], dim=-1)
        pred_return = self.regressor(combined).squeeze(-1)
        up_logit = self.classifier(combined).squeeze(-1)

        return {
            "final_score": final_score,
            "fund_score": score_f,
            "chart_score": score_c,
            "market_score": score_m,
            "pred_return": pred_return,
            "up_logit": up_logit,
            "attn_fund": attn_f,
            "attn_chart": attn_c,
            "attn_market": attn_m,
            "weights": weights,
        }


class MultiTaskLoss(nn.Module):
    """회귀 + 분류 + 점수-수익률 정합 손실."""

    def __init__(self, cls_weight: float = None):
        super().__init__()
        self.cls_weight = cls_weight if cls_weight is not None else CFG.model.cls_loss_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs: dict, targets: torch.Tensor) -> dict:
        reg_loss = self.mse(outputs["pred_return"], targets)
        labels_up = (targets > 0).float()
        cls_loss = self.bce(outputs["up_logit"], labels_up)

        # 정합 손실
        score_norm = outputs["final_score"]
        if len(score_norm) > 1:
            score_centered = score_norm - score_norm.mean()
            target_centered = targets - targets.mean()
            cov = (score_centered * target_centered).mean()
            score_std = score_centered.std() + 1e-6
            target_std = target_centered.std() + 1e-6
            corr = cov / (score_std * target_std)
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
    B, T = 4, 60
    model = MultiEncoderLSTM()

    fund = torch.randn(B, T, CFG.model.fundamental_dim)
    chart = torch.randn(B, T, CFG.model.chart_dim)
    market = torch.randn(B, T, CFG.model.market_dim)

    out = model(fund, chart, market)

    print("=" * 50)
    print("MultiEncoderLSTM 검증")
    print("=" * 50)
    print(f"파라미터: {sum(p.numel() for p in model.parameters()):,}")
    print(f"가중치 (6:2:2): {out['weights'].tolist()}")
    print(f"\n출력:")
    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}")

    targets = torch.randn(B) * 0.3
    loss_fn = MultiTaskLoss()
    losses = loss_fn(out, targets)
    print(f"\n손실: total={losses['total'].item():.4f}")
