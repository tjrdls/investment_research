"""
학습 루프 (맥북 에어 M5 / MPS 최적화)
========================================
- Apple Silicon에서 MPS 가속 자동 활성화
- 시간순 분할 (룩어헤드 방지)
- IC (Information Coefficient) 모니터링
- 베스트 모델 자동 저장
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import CFG, MODEL_PATH, get_device
from src.model.lstm import MultiEncoderLSTM, MultiTaskLoss

logger = logging.getLogger(__name__)


class MultiInputDataset(Dataset):
    def __init__(self, fund, chart, market, y):
        self.fund = torch.from_numpy(fund).float()
        self.chart = torch.from_numpy(chart).float()
        self.market = torch.from_numpy(market).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.fund[idx], self.chart[idx], self.market[idx], self.y[idx]


class GroupScaler:
    """3개 그룹 각각 별도 mean/std (룩어헤드 방지)."""

    def __init__(self):
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, **groups: np.ndarray) -> "GroupScaler":
        for name, X in groups.items():
            flat = X.reshape(-1, X.shape[-1])
            mean = np.nanmean(flat, axis=0)
            std = np.nanstd(flat, axis=0) + 1e-6
            self.stats[name] = (mean, std)
        return self

    def transform(self, name: str, X: np.ndarray) -> np.ndarray:
        mean, std = self.stats[name]
        out = (X - mean) / std
        return np.nan_to_num(out, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)

    def to_dict(self) -> dict:
        return {f"{k}_mean": v[0] for k, v in self.stats.items()} | \
               {f"{k}_std": v[1] for k, v in self.stats.items()}


def time_split(meta: pd.DataFrame):
    dd = pd.to_datetime(meta["decision_date"])
    train_idx = np.where(dd <= CFG.train.train_end)[0]
    val_idx = np.where((dd > CFG.train.train_end) & (dd <= CFG.train.val_end))[0]
    test_idx = np.where(dd > CFG.train.val_end)[0]
    return train_idx, val_idx, test_idx


def train(dataset: dict, save_path: Path = MODEL_PATH, device: Optional[str] = None) -> dict:
    device = device or get_device()
    logger.info("디바이스: %s", device)
    if device == "mps":
        logger.info("✓ Apple Silicon MPS 가속 사용")

    meta = dataset["meta"]
    train_idx, val_idx, test_idx = time_split(meta)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"분할 부족 — train={len(train_idx)}, val={len(val_idx)}. "
            f"train_end/val_end 조정 필요."
        )
    logger.info("분할: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))

    scaler = GroupScaler().fit(
        fund=dataset["fund"][train_idx],
        chart=dataset["chart"][train_idx],
        market=dataset["market"][train_idx],
    )
    fund_s = scaler.transform("fund", dataset["fund"])
    chart_s = scaler.transform("chart", dataset["chart"])
    market_s = scaler.transform("market", dataset["market"])

    bs = CFG.train.batch_size
    train_ds = MultiInputDataset(fund_s[train_idx], chart_s[train_idx],
                                  market_s[train_idx], dataset["y"][train_idx])
    val_ds = MultiInputDataset(fund_s[val_idx], chart_s[val_idx],
                                market_s[val_idx], dataset["y"][val_idx])

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=bs)

    model = MultiEncoderLSTM().to(device)
    loss_fn = MultiTaskLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.train.lr,
        weight_decay=CFG.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.train.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("모델 파라미터: %s", f"{n_params:,}")

    history = {"train_loss": [], "val_loss": [], "val_ic": [], "val_dir_acc": []}
    best_ic = -float("inf")
    best_state = None

    for ep in range(1, CFG.train.epochs + 1):
        model.train()
        train_losses = []
        for fund, chart, market, y in train_dl:
            fund = fund.to(device); chart = chart.to(device)
            market = market.to(device); y = y.to(device)

            optimizer.zero_grad()
            out = model(fund, chart, market)
            losses = loss_fn(out, y)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.train.grad_clip)
            optimizer.step()
            train_losses.append(losses["total"].item())
        scheduler.step()

        model.eval()
        val_losses, preds, trues = [], [], []
        with torch.no_grad():
            for fund, chart, market, y in val_dl:
                fund = fund.to(device); chart = chart.to(device)
                market = market.to(device); y = y.to(device)
                out = model(fund, chart, market)
                losses = loss_fn(out, y)
                val_losses.append(losses["total"].item())
                preds.append(out["pred_return"].cpu().numpy())
                trues.append(y.cpu().numpy())

        p = np.concatenate(preds) if preds else np.array([])
        t = np.concatenate(trues) if trues else np.array([])

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        ic = float(pd.Series(p).rank().corr(pd.Series(t).rank())) if len(p) > 1 else 0.0
        dir_acc = float(((p > 0) == (t > 0)).mean()) if len(p) > 0 else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ic"].append(ic)
        history["val_dir_acc"].append(dir_acc)

        if ic > best_ic:
            best_ic = ic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        logger.info(
            "[Ep %02d] train=%.4f val=%.4f IC=%.4f dir_acc=%.3f lr=%.2e",
            ep, train_loss, val_loss, ic, dir_acc, scheduler.get_last_lr()[0],
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = {}
    if len(test_idx) > 0:
        test_ds = MultiInputDataset(fund_s[test_idx], chart_s[test_idx],
                                     market_s[test_idx], dataset["y"][test_idx])
        test_dl = DataLoader(test_ds, batch_size=bs)
        model.eval()
        preds, trues, scores = [], [], []
        with torch.no_grad():
            for fund, chart, market, y in test_dl:
                fund = fund.to(device); chart = chart.to(device); market = market.to(device)
                out = model(fund, chart, market)
                preds.append(out["pred_return"].cpu().numpy())
                scores.append(out["final_score"].cpu().numpy())
                trues.append(y.numpy())
        p = np.concatenate(preds); t = np.concatenate(trues); s = np.concatenate(scores)
        test_metrics = {
            "ic_pred": float(pd.Series(p).rank().corr(pd.Series(t).rank())),
            "ic_score": float(pd.Series(s).rank().corr(pd.Series(t).rank())),
            "dir_acc": float(((p > 0) == (t > 0)).mean()),
            "top10pct_avg_return": float(t[s >= np.quantile(s, 0.9)].mean()) if len(s) >= 10 else 0.0,
        }
        logger.info("[Test] %s", test_metrics)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "scaler_stats": scaler.to_dict(),
        "config_snapshot": {
            "weights": [CFG.weights.fundamental, CFG.weights.chart, CFG.weights.market],
            "seq_len": CFG.model.seq_len,
            "horizon": CFG.model.horizon,
            "fundamental_dim": CFG.model.fundamental_dim,
            "chart_dim": CFG.model.chart_dim,
            "market_dim": CFG.model.market_dim,
        },
        "best_val_ic": best_ic,
        "test_metrics": test_metrics,
    }, save_path)
    logger.info("✓ 모델 저장: %s", save_path)

    return {
        "history": history,
        "best_val_ic": best_ic,
        "test_metrics": test_metrics,
        "scaler": scaler,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from src.data.feature_engineer import FeatureEngineer

    print(">>> 데이터셋 생성")
    fe = FeatureEngineer()
    ds = fe.build_dataset(start="2015-01-01")

    print(">>> 학습 시작")
    result = train(ds)
    print(f"베스트 IC: {result['best_val_ic']:.4f}")
    print(f"테스트: {result['test_metrics']}")
