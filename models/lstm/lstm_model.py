# -*- coding: utf-8 -*-
"""
역할: 기술적 지표 시퀀스만 사용하는 LSTM 모델 (Late Fusion 설계).

Late Fusion 구조에서 LSTM은 기술지표 패턴 신호만 담당한다.
재무 텍스트 임베딩은 LLM 입력으로 직접 전달되며, LSTM 내부에서 결합하지 않는다.

  LSTM(기술지표 시퀀스) → 상승/하락/횡보 확률 + confidence
  재무 텍스트 임베딩   → (LLM 단계에서 별도로 활용)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from openai import OpenAI
from dotenv import load_dotenv

from config import (
    LSTM_NUM_FEAT, LSTM_HIDDEN, LSTM_TEXT_DIM, LSTM_PROJ_DIM,
    LSTM_N_LAYERS, LSTM_DROPOUT, LSTM_SEQ_LEN, LSTM_PRED_DAYS, LSTM_THRESHOLD,
)

load_dotenv()

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockPredictor(nn.Module):
    """
    기술적 지표 시퀀스 → 상승/하락/횡보 3분류 LSTM 모델.

    Late Fusion 설계에 따라 텍스트 branch를 제거하고
    순수 기술지표 시계열만 입력으로 사용한다.
    """

    def __init__(
        self,
        num_feat: int = LSTM_NUM_FEAT,
        hidden: int = LSTM_HIDDEN,
        n_cls: int = 3,
        n_layers: int = LSTM_N_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_feat,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.lstm_drop = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, n_cls),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x_seq)
        return self.classifier(self.lstm_drop(out[:, -1, :]))


# 하위 호환성 alias (기존 코드에서 MultimodalStockPredictor를 참조하는 경우)
MultimodalStockPredictor = StockPredictor


class StockDataset(Dataset):
    def __init__(self, seqs: np.ndarray, labels: np.ndarray):
        self.seqs = torch.from_numpy(seqs)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.seqs[idx], self.labels[idx]


def prepare_dataset(
    price_data: dict,
    seq_len: int = LSTM_SEQ_LEN,
    pred_days: int = LSTM_PRED_DAYS,
    threshold: float = LSTM_THRESHOLD,
) -> tuple:
    """
    학습용 데이터셋 준비 (기술지표 시퀀스만 사용).

    :return: (X_seq, Y, scalers)
    """
    feature_cols = ["open", "high", "low", "close", "volume",
                    "sma_20", "rsi", "bb_pct_b", "macd", "macd_hist", "volatility"]

    X_seq, Y = [], []
    scalers: dict = {}

    for ticker, df in price_data.items():
        if len(df) < seq_len + pred_days:
            continue

        available_cols = [c for c in feature_cols if c in df.columns]
        if not available_cols:
            continue

        feat = df[available_cols].values.astype(np.float32)
        close = df["close"].values.astype(np.float32)

        scaler = StandardScaler().fit(feat)
        scalers[ticker] = scaler
        feat_scaled = scaler.transform(feat)

        labels = []
        for i in range(len(close) - pred_days):
            ret = (close[i + pred_days] - close[i]) / (close[i] + 1e-8)
            if ret > threshold:
                labels.append(0)
            elif ret < -threshold:
                labels.append(1)
            else:
                labels.append(2)

        labels_arr = np.array(labels, dtype=np.int64)

        for i in range(seq_len, len(feat_scaled) - pred_days):
            label_idx = i - seq_len
            if label_idx >= len(labels_arr):
                break
            X_seq.append(feat_scaled[i - seq_len: i])
            Y.append(labels_arr[label_idx])

    if not X_seq:
        raise ValueError("No valid data sequences generated")

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(Y, dtype=np.int64),
        scalers,
    )


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.001,
    patience: int = 20,
    class_weights: Optional[torch.Tensor] = None,
) -> tuple:
    """
    모델 학습.

    :return: (trained_model, history)
    """
    model = StockPredictor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
<<<<<<< HEAD

    # 클래스 불균형 보정
    labels_arr = train_loader.dataset.labels.numpy()
    n_total = len(labels_arr)
    cls_cnt = np.bincount(labels_arr, minlength=3).astype(np.float32)
    cls_cnt = np.where(cls_cnt == 0, 1.0, cls_cnt)
    cls_weights = torch.FloatTensor((1.0 / cls_cnt) * n_total / 3).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cls_weights)
    logger.info("클래스 가중치: 상승=%.3f 하락=%.3f 횡보=%.3f", cls_weights[0], cls_weights[1], cls_weights[2])
=======
    weights = class_weights.to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights)
>>>>>>> 803219d (train변경)

    history = {"tr_loss": [], "val_loss": [], "tr_acc": [], "val_acc": []}
    es_counter = 0
    best_val = 0.0
    best_state = None

    logger.info("모델 학습 시작 (%d epochs)...", epochs)

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for seq, lbl in train_loader:
            seq, lbl = seq.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            logit = model(seq)
            loss = criterion(logit, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
            tr_correct += (logit.argmax(1) == lbl).sum().item()
            tr_total += len(lbl)

        tr_loss /= max(len(train_loader), 1)
        tr_acc = tr_correct / max(tr_total, 1)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for seq, lbl in val_loader:
                seq, lbl = seq.to(DEVICE), lbl.to(DEVICE)
                logit = model(seq)
                loss = criterion(logit, lbl)
                val_loss += loss.item()
                val_correct += (logit.argmax(1) == lbl).sum().item()
                val_total += len(lbl)

        val_loss /= max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)

        history["tr_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["tr_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()
            es_counter = 0
        else:
            es_counter += 1

        if (ep % 10 == 0 or es_counter >= patience) and ep > 1:
            logger.info("Epoch %3d: tr_loss=%.4f tr_acc=%.3f | val_loss=%.4f val_acc=%.3f",
                        ep, tr_loss, tr_acc, val_loss, val_acc)

        if es_counter >= patience:
            logger.info("Early stopping at epoch %d", ep)
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(model: StockPredictor, loader: DataLoader) -> dict:
    """
    Test set 평가. Accuracy + 클래스별 F1/Precision/Recall 반환.

    :return: {"accuracy": float, "report": str}
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seq, lbl in loader:
            seq, lbl = seq.to(DEVICE), lbl.to(DEVICE)
            preds = model(seq).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = float((all_preds == all_labels).mean())
    report = classification_report(
        all_labels, all_preds,
        target_names=["상승", "하락", "횡보"],
        zero_division=0,
    )
    return {"accuracy": accuracy, "report": report}


def predict_next_trend(
    model: Optional[StockPredictor],
    indicators_df: pd.DataFrame,
    seq_len: int = LSTM_SEQ_LEN,
    scalers: Optional[dict] = None,
    stock_code: Optional[str] = None,
) -> dict:
    """
    다음 추세 예측 (기술지표 시퀀스만 사용).

    :return: dict with prediction, probabilities, confidence
    """
    if len(indicators_df) < seq_len:
        return {"prediction": "데이터 부족", "probabilities": {}}

    feature_cols = ["open", "high", "low", "close", "volume",
                    "sma_20", "rsi", "bb_pct_b", "macd", "macd_hist", "volatility"]

    available_cols = [c for c in feature_cols if c in indicators_df.columns]
    feat = indicators_df[available_cols].values.astype(np.float32)

    if scalers and stock_code and stock_code in scalers:
        scaler = scalers[stock_code]
    else:
        scaler = StandardScaler().fit(feat)

    feat_scaled = scaler.transform(feat)
    x_seq = torch.FloatTensor(feat_scaled[-seq_len:]).unsqueeze(0).to(DEVICE)

    if model is None:
        logger.warning("모델 미설정: 기본 예측을 사용합니다.")
        return {
            "prediction": "모델 없음",
            "probabilities": {"상승": 0.33, "하락": 0.33, "횡보": 0.34},
            "confidence": 0.33,
        }

    model.eval()
    with torch.no_grad():
        logits = model(x_seq)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    class_names = ["📈 상승", "📉 하락", "➡️  횡보"]
    pred_class = int(np.argmax(probs))

    return {
        "prediction": class_names[pred_class],
        "probabilities": {"상승": float(probs[0]), "하락": float(probs[1]), "횡보": float(probs[2])},
        "confidence": float(probs[pred_class]),
    }


if __name__ == "__main__":
    logger.info("LSTM 모듈 로드 완료 (Late Fusion 구조)")
