# -*- coding: utf-8 -*-
"""
역할: LSTM과 텍스트 임베딩을 결합한 멀티모달 모델.
기술적 지표 시퀀스와 DART 재무 정보 임베딩을 함께 사용하여 
상승/하락/횡보를 3분류 예측한다.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultimodalStockPredictor(nn.Module):
    """
    멀티모달 주식 예측 모델
    
    LSTM Branch: 기술적 지표 시퀀스 (seq_len × num_features)
                → LSTM 레이어 → hidden state
    
    Text Branch: DART 재무 정보 임베딩 (1536-d)
               → Dense layers → 64-d feature
    
    Fusion: LSTM output + Text feature concat
          → Dense layers → [상승, 하락, 횡보] 3분류
    """
    
    def __init__(self, num_feat=11, hidden=128, text_in=1536, proj=64,
                 n_cls=3, n_layers=2, dropout=0.3):
        super(MultimodalStockPredictor, self).__init__()
        
        # LSTM Branch
        self.lstm = nn.LSTM(
            input_size=num_feat,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.lstm_drop = nn.Dropout(dropout)
        
        # Text Branch (1536-d → 512 → 256 → 64)
        self.text_proj = nn.Sequential(
            nn.Linear(text_in, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, proj),
            nn.GELU()
        )
        
        # Fusion & Output (128 + 64 → 128 → 64 → 3)
        self.fusion = nn.Sequential(
            nn.Linear(hidden + proj, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, n_cls)
        )
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
    
    def forward(self, x_seq, x_text):
        """
        :param x_seq: (batch, seq_len, num_feat)
        :param x_text: (batch, 1536)
        :return: (batch, 3) logits
        """
        # LSTM Branch
        out, _ = self.lstm(x_seq)
        lstm_f = self.lstm_drop(out[:, -1, :])  # 마지막 타임스텝 (batch, hidden)
        
        # Text Branch
        text_f = self.text_proj(x_text)  # (batch, 64)
        
        # Fusion
        fused = torch.cat([lstm_f, text_f], dim=1)  # (batch, 128+64)
        return self.fusion(fused)


class StockDataset(Dataset):
    """PyTorch Dataset for stock prediction"""
    
    def __init__(self, seqs, texts, labels):
        self.seqs = torch.from_numpy(seqs)
        self.texts = torch.from_numpy(texts)
        self.labels = torch.from_numpy(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.seqs[idx], self.texts[idx], self.labels[idx]


def prepare_dataset(price_data, text_embeddings, seq_len=20, pred_days=5, threshold=0.02):
    """
    학습용 데이터셋 준비
    
    :param price_data: dict {ticker: DataFrame with indicators}
    :param text_embeddings: dict {ticker: np.array (1536-d)}
    :param seq_len: LSTM input sequence length
    :param pred_days: 예측 기간 (거래일)
    :param threshold: 상승/하락 임계값
    :return: (X_seq, X_txt, Y, scalers)
    """
    
    feature_cols = [
        "open", "high", "low", "close", "volume",
        "sma_20", "rsi", "bb_pct_b", "macd", "macd_hist", "volatility"
    ]
    
    X_seq, X_txt, Y = [], [], []
    scalers = {}
    
    for ticker, df in price_data.items():
        if ticker not in text_embeddings or len(df) < seq_len + pred_days:
            continue
        
        # 컬럼 확인 및 추출
        available_cols = [c for c in feature_cols if c in df.columns]
        if not available_cols:
            continue
        
        feat = df[available_cols].values.astype(np.float32)
        close = df["close"].values.astype(np.float32)
        emb = text_embeddings[ticker]
        
        # Normalization
        scaler = StandardScaler().fit(feat)
        scalers[ticker] = scaler
        feat_scaled = scaler.transform(feat)
        
        # Label 생성 (상승/하락/횡보)
        labels = []
        for i in range(len(close) - pred_days):
            ret = (close[i + pred_days] - close[i]) / (close[i] + 1e-8)
            if ret > threshold:
                labels.append(0)  # 상승
            elif ret < -threshold:
                labels.append(1)  # 하락
            else:
                labels.append(2)  # 횡보
        
        labels = np.array(labels, dtype=np.int64)
        
        # Sequence 생성
        for i in range(seq_len, len(feat_scaled) - pred_days):
            label_idx = i - seq_len
            if label_idx >= len(labels):
                break
            
            X_seq.append(feat_scaled[i - seq_len : i])
            X_txt.append(emb)
            Y.append(labels[label_idx])
    
    if not X_seq:
        raise ValueError("No valid data sequences generated")
    
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_txt, dtype=np.float32),
        np.array(Y, dtype=np.int64),
        scalers
    )


def train_model(train_loader, val_loader, epochs=100, lr=0.001, patience=20):
    """
    모델 학습
    
    :param train_loader: DataLoader
    :param val_loader: DataLoader
    :param epochs: 에포크 수
    :param lr: 학습률
    :param patience: Early stopping patience
    :return: (trained_model, history)
    """
    
    model = MultimodalStockPredictor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    
    criterion = nn.CrossEntropyLoss()
    
    history = {"tr_loss": [], "val_loss": [], "tr_acc": [], "val_acc": []}
    es_counter = 0
    best_val = 0.0
    best_state = None
    
    print("   🤖 모델 학습 시작 ({} epochs)...".format(epochs))
    
    for ep in range(1, epochs + 1):
        # Training
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        
        for seq, txt, lbl in train_loader:
            seq, txt, lbl = seq.to(DEVICE), txt.to(DEVICE), lbl.to(DEVICE)
            
            optimizer.zero_grad()
            logit = model(seq, txt)
            loss = criterion(logit, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            tr_loss += loss.item()
            tr_correct += (logit.argmax(1) == lbl).sum().item()
            tr_total += len(lbl)
        
        tr_loss = tr_loss / max(len(train_loader), 1)
        tr_acc = tr_correct / max(tr_total, 1)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for seq, txt, lbl in val_loader:
                seq, txt, lbl = seq.to(DEVICE), txt.to(DEVICE), lbl.to(DEVICE)
                logit = model(seq, txt)
                loss = criterion(logit, lbl)
                
                val_loss += loss.item()
                val_correct += (logit.argmax(1) == lbl).sum().item()
                val_total += len(lbl)
        
        val_loss = val_loss / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)
        
        history["tr_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["tr_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        
        scheduler.step()
        
        # Early stopping
        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()
            es_counter = 0
        else:
            es_counter += 1
        
        if (ep % 10 == 0 or es_counter >= patience) and ep > 1:
            print("    Epoch {:3d}: tr_loss={:.4f} tr_acc={:.3f} | val_loss={:.4f} val_acc={:.3f}".format(
                ep, tr_loss, tr_acc, val_loss, val_acc
            ))
        
        if es_counter >= patience:
            print("    Early stopping at epoch {}".format(ep))
            break
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    return model, history


def predict_next_trend(model, indicators_df, text_embedding, seq_len=20, scalers=None, stock_code=None):
    """
    다음 추세 예측
    
    :param model: trained model
    :param indicators_df: DataFrame with technical indicators
    :param text_embedding: np.array (1536-d)
    :param seq_len: sequence length
    :param scalers: dict of scalers {ticker: scaler}
    :param stock_code: stock code for scaler lookup
    :return: dict with prediction results
    """
    
    if len(indicators_df) < seq_len:
        return {"prediction": "데이터 부족", "probabilities": {}}
    
    feature_cols = [
        "open", "high", "low", "close", "volume",
        "sma_20", "rsi", "bb_pct_b", "macd", "macd_hist", "volatility"
    ]
    
    available_cols = [c for c in feature_cols if c in indicators_df.columns]
    feat = indicators_df[available_cols].values.astype(np.float32)
    
    # 정규화
    if scalers and stock_code and stock_code in scalers:
        scaler = scalers[stock_code]
    else:
        scaler = StandardScaler().fit(feat)
    
    feat_scaled = scaler.transform(feat)
    
    # 최근 시퀀스
    x_seq = torch.FloatTensor(feat_scaled[-seq_len:]).unsqueeze(0).to(DEVICE)
    x_text = torch.FloatTensor(text_embedding).unsqueeze(0).to(DEVICE)

    if model is None:
        print("    [경고] 모델 미설정: 기본 예측을 사용합니다.")
        return {
            "prediction": "모델 없음",
            "probabilities": {
                "상승": 0.33,
                "하락": 0.33,
                "횡보": 0.34
            },
            "confidence": 0.33
        }

    model.eval()
    with torch.no_grad():
        logits = model(x_seq, x_text)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    class_names = ["📈 상승", "📉 하락", "➡️  횡보"]
    pred_class = int(np.argmax(probs))
    
    return {
        "prediction": class_names[pred_class],
        "probabilities": {
            "상승": float(probs[0]),
            "하락": float(probs[1]),
            "횡보": float(probs[2])
        },
        "confidence": float(probs[pred_class])
    }


if __name__ == "__main__":
    print("LSTM 모듈 로드 완료")

    print("LLM Analysis:", analysis)