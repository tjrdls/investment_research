# -*- coding: utf-8 -*-
"""
역할: LSTM 모델 학습 스크립트
여러 종목의 데이터를 수집하여 멀티모달 LSTM 모델을 학습한다.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(__file__))

from data.price.data_collector import collect_price_data
from data.financial.financial_collector import collect_financial_data, get_corp_code_map
from analysis.indicators.technical_indicators import calculate_indicators
from models.lstm.lstm_model import (
    MultimodalStockPredictor,
    prepare_dataset,
    train_model,
    StockDataset
)
from pipeline.prediction_pipeline import get_text_embedding
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_multimodal_stock_model.pth"
SEQ_LEN = 20
PRED_DAYS = 5
THRESHOLD = 0.02


def collect_training_data():
    """
    여러 종목의 학습용 데이터 수집
    
    :return: dict {ticker: indicators_df}, dict {ticker: text_embedding}
    """
    
    # 학습용 종목 (상위 10개)
    train_stocks = [
        ("005930", "삼성전자"),
        ("000660", "SK하이닉스"),
        ("373220", "LG에너지솔루션"),
        ("207940", "삼성바이오로직스"),
        ("005380", "현대차"),
        ("051910", "LG화학"),
        ("035420", "NAVER"),
        ("028260", "삼성물산"),
        ("000270", "기아"),
        ("055550", "신한지주")
    ]
    
    price_data = {}
    text_embeddings = {}
    corp_map = get_corp_code_map()
    
    print("\n" + "="*60)
    print("  📊 LSTM 학습용 데이터 수집")
    print("="*60)
    
    for stock_code, stock_name in train_stocks:
        print("\n  [{}/{}] {} ({})".format(
            train_stocks.index((stock_code, stock_name)) + 1,
            len(train_stocks),
            stock_name,
            stock_code
        ))
        
        try:
            # 1. 주가 데이터 수집
            print("    📈 주가 데이터 수집...")
            price_df = collect_price_data(stock_code, period="3y")
            
            if price_df.empty or len(price_df) < SEQ_LEN + PRED_DAYS:
                print("    ⚠️  데이터 부족 (최소 {} 거래일 필요)".format(SEQ_LEN + PRED_DAYS))
                continue
            
            # 2. 기술적 지표 계산
            print("    📊 기술적 지표 계산...")
            indicators_df = calculate_indicators(price_df)
            
            if indicators_df.empty:
                print("    ⚠️  지표 계산 실패")
                continue
            
            price_data[stock_code] = indicators_df
            print("    ✅ {} 거래일 확보".format(len(indicators_df)))
            
            # 3. 텍스트 임베딩 생성 (회사명 기본)
            text_emb = get_text_embedding(stock_name)
            text_embeddings[stock_code] = text_emb
            print("    ✅ 텍스트 임베딩 생성")
            
        except Exception as e:
            print("    ❌ 오류: {}".format(str(e)))
            continue
    
    print("\n  📦 총 {} 종목 데이터 확보".format(len(price_data)))
    return price_data, text_embeddings


def train_lstm():
    """
    LSTM 모델 학습
    """
    
    print("\n" + "="*60)
    print("  🤖 LSTM 모델 학습 시작")
    print("="*60)
    
    # 1. 데이터 수집
    price_data, text_embeddings = collect_training_data()
    
    if len(price_data) < 2:
        print("\n  ❌ 학습용 데이터 부족 (최소 2개 종목 필요)")
        return False
    
    # 2. 데이터셋 준비
    print("\n  📦 데이터셋 준비 중...")
    try:
        X_seq, X_txt, Y, scalers = prepare_dataset(
            price_data,
            text_embeddings,
            seq_len=SEQ_LEN,
            pred_days=PRED_DAYS,
            threshold=THRESHOLD
        )
        print("  ✅ 학습 샘플: {}".format(len(X_seq)))
    except Exception as e:
        print("  ❌ 데이터셋 준비 실패: {}".format(str(e)))
        return False
    
    # 3. 학습/검증 데이터 분할
    split_idx = int(len(X_seq) * 0.8)
    X_train_seq, X_val_seq = X_seq[:split_idx], X_seq[split_idx:]
    X_train_txt, X_val_txt = X_txt[:split_idx], X_txt[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    
    train_dataset = StockDataset(X_train_seq, X_train_txt, Y_train)
    val_dataset = StockDataset(X_val_seq, X_val_txt, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("  ├─ 학습 샘플: {}".format(len(train_dataset)))
    print("  └─ 검증 샘플: {}".format(len(val_dataset)))
    
    # 4. 모델 생성
    print("\n  🏗️  모델 생성 중...")
    model = MultimodalStockPredictor().to(DEVICE)
    print("  ✅ 모델 생성 완료 (Device: {})".format(DEVICE))
    print("  └─ 파라미터: {:,}개".format(sum(p.numel() for p in model.parameters())))
    
    # 5. 모델 학습
    print("\n  🎓 모델 학습 중...")
    model, history = train_model(
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001,
        patience=10
    )
    
    # 6. 모델 저장
    print("\n  💾 모델 저장 중...")
    torch.save(model.state_dict(), MODEL_PATH)
    print("  ✅ 모델 저장 완료: {}".format(MODEL_PATH))
    
    # 7. 결과 요약
    print("\n" + "="*60)
    print("  ✅ 학습 완료!")
    print("="*60)
    print("  📊 최종 성능:")
    if history["val_acc"]:
        print("  ├─ 최고 검증 정확도: {:.2%}".format(max(history["val_acc"])))
        print("  └─ 최저 검증 손실: {:.4f}".format(min(history["val_loss"])))
    
    return True


if __name__ == "__main__":
    try:
        success = train_lstm()
        if success:
            print("\n  💡 이제 main.py를 실행하면 학습된 모델을 사용합니다.")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n  ⚠️  학습 중단됨")
        sys.exit(1)
    except Exception as e:
        print("\n  ❌ 오류 발생: {}".format(str(e)))
        import traceback
        traceback.print_exc()
        sys.exit(1)
