# -*- coding: utf-8 -*-
"""
역할: LSTM 모델 학습 스크립트
여러 종목의 데이터를 수집하여 멀티모달 LSTM 모델을 학습한다.
"""

import logging
import sys
import os

import torch
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from utils.logger import configure_root_logger
configure_root_logger()

from config import MODEL_PATH, LSTM_SEQ_LEN, LSTM_PRED_DAYS, LSTM_THRESHOLD
from data_loader.price.data_collector import collect_price_data
from data_loader.financial.financial_collector import get_corp_code_map
from analysis.indicators.technical_indicators import calculate_indicators
from models.lstm.lstm_model import MultimodalStockPredictor, prepare_dataset, train_model, StockDataset
from pipeline.prediction_pipeline import get_text_embedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

# 학습용 종목 (상위 10개)
TRAIN_STOCKS = [
    ("005930", "삼성전자"),
    ("000660", "SK하이닉스"),
    ("373220", "LG에너지솔루션"),
    ("207940", "삼성바이오로직스"),
    ("005380", "현대차"),
    ("051910", "LG화학"),
    ("035420", "NAVER"),
    ("028260", "삼성물산"),
    ("000270", "기아"),
    ("055550", "신한지주"),
]


def collect_training_data() -> tuple:
    """
    여러 종목의 학습용 데이터 수집.

    :return: (price_data dict, text_embeddings dict)
    """
    price_data: dict = {}
    text_embeddings: dict = {}
    get_corp_code_map()  # 기업코드 캐시 warming

    logger.info("=" * 60)
    logger.info("📊 LSTM 학습용 데이터 수집")
    logger.info("=" * 60)

    for idx, (stock_code, stock_name) in enumerate(TRAIN_STOCKS, 1):
        logger.info("[%d/%d] %s (%s)", idx, len(TRAIN_STOCKS), stock_name, stock_code)
        try:
            price_df = collect_price_data(stock_code, period="3y")
            if price_df.empty or len(price_df) < LSTM_SEQ_LEN + LSTM_PRED_DAYS:
                logger.warning("데이터 부족 (최소 %d 거래일 필요)", LSTM_SEQ_LEN + LSTM_PRED_DAYS)
                continue

            indicators_df = calculate_indicators(price_df)
            if indicators_df.empty:
                logger.warning("지표 계산 실패: %s", stock_code)
                continue

            price_data[stock_code] = indicators_df
            text_embeddings[stock_code] = get_text_embedding(stock_name)
            logger.info("✅ %d 거래일 확보", len(indicators_df))

        except Exception as e:
            logger.error("❌ 오류 [%s]: %s", stock_code, e)

    logger.info("📦 총 %d 종목 데이터 확보", len(price_data))
    return price_data, text_embeddings


def train_lstm() -> bool:
    """LSTM 모델 학습."""
    logger.info("=" * 60)
    logger.info("🤖 LSTM 모델 학습 시작")
    logger.info("=" * 60)

    price_data, text_embeddings = collect_training_data()

    if len(price_data) < 2:
        logger.error("학습용 데이터 부족 (최소 2개 종목 필요)")
        return False

    logger.info("📦 데이터셋 준비 중...")
    try:
        X_seq, X_txt, Y, _ = prepare_dataset(
            price_data, text_embeddings,
            seq_len=LSTM_SEQ_LEN, pred_days=LSTM_PRED_DAYS, threshold=LSTM_THRESHOLD,
        )
        logger.info("✅ 학습 샘플: %d", len(X_seq))
    except Exception as e:
        logger.error("데이터셋 준비 실패: %s", e)
        return False

    split_idx = int(len(X_seq) * 0.8)
    train_loader = DataLoader(
        StockDataset(X_seq[:split_idx], X_txt[:split_idx], Y[:split_idx]),
        batch_size=32, shuffle=True,
    )
    val_loader = DataLoader(
        StockDataset(X_seq[split_idx:], X_txt[split_idx:], Y[split_idx:]),
        batch_size=32, shuffle=False,
    )
    logger.info("학습 샘플: %d  검증 샘플: %d", len(train_loader.dataset), len(val_loader.dataset))

    model = MultimodalStockPredictor().to(DEVICE)
    logger.info("모델 생성 완료 (Device: %s, 파라미터: %s개)",
                DEVICE, f"{sum(p.numel() for p in model.parameters()):,}")

    model, history = train_model(train_loader, val_loader, epochs=50, lr=0.001, patience=10)

    torch.save(model.state_dict(), MODEL_PATH)
    logger.info("✅ 모델 저장 완료: %s", MODEL_PATH)

    if history["val_acc"]:
        logger.info("최고 검증 정확도: %.2f%%  최저 검증 손실: %.4f",
                    max(history["val_acc"]) * 100, min(history["val_loss"]))

    return True


if __name__ == "__main__":
    try:
        success = train_lstm()
        if success:
            logger.info("💡 이제 main.py를 실행하면 학습된 모델을 사용합니다.")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("학습 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error("오류 발생: %s", e, exc_info=True)
        sys.exit(1)
