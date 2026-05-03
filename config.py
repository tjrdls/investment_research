# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple

DEFAULT_STOCKS: List[Tuple[str, str]] = [
    ("005930", "삼성전자"),
    ("000660", "SK하이닉스"),
    ("373220", "LG에너지솔루션"),
    ("207940", "삼성바이오로직스"),
    ("005380", "현대차"),
]

STOCK_CODE_TO_NAME: Dict[str, str] = dict(DEFAULT_STOCKS)

PERIOD_DAYS: Dict[str, int] = {
    "3y": 1095,
    "1y": 365,
    "6m": 180,
}

PERIOD_OPTIONS: Dict[str, str] = {
    "3년": "3y",
    "1년": "1y",
    "6개월": "6m",
}

# LSTM hyperparameters
LSTM_NUM_FEAT: int = 11
LSTM_HIDDEN: int = 128
LSTM_TEXT_DIM: int = 1536
LSTM_PROJ_DIM: int = 64
LSTM_N_LAYERS: int = 2
LSTM_DROPOUT: float = 0.3
LSTM_SEQ_LEN: int = 20
LSTM_PRED_DAYS: int = 5
LSTM_THRESHOLD: float = 0.02

# OpenAI
GPT_MODEL: str = "gpt-4o-mini"
EMBEDDING_MODEL: str = "text-embedding-ada-002"
EMBEDDING_DIM: int = 1536
GPT_TEMPERATURE: float = 0.1
GPT_MAX_TOKENS_NEWS: int = 500
GPT_MAX_TOKENS_ANALYSIS: int = 800
GPT_MAX_TOKENS_VALUATION: int = 300

# File paths
MODEL_PATH: str = "best_multimodal_stock_model.pth"
CACHE_PATH: str = "text_embeddings_cache.pkl"
DATA_DIR: str = "data/downloads"
