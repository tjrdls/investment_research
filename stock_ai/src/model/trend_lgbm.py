"""2달 주기 추세 예측 — LightGBM Trend Classifier.

ChartLSTM 대체. 정예 6 피처 + 42거래일 라벨 + 시간순 분할 (42일 갭).
KR + US 통합 학습, 평가는 KR/US 분리.

사용:
    from src.model.trend_lgbm import build_dataset, train, evaluate
    df = build_dataset()
    model, info = train(df)
    evaluate(model, df, info)
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DB_PATH, MODEL_DIR

logger = logging.getLogger(__name__)

# ── 분할 기준 (사용자 명시) ─────────────────────────────
TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"
VAL_START   = "2022-03-01"  # ≈ TRAIN_END + 42 거래일 갭
VAL_END     = "2023-12-31"
TEST_START  = "2024-03-01"  # ≈ VAL_END + 42 거래일 갭
TEST_END    = "2026-05-31"
TARGET_HORIZON = 42  # 거래일

FEATURE_COLS = [
    "price_to_ma200",
    "ma60_slope",
    "ma20_to_ma60",
    "bb_position",
    "volume_ratio",
    "macd_hist_ratio",
]


# ============================================================
# 1. 데이터 로드 + market 매핑
# ============================================================
def load_ohlcv(db_path: Path = DB_PATH) -> pd.DataFrame:
    """tickers 의 market 을 KR/US 로 매핑하여 통합 OHLCV 반환."""
    with sqlite3.connect(db_path) as c:
        df = pd.read_sql_query("""
            SELECT o.ticker AS code, t.market AS market_raw,
                   o.date, o.open, o.high, o.low, o.close, o.volume
            FROM ohlcv o
            JOIN tickers t ON t.ticker = o.ticker
            WHERE t.market IN ('KOSPI','KOSDAQ','SP500','NASDAQ100')
              AND o.date >= ?
            ORDER BY t.market, o.ticker, o.date
        """, c, params=[TRAIN_START], parse_dates=["date"])
    df["market"] = df["market_raw"].map({
        "KOSPI": "KR", "KOSDAQ": "KR",
        "SP500": "US", "NASDAQ100": "US",
    })
    df = df.drop(columns=["market_raw"])
    df = df[df["volume"] > 0].reset_index(drop=True)
    logger.info("OHLCV 로드: %d 행, %d 종목 (KR %d / US %d)",
                len(df), df["code"].nunique(),
                df[df["market"] == "KR"]["code"].nunique(),
                df[df["market"] == "US"]["code"].nunique())
    return df


# ============================================================
# 2. 정예 6 피처 + 42일 라벨 (groupby market+code)
# ============================================================
def _add_features_per_group(g: pd.DataFrame) -> pd.DataFrame:
    """한 종목에 대해 6 피처 + 42일 라벨 추가. g 는 date 정렬되어 있다고 가정."""
    close = g["close"]
    high  = g["high"]
    low   = g["low"]
    vol   = g["volume"]

    # 이동평균
    ma20  = close.rolling(20).mean()
    ma60  = close.rolling(60).mean()
    ma200 = close.rolling(200).mean()

    # 1) 장기 뼈대
    g["price_to_ma200"] = close / ma200

    # 2) 중기 트렌드 (60일선 5일 기울기)
    g["ma60_slope"] = (ma60 - ma60.shift(5)) / ma60.shift(5)

    # 3) 단기 엔진
    g["ma20_to_ma60"] = ma20 / ma60

    # 4) 볼린저밴드 위치 (20일)
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    g["bb_position"] = (close - lower) / (upper - lower + 1e-9)

    # 5) 거래량 비율
    vol_ma20 = vol.rolling(20).mean()
    g["volume_ratio"] = vol / (vol_ma20 + 1e-9)

    # 6) MACD 히스토그램 비율
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    g["macd_hist_ratio"] = (macd_line - signal_line) / close

    # 타깃: 42거래일 후 수익률
    future_close = close.shift(-TARGET_HORIZON)
    g["target_return"] = (future_close - close) / close
    g["target_label"] = (g["target_return"] > 0).astype(int)

    return g


def build_dataset(db_path: Path = DB_PATH) -> pd.DataFrame:
    df = load_ohlcv(db_path)
    logger.info("피처 계산 중 (groupby market+code)...")
    df = df.sort_values(["market", "code", "date"]).reset_index(drop=True)
    df = df.groupby(["market", "code"], group_keys=False).apply(_add_features_per_group)
    logger.info("✓ 피처 생성 완료: %d 행", len(df))
    return df


# ============================================================
# 3. 시간순 분할 + 42거래일 갭
# ============================================================
@dataclass
class SplitInfo:
    train_idx: np.ndarray
    val_idx:   np.ndarray
    test_idx:  np.ndarray
    n_train:   int
    n_val:     int
    n_test:    int


def make_splits(df: pd.DataFrame) -> SplitInfo:
    """날짜 기준 분할. 사용자 명시 구간 사이는 자연스럽게 42거래일 갭."""
    d = df["date"]
    train_mask = (d >= TRAIN_START) & (d <= TRAIN_END)
    val_mask   = (d >= VAL_START)   & (d <= VAL_END)
    test_mask  = (d >= TEST_START)  & (d <= TEST_END)

    # 결측 (피처 NaN, 라벨 NaN) 제거
    feat_ok = df[FEATURE_COLS + ["target_label", "target_return"]].notna().all(axis=1)

    train_idx = np.where(train_mask & feat_ok)[0]
    val_idx   = np.where(val_mask   & feat_ok)[0]
    test_idx  = np.where(test_mask  & feat_ok)[0]

    logger.info("=== 분할 (날짜 기준 + 42거래일 갭) ===")
    logger.info("  TRAIN [%s ~ %s] : %d 행", TRAIN_START, TRAIN_END, len(train_idx))
    logger.info("  GAP   ~%d 거래일", TARGET_HORIZON)
    logger.info("  VAL   [%s ~ %s] : %d 행", VAL_START, VAL_END, len(val_idx))
    logger.info("  GAP   ~%d 거래일", TARGET_HORIZON)
    logger.info("  TEST  [%s ~ %s] : %d 행", TEST_START, TEST_END, len(test_idx))
    return SplitInfo(train_idx, val_idx, test_idx,
                     len(train_idx), len(val_idx), len(test_idx))


# ============================================================
# 4. LightGBM 학습
# ============================================================
def train(df: pd.DataFrame, save_path: Optional[Path] = None) -> tuple:
    import lightgbm as lgb

    split = make_splits(df)
    X = df[FEATURE_COLS].values
    y = df["target_label"].values

    train_set = lgb.Dataset(X[split.train_idx], label=y[split.train_idx])
    val_set   = lgb.Dataset(X[split.val_idx],   label=y[split.val_idx], reference=train_set)

    # 규제 파라미터 (오버피팅 방지)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "max_depth": 6,
        "num_leaves": 31,
        "min_data_in_leaf": 200,     # 큰 값 = 강한 규제
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "num_threads": -1,
    }
    logger.info("LightGBM 학습 시작 (max_depth=%d, min_data_in_leaf=%d)",
                params["max_depth"], params["min_data_in_leaf"])
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    save_path = save_path or (MODEL_DIR / "trend_lgbm.txt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(save_path))
    logger.info("✓ 모델 저장: %s (best iter=%d)", save_path, model.best_iteration)
    return model, split


# ============================================================
# 5. 평가 (전체/KR/US 분리)
# ============================================================
def _metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, target_ret: np.ndarray) -> dict:
    """Accuracy, Rank IC, 상하 10% spread."""
    from scipy.stats import spearmanr
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = float((y_pred == y_true).mean())
    # Rank IC (Spearman)
    if y_pred_prob.std() < 1e-9 or target_ret.std() < 1e-9:
        rank_ic = 0.0
    else:
        rho, _ = spearmanr(y_pred_prob, target_ret)
        rank_ic = float(rho)
    # 상위/하위 10%
    top_thr = np.percentile(y_pred_prob, 90)
    bot_thr = np.percentile(y_pred_prob, 10)
    top_mask = y_pred_prob >= top_thr
    bot_mask = y_pred_prob <= bot_thr
    top_ret = float(target_ret[top_mask].mean()) if top_mask.sum() > 0 else 0.0
    bot_ret = float(target_ret[bot_mask].mean()) if bot_mask.sum() > 0 else 0.0
    return dict(
        accuracy=acc, rank_ic=rank_ic,
        top10_ret=top_ret, bot10_ret=bot_ret,
        spread=top_ret - bot_ret,
        n=len(y_true),
    )


def evaluate(model, df: pd.DataFrame, split: SplitInfo) -> dict:
    X = df[FEATURE_COLS].values
    y = df["target_label"].values
    ret = df["target_return"].values
    mkt = df["market"].values

    test_X = X[split.test_idx]
    test_y = y[split.test_idx]
    test_ret = ret[split.test_idx]
    test_mkt = mkt[split.test_idx]
    pred = model.predict(test_X)

    print("\n" + "=" * 100)
    print(f"평가 (TEST: {TEST_START} ~ {TEST_END}, n={len(test_y):,})")
    print("=" * 100)

    out = {}
    print(f"\n{'segment':<10}{'n':>10}{'Accuracy':>12}{'Rank IC':>12}{'Top10%':>12}{'Bot10%':>12}{'Spread':>12}")
    print("─" * 90)
    for label, mask in [
        ("ALL",  np.ones_like(test_mkt, dtype=bool)),
        ("KR",   test_mkt == "KR"),
        ("US",   test_mkt == "US"),
    ]:
        if mask.sum() == 0:
            print(f"{label:<10}{'(no data)':>10}")
            continue
        m = _metrics(test_y[mask], pred[mask], test_ret[mask])
        out[label] = m
        print(f"{label:<10}{m['n']:>10,}{m['accuracy']*100:>11.2f}%"
              f"{m['rank_ic']:>12.4f}{m['top10_ret']*100:>11.2f}%{m['bot10_ret']*100:>11.2f}%"
              f"{m['spread']*100:>11.2f}%")

    # 피처 중요도
    print("\n" + "─" * 90)
    print("피처 중요도 (gain)")
    print("─" * 90)
    imp = model.feature_importance(importance_type="gain")
    for f, v in sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1]):
        bar = "█" * int(v / max(imp) * 40)
        print(f"  {f:<22} {v:>10.0f}  {bar}")
    return out


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    import time
    t0 = time.time()

    print("\n>>> [1/3] 데이터셋 구축...")
    df = build_dataset()
    print(f"  ({time.time()-t0:.0f}s 누적)")

    print("\n>>> [2/3] LightGBM 학습...")
    t1 = time.time()
    model, split = train(df)
    print(f"  ({time.time()-t1:.0f}s 학습)")

    print("\n>>> [3/3] 평가...")
    evaluate(model, df, split)
    print(f"\n총 {time.time()-t0:.0f}초")
