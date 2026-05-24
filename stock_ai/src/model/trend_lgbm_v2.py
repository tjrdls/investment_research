"""LightGBM Trend Ranker v2 — Cross-Sectional Rank Regression.

기존 trend_lgbm.py 의 한계를 극복하기 위한 3가지 고도화:
  1) 마켓 지수 매크로 피처 2개 (index_ma60_slope, index_mdd_20)
  2) 개별 종목 정예 피처 6 → 9개 (rsi_signal_ratio, stoch_60_pos, obv_slope_20 추가)
  3) 이진 분류 → Cross-Sectional Rank Regression (groupby('date').rank(pct=True))

인덱스: KR=069500 (KODEX200), US=QQQ (Invesco QQQ)
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

# ── 분할 (기존 trend_lgbm 와 동일) ──────────────────────
TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"
VAL_START   = "2022-03-01"   # ≈ TRAIN_END + 42일 갭
VAL_END     = "2023-12-31"
TEST_START  = "2024-03-01"   # ≈ VAL_END + 42일 갭
TEST_END    = "2026-05-31"
TARGET_HORIZON = 42

INDEX_TICKER = {"KR": "069500", "US": "QQQ"}

FEATURE_COLS = [
    # 기존 6 (개별 종목)
    "price_to_ma200",
    "ma60_slope",
    "ma20_to_ma60",
    "bb_position",
    "volume_ratio",
    "macd_hist_ratio",
    # 신규 3 (2달 관성 정밀 포착)
    "rsi_signal_ratio",
    "stoch_60_pos",
    "obv_slope_20",
    # 신규 2 (마켓 매크로)
    "index_ma60_slope",
    "index_mdd_20",
]


# ============================================================
# 1. 데이터 로드
# ============================================================
def _load_ohlcv(db_path: Path = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as c:
        df = pd.read_sql_query("""
            SELECT o.ticker AS code, t.market AS market_raw,
                   o.date, o.open, o.high, o.low, o.close, o.volume
            FROM ohlcv o JOIN tickers t ON t.ticker = o.ticker
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
    return df


def _load_index(db_path: Path = DB_PATH) -> pd.DataFrame:
    """KR/US 인덱스 일별 종가 + 매크로 피처."""
    rows = []
    with sqlite3.connect(db_path) as c:
        for mk, tk in INDEX_TICKER.items():
            df = pd.read_sql_query(
                "SELECT date, close FROM ohlcv WHERE ticker=? AND date >= ? ORDER BY date",
                c, params=[tk, TRAIN_START], parse_dates=["date"])
            df["market"] = mk
            df = df.rename(columns={"close": "index_close"})
            # 매크로 피처 1: 지수 MA60 5일 기울기
            ma60 = df["index_close"].rolling(60).mean()
            df["index_ma60_slope"] = (ma60 - ma60.shift(5)) / ma60.shift(5)
            # 매크로 피처 2: 최근 20거래일 지수 MDD
            roll_max = df["index_close"].rolling(20).max()
            df["index_mdd_20"] = (df["index_close"] - roll_max) / roll_max  # ≤ 0
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ============================================================
# 2. 개별 종목 9 피처 (groupby market+code 내부)
# ============================================================
def _features_per_group(g: pd.DataFrame) -> pd.DataFrame:
    close = g["close"]; high = g["high"]; low = g["low"]; vol = g["volume"]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma200 = close.rolling(200).mean()

    # 기존 6
    g["price_to_ma200"] = close / ma200
    g["ma60_slope"] = (ma60 - ma60.shift(5)) / ma60.shift(5)
    g["ma20_to_ma60"] = ma20 / ma60
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20; lower = ma20 - 2 * std20
    g["bb_position"] = (close - lower) / (upper - lower + 1e-9)
    vol_ma20 = vol.rolling(20).mean()
    g["volume_ratio"] = vol / (vol_ma20 + 1e-9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    g["macd_hist_ratio"] = (macd_line - signal) / close

    # 신규 3
    # 3-1) RSI(14) / RSI_Signal(RSI 9일 MA)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi14 = 100 - 100 / (1 + rs)
    rsi_sig = rsi14.rolling(9).mean()
    g["rsi_signal_ratio"] = rsi14 / (rsi_sig + 1e-9)

    # 3-2) Stochastic 60일 정규화 위치
    low60 = low.rolling(60).min()
    high60 = high.rolling(60).max()
    g["stoch_60_pos"] = (close - low60) / (high60 - low60 + 1e-9)

    # 3-3) OBV 20일 기울기 비율
    obv_diff = np.sign(close.diff().fillna(0)) * vol
    obv = obv_diff.cumsum()
    obv_20 = obv.shift(20)
    g["obv_slope_20"] = (obv - obv_20) / (obv_20.abs() + 1e-9)

    # 타깃: 42일 후 수익률 (rank 는 나중에 cross-sectional 로)
    future_close = close.shift(-TARGET_HORIZON)
    g["target_return"] = (future_close - close) / close
    return g


# ============================================================
# 3. 통합 dataset 빌드
# ============================================================
def build_dataset(db_path: Path = DB_PATH) -> pd.DataFrame:
    logger.info("OHLCV 로드 중...")
    df = _load_ohlcv(db_path)
    logger.info("  %d 행, %d 종목", len(df), df["code"].nunique())

    logger.info("인덱스 로드 중 (KR=%s, US=%s)...", INDEX_TICKER["KR"], INDEX_TICKER["US"])
    idx = _load_index(db_path)

    logger.info("개별 종목 9 피처 계산 중 (groupby market+code)...")
    df = df.sort_values(["market", "code", "date"]).reset_index(drop=True)
    df = df.groupby(["market", "code"], group_keys=False).apply(_features_per_group)

    logger.info("매크로 피처 merge 중...")
    df = df.merge(idx[["market", "date", "index_ma60_slope", "index_mdd_20"]],
                  on=["market", "date"], how="left")

    # Cross-sectional rank target
    logger.info("Cross-sectional rank 타깃 생성 중...")
    df["target_rank"] = df.groupby("date")["target_return"].rank(pct=True)
    logger.info("✓ 데이터셋 완성: %d 행, %d 피처", len(df), len(FEATURE_COLS))
    return df


# ============================================================
# 4. 분할
# ============================================================
@dataclass
class SplitInfo:
    train_idx: np.ndarray; val_idx: np.ndarray; test_idx: np.ndarray
    n_train: int; n_val: int; n_test: int


def make_splits(df: pd.DataFrame) -> SplitInfo:
    d = df["date"]
    train_mask = (d >= TRAIN_START) & (d <= TRAIN_END)
    val_mask   = (d >= VAL_START)   & (d <= VAL_END)
    test_mask  = (d >= TEST_START)  & (d <= TEST_END)
    feat_ok = df[FEATURE_COLS + ["target_rank", "target_return"]].notna().all(axis=1)
    tr = np.where(train_mask & feat_ok)[0]
    va = np.where(val_mask & feat_ok)[0]
    te = np.where(test_mask & feat_ok)[0]
    logger.info("=== 분할 (시간순 walk-forward + 42거래일 갭) ===")
    logger.info("  TRAIN [%s ~ %s]: %d 행", TRAIN_START, TRAIN_END, len(tr))
    logger.info("  VAL   [%s ~ %s]: %d 행", VAL_START,   VAL_END,   len(va))
    logger.info("  TEST  [%s ~ %s]: %d 행", TEST_START,  TEST_END,  len(te))
    return SplitInfo(tr, va, te, len(tr), len(va), len(te))


# ============================================================
# 5. LGBMRegressor 학습
# ============================================================
def train(df: pd.DataFrame, save_path: Optional[Path] = None):
    from lightgbm import LGBMRegressor
    split = make_splits(df)
    X = df[FEATURE_COLS].values
    y = df["target_rank"].values

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_estimators=500,
        n_jobs=-1,
        verbose=-1,
    )
    logger.info("LGBMRegressor 학습 (rank regression)...")
    model = LGBMRegressor(**params)
    model.fit(
        X[split.train_idx], y[split.train_idx],
        eval_set=[(X[split.val_idx], y[split.val_idx])],
        eval_metric="rmse",
        callbacks=[
            __import__("lightgbm").early_stopping(30, verbose=False),
            __import__("lightgbm").log_evaluation(50),
        ],
    )
    save_path = save_path or (MODEL_DIR / "trend_lgbm_v2.txt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(save_path))
    logger.info("✓ 저장: %s (best iter=%d)", save_path, model.best_iteration_)
    return model, split


# ============================================================
# 6. 평가 (segment 별)
# ============================================================
def _metrics(pred: np.ndarray, target_ret: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    if pred.std() < 1e-9 or target_ret.std() < 1e-9:
        rank_ic = 0.0
    else:
        rho, _ = spearmanr(pred, target_ret)
        rank_ic = float(rho)
    top_thr = np.percentile(pred, 90); bot_thr = np.percentile(pred, 10)
    top_mask = pred >= top_thr; bot_mask = pred <= bot_thr
    top_ret = float(target_ret[top_mask].mean()) if top_mask.sum() > 0 else 0.0
    bot_ret = float(target_ret[bot_mask].mean()) if bot_mask.sum() > 0 else 0.0
    return dict(rank_ic=rank_ic, top10_ret=top_ret, bot10_ret=bot_ret,
                spread=top_ret - bot_ret, n=len(target_ret))


def evaluate(model, df: pd.DataFrame, split: SplitInfo):
    X = df[FEATURE_COLS].values
    ret = df["target_return"].values
    mkt = df["market"].values

    test_X = X[split.test_idx]
    test_ret = ret[split.test_idx]
    test_mkt = mkt[split.test_idx]
    pred = model.predict(test_X)

    print("\n" + "=" * 100)
    print(f"v2 평가 (TEST: {TEST_START} ~ {TEST_END}, n={len(test_ret):,})")
    print("=" * 100)
    print(f"\n{'segment':<10}{'n':>11}{'Rank IC':>11}{'Top10%':>11}{'Bot10%':>11}{'Spread':>11}")
    print("─" * 70)
    out = {}
    for label, mask in [
        ("ALL", np.ones_like(test_mkt, dtype=bool)),
        ("KR",  test_mkt == "KR"),
        ("US",  test_mkt == "US"),
    ]:
        if mask.sum() == 0:
            continue
        m = _metrics(pred[mask], test_ret[mask])
        out[label] = m
        print(f"{label:<10}{m['n']:>10,} {m['rank_ic']:>11.4f}"
              f"{m['top10_ret']*100:>10.2f}%{m['bot10_ret']*100:>10.2f}%{m['spread']*100:>10.2f}%")

    print("\n[피처 중요도 (gain)]")
    imp = model.booster_.feature_importance(importance_type="gain")
    for f, v in sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1]):
        bar = "█" * max(1, int(v / max(imp) * 40))
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
    print("\n>>> [1/3] v2 데이터셋 구축 (11 피처 + rank 타깃)...")
    df = build_dataset()
    print(f"  ({time.time()-t0:.0f}s)")

    print("\n>>> [2/3] LGBMRegressor 학습...")
    t1 = time.time()
    model, split = train(df)
    print(f"  ({time.time()-t1:.0f}s)")

    print("\n>>> [3/3] 평가...")
    evaluate(model, df, split)

    # 기존 v1 (이진 분류 + 6 피처) 결과와 비교
    print("\n" + "=" * 100)
    print("vs 기존 v1 (이진 분류 + 6 피처) 결과")
    print("=" * 100)
    print(f"{'segment':<10}{'v1 IC':>11}{'v1 Spread':>13}{'v2 IC':>11}{'v2 Spread':>13}")
    print("─" * 60)
    v1 = {"ALL": (0.0819, 2.06), "KR": (0.0748, 2.52), "US": (0.0029, -0.25)}
    for seg in ("ALL", "KR", "US"):
        pred_mask = (split.test_idx, None)
        # v2 결과 다시 계산
        pass  # 평가 출력에서 이미 표시됨

    print(f"\n총 {time.time()-t0:.0f}초")
