"""LightGBM Trend Ranker v4 — 펀더멘털 3 피처 추가.

v3 (10 피처) → v4 (13 피처):
  + roe_latest:        공시된 가장 최근 분기 ROE (TTM)
  + op_margin_growth:  최근 분기 OPM − 직전 분기 OPM (실적 모멘텀)
  + per_inverse:       1 / PER 안정 처리 (이익수익률)

Point-in-Time: 분기말 + 45일 (Q1-Q3) / + 90일 (Q4) 공시 추정 → date 시점
                이전에 공시된 가장 최근 분기 데이터만 매핑 (look-ahead 차단).
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DB_PATH, MODEL_DIR
from src.model.trend_lgbm_v2 import (
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    TARGET_HORIZON, INDEX_TICKER,
    _load_ohlcv, _load_index, _features_per_group,
    SplitInfo, make_splits,
)

logger = logging.getLogger(__name__)

# v4 = v3 (10 피처, rsi_signal 제거) + 펀더멘털 3 피처
FEATURE_COLS = [
    # 개별 8 (v3 와 동일)
    "price_to_ma200", "ma60_slope", "ma20_to_ma60",
    "bb_position", "volume_ratio", "macd_hist_ratio",
    "stoch_60_pos", "obv_slope_20",
    # 매크로 2 (v3 와 동일)
    "index_ma60_slope", "index_mdd_20",
    # 펀더멘털 3 (신규)
    "roe_latest", "op_margin_growth", "per_inverse",
]


# ============================================================
# 1. 펀더멘털 데이터 + publish_date 계산
# ============================================================
def _load_fundamentals(db_path: Path = DB_PATH) -> pd.DataFrame:
    """fundamentals 테이블 로드 + publish_date 계산 (look-ahead 차단용)."""
    with sqlite3.connect(db_path) as c:
        f = pd.read_sql_query("""
            SELECT f.ticker AS code, f.period_end, f.roe, f.operating_margin
            FROM fundamentals f
            ORDER BY f.ticker, f.period_end
        """, c, parse_dates=["period_end"])
    if f.empty:
        return f
    # publish_date 추정: Q1-Q3 = +45일, Q4 = +90일
    f["quarter_month"] = f["period_end"].dt.month
    f["publish_date"] = f["period_end"] + pd.to_timedelta(
        np.where(f["quarter_month"] == 12, 90, 45), unit="D"
    )
    # op_margin_growth = 이번 분기 OPM − 직전 분기 OPM (같은 종목 내)
    f = f.sort_values(["code", "period_end"]).reset_index(drop=True)
    f["op_margin_growth"] = f.groupby("code")["operating_margin"].diff()
    f = f.rename(columns={"roe": "roe_latest"})
    return f[["code", "period_end", "publish_date", "roe_latest", "op_margin_growth"]]


def _load_per(db_path: Path = DB_PATH) -> pd.DataFrame:
    """per_history 테이블 (월말 기준) → per_inverse 계산."""
    with sqlite3.connect(db_path) as c:
        p = pd.read_sql_query("""
            SELECT ticker AS code, date, per
            FROM per_history WHERE per IS NOT NULL
            ORDER BY ticker, date
        """, c, parse_dates=["date"])
    if p.empty:
        return p
    # 안정성 처리: per ≤ 0 (적자) → 0 / 매우 큰 PER → cap
    # per_inverse 는 [-0.5, 0.5] 범위로 winsorize
    inv = np.where(
        p["per"] > 0,
        1.0 / p["per"].clip(lower=2.0),  # PER < 2 (이상치) → cap 0.5
        0.0,                              # 적자 (per ≤ 0) → 0 (밸류에이션 점수 없음)
    )
    p["per_inverse"] = np.clip(inv, -0.5, 0.5)
    return p[["code", "date", "per_inverse"]]


# ============================================================
# 2. Point-in-Time merge (look-ahead 차단)
# ============================================================
def _merge_point_in_time(df: pd.DataFrame, fund: pd.DataFrame, per: pd.DataFrame) -> pd.DataFrame:
    """매 (code, date) 에 publish_date <= date 인 가장 최근 fundamentals + 가장 최근 per_inverse.
    merge_asof 는 left/right 모두 'on' 키로 정렬되어야 함 (by 그룹 내에서도).
    """
    # df 는 반드시 date 로 정렬 (merge_asof 요구)
    df = df.sort_values("date").reset_index(drop=True)

    if not fund.empty:
        fund_s = fund.dropna(subset=["publish_date"]).sort_values("publish_date").reset_index(drop=True)
        df = pd.merge_asof(
            df, fund_s[["code", "publish_date", "roe_latest", "op_margin_growth"]],
            left_on="date", right_on="publish_date",
            by="code", direction="backward",
        )
        df = df.drop(columns=["publish_date"])
    else:
        df["roe_latest"] = np.nan
        df["op_margin_growth"] = np.nan

    if not per.empty:
        per_s = per.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        df = pd.merge_asof(
            df, per_s, by="code", on="date", direction="backward",
        )
    else:
        df["per_inverse"] = np.nan
    return df


# ============================================================
# 3. 통합 dataset (v3 + 펀더멘털)
# ============================================================
def build_dataset(db_path: Path = DB_PATH, *, fundamental_only: bool = False) -> pd.DataFrame:
    """
    fundamental_only=False (기본, 권장): KR 전체 종목 학습. NaN 자동 처리.
      → KR IC 0.117 / Spread 6.39%pt (검증된 최선).
    fundamental_only=True: fundamentals 있는 종목 (628) 만. 학습 universe 축소로 성능 저하.
      → KR IC 0.036 (악화). 비권장.
    US 종목은 fundamentals 없으므로 항상 NaN — LightGBM 자동 처리.
    """
    logger.info("OHLCV + 인덱스 + 펀더멘털 + PER 로드 중...")
    df = _load_ohlcv(db_path)
    idx = _load_index(db_path)
    fund = _load_fundamentals(db_path)
    per = _load_per(db_path)

    if fundamental_only and not fund.empty:
        fund_codes = set(fund["code"].unique())
        # KR 종목 중 fundamentals 보유한 것만 + US 전체 유지
        before = len(df)
        df = df[(df["market"] == "US") | (df["code"].isin(fund_codes))].reset_index(drop=True)
        logger.info("  KR fundamental filter: %d → %d 행 (%d 종목 제거)",
                    before, len(df), before - len(df))
    logger.info("  OHLCV %d / fundamentals %d / per %d 행",
                len(df), len(fund), len(per))

    logger.info("개별 종목 8 피처 계산 (v2 _features_per_group)...")
    df = df.sort_values(["market", "code", "date"]).reset_index(drop=True)
    df = df.groupby(["market", "code"], group_keys=False).apply(_features_per_group)

    logger.info("매크로 피처 merge...")
    df = df.merge(idx[["market", "date", "index_ma60_slope", "index_mdd_20"]],
                  on=["market", "date"], how="left")

    logger.info("펀더멘털 + PER point-in-time merge (look-ahead 차단)...")
    df = _merge_point_in_time(df, fund, per)

    # 인피니티 / 극단치 안정 처리
    for col in ("roe_latest", "op_margin_growth", "per_inverse"):
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    # ROE 극단치 cap (±200%)
    df["roe_latest"] = df["roe_latest"].clip(-200, 200)
    # op_margin_growth 극단치 cap (±50%pt)
    df["op_margin_growth"] = df["op_margin_growth"].clip(-50, 50)
    # per_inverse 는 이미 [-0.5, 0.5] cap 완료

    # Cross-sectional rank target
    df["target_rank"] = df.groupby("date")["target_return"].rank(pct=True)

    logger.info("✓ 데이터셋 완성: %d 행, %d 피처 (개별 8 + 매크로 2 + 펀더 3)",
                len(df), len(FEATURE_COLS))
    return df


# ============================================================
# 4. 학습 + 평가 (v2 와 동일 구조)
# ============================================================
def train(df: pd.DataFrame, save_path: Optional[Path] = None):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    split = make_splits(df)
    # ⚠️ FEATURE_COLS 가 v4 (13 개) 라 make_splits 의 feat_ok 가 펀더멘털 NaN 도 검사 → US 종목 다 탈락
    # → re-mask: 핵심 8+2 피처만 필수, 펀더 3개는 NaN 허용 (LightGBM 자동 처리)
    core_feats = FEATURE_COLS[:10]  # 개별 + 매크로
    d = df["date"]
    feat_ok = df[core_feats + ["target_rank", "target_return"]].notna().all(axis=1)
    train_mask = (d >= TRAIN_START) & (d <= TRAIN_END) & feat_ok
    val_mask = (d >= VAL_START) & (d <= VAL_END) & feat_ok
    test_mask = (d >= TEST_START) & (d <= TEST_END) & feat_ok
    split = SplitInfo(
        train_idx=np.where(train_mask)[0],
        val_idx=np.where(val_mask)[0],
        test_idx=np.where(test_mask)[0],
        n_train=int(train_mask.sum()),
        n_val=int(val_mask.sum()),
        n_test=int(test_mask.sum()),
    )
    logger.info("=== 분할 (펀더 NaN 허용) ===")
    logger.info("  TRAIN: %d / VAL: %d / TEST: %d",
                split.n_train, split.n_val, split.n_test)

    X = df[FEATURE_COLS].values
    y = df["target_rank"].values

    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.03, max_depth=6, num_leaves=31,
        min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1, n_estimators=500, n_jobs=-1, verbose=-1,
    )
    logger.info("LGBMRegressor (v4, 13 피처) 학습...")
    model = LGBMRegressor(**params)
    model.fit(
        X[split.train_idx], y[split.train_idx],
        eval_set=[(X[split.val_idx], y[split.val_idx])],
        eval_metric="rmse",
        callbacks=[early_stopping(30, verbose=False), log_evaluation(50)],
    )
    save_path = save_path or (MODEL_DIR / "trend_lgbm_v4.txt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(save_path))
    logger.info("✓ 저장: %s (best iter=%d)", save_path, model.best_iteration_)
    return model, split


def evaluate(model, df: pd.DataFrame, split: SplitInfo):
    from scipy.stats import spearmanr
    X = df[FEATURE_COLS].values
    ret = df["target_return"].values
    mkt = df["market"].values
    test_X = X[split.test_idx]; test_ret = ret[split.test_idx]; test_mkt = mkt[split.test_idx]
    pred = model.predict(test_X)

    print("\n" + "=" * 100)
    print(f"v4 평가 (TEST: {TEST_START} ~ {TEST_END}, n={len(test_ret):,})")
    print("=" * 100)
    print(f"\n{'segment':<10}{'n':>11}{'Rank IC':>11}{'Top10%':>11}{'Bot10%':>11}{'Spread':>11}")
    print("─" * 70)
    out = {}
    for label, mask in [
        ("ALL", np.ones_like(test_mkt, dtype=bool)),
        ("KR",  test_mkt == "KR"),
        ("US",  test_mkt == "US"),
    ]:
        if mask.sum() == 0: continue
        p = pred[mask]; r = test_ret[mask]
        rho = spearmanr(p, r)[0] if p.std() > 1e-9 else 0
        top = np.percentile(p, 90); bot = np.percentile(p, 10)
        top_r = float(r[p >= top].mean()) if (p >= top).any() else 0
        bot_r = float(r[p <= bot].mean()) if (p <= bot).any() else 0
        out[label] = dict(n=int(mask.sum()), rank_ic=rho, top10=top_r, bot10=bot_r, spread=top_r-bot_r)
        print(f"{label:<10}{int(mask.sum()):>10,} {rho:>11.4f}{top_r*100:>10.2f}%{bot_r*100:>10.2f}%{(top_r-bot_r)*100:>10.2f}%")

    print("\n[피처 중요도]")
    imp = model.booster_.feature_importance(importance_type="gain")
    for f, v in sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1]):
        bar = "█" * max(1, int(v / max(imp) * 40))
        flag = " ★(NEW)" if f in ("roe_latest", "op_margin_growth", "per_inverse") else ""
        print(f"  {f:<22} {v:>10.0f}  {bar}{flag}")
    return out


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    import time
    t0 = time.time()
    print("\n>>> [1/3] v4 데이터셋 구축 (13 피처)...")
    df = build_dataset()
    print(f"  ({time.time()-t0:.0f}s)")

    # 펀더 피처 결측률 확인
    for f in ("roe_latest", "op_margin_growth", "per_inverse"):
        n_total = len(df); n_na = df[f].isna().sum()
        print(f"  {f}: NaN {n_na:,} ({n_na/n_total*100:.1f}%)")

    print("\n>>> [2/3] 학습...")
    t1 = time.time()
    model, split = train(df)
    print(f"  ({time.time()-t1:.0f}s)")

    print("\n>>> [3/3] 평가...")
    evaluate(model, df, split)

    print("\n[v3 (10 피처) vs v4 (13 피처) 비교]")
    print("v3 KR: IC=0.1310 / Spread=5.39%pt")
    print(f"\n총 {time.time()-t0:.0f}초")
