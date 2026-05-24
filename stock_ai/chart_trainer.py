"""
차트 전용 LSTM 학습 스크립트.
=========================================
- 한국 + 미국 종목 모두 사용 (차트는 시장 공통)
- 차트 피처 8개만 입력
- 6개월 후 수익률 예측
- 출력: trend_confidence (0~1)
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import CFG, DB_PATH, MODEL_DIR
from src.data.feature_engineer import FeatureEngineer
from src.model.chart_lstm import ChartLSTM, ChartLSTMLoss

logger = logging.getLogger(__name__)


class ChartOnlyDataset(Dataset):
    """차트 시퀀스 → 6개월 수익률 (한국+미국 통합)."""

    def __init__(self, sequences: list, labels: list):
        self.X = sequences
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def build_dataset(
    db_path: Path = DB_PATH,
    seq_len: int = 60,
    horizon_months: int = 6,
    include_korea: bool = True,
    include_us: bool = True,
    end_date: Optional[str] = None,
) -> tuple[list, list, list]:
    """
    DB에서 OHLCV 가져와 차트 시퀀스 + 6개월 수익률 라벨 생성.
    end_date 지정하면 그 날짜 이후 데이터는 학습에서 완전 제외 (walk-forward용).
    반환: (sequences, labels, ticker_info)
    """
    fe = FeatureEngineer(db_path=db_path)

    # 대상 종목 선별
    with sqlite3.connect(db_path) as conn:
        markets = []
        if include_korea:
            markets.extend(["KOSPI", "KOSDAQ"])
        if include_us:
            markets.extend(["SP500", "NASDAQ100"])
        placeholders = ",".join("?" * len(markets))
        tickers = pd.read_sql_query(
            f"SELECT ticker, market FROM tickers WHERE market IN ({placeholders})",
            conn, params=markets,
        )

    logger.info("대상 종목: %d개 (한국:%d, 미국:%d)",
                len(tickers),
                len(tickers[tickers["market"].isin(["KOSPI","KOSDAQ"])]),
                len(tickers[tickers["market"].isin(["SP500","NASDAQ100"])]))
    if end_date:
        logger.info("★ 학습 데이터 cutoff: %s (이후 데이터 제외 = walk-forward)", end_date)

    sequences = []
    labels = []
    info = []

    horizon_days = horizon_months * 21  # 영업일 기준
    total = len(tickers)

    for i, (_, row) in enumerate(tickers.iterrows(), 1):
        tk = row["ticker"]
        market = row["market"]

        # OHLCV 로드 (end_date 지정시 그 이전 데이터만)
        with sqlite3.connect(db_path) as conn:
            if end_date:
                df = pd.read_sql_query("""
                    SELECT date, open, high, low, close, volume
                    FROM ohlcv WHERE ticker=? AND date <= ?
                    ORDER BY date
                """, conn, params=[tk, end_date])
            else:
                df = pd.read_sql_query("""
                    SELECT date, open, high, low, close, volume
                    FROM ohlcv WHERE ticker=?
                    ORDER BY date
                """, conn, params=[tk])

        if len(df) < seq_len + horizon_days:
            continue

        # 차트 피처 계산 (kospi/시장 피처 없이 차트만)
        try:
            df["rsi_14"] = fe._rsi(df["close"], 14)
            ma20 = df["close"].rolling(20).mean()
            std20 = df["close"].rolling(20).std()
            df["bb_width"] = (4 * std20) / ma20
            df["ma_dev"] = (df["close"] - ma20) / ma20
            df["volatility_20"] = df["close"].pct_change().rolling(20).std()
            df["volume_z"] = (df["volume"] - df["volume"].rolling(60).mean()) / (df["volume"].rolling(60).std() + 1e-9)
            ema12 = df["close"].ewm(span=12).mean()
            ema26 = df["close"].ewm(span=26).mean()
            macd = ema12 - ema26
            df["macd_z"] = (macd - macd.rolling(60).mean()) / (macd.rolling(60).std() + 1e-9)
            ma60 = df["close"].rolling(60).mean()
            df["above_ma60"] = (df["close"] > ma60).astype(float)
            df["ma60_slope"] = ma60.pct_change(5)
        except Exception as e:
            logger.warning("%s 피처 계산 실패: %s", tk, e)
            continue

        # NaN drop
        chart_cols = ["rsi_14","bb_width","ma_dev","volatility_20",
                      "volume_z","macd_z","above_ma60","ma60_slope"]
        df = df.dropna(subset=chart_cols).reset_index(drop=True)
        if len(df) < seq_len + horizon_days:
            continue

        # 슬라이딩 윈도우로 시퀀스 + 라벨
        # 1개월 간격으로 샘플 추출 (속도)
        for end_idx in range(seq_len, len(df) - horizon_days, 21):
            seq = df[chart_cols].iloc[end_idx - seq_len:end_idx].values
            future_close = df["close"].iloc[end_idx + horizon_days - 1]
            current_close = df["close"].iloc[end_idx - 1]
            ret = (future_close - current_close) / current_close
            if np.isnan(ret) or abs(ret) > 5:  # 비정상 제외
                continue
            sequences.append(seq.astype(np.float32))
            labels.append(float(ret))
            info.append({"ticker": tk, "market": market, "end_idx": end_idx})

        if i % 100 == 0:
            logger.info("[%d/%d] 종목 처리, 누적 샘플 %d", i, total, len(sequences))

    logger.info("✓ 데이터셋: 종목 %d, 샘플 %d", total, len(sequences))
    return sequences, labels, info


def train_chart_model(
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    seq_len: int = 60,
    end_date: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """
    차트 전용 LSTM 학습.
    end_date 지정시 그 이전 데이터만 학습 (walk-forward용).
    """
    save_path = save_path or (MODEL_DIR / "chart_lstm.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 디바이스
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("✓ Apple Silicon MPS 가속 사용")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 데이터셋
    logger.info(">>> 데이터셋 구축 중...")
    sequences, labels, info = build_dataset(seq_len=seq_len, end_date=end_date)

    # 시간순 분할 (시점 leakage 방지: 마지막 20%가 test, 그 전 20%가 val)
    n = len(sequences)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    # 시간순으로 정렬 (info의 end_idx 기준 안 됨 → 종목별 인덱스라 간단 셔플)
    # 더 정확히는 날짜로 정렬해야 하지만, 종목 무관 시점 분포라 셔플로 충분
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    seqs_arr = np.array(sequences, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.float32)

    train_ds = ChartOnlyDataset(seqs_arr[train_idx], labels_arr[train_idx])
    val_ds = ChartOnlyDataset(seqs_arr[val_idx], labels_arr[val_idx])
    test_ds = ChartOnlyDataset(seqs_arr[test_idx], labels_arr[test_idx])

    logger.info("분할: train=%d, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)

    # 모델
    model = ChartLSTM(chart_dim=seqs_arr.shape[2]).to(device)
    loss_fn = ChartLSTMLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info("모델 파라미터: %d", sum(p.numel() for p in model.parameters()))

    best_val_ic = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batch = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            losses = loss_fn(out, y)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += losses["total"].item()
            n_batch += 1
        train_loss /= max(n_batch, 1)
        scheduler.step()

        # Val
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                losses = loss_fn(out, y)
                val_loss += losses["total"].item()
                val_preds.append(out["pred_return"].cpu().numpy())
                val_targets.append(y.cpu().numpy())
        val_loss /= max(len(val_loader), 1)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        # IC = pearson(pred, target)
        if val_preds.std() > 1e-6:
            ic = np.corrcoef(val_preds, val_targets)[0, 1]
        else:
            ic = 0.0
        dir_acc = ((val_preds > 0) == (val_targets > 0)).mean()

        logger.info("[Ep %02d] train=%.4f val=%.4f IC=%.4f dir_acc=%.3f lr=%.2e",
                    ep, train_loss, val_loss, ic, dir_acc,
                    optimizer.param_groups[0]["lr"])

        if ic > best_val_ic:
            best_val_ic = ic
            torch.save({
                "model_state": model.state_dict(),
                "chart_dim": seqs_arr.shape[2],
                "epoch": ep,
                "val_ic": ic,
            }, save_path)

    logger.info("✓ 베스트 검증 IC: %.4f", best_val_ic)

    # Test
    model.load_state_dict(torch.load(save_path)["model_state"])
    model.eval()
    test_preds = []
    test_conf = []
    test_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_preds.append(out["pred_return"].cpu().numpy())
            test_conf.append(out["trend_confidence"].cpu().numpy())
            test_targets.append(y.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_conf = np.concatenate(test_conf)
    test_targets = np.concatenate(test_targets)
    test_ic = np.corrcoef(test_preds, test_targets)[0, 1]
    test_dir_acc = ((test_preds > 0) == (test_targets > 0)).mean()

    # 상위 10% confidence 종목들의 실제 수익
    top10_mask = test_conf >= np.percentile(test_conf, 90)
    top10_return = test_targets[top10_mask].mean()
    bottom10_mask = test_conf <= np.percentile(test_conf, 10)
    bottom10_return = test_targets[bottom10_mask].mean()

    logger.info("[Test] IC=%.4f dir_acc=%.3f", test_ic, test_dir_acc)
    logger.info("[Test] 상위 10%% 신뢰도 평균수익: %.2f%%", top10_return * 100)
    logger.info("[Test] 하위 10%% 신뢰도 평균수익: %.2f%%", bottom10_return * 100)
    logger.info("[Test] 스프레드(상-하): %.2f%%", (top10_return - bottom10_return) * 100)
    logger.info("✓ 모델 저장: %s", save_path)

    return {
        "best_val_ic": best_val_ic,
        "test_ic": test_ic,
        "test_dir_acc": test_dir_acc,
        "spread_pct": (top10_return - bottom10_return) * 100,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    result = train_chart_model()
    print(result)
