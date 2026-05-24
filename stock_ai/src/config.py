"""
중앙 설정 파일 — 모든 하이퍼파라미터를 한 곳에서 관리
=====================================================
당신이 "6:2:2 비율은 조절을 계속 해봐야 한다"고 했으니,
SCORE_WEIGHTS만 바꾸면 추천 결과가 즉시 반영됩니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ============================================================
# 1. 경로
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "cache.db"
MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "lstm_best.pt"


# ============================================================
# 2. 하드 필터 — 통과 못하면 추천 후보에서 탈락
# ============================================================
@dataclass
class HardFilter:
    roe_min: float = 20.0              # ROE ≥ 20% (조정: 30→20, 한국시장 우량주 현실 반영)
    per_max: float = 60.0              # PER ≤ 60배 (조정: 50→60, 성장주 포함 위해 확장)
    per_min: float = 0.0               # 적자 종목 제외
    market_cap_min_krw: float = 5e11   # 시총 5,000억+ (조정: 1조→5000억, 중대형주 포함)
    revenue_growth_required: bool = True   # 매출 성장 필수
    profit_growth_required: bool = True    # 순이익 성장 필수
    growth_period: str = "yoy"         # "yoy" 또는 "qoq"


# ============================================================
# 3. 점수 가중치 — 당신이 정한 6:2:2
# ============================================================
@dataclass
class ScoreWeights:
    fundamental: float = 0.6   # ROE, 성장률, PER 매력도
    chart: float = 0.2         # 과열도
    market: float = 0.2        # 금리 + 유동성

    def normalize(self) -> "ScoreWeights":
        s = self.fundamental + self.chart + self.market
        return ScoreWeights(
            fundamental=self.fundamental / s,
            chart=self.chart / s,
            market=self.market / s,
        )


# ============================================================
# 4. 추천 설정
# ============================================================
@dataclass
class RecommendConfig:
    """매년 상위 20개 분석 → 그중 10개 압축 추천 (당신 요구)."""
    analyze_top_n: int = 20
    final_top_n: int = 10


# ============================================================
# 5. 모델 하이퍼파라미터
# ============================================================
@dataclass
class ModelConfig:
    seq_len: int = 60          # 입력 시퀀스 길이
    horizon: int = 126         # 예측 기간 (≈6개월)

    fundamental_dim: int = 4   # ROE, 매출성장률, 순이익성장률, PER
    chart_dim: int = 8         # RSI, 볼린저폭, MA이격, 변동성, 거래량z, MACD, above_ma60, ma60_slope
    market_dim: int = 3        # 금리, M2증감, KOSPI

    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True

    cls_loss_weight: float = 0.3   # 분류 보조 손실 비중


# ============================================================
# 6. 학습 설정 (M5 MPS 기준)
# ============================================================
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # 시간 분할 (룩어헤드 방지)
    train_end: str = "2018-12-31"
    val_end: str = "2021-12-31"

    sample_stride: int = 5     # 5거래일마다 한 샘플


# ============================================================
# 7. 백테스트
# ============================================================
@dataclass
class BacktestConfig:
    start_year: int = 2010
    end_year: int = 2024
    txn_cost: float = 0.0025   # 단방향 0.25%
    benchmark_ticker: str = "069500"   # KODEX 200 ETF


# ============================================================
# 8. 데이터 수집
# ============================================================
@dataclass
class DataConfig:
    ohlcv_start: str = "2000-01-01"
    pykrx_request_sleep: float = 0.15
    pykrx_retry: int = 3


# ============================================================
# 통합 설정
# ============================================================
@dataclass
class Config:
    hard_filter: HardFilter = field(default_factory=HardFilter)
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    recommend: RecommendConfig = field(default_factory=RecommendConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    data: DataConfig = field(default_factory=DataConfig)


CFG = Config()


# ============================================================
# 디바이스 자동 선택 (맥북 M-시리즈 → MPS)
# ============================================================
def get_device() -> str:
    """맥북 에어 M5 → 'mps' (Metal 가속). CUDA 있으면 'cuda', 없으면 'cpu'."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


if __name__ == "__main__":
    cfg = Config()
    print("=" * 50)
    print("Stock AI 설정 검토")
    print("=" * 50)
    print("[하드 필터]")
    print(f"  ROE ≥ {cfg.hard_filter.roe_min}%")
    print(f"  PER ≤ {cfg.hard_filter.per_max}배")
    print(f"  시총 ≥ {cfg.hard_filter.market_cap_min_krw/1e12:.1f}조원")
    print(f"  매출/순이익 성장: {cfg.hard_filter.growth_period.upper()}")
    print("\n[가중치 (6:2:2)]")
    w = cfg.weights.normalize()
    print(f"  펀더멘털: {w.fundamental*100:.0f}%")
    print(f"  차트:    {w.chart*100:.0f}%")
    print(f"  시장:    {w.market*100:.0f}%")
    print(f"\n[추천] 상위 {cfg.recommend.analyze_top_n}개 → 최종 {cfg.recommend.final_top_n}개")
    print(f"[디바이스] {get_device()}")
