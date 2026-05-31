"""
Stock AI — 통합 CLI 엔트리포인트
=================================
사용법:
    python main.py collect              # PYKRX OHLCV + 시총 수집 (전체, 수 시간)
    python main.py update               # 어제까지 증분 업데이트 (1-2분, 권장)
    python main.py collect-dart         # DART 펀더멘털 수집 (API 키 필요)
    python main.py features             # 학습 데이터셋 생성 테스트
    python main.py train                # LSTM 학습
    python main.py recommend            # 오늘자 추천
    python main.py backtest             # 과거 검증
    python main.py dashboard            # Streamlit GUI
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# .env 파일 자동 로드 (KRX_ID, KRX_PW, DART_API_KEY 등)
try:
    from dotenv import load_dotenv
    env_file = ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # python-dotenv 없으면 OS 환경변수만 사용

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# pykrx 라이브러리가 root logger에 시끄럽게 쓰는 메시지들 억제
logging.getLogger("pykrx").setLevel(logging.WARNING)


def _check_krx_credentials():
    """KRX OpenAPI 인증키 확인. 없으면 명확한 안내."""
    if os.environ.get("KRX_AUTH_KEY"):
        return True
    # 레거시: KRX_ID/PW 도 허용 (단, pykrx 사이트 로그인은 현재 게이트로 불안정)
    if os.environ.get("KRX_ID") and os.environ.get("KRX_PW"):
        return True
    print("\n" + "!" * 70)
    print("⚠ KRX 인증 정보가 없습니다")
    print("!" * 70)
    print("해결 (권장 — KRX OpenAPI 인증키):")
    print("  1. https://openapi.krx.co.kr 접속 → 로그인 → 인증키 신청")
    print("  2. MyPage 에서 사용할 API '활용신청' → 승인 (보통 1일 내, 이메일 통지)")
    print("     · 유가증권 일별매매정보 (sto/stk_bydd_trd)")
    print("     · 코스닥 일별매매정보 (sto/ksq_bydd_trd)")
    print("  3. .env 파일에 추가:  KRX_AUTH_KEY=발급받은_인증키")
    print("!" * 70 + "\n")
    return False


def _filter_pykrx_noise(record: logging.LogRecord) -> bool:
    """pykrx가 root logger로 쏟아내는 노이즈 필터링."""
    if record.name != "root":
        return True
    msg = str(record.msg) if not isinstance(record.msg, str) else record.msg
    # pykrx의 알려진 노이즈 패턴
    if "Expecting value" in msg or isinstance(record.msg, tuple):
        return False
    return True


for handler in logging.root.handlers:
    handler.addFilter(_filter_pykrx_noise)


# ============================================================
# 1. PYKRX 데이터 수집
# ============================================================
def cmd_collect(args: argparse.Namespace) -> None:
    """KRX OpenAPI 로 OHLCV + 시총 + 유니버스를 날짜-bulk 로 수집 (cache.db 생성).

    --legacy-pykrx 지정 시 옛 pykrx 종목별 수집 경로 사용 (사이트 로그인 필요).
    """
    if not _check_krx_credentials():
        return

    if getattr(args, "legacy_pykrx", False):
        from src.data.pykrx_loader import PyKrxLoader
        loader = PyKrxLoader()
        print(">>> [1/2] 종목 유니버스 갱신 (KOSPI + KOSDAQ)")
        loader.update_universe(anchor=args.end_date)
        print(">>> [2/2] OHLCV + 시총 수집 (수 시간 소요 가능)")
        stats = loader.collect_all(start=args.start, end=args.end_date)
        print(f"\n완료: {stats}")
        return

    from src.data.krx_openapi_loader import KrxOpenApiLoader
    from src.data.daily_update import _backfill_etfs

    loader = KrxOpenApiLoader()
    print(">>> [1/2] 주식 OHLCV + 시총 + 유니버스 수집 (KRX OpenAPI, 날짜-bulk)")
    res = loader.collect_range(args.start, args.end_date)
    if not res.get("ok"):
        err = res.get("error", "")
        print(f"\n⚠ 수집 중단: {err}")
        if "401" in str(err) or "승인" in str(err):
            print("\n" + "─" * 64)
            print("  아직 KRX OpenAPI 승인 전입니다. 두 가지 방법:")
            print("  1) 승인 상태 확인:  python check_krx_api.py")
            print("  2) 승인 전 임시 데이터로 먼저 돌려보기:")
            print("       python main.py seed          # 하드코딩 우량주 26개 (KRX 인증 불필요)")
            print("       python main.py collect-dart  # 재무 수집")
            print("       python main.py recommend     # 추천 / dashboard")
            print("  → 승인되면 그때 python main.py collect 로 전종목 교체")
            print("─" * 64)
        return
    print(f"    주식: {res['days_added']}거래일 · OHLCV {res['ohlcv_rows']:,}행 · "
          f"시총 {res['cap_rows']:,}행 · API {res.get('api_calls')}회")
    print(">>> [2/2] 벤치마크/금헤지 ETF 수집 (pykrx 종목별)")
    etf = _backfill_etfs(res.get("from"), res.get("to"))
    print(f"    ETF: {etf:,}행 ({', '.join(['069500', '132030'])})")
    print(f"\n완료: {res['from']} ~ {res['to']} ({res['elapsed_sec']:.0f}초)")


# ============================================================
# 1-B. 어제까지 증분 업데이트 (대시보드용 빠른 갱신)
# ============================================================
def cmd_update(args: argparse.Namespace) -> None:
    """DB 의 마지막 거래일 다음 날부터 어제(또는 --until)까지 일별 일괄 적재.
    `collect` 가 종목별 호출이라 수 시간 걸리는 것과 달리, 이 명령은
    pykrx 의 `get_market_ohlcv_by_ticker(date)` 일자별 bulk API 로 분 단위에 마무리.
    """
    from datetime import datetime as _dt
    from src.data.daily_update import status_summary, update_to

    if not _check_krx_credentials():
        return

    s = status_summary()
    print(f"DB OHLCV 마지막: {s['ohlcv_max'] or '없음'}  →  목표: {s['target']}  "
          f"({s['days_behind']}일 뒤처짐)")

    target = None
    if args.until:
        try:
            target = _dt.strptime(args.until, "%Y-%m-%d").date()
        except ValueError:
            print(f"⚠ --until 형식 오류: '{args.until}' (예: 2026-05-18)")
            return

    if not s["is_stale"] and target is None:
        print("✓ 이미 최신 상태 — 업데이트 불필요")
        return

    print(">>> 증분 적재 시작 (일별 bulk API)")
    r = update_to(target_date=target)
    if not r["ok"]:
        print(f"⚠ 실패: {r['error']}")
        return

    print(f"\n완료: {r['from']} → {r['to']}")
    print(f"  거래일 적재: {r['days_added']}일 (전체 {r['days_processed']}일 중)")
    print(f"  OHLCV {r['ohlcv_rows']:,}행 · 시총 {r['cap_rows']:,}행")
    if r["skipped"]:
        print(f"  휴장일 스킵: {len(r['skipped'])}일")
    print(f"  소요: {r['elapsed_sec']:.1f}초")


# ============================================================
# 2. DART 펀더멘털 수집
# ============================================================
def cmd_collect_dart(args: argparse.Namespace) -> None:
    try:
        from src.data.dart_loader import DartLoader
    except ImportError as e:
        print(f"DART 모듈 임포트 실패: {e}")
        return

    try:
        dart = DartLoader()
    except ValueError as e:
        print(f"\n⚠ {e}\n")
        print("발급 후 .env 파일에 DART_API_KEY=발급받은_키 설정")
        return

    print(">>> [1/3] DART 기업 코드 매핑 다운로드")
    dart.update_corp_codes()

    print(">>> [2/3] 종목별 재무제표 수집")
    import sqlite3
    from src.config import DB_PATH, CFG
    cap_min = args.market_cap_min if args.market_cap_min is not None else CFG.hard_filter.market_cap_min_krw
    with sqlite3.connect(DB_PATH) as conn:
        tickers = [r[0] for r in conn.execute(f"""
            SELECT DISTINCT ticker FROM market_cap m
            WHERE m.market_cap >= {cap_min}
              AND m.date = (SELECT MAX(date) FROM market_cap WHERE ticker=m.ticker)
        """).fetchall()]

    print(f"    대상: {len(tickers)}개 종목 (시총 {cap_min/1e8:.0f}억+, DART API는 종목당 ~3초)")

    for i, tk in enumerate(tickers, 1):
        try:
            n = dart.collect_ticker(tk, start_year=args.start_year)
            if i % 20 == 0:
                print(f"    [{i}/{len(tickers)}] {tk}: 신규 {n}분기")
        except Exception as e:
            logging.warning("종목 %s 실패: %s", tk, e)

    print(">>> [3/3] 펀더멘털 지표 계산 (ROE, 성장률 등)")
    dart.compute_fundamentals()
    print("✓ DART 수집 완료")


# ============================================================
# 2-B. 시드 수집 (KRX OpenAPI 승인 전 임시 — 하드코딩 소수 종목)
# ============================================================
def cmd_seed(args: argparse.Namespace) -> None:
    """KRX OpenAPI 승인 전, 하드코딩 우량주만으로 cache.db 적재 (KRX 인증 불필요).

    OHLCV=pykrx 종목별, 상장주식수=yfinance, 시총=종가×주식수. 이후 collect-dart 연결 가능.
    """
    from src.data.seed_loader import SEED_UNIVERSE, SeedLoader, _yesterday
    from src.data.daily_update import _backfill_etfs

    end = args.end_date or _yesterday().strftime("%Y-%m-%d")
    n_uni = len(SEED_UNIVERSE)
    print(f">>> [1/2] 시드 종목 OHLCV + 시총 적재 ({n_uni}종목, KRX 인증 불필요)")
    print(f"    소스: pykrx OHLCV + yfinance 상장주식수 · {args.start} ~ {end}")
    loader = SeedLoader()
    stats = loader.collect(start=args.start, end=end)
    print(f"    종목 {stats['tickers']}/{n_uni} · OHLCV {stats['ohlcv_rows']:,}행 · "
          f"시총 {stats['cap_rows']:,}행")
    if stats["no_shares"]:
        print(f"    ⚠ 상장주식수 미확보(시총 없음): {', '.join(stats['no_shares'])}")
    if stats["failed"]:
        print(f"    ⚠ 수집 실패: {', '.join(stats['failed'])}")

    print(">>> [2/2] 벤치마크/금헤지 ETF 적재 (069500, 132030)")
    etf = _backfill_etfs(args.start, end)
    print(f"    ETF: {etf:,}행")

    print(f"\n완료 ({stats['elapsed_sec']:.0f}초). 다음 단계:")
    print("    python main.py collect-dart      # 재무 수집 (시드 유니버스 대상)")
    print("    python main.py recommend         # 추천 실행")
    print("\n  ※ KRX OpenAPI 승인되면: python main.py collect 로 전종목 교체")


# ============================================================
# 3. 미국 주식 OHLCV 수집 (S&P500 + Nasdaq100)
# ============================================================
def cmd_collect_us(args: argparse.Namespace) -> None:
    try:
        from src.data.yfinance_loader import YFinanceLoader
    except ImportError as e:
        print(f"⚠ {e}")
        return

    loader = YFinanceLoader()
    print(f">>> 미국 주식 수집 시작 ({args.start} ~ {args.end or '오늘'})")
    print("    S&P500 + Nasdaq100 약 500종목 (yfinance, 무료)")
    result = loader.collect_all(
        start=args.start,
        end=args.end,
        sp500=not args.no_sp500,
        nasdaq100=not args.no_nasdaq100,
    )
    print(f"\n✓ 완료: {result}")


# ============================================================
# 3. 피처 데이터셋 생성 테스트
# ============================================================
def cmd_features(args: argparse.Namespace) -> None:
    from src.data.feature_engineer import FeatureEngineer

    fe = FeatureEngineer()
    ds = fe.build_dataset(start=args.start)
    print(f"\n생성된 데이터셋:")
    print(f"  fund:   {ds['fund'].shape}")
    print(f"  chart:  {ds['chart'].shape}")
    print(f"  market: {ds['market'].shape}")
    print(f"  y:      {ds['y'].shape}")
    print(f"  종목 수: {ds['meta']['ticker'].nunique()}")
    print(f"  기간:   {ds['meta']['decision_date'].min()} ~ {ds['meta']['decision_date'].max()}")


# ============================================================
# 4. LSTM 학습
# ============================================================
def cmd_train(args: argparse.Namespace) -> None:
    from src.data.feature_engineer import FeatureEngineer
    from src.model.trainer import train

    print(">>> 데이터셋 생성")
    fe = FeatureEngineer()
    ds = fe.build_dataset(start=args.start)

    print(">>> 학습 시작 (M5 MPS: 약 30~60분)")
    result = train(ds)
    print(f"\n✅ 베스트 검증 IC: {result['best_val_ic']:.4f}")
    print(f"✅ 테스트 성능: {result['test_metrics']}")


def cmd_train_chart(args: argparse.Namespace) -> None:
    """차트 전용 LSTM (한국+미국 통합 학습)."""
    from src.model.chart_trainer import train_chart_model
    from src.config import MODEL_DIR

    print(">>> 차트 전용 LSTM 학습 (한국 + 미국, M5 MPS)")
    print(f"    epochs={args.epochs}, batch_size={args.batch_size}, seq_len={args.seq_len}")
    if args.end_date:
        print(f"    ★ 학습 cutoff: {args.end_date} (walk-forward)")
    print(f"    저장: data/models/{args.save_name}")

    save_path = MODEL_DIR / args.save_name
    result = train_chart_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        end_date=args.end_date,
        save_path=save_path,
    )
    print(f"\n✅ 베스트 검증 IC: {result['best_val_ic']:.4f}")
    print(f"✅ 테스트 IC: {result['test_ic']:.4f}")
    print(f"✅ 방향성 정확도: {result['test_dir_acc']:.3f}")
    print(f"✅ 상위-하위 스프레드: {result['spread_pct']:.2f}%")


# ============================================================
# 5. 추천
# ============================================================
def cmd_recommend(args: argparse.Namespace) -> None:
    from src.config import CFG
    from src.recommend.recommender import Recommender, format_table

    # CLI 인자로 임시 override (당신이 빠르게 다른 조건 시험할 때)
    if args.roe_min is not None:
        CFG.hard_filter.roe_min = args.roe_min
    if args.per_max is not None:
        CFG.hard_filter.per_max = args.per_max
    if args.top_n is not None:
        CFG.recommend.final_top_n = args.top_n

    rec = Recommender()
    result = rec.recommend()

    if result.empty:
        print("추천 종목 없음 (하드 필터 통과 종목 없음)")
        print("→ ROE/PER 조건을 완화하거나 데이터 수집을 먼저 확인하세요.")
        return

    print()
    print("=" * 80)
    print(f"🤖 AI 추천 종목 — 최종 {len(result)}개")
    print("=" * 80)
    print(format_table(result))


# ============================================================
# 6. 6개월 리밸런싱 백테스트 (★ 핵심)
# ============================================================
def cmd_backtest(args: argparse.Namespace) -> None:
    """6개월마다 리밸런싱하는 백테스트.
    --strategy로 규칙(rule) / AI(ai) / 둘 다(both) 선택.
    """
    from src.backtest.rebalance import run_rule_based_backtest

    if args.strategy in ("rule", "both"):
        print("\n" + "="*70)
        print("📊 규칙 기반 백테스트 (PER↓ + ROE↑ + 매출/순이익 성장↑)")
        print("="*70)
        result_rule = run_rule_based_backtest(
            start_year=args.start_year,
            end_year=args.end_year,
            top_n=args.top_n,
            weight_scheme=args.weight,
            replacement_rule=args.replacement,
            score_diff_pct=args.score_diff_pct,
            market_split=args.market_split,
            trend_filter=args.trend_filter,
            period_months=args.period_months,
            trend_stop_loss=args.trend_stop_loss,
            trend_bonus=args.trend_bonus,
            market_cap_min=args.market_cap_min,
            market_cap_percentile=args.market_cap_percentile,
            market_ratio=args.market_ratio,
        )
        print(result_rule.summary())
        if not result_rule.yearly.empty:
            print("\n[연도별 수익률]")
            yr = result_rule.yearly.copy()
            for c in ("portfolio_return", "benchmark_return", "alpha"):
                yr[c] = (yr[c] * 100).round(2).astype(str) + "%"
            print(yr.to_string(index=False))

    if args.strategy == "ensemble":
        # 앙상블: 규칙 + LGBM Hybrid (v3+v4) — README production. 동봉 모델, 학습 불필요.
        from src.backtest.rebalance import run_rule_based_backtest
        print("\n" + "="*70)
        print(f"🤖 LGBM Hybrid 앙상블 백테스트 (규칙 {(1-args.ai_weight)*100:.0f}% + AI {args.ai_weight*100:.0f}%)")
        print("="*70)
        try:
            result = run_rule_based_backtest(
                start_year=args.start_year,
                end_year=args.end_year,
                top_n=args.top_n,
                weight_scheme=args.weight,
                replacement_rule=args.replacement,
                score_diff_pct=args.score_diff_pct,
                market_split=args.market_split,
                trend_filter=args.trend_filter,
                period_months=args.period_months,
                trend_stop_loss=args.trend_stop_loss,
                trend_bonus=args.trend_bonus,
                market_cap_min=args.market_cap_min,
                market_cap_percentile=args.market_cap_percentile,
                market_ratio=args.market_ratio,
                ai_weight=args.ai_weight,        # ★ LGBM 앙상블 활성화
            )
            print(result.summary())
            if not result.yearly.empty:
                print("\n[연도별 수익률]")
                yr = result.yearly.copy()
                for c in ("portfolio_return", "benchmark_return", "alpha"):
                    yr[c] = (yr[c] * 100).round(2).astype(str) + "%"
                print(yr.to_string(index=False))
        except FileNotFoundError as e:
            print(f"\n⚠ LGBM 모델 없음: {e}")
            print("  → data/models/trend_lgbm_v3.txt · v4.txt 확인 (동봉됨)")
        return

    if args.strategy == "chart-lstm":
        # (레거시) 차트 ChartLSTM 앙상블 — train-chart 로 학습된 chart_lstm.pt 필요
        from src.recommend.ensemble import run_ensemble_backtest
        print("\n" + "="*70)
        print(f"🤖 (레거시) ChartLSTM 앙상블 (규칙 {(1-args.ai_weight)*100:.0f}% + 차트 AI {args.ai_weight*100:.0f}%)")
        if args.model_path:
            print(f"   모델: {args.model_path}")
        print("="*70)
        try:
            model_path = Path(args.model_path) if args.model_path else None
            result = run_ensemble_backtest(
                ai_weight=args.ai_weight,
                start_year=args.start_year,
                end_year=args.end_year,
                top_n=args.top_n,
                replacement_rule=args.replacement,
                market_split=args.market_split,
                trend_filter=args.trend_filter,
                period_months=args.period_months,
                market_cap_min=args.market_cap_min,
                trend_bonus=args.trend_bonus,
                model_path=model_path,
                market_ratio=args.market_ratio,
            )
            print(result.summary())
            if not result.yearly.empty:
                print("\n[연도별 수익률]")
                yr = result.yearly.copy()
                for c in ("portfolio_return", "benchmark_return", "alpha"):
                    yr[c] = (yr[c] * 100).round(2).astype(str) + "%"
                print(yr.to_string(index=False))
        except FileNotFoundError as e:
            print(f"\n⚠ {e}")
            print("  → 먼저 `python main.py train-chart`로 차트 모델을 학습하세요.")
        return

    if args.strategy in ("ai", "both"):
        # AI 백테스트는 학습된 모델이 있을 때만
        from src.config import MODEL_PATH
        if not MODEL_PATH.exists():
            print(f"\n⚠ AI 모델이 없습니다: {MODEL_PATH}")
            print("  → 먼저 `python main.py train`으로 모델을 학습하세요.")
        else:
            print("\n" + "="*70)
            print("🤖 AI(LSTM) 기반 백테스트")
            print("="*70)
            from src.backtest.rebalance import RebalanceBacktest
            from src.recommend.recommender import Recommender
            rec = Recommender()

            def ai_picker(as_of: str, n: int):
                return rec.recommend(as_of=as_of)

            result_ai = RebalanceBacktest().run(
                picker=ai_picker,
                start_year=args.start_year,
                end_year=args.end_year,
                top_n=args.top_n,
                weight_scheme=args.weight,
            )
            print(result_ai.summary())
            if not result_ai.yearly.empty:
                print("\n[연도별 수익률]")
                yr = result_ai.yearly.copy()
                for c in ("portfolio_return", "benchmark_return", "alpha"):
                    yr[c] = (yr[c] * 100).round(2).astype(str) + "%"
                print(yr.to_string(index=False))


# ============================================================
# 6-2. 한 시점 종목 선정 미리보기
# ============================================================
def cmd_screen(args: argparse.Namespace) -> None:
    """특정 날짜 기준으로 규칙 기반 스크리너가 어떤 종목을 고르는지 확인."""
    import pandas as pd
    from src.screener.rule_based import RuleBasedScreener

    s = RuleBasedScreener()
    picks = s.select_top_n(as_of=args.date, top_n=args.top_n)
    if picks.empty:
        print(f"[{args.date}] 조건 통과 종목 없음")
        return

    print(f"\n[{args.date}] 규칙 기반 상위 {len(picks)}개 종목:")
    cols = [c for c in [
        "ticker", "name", "rule_score",
        "per_score", "roe_score", "revenue_score", "profit_score",
        "roe", "per", "revenue_growth_yoy", "profit_growth_yoy",
    ] if c in picks.columns]

    fmt = picks[cols].copy()
    # 점수 컬럼 (PER 비활성 시 per_score는 None일 수 있음 → numeric 변환 후 round)
    for c in ("rule_score", "per_score", "roe_score", "revenue_score", "profit_score"):
        if c in fmt.columns:
            fmt[c] = pd.to_numeric(fmt[c], errors="coerce").round(1)
    # 퍼센트 컬럼
    for c in ("roe", "revenue_growth_yoy", "profit_growth_yoy"):
        if c in fmt.columns:
            num = pd.to_numeric(fmt[c], errors="coerce").round(1)
            fmt[c] = num.where(num.isna(), num.astype(str) + "%").fillna("-")
    if "per" in fmt.columns:
        fmt["per"] = pd.to_numeric(fmt["per"], errors="coerce").round(1)
    print(fmt.to_string(index=False))


# ============================================================
# 6-3. 단일 종목 멀티모달 분석 (★ 멀티모달 병합)
# ============================================================
def cmd_analyze(args: argparse.Namespace) -> None:
    """단일 종목을 5개 모달리티(차트·펀더·밸류·뉴스·매크로)로 분석·융합.

    데이터/키가 없는 모달리티는 자동 제외되고 나머지로 진행한다.
    """
    from src.modality.analyzer import MultimodalAnalyzer

    ticker = args.ticker
    name = args.name
    if not name:
        # DB 또는 config 에서 종목명 조회 시도
        try:
            import sqlite3
            from src.config import DB_PATH
            with sqlite3.connect(DB_PATH) as c:
                row = c.execute("SELECT name FROM tickers WHERE ticker=?", (ticker,)).fetchone()
                name = row[0] if row else ticker
        except Exception:
            name = ticker

    analyzer = MultimodalAnalyzer(ai_weight=args.ai_weight)
    out = analyzer.analyze(ticker, name, as_of=args.date)
    print("\n" + out["report"])

    fusion = out["fusion"]
    if fusion.skipped:
        print(f"\n(데이터/키 부재로 제외된 모달리티: {', '.join(fusion.skipped)})")
    print(f"기여도: " + ", ".join(f"{k} {v*100:.0f}%" for k, v in fusion.contributions.items()))


# ============================================================
# 7. 대시보드
# ============================================================
def cmd_train_fusion(args: argparse.Namespace) -> None:
    """학습된 Late Fusion 게이트 학습 (옵션). 기본 가중평균 fusion 은 그대로 유지."""
    from src.modality.fusion_dataset import build_samples, month_grid
    from src.modality.learned_fusion import LearnedFusion, DEFAULT_GATE_PATH

    dates = month_grid(args.start, args.end, day=args.day)
    print(f">>> [1/2] 학습 샘플 생성 ({len(dates)}개 시점, horizon {args.horizon}거래일)")
    print("    모달: chart(AI)·fundamental(Rule)·valuation·macro (뉴스 제외 — 과거 재구성 불가)")
    X, M, S, Y = build_samples(dates, horizon=args.horizon, ai_weight=args.ai_weight,
                               market_cap_min=args.market_cap_min)
    print(f"    샘플 {len(Y)}개")
    if len(Y) < 8:
        print("\n⚠ 샘플 부족 — 데이터가 더 필요합니다.")
        print("  · 승인 전이면 `python main.py collect-us` 로 미국 데이터를 늘리거나,")
        print("  · KRX 승인 후 `python main.py collect` 로 전종목 적재 후 재시도하세요.")
        return

    print(">>> [2/2] 게이트 신경망 학습")
    lf = LearnedFusion()
    metrics = lf.fit(X, M, S, Y, epochs=args.epochs)
    path = lf.save()
    print(f"    {metrics}")
    print(f"\n완료 → {path}")
    print("  이제 멀티모달 분석/대시보드에서 fusion_mode='learned'(또는 auto) 로 사용됩니다.")


def cmd_dashboard(args: argparse.Namespace) -> None:
    # `streamlit` 이 PATH 에 없을 수 있으므로 현재 파이썬의 -m 모듈 실행으로 호출.
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run",
         str(ROOT / "src" / "ui" / "dashboard.py")],
        check=False,
    )


# ============================================================
# CLI parser
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stock AI — 펀더멘털 + LSTM 한국 주식 추천")
    sub = p.add_subparsers(dest="command", required=True)

    # collect
    pc = sub.add_parser("collect", help="OHLCV + 시총 + 유니버스 수집 (KRX OpenAPI)")
    pc.add_argument("--start", default="2015-01-01",
                    help="수집 시작일 (기본: 2015-01-01; KRX OpenAPI 는 ~2010년+ 제공)")
    pc.add_argument("--end-date", default=None,
                    help="수집 종료일 (기본: 어제). 예: 2026-05-28")
    pc.add_argument("--legacy-pykrx", action="store_true",
                    help="옛 pykrx 종목별 수집 경로 사용 (KRX 사이트 로그인 필요, 느림)")
    pc.set_defaults(func=cmd_collect)

    # seed (KRX OpenAPI 승인 전 임시 — 하드코딩 소수 우량주)
    ps = sub.add_parser("seed",
                        help="승인 전 임시: 하드코딩 우량주만 적재 (KRX 인증 불필요)")
    ps.add_argument("--start", default="2015-01-01", help="수집 시작일 (기본: 2015-01-01)")
    ps.add_argument("--end-date", default=None, help="수집 종료일 (기본: 어제)")
    ps.set_defaults(func=cmd_seed)

    # update (증분 — 대시보드용 빠른 갱신)
    pu = sub.add_parser("update",
                        help="DB 마지막 날짜 → 어제까지 증분 적재 (일별 bulk, 분 단위)")
    pu.add_argument("--until", default=None,
                    help="목표일 (YYYY-MM-DD, 기본=어제). 예: 2026-05-18")
    pu.set_defaults(func=cmd_update)

    # collect-dart
    pcd = sub.add_parser("collect-dart", help="DART 재무제표 수집")
    pcd.add_argument("--start-year", type=int, default=2015)
    pcd.add_argument("--market-cap-min", type=float, default=None,
                     help="대상 종목 시총 하한 (원). 기본=config.py 값 (5천억). "
                          "예: 3e11 (3천억), 1e12 (1조)")
    pcd.set_defaults(func=cmd_collect_dart)

    # train-fusion (학습된 Late Fusion 게이트 — 옵션)
    ptf = sub.add_parser("train-fusion",
                         help="학습된 Late Fusion 게이트 학습 (옵션; 기본 가중평균은 유지)")
    ptf.add_argument("--start", default="2015-01-01", help="학습 시작일")
    ptf.add_argument("--end", default="2026-01-01", help="학습 종료일 (horizon 여유 두기)")
    ptf.add_argument("--day", type=int, default=9, help="매월 샘플 시점 (기본: 9일)")
    ptf.add_argument("--horizon", type=int, default=42, help="타깃 forward 거래일 (기본: 42)")
    ptf.add_argument("--ai-weight", type=float, default=0.2)
    ptf.add_argument("--market-cap-min", type=float, default=None)
    ptf.add_argument("--epochs", type=int, default=400)
    ptf.set_defaults(func=cmd_train_fusion)

    # collect-us (미국 주식 OHLCV)
    pcu = sub.add_parser("collect-us", help="미국 주식 OHLCV 수집 (S&P500 + Nasdaq100)")
    pcu.add_argument("--start", default="2015-01-01", help="시작 날짜")
    pcu.add_argument("--end", default=None, help="종료 날짜 (기본: 오늘)")
    pcu.add_argument("--no-sp500", action="store_true", help="S&P500 수집 제외")
    pcu.add_argument("--no-nasdaq100", action="store_true", help="Nasdaq100 수집 제외")
    pcu.set_defaults(func=cmd_collect_us)

    # features
    pf = sub.add_parser("features", help="학습 데이터셋 생성 테스트")
    pf.add_argument("--start", default="2015-01-01")
    pf.set_defaults(func=cmd_features)

    # train
    pt = sub.add_parser("train", help="LSTM 학습")
    pt.add_argument("--start", default="2015-01-01")
    pt.set_defaults(func=cmd_train)

    # train-chart (차트 전용 가벼운 모델)
    ptc = sub.add_parser("train-chart", help="차트 전용 LSTM 학습 (한국+미국)")
    ptc.add_argument("--epochs", type=int, default=25)
    ptc.add_argument("--batch-size", type=int, default=256)
    ptc.add_argument("--lr", type=float, default=1e-3)
    ptc.add_argument("--seq-len", type=int, default=60)
    ptc.add_argument("--end-date", default=None,
                     help="학습 데이터 cutoff (예: '2024-06-30'). 이후 데이터는 학습 제외 (walk-forward용)")
    ptc.add_argument("--save-name", default="chart_lstm.pt",
                     help="저장 파일명. 예: chart_lstm_walkforward.pt")
    ptc.set_defaults(func=cmd_train_chart)

    # recommend
    pr = sub.add_parser("recommend", help="AI 추천")
    pr.add_argument("--top-n", type=int, default=None)
    pr.add_argument("--roe-min", type=float, default=None,
                    help="ROE 최소 (기본: config.py의 30)")
    pr.add_argument("--per-max", type=float, default=None,
                    help="PER 최대 (기본: config.py의 50)")
    pr.set_defaults(func=cmd_recommend)

    # backtest (6개월 리밸런싱)
    pb = sub.add_parser("backtest", help="6개월 리밸런싱 백테스트")
    pb.add_argument("--start-year", type=int, default=2015,
                    help="시작 연도 (기본: 2015)")
    pb.add_argument("--end-year", type=int, default=2024,
                    help="종료 연도 (기본: 2024)")
    pb.add_argument("--top-n", type=int, default=10,
                    help="매 리밸런싱 시점의 보유 종목 수 (기본: 10)")
    pb.add_argument("--strategy", choices=["rule", "ai", "both", "ensemble", "chart-lstm"],
                    default="rule",
                    help="rule=규칙 / ensemble=규칙+LGBM Hybrid(v3+v4, 동봉) / "
                         "chart-lstm=레거시 ChartLSTM(학습필요) / ai,both=멀티모달 .pth")
    pb.add_argument("--ai-weight", type=float, default=0.2,
                    help="앙상블에서 AI 비중 (0.0~1.0). 예: 0.2=80:20, 0.3=70:30")
    pb.add_argument("--model-path", default=None,
                    help="앙상블 백테스트용 차트 모델 경로 (기본: data/models/chart_lstm.pt)")
    pb.add_argument("--weight", choices=["rank", "equal", "score"], default="rank",
                    help="포트폴리오 비중 방식 (기본: rank — 1위가 가장 많이)")
    pb.add_argument("--replacement",
                    choices=["always", "keep_simple", "score_diff", "three_cond"],
                    default="always",
                    help=("종목 교체 규칙: always=매번 새로(기본) | "
                          "keep_simple=기존 통과 시 유지 | "
                          "score_diff=점수 차이 클 때만 교체 | "
                          "three_cond=시총↑+PER↓+ROE↑ 모두 만족 시만 교체"))
    pb.add_argument("--score-diff-pct", type=float, default=15.0,
                    help="score_diff 모드에서 교체 임계 퍼센트 (기본 15)")
    pb.add_argument("--market-split", action="store_true",
                    help="코스피/코스닥 절반씩 고정 (예: top_n=10이면 코스피 5 + 코스닥 5)")
    pb.add_argument("--trend-filter", action="store_true",
                    help="60일선 위 종목만 통과 (사용자 의도: 정배열만)")
    pb.add_argument("--period-months", type=int, default=6,
                    help="리밸런싱 주기 개월 (6=반기 기본, 4=4개월, 3=분기)")
    pb.add_argument("--trend-stop-loss", action="store_true",
                    help="보유 중 매월 추세 점검 → 60일선 깬 종목 즉시 매도")
    pb.add_argument("--trend-bonus", type=float, default=0.0,
                    help="5/20/60 정배열 종목 점수 가산 (0.0~1.0). 예: 0.5 = 정배열은 50퍼센트 점수 가산")
    pb.add_argument("--market-cap-min", type=float, default=None,
                    help="시총 하한 (원). 예: 1e12=1조, 5e11=5천억, 3e11=3천억. "
                         "미지정시 config.py 값 사용 (5천억)")
    pb.add_argument("--market-cap-percentile", type=float, default=None,
                    help="시총 상위 N%% 만 통과 (0.0~1.0). 예: 0.30 = 상위 30퍼센트. "
                         "시간 가변 필터 — 시점마다 자동 조정")
    pb.add_argument("--market-ratio", default=None,
                    help="코스피:코스닥 비율 (예: '6:4', '7:3'). market-split보다 우선")
    pb.set_defaults(func=cmd_backtest)

    # screen (특정 시점 종목 미리보기)
    ps = sub.add_parser("screen", help="특정 날짜 기준 규칙 기반 종목 선정 미리보기")
    ps.add_argument("--date", default="2024-12-30",
                    help="기준 날짜 (예: 2020-01-02)")
    ps.add_argument("--top-n", type=int, default=10)
    ps.set_defaults(func=cmd_screen)

    # analyze (단일 종목 멀티모달 분석)
    pa = sub.add_parser("analyze", help="단일 종목 멀티모달 분석 (차트·펀더·밸류·뉴스·매크로 융합)")
    pa.add_argument("ticker", help="종목 코드 (예: 005930)")
    pa.add_argument("--name", default=None, help="종목명 (미지정 시 DB 에서 조회)")
    pa.add_argument("--date", default=None, help="기준일 YYYY-MM-DD (기본: 오늘)")
    pa.add_argument("--ai-weight", type=float, default=0.2,
                    help="차트 AI 점수 산출 시 앙상블 AI 비중 (기본 0.2)")
    pa.set_defaults(func=cmd_analyze)

    # dashboard
    pd_ = sub.add_parser("dashboard", help="Streamlit 대시보드")
    pd_.set_defaults(func=cmd_dashboard)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
