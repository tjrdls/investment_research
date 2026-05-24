#!/bin/bash
# Stock AI 전체 파이프라인 자동 실행
# 노트북 닫아도 돌아가고, 한 단계 끝나면 다음 단계로 자동 진행

set -e   # 에러 나면 즉시 중단

cd ~/Downloads/stock_ai
source .venv/bin/activate

LOG_DIR="$HOME/Downloads/stock_ai/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Stock AI 자동 실행 시작: $(date)"
echo "로그 위치: $LOG_DIR"
echo "=========================================="

# [1/3] OHLCV + 시총 수집 (4시간 예상)
echo ""
echo ">>> [1/3] OHLCV + 시총 수집 시작: $(date)"
python main.py collect --start 2015-01-01 --end-date 2024-12-30 \
    2>&1 | tee "$LOG_DIR/1_collect_$TS.log"
echo ">>> [1/3] 완료: $(date)"

# [2/3] DART 재무제표 수집 (1~2시간 예상)
echo ""
echo ">>> [2/3] DART 재무제표 수집 시작: $(date)"
python main.py collect-dart --start-year 2015 \
    2>&1 | tee "$LOG_DIR/2_dart_$TS.log"
echo ">>> [2/3] 완료: $(date)"

# [3/3] 한 시점 미리보기 + 백테스트
echo ""
echo ">>> [3/3] 종목 미리보기 + 백테스트 시작: $(date)"
python main.py screen --date 2024-12-30 --top-n 10 \
    2>&1 | tee "$LOG_DIR/3_screen_$TS.log"
python main.py backtest --start-year 2015 --end-year 2024 --strategy rule \
    2>&1 | tee "$LOG_DIR/4_backtest_$TS.log"
echo ">>> [3/3] 완료: $(date)"

echo ""
echo "=========================================="
echo "🎉 전체 파이프라인 완료: $(date)"
echo "=========================================="

# macOS 알림 (작업 끝나면 알림 띄움)
osascript -e 'display notification "Stock AI 백테스트 완료!" with title "Stock AI" sound name "Glass"'
