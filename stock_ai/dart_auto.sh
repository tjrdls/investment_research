#!/bin/bash
# DART 수집 자동 재시도 스크립트
# - 와이파이 끊김/중단 자동 감지 → 30초 대기 후 재시도
# - 이미 받은 데이터는 자동 스킵 (증분 수집)
# - 최대 10회 시도

cd ~/Downloads/stock_ai
source .venv/bin/activate

LOG="logs/expand/dart_auto.log"
mkdir -p logs/expand

MAX_TRIES=10
TRY=1

while [ $TRY -le $MAX_TRIES ]; do
    echo "" | tee -a "$LOG"
    echo "===== 시도 $TRY/$MAX_TRIES — $(date) =====" | tee -a "$LOG"

    # 인터넷 대기 (최대 5분)
    WAIT=0
    while ! ping -c 1 -t 5 8.8.8.8 > /dev/null 2>&1; do
        if [ $WAIT -ge 300 ]; then break; fi
        echo "  네트워크 대기... ($WAIT초)" | tee -a "$LOG"
        sleep 30
        WAIT=$((WAIT + 30))
    done

    # 진척률 확인
    BEFORE=$(sqlite3 data/cache.db "SELECT COUNT(DISTINCT ticker) FROM financials" 2>/dev/null)
    echo "  현재: $BEFORE 종목" | tee -a "$LOG"

    # 수집 실행
    python main.py collect-dart --start-year 2015 --market-cap-min 3e11 \
        2>&1 | tee -a "$LOG"

    # 결과 확인
    AFTER=$(sqlite3 data/cache.db "SELECT COUNT(DISTINCT ticker) FROM financials" 2>/dev/null)
    echo "  결과: $AFTER 종목 (+$((AFTER - BEFORE)))" | tee -a "$LOG"

    # 종료 코드 확인
    if grep -q "DART 수집 완료" "$LOG" 2>/dev/null; then
        echo "✅ 정상 완료" | tee -a "$LOG"
        osascript -e 'display notification "DART 수집 완료!" sound name "Glass"' 2>/dev/null
        break
    fi

    echo "  중단됨, 30초 후 재시도..." | tee -a "$LOG"
    TRY=$((TRY + 1))
    sleep 30
done

echo "" | tee -a "$LOG"
echo "===== 최종 종목 수 =====" | tee -a "$LOG"
sqlite3 data/cache.db "SELECT COUNT(DISTINCT ticker), COUNT(*) FROM financials" | tee -a "$LOG"
