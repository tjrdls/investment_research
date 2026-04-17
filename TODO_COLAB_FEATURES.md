# TODO: Colab 노트북 기능을 프로젝트에 적용하기

이 문서는 `Colab_시작하기의_사본.ipynb`에 있는 기능을 이 프로젝트에 반영하기 위한 작업 목록입니다.

## 1. 데이터 수집 및 종목 선정 강화

- [ ] `data_loader/price/data_collector.py`의 `get_top_stocks()`에 KRX 시총 조회 실패 처리 개선
  - 현재 `pykrx_stock.get_market_cap_by_ticker()` 예외 시 하드코딩 fallback은 있지만, 컬럼 오류나 빈 응답에 대한 방어가 부족함.
  - KRX API 실패 시 `KRX_ID`/`KRX_PW` 없이도 동작하는 안정적 조회 로직 추가.
- [ ] KOSPI 상위 종목 조회 시 최근 7일 종가 확인 로직 추가
  - `pykrx_stock.get_market_ohlcv()`으로 정상 종가가 있는 종목만 결과에 포함.
- [ ] `collect_price_data()`에 `period` 외에 명시적 `start_date`/`end_date` 동적 설정 로직 보강
  - 노트북에서는 5년 `START_DATE`/`END_DATE` 동적 계산을 사용함.

## 2. 기술 지표 확장

- [ ] `analysis/indicators/technical_indicators.py`에 추가 지표 도입
  - `bb_width_zscore` (볼린저 밴드 폭 z-score)
  - `ichi_chikou_diff` (후행스팬과 26일 전 가격 차이)
  - `ichi_cloud_thick`, `ichi_cloud_bull` (일목 구름 두께 및 색)
  - `volume_ratio` (거래량 / 20일 평균 거래량)
- [ ] 기존 지표 계산을 `calculate_indicators()`에서 일관되게 지원하도록 확장
- [ ] `get_technical_signals()`에 고급 시그널 추가
  - 볼린저 스퀴즈 감지 + 돌파/하락 전환
  - 일목균형표 3일 연속 구름 위/아래 확인
  - 후행스팬, 구름 색, 구름 두께 기반 신호

## 3. 재무 데이터 및 밸류에이션 강화

- [ ] `data_loader/financial/financial_collector.py`에서 DART 기업 코드, 공시 검색, 분기/반기/연간 보고서 분류 기능 검토
  - 노트북의 `get_corp_code_map()`, `find_available_reports()`, `get_financial_statements()` 방식과 비교하여 누락 요소 보완.
- [ ] `analysis/valuation_analyzer.py`에 업종 기반 GPT 밸류에이션 출력 확장
  - 업종 평균 PER/PBR/ROE, `valuation_status`, `correction_per`, `comment` 추가.
- [ ] `pipeline/prediction_pipeline.py`에서 밸류 분석 단계에 업종 정보를 전달하는 옵션 추가.

## 4. 뉴스 분석 강화

- [ ] `analysis/news_analyzer.py`에 글로벌/매크로 뉴스 수집 쿼리 추가
  - 영어/한국어 뉴스 쿼리 세트와 `NewsAPI` 호출 확장
- [ ] 뉴스 중복 제거 로직 추가
- [ ] GPT 분석 결과에 `bullish_prob`, `bearish_prob`, `caution_prob` 출력 추가
- [ ] 뉴스 분석 요약 및 추천 메시지 구조 강화

## 5. 멀티모달 LSTM 모델 학습 파이프라인

- [ ] `models/lstm/lstm_model.py` `prepare_dataset()`에 더 많은 피처 허용
  - Notebook의 `FEATURE_COLS` 목록을 참고하여 모델 입력 변수 확장.
- [ ] 동적 임계값 `threshold`를 종목별로 적용하는 라벨링 로직 추가
  - 현재는 고정 `THRESHOLD = 0.02`만 사용함.
- [ ] `train_lstm.py`에 학습/검증/테스트 분리, 클래스 불균형 보정, 스케줄러, 얼리스톱핑 추가
- [ ] `pipeline/prediction_pipeline.py`에서 `text_embedding` 캐시 파일(`text_embeddings_cache.pkl`) 사용 옵션 도입
- [ ] 학습 결과 저장 및 플롯 저장 기능 추가

## 6. UI/세션 상태 및 결과 표시 개선

- [ ] `web/analysis/llm.py`에서 LSTM 확률을 0~1 비율에서 퍼센트로 변환하여 표시
  - 이미 문제를 수정했지만, UI 전체 경로 점검 필요.
- [ ] `web/ui/pipeline_view.py`에 `AI 종합 분석` 결과에 `bullish_prob`, `bearish_prob`, `caution_prob` 등 추가 출력
- [ ] 중간 결과(`technical`, `news`, `valuation`, `lstm_prediction`)를 세션 상태에 명확히 저장하고 후속 단계에서 재사용

## 7. 문서 및 설정 정리

- [ ] `README.md` / `README_IMPLEMENTATION.md`에 다음 항목 반영
  - `NEWSAPI_KEY`, `OPENAI_API_KEY`, DART API 키 설정 방법
  - KRX/pykrx 기반 시총 조회 예외 처리
  - 텍스트 임베딩 캐시 사용
  - 노트북 기반 학습 파이프라인 요약
- [ ] `TODO_COLAB_FEATURES.md`에 작업 진행 상태 기입

## 적용 우선순위 제안

1. UI 확률 표시 버그 수정 및 `technical`/`news`/`valuation` 세션 저장 확인
2. 기술 지표 확장 및 고급 시그널 추가
3. 뉴스 분석 GPT 출력 강화
4. 멀티모달 LSTM 학습 파이프라인 보강
5. 문서화 및 환경 변수 안내 보완
