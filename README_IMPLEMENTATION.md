# AI 주식 투자 분석 시스템

## 프로젝트 구조

```
investment_analysis/
├── main.py                          # 메인 진입점
├── requirements.txt                 # 의존성
├── .env                            # API 키 설정
│
├── data/                           # 데이터 수집
│   ├── price/
│   │   └── data_collector.py      # pykrx로 주가 수집
│   └── financial/
│       └── financial_collector.py # DART API로 재무 데이터 수집
│
├── analysis/                       # 데이터 분석
│   ├── indicators/
│   │   └── technical_indicators.py # 기술적 지표 (RSI, MACD, 일목 등)
│   ├── news_analyzer.py           # NewsAPI + GPT 뉴스 분석
│   └── valuation_analyzer.py      # TTM 밸류에이션 분석
│
├── models/                         # 기계학습 모델
│   ├── lstm/
│   │   └── lstm_model.py          # 멀티모달 LSTM (기술적 지표 + 텍스트)
│   └── llm/
│       └── llm_analyzer.py        # GPT-4o-mini 종합 분석
│
└── pipeline/
    └── prediction_pipeline.py     # 전체 파이프라인 통합
```

## 설정 및 설치

### 1. 필수 API 키 설정 (.env 파일)

```
OPENAI_API_KEY=sk-proj-...
DART_API_KEY=...
NEWSAPI_KEY=...
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

주요 라이브러리:
- **pykrx**: 한국 주식 데이터 (실시간 주가, 기업정보)
- **torch**: 멀티모달 LSTM 모델
- **openai**: GPT-4o-mini, Embedding API
- **requests**: DART API, NewsAPI 호출
- **pandas/numpy**: 데이터 처리

## 실행 방법

### 단일 종목 분석
```bash
python main.py
# 선택: 1 (단일 종목)
# 종목 코드 입력: 005930
```

### 상위 5개 종목 일괄 분석
```bash
python main.py
# 선택: 2
```

### 파이썬 코드에서 직접 호출
```python
from pipeline.prediction_pipeline import run_analysis

result = run_analysis("005930")  # 삼성전자
print(result['final_analysis'])
```

## 분석 흐름

각 종목마다 다음 8단계 분석 수행:

1. **주가 데이터 수집** (pykrx)
   - OHLCV 데이터 (최근 3년)
   
2. **기술적 지표 계산**
   - 이동평균 (5, 20, 50, 200일)
   - RSI, MACD, 볼린저밴드, 일목균형표
   - 변동성 지표
   
3. **재무 데이터 수집** (DART API)
   - 매출, 영업이익, 순이익
   - 자본, 부채
   - 분기/반기/연간 보고서
   
4. **텍스트 임베딩 생성**
   - 재무 정보를 OpenAI Embedding으로 벡터화 (1536차원)
   
5. **LSTM 예측**
   - 기술적 지표 시퀀스 → LSTM → 숨겨진 상태
   - 재무 임베딩 → Dense layer → 압축 벡터
   - 두 벡터 결합 → 3분류 예측 (상승/하락/횡보)
   
6. **뉴스 분석** (NewsAPI + GPT)
   - 종목 관련 뉴스 수집
   - 글로벌 매크로 뉴스 수집
   - GPT로 호재/악재 판정 및 확률 계산
   
7. **밸류에이션 분석**
   - TTM (Trailing Twelve Months) 지표 계산
   - PER, PBR, ROE, PSR, 부채비율
   
8. **최종 LLM 분석**
   - 모든 분석 결과 종합
   - 투자 의견 및 전략 생성
   - 리스크/기회 분석

## 출력 예시

```
╔══════════════════════════════════════════════════════════╗
  📊 삼성전자 (005930)
╚══════════════════════════════════════════════════════════╝

▸ 현재 상태:
  반도체 경기 회복 신호와 기술적 정렬로 단기 상승 기대

▸ 투자 의견: 매수 (신뢰도 72%)
  목표: +15% / 위험: -10%

▸ 주요 리스크:
  ⛔ 글로벌 반도체 공급 과잉
  ⛔ 미국 규제 리스크
  ⛔ 환율 변동성

▸ 주요 기회:
  ✅ AI 칩 수요 확대
  ✅ 메모리칩 가격 회복

▸ 투자 전략:
  3월 저항선 38,000원 돌파 시 매수, 
  반도체 지수 기준 익절 또는 손절 설정

▸ 주시사항:
  📌 3월 실적 발표 (변동성 확대 예상)
  📌 경쟁사 (TSMC) 실적 추이
```

## 핵심 기능

### ✨ 멀티모달 분석
- **기술적 지표** (LSTM)
- **재무 정보** (임베딩)
- **뉴스 감정** (GPT)
- **밸류에이션** (메트릭)

### 🤖 AI 기반 의사결정
- LSTM으로 패턴 인식
- GPT-4o-mini로 전문가 수준 분석
- Zero-shot 학습으로 신규 시장 상황 대응

### 📊 실시간 한국 주식 데이터
- pykrx (공식 거래소 API)
- DART (공식 공시)
- NewsAPI (한글 뉴스)

## 주요 코드 기능

### 기술적 지표 함수
```python
from analysis.indicators.technical_indicators import calculate_indicators

indicators_df = calculate_indicators(price_data)
signals, warnings, score = get_technical_signals(indicators_df)
```

### 재무 데이터 수집
```python
from data.financial.financial_collector import collect_financial_data

financial_df = collect_financial_data(stock_code, corp_code)
```

### 뉴스 분석
```python
from analysis.news_analyzer import collect_stock_news, analyze_news_with_gpt

stock_news = collect_stock_news("삼성전자")
analysis = analyze_news_with_gpt("삼성전자", stock_news, macro_news)
```

### 밸류에이션
```python
from analysis.valuation_analyzer import calculate_ttm_metrics

metrics = calculate_ttm_metrics(financial_records, current_price)
# PER, PBR, ROE, PSR 등
```

### 통합 분석
```python
from pipeline.prediction_pipeline import run_analysis

result = run_analysis("005930")
# 완전한 분석 결과 반환
```

## API 비용

### 무료/저비용
- **pykrx**: 무료 (공식 거래소 API)
- **DART**: 무료 (공식 공시 API)

### 유료 (소량 사용 시 무료)
- **OpenAI**: 
  - Embedding API: $0.02/1M tokens
  - GPT-4o-mini: $0.15/1M input, $0.60/1M output
  - 월 $5 크레딧 포함
  
- **NewsAPI**: 
  - 무료 플랜: 100 요청/일
  - 유료: $29/월부터

## 주의사항

⚠️ 본 분석은 참고용이며, 투자 결정은 본인의 책임입니다.

⚠️ 과거 데이터 기반 분석으로 미래 수익을 보장하지 않습니다.

⚠️ API 키는 절대 공개하지 마세요.

## 개선 방안

- [ ] 크롤링 기반 실시간 뉴스 분석
- [ ] 한국 증시 특화 sentiment 분석
- [ ] 옵션 시장 데이터 통합
- [ ] 포트폴리오 최적화 기능
- [ ] 백테스팅 및 성과 분석
- [ ] 웹 대시보드 구축

## 라이센스

MIT License
