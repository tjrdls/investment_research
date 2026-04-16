# 투자 분석 시스템

이 시스템은 주가 데이터와 재무제표를 기반으로 투자 분석을 수행하는 최소 MVP입니다.

## 구조
- 데이터 수집 → 데이터 분석 → LSTM 예측 → RAG 재무제표 검색 → LLM 최종 분석 → 결과 출력

## 설치
1. 의존성 설치: `pip install -r requirements.txt`
2. .env 파일 생성 및 API 키 설정:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DART_API_KEY=your_dart_api_key
   NEWSAPI_KEY=your_newsapi_key
   ```
3. **LSTM 모델 학습** (처음 한 번만):
   ```bash
   python models/lstm/train_lstm.py
   ```
   이 과정에서 상위 10개 종목의 3년 데이터를 학습하여 `best_multimodal_stock_model.pth` 파일을 생성합니다.

## 사용법
```bash
python main.py
```

### 웹 UI 실행
```bash
streamlit run web/app.py
```

### 분석 모드
- **모드 1**: 단일 종목 분석 (예: 005930)
- **모드 2**: 상위 5개 종목 일괄 분석

### 출력 예시
```
  📊 삼성전자 (005930) 분석

  [1/6] 주가 데이터 수집... ✅ 730 거래일
  [2/6] 기술적 지표 계산... ✅ 신호 1, 경고 0
  [3/6] 재무 데이터 수집... ✅ 4 기간
  [4/6] 텍스트 임베딩... ✅ 1536차원
  [5/6] LSTM 예측... ✅ 📈 상승: 신뢰도 65.3%
  [6/6] 뉴스 분석... ✅ 호재: 매수적극
  [7/6] 밸류에이션... ✅ PER: 13.53 PBR: 3.18 ROE: 23.53%
  [8/6] 최종 LLM 분석... ✅ 매수 추천
```

## LSTM 모델 상세 정보

[LSTM_GUIDE.md](LSTM_GUIDE.md) 참고

**빠른 요약:**
- 모델 파일: `best_multimodal_stock_model.pth` (11.2MB)
- 학습 데이터: 상위 10개 종목 × 3년 × 기술적 지표
- 예측: 상승/하락/횡보 3분류
- 정확도: ~65-70%

## 폴더 구조
- data/: 데이터 저장
- models/: 모델 파일
- rag/: RAG 시스템
- analysis/: 분석 모듈
- pipeline/: 파이프라인