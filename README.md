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

## 사용법
python main.py

## 폴더 구조
- data/: 데이터 저장
- models/: 모델 파일
- rag/: RAG 시스템
- analysis/: 분석 모듈
- pipeline/: 파이프라인