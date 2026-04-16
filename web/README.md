# 웹 UI

이 폴더에는 Streamlit 기반 웹 서비스 진입점이 있습니다.

## 실행 방법

1. 필요한 라이브러리를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```
2. 프로젝트 루트에서 Streamlit 앱을 실행합니다:
   ```bash
   streamlit run web/app.py
   ```

## 기능

- 사이드바에서 종목 선택, 기간 선택 후 분석 시작
- 종합 분석, 차트, 기술 분석, 재무 분석, 뉴스 탭 제공
- Plotly 캔들차트 및 거래량 시각화
- LSTM, 기술적 지표, 재무 지표, 뉴스 분석 통합 출력
