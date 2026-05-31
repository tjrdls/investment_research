# 멀티모달 분석 레이어 (`src/modality/`)

> 구버전 `stockAI/`(단일 종목 LSTM+RAG+LLM)의 **멀티모달 철학**을,
> 개선본 `stock_ai/`(Rule + LGBM 포트폴리오 + 정직한 백테스트 검증) 위에 병합한 레이어.

## 왜 병합인가

| | 구버전 stockAI | 개선본 stock_ai (병합 전) |
|---|---|---|
| 강점 | 차트·재무·뉴스·공시·매크로 **멀티모달** Late Fusion | 데이터 버그 수정 · walk-forward · 일별 MDD · 백테스트 |
| 약점 | 출력이 단일 종목 텍스트 → **검증 불가** | LSTM·뉴스·텍스트 제거 → **더 이상 멀티모달 아님** |

→ 두 시스템의 약점이 정확히 직교한다. 합치면 **멀티모달성 + 검증가능성**을 모두 얻는다.

## 아키텍처

```
                          ┌─ [차트]     LGBM v3+v4 (횡단면 랭킹)      ─┐
                          ├─ [펀더멘털] Rule 4팩터 점수                ├─→ ModalitySignal
종목 ─→ 모달리티 인코더 ──┼─ [밸류에이션] TTM PER/ROE/PBR             ─┤   (0~100, confidence)
                          ├─ [뉴스]     NewsAPI + GPT 감성             ─┤
                          └─ [매크로]   KOSPI200 레짐                 ─┘
                                              │
                                              ▼
                                   LateFusion (가중 결합 + 충돌 감지)   ← fusion.py (순수, 테스트됨)
                                              │
                                              ▼
                                   llm.narrate (근거 설명)              ← LLM 또는 템플릿 fallback
```

**핵심 설계 결정**: 구버전은 LLM 이 '종합 판단'까지 했지만(블랙박스),
여기서는 **정량 Late Fusion 이 점수·판정을 확정**하고 LLM 은 **근거 설명**만 한다.
→ 점수는 재현·검증 가능, 설명은 풍부.

## 파일

| 파일 | 역할 | I/O 의존 |
|---|---|---|
| `base.py` | `ModalitySignal` 표준 + 스케일 변환 | **없음 (순수)** |
| `fusion.py` | `LateFusion` 가중 결합·충돌 감지·판정 | **없음 (순수)** |
| `valuation.py` | TTM 지표 계산 + 신호 변환 / 업종 GPT | 계산=순수, 업종평가=OpenAI |
| `news.py` | 뉴스 결과→신호 / 수집·GPT 감성 | 변환=순수, 수집=NewsAPI+OpenAI |
| `llm.py` | 융합 결과 → 내러티브 | OpenAI (없으면 템플릿) |
| `analyzer.py` | DB·DART·뉴스에서 신호 빌드 → 융합 | DB + 키 (각자 degrade) |

## 우아한 degrade

데이터/키가 없는 모달리티는 `ModalitySignal.unavailable` 을 내고
`LateFusion` 이 **남은 모달리티로 가중치를 재정규화**한다. 즉:

- `OPENAI_API_KEY` 없음 → 뉴스·업종평가 제외, LLM 내러티브는 템플릿으로 fallback
- `NEWSAPI_KEY` 없음 → 뉴스 모달리티만 제외
- `cache.db` 없음 → 차트·펀더·밸류·매크로 제외 (전부 없으면 중립 반환, 크래시 없음)

## 사용법

```bash
# 단일 종목 멀티모달 분석 (CLI)
python main.py analyze 005930 --name 삼성전자

# 코드에서
from src.modality.analyzer import MultimodalAnalyzer
out = MultimodalAnalyzer().analyze("005930", "삼성전자")
print(out["report"])          # 텍스트 리포트
print(out["fusion"].score)    # 0~100 융합점수
```

## 설정

`src/config.py`:
- `CFG.fusion` — 모달리티별 가중치 (`FusionWeights`)
- `CFG.llm` — GPT 모델·토큰·on/off (`LLMConfig`)

`.env` — `OPENAI_API_KEY`, `NEWSAPI_KEY` (없어도 동작)

## 테스트 (DB/키 불필요)

```bash
python -m pytest tests/ -q
```

- `test_modality_base.py` — 신호 표준·스케일 변환
- `test_fusion.py` — 가중 결합·재정규화·충돌 감지·판정 임계값
- `test_valuation.py` — TTM 지표 계산 골든 케이스·신호 변환
- `test_news_and_analyzer.py` — 뉴스 변환·오케스트레이터 템플릿 fallback

순수 로직(`base`, `fusion`, `valuation` 계산)이 전부 테스트되므로
DB·API 키가 없는 환경에서도 병합 핵심을 검증할 수 있다.

## 남은 작업 (DB/키 생긴 뒤)

- [ ] LSTM 가중치(`chart_lstm_wf.pt`) 재학습 → 차트 모달에 LSTM 옵션 추가 (현재는 LGBM)
- [ ] 공시(DART) 텍스트 임베딩 RAG 를 별도 모달리티로 추가 (현재 5개 → 6개)
- [ ] **모달별 ablation**: 차트만 / +펀더 / +뉴스 ... 단계별 백테스트 알파 기여도 측정
- [ ] 대시보드에 "멀티모달 분석" 탭 추가 (종목 검색 탭 확장)
- [ ] `text_embeddings_cache.pkl` 캐시로 임베딩 재사용
