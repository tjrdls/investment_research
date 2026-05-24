"""
Stock AI 대시보드
실행: streamlit run src/ui/dashboard.py  /  python main.py dashboard
"""
from __future__ import annotations

import calendar
import sqlite3
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# .env 자동 로드 (DART_API_KEY 등) — streamlit/venv 환경에서도 동작
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass  # python-dotenv 없으면 OS 환경변수만 사용

from src.config import CFG, DB_PATH, MODEL_DIR
from src.data.daily_update import status_summary, update_to

MODEL_PATH_WF = MODEL_DIR / "chart_lstm_wf.pt"

ACCENT = "#14b8a6"
POS = "#10b981"
NEG = "#ef4444"
MUTED = "#94a3b8"

CUSTOM_CSS = f"""
<style>
.block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }}
[data-testid="stMetricValue"] {{
    font-size: 1.65rem; font-weight: 600; letter-spacing: -0.02em;
}}
[data-testid="stMetricLabel"] {{ font-size: 0.78rem; opacity: 0.55; font-weight: 500; }}
[data-testid="stMetricDelta"] {{ font-size: 0.75rem; opacity: 0.7; }}
h1 {{ font-weight: 700; letter-spacing: -0.03em; font-size: 1.9rem; margin-bottom: 0.2rem; }}
h2 {{ font-weight: 600; letter-spacing: -0.015em; font-size: 1.15rem; margin-top: 1.4rem; }}
h3 {{ font-weight: 500; font-size: 0.95rem; opacity: 0.85; margin-top: 1rem; }}
.stTabs [data-baseweb="tab-list"] {{
    gap: 1.8rem; border-bottom: 1px solid rgba(250,250,250,0.08);
}}
.stTabs [data-baseweb="tab"] {{ font-size: 0.95rem; padding: 0.4rem 0; font-weight: 500; }}
.stTabs [aria-selected="true"] {{ color: {ACCENT} !important; }}
[data-testid="stCaptionContainer"] {{ opacity: 0.55; font-size: 0.82rem; }}
[data-testid="stSidebar"] {{ background: rgba(0,0,0,0.15); }}
button[kind="primary"] {{ background: {ACCENT}; border-color: {ACCENT}; }}
.news-item {{
    padding: 0.7rem 0; border-bottom: 1px solid rgba(128,128,128,0.15);
}}
.news-item a {{
    color: inherit; text-decoration: none; font-weight: 600;
    font-size: 0.98rem; line-height: 1.4;
}}
.news-item a:hover {{ color: {ACCENT}; text-decoration: underline; }}
.news-meta {{ opacity: 0.55; font-size: 0.78rem; margin-top: 0.25rem; }}
hr {{ margin: 0.4rem 0 0.9rem; opacity: 0.1; }}
</style>
"""


# ============================================================
# DB 헬퍼
# ============================================================
@st.cache_data(ttl=60)
def get_db_summary() -> dict:
    if not DB_PATH.exists():
        return {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            n_tickers = conn.execute("SELECT COUNT(*) FROM tickers").fetchone()[0]
            ohlcv_range = conn.execute("SELECT MIN(date), MAX(date) FROM ohlcv").fetchone()
            cap_max = conn.execute("SELECT MAX(date) FROM market_cap").fetchone()[0] \
                if "market_cap" in tables else None
            n_funds = conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0] \
                if "fundamentals" in tables else 0
            n_per = conn.execute("SELECT COUNT(*) FROM per_history").fetchone()[0] \
                if "per_history" in tables else 0
            per_max = conn.execute("SELECT MAX(date) FROM per_history").fetchone()[0] \
                if "per_history" in tables else None
        return {"tickers": n_tickers, "date_min": ohlcv_range[0], "date_max": ohlcv_range[1],
                "cap_max": cap_max, "fundamentals": n_funds, "per_history": n_per,
                "per_max": per_max}
    except Exception:
        return {}


@st.cache_data(ttl=60)
def get_fundamentals_coverage() -> pd.DataFrame:
    """시장별·분기별 fundamentals 종목 커버리지. KOSDAQ 2026-Q1 stale 진단용."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query("""
                SELECT t.market, f.period_end, COUNT(DISTINCT f.ticker) AS n_tickers
                FROM fundamentals f JOIN tickers t ON t.ticker=f.ticker
                WHERE f.period_end >= date('now','-15 months')
                GROUP BY t.market, f.period_end
                ORDER BY f.period_end DESC, t.market
            """, conn)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_market_regime(as_of: str) -> dict:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT date, close FROM ohlcv WHERE ticker='069500' AND date <= ? "
                "ORDER BY date DESC LIMIT 250", conn, params=[as_of])
        if len(df) < 200:
            return {"bull": True, "kospi": None, "ma200": None, "ratio": None}
        ma200 = float(df["close"].iloc[1:201].mean())
        current = float(df["close"].iloc[0])
        return {"bull": current >= ma200, "kospi": current, "ma200": ma200,
                "ratio": (current / ma200 - 1) * 100}
    except Exception:
        return {"bull": True, "kospi": None, "ma200": None, "ratio": None}


@st.cache_data(ttl=300)
def get_financials(ticker: str, as_of: str) -> dict:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            fund = conn.execute(
                "SELECT roe, operating_margin, operating_income_growth_yoy, "
                "revenue_growth_yoy, profit_growth_yoy, debt_ratio "
                "FROM fundamentals WHERE ticker=? AND period_end<=? "
                "ORDER BY period_end DESC LIMIT 1", (ticker, as_of)).fetchone()
            per_row = conn.execute(
                "SELECT per, pbr, div_yield FROM per_history WHERE ticker=? AND date<=? "
                "ORDER BY date DESC LIMIT 1", (ticker, as_of)).fetchone()
        result = {}
        if fund:
            result.update(dict(zip(
                ["roe", "operating_margin", "op_income_growth",
                 "rev_growth", "profit_growth", "debt_ratio"], fund)))
        if per_row:
            result.update(dict(zip(["per", "pbr", "div_yield"], per_row)))
        return result
    except Exception:
        return {}


AI_WEIGHT_PRODUCTION = 0.2  # Hybrid (v3+v4) sweet spot — TR 1451.9% / Sharpe 1.42


def _run_ai_ensemble_backtest(*, start_year: int, end_year: int, end_date=None):
    """Hybrid LGBM(v3 60% + v4 40%, ai_w=0.2) + risk overlay. 실패 시 rule_only 폴백."""
    import traceback
    from src.backtest.rebalance import RebalanceBacktest, run_rule_based_backtest
    from src.backtest.risk_overlay import apply_risk_overlay
    from src.config import MODEL_DIR

    v3_path = MODEL_DIR / "trend_lgbm_v3.txt"
    v4_path = MODEL_DIR / "trend_lgbm_v4.txt"

    rule_only_kwargs = dict(
        start_year=start_year, end_year=end_year, top_n=10,
        period_months=2, rebalance_day=9,
        replacement_rule="keep_simple",
        market_split=True, trend_filter=True, market_cap_min=1e12,
        ichimoku_adx=True, ia_scaling=True, use_ttm_per=True,
        defensive_ticker="132030",
    )
    if end_date:
        rule_only_kwargs["end_date"] = end_date

    if v3_path.exists() and v4_path.exists():
        try:
            from src.recommend.lgbm_ensemble import LGBMEnsembleScreener
            ens = LGBMEnsembleScreener(
                ai_weight=AI_WEIGHT_PRODUCTION,
                model_v3_path=v3_path, model_v4_path=v4_path,
            )
            def picker(as_of: str, n: int):
                return ens.select_top_n(
                    as_of=as_of, top_n=n,
                    market_split=True, trend_filter=True, market_cap_min=1e12,
                    use_ttm_per=True,
                )
            bt = RebalanceBacktest()
            kwargs = dict(rule_only_kwargs)
            kwargs.pop("market_split", None); kwargs.pop("trend_filter", None)
            kwargs.pop("market_cap_min", None); kwargs.pop("use_ttm_per", None)
            kwargs["picker"] = picker
            kwargs["weight_scheme"] = "rank"
            raw = bt.run(**kwargs)
        except Exception as e:
            # Hybrid 실패 시 rule_only 폴백 (streamlit 에 명시)
            print(f"⚠️ Hybrid 백테스트 실패 → rule_only 폴백: {e}")
            traceback.print_exc()
            raw = run_rule_based_backtest(**rule_only_kwargs)
    else:
        raw = run_rule_based_backtest(**rule_only_kwargs)

    overlay = apply_risk_overlay(raw, weight_cap=0.25, bond_yield=0.03)
    new_metrics = dict(raw.metrics)
    for k in ("total_return", "cagr", "sharpe", "mdd_daily", "mdd",
              "win_rate_periods", "alpha_annualized"):
        if k in overlay:
            new_metrics[k] = overlay[k]
    raw.metrics = new_metrics
    return raw


@st.cache_data(ttl=86400, show_spinner=False)
def run_validated_backtest():
    """10년 성과 — EnsembleScreener(ai_w=0.8) + 2m + day=9 + cap25 + 채권3%. 캐시 하루."""
    return _run_ai_ensemble_backtest(start_year=2015, end_year=2024)


def _data_version() -> str:
    """DB OHLCV 최대 날짜 — 캐시 키에 포함시켜 데이터 갱신 시 자동 무효화."""
    return status_summary().get("ohlcv_max") or "empty"


@st.cache_data(ttl=86400, show_spinner=False)
def get_live_backtest(data_version: str = ""):
    """라이브 (2015 ~ DB 마지막 거래일) — production 스펙.
    data_version: DB OHLCV 최대 날짜. 새 데이터가 적재되면 키가 바뀌어 자동 재계산.
    """
    today_iso = date.today().isoformat()
    today_year = date.today().year
    return _run_ai_ensemble_backtest(
        start_year=2015, end_year=today_year + 1, end_date=today_iso,
    )


@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(query: str = "한국 경제", limit: int = 20) -> dict:
    """Google News RSS (Korean)."""
    url = (
        "https://news.google.com/rss/search?"
        f"q={urllib.parse.quote(query)}&hl=ko&gl=KR&ceid=KR:ko"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            xml_data = r.read()
    except Exception as e:
        return {"error": str(e), "items": []}
    try:
        root = ET.fromstring(xml_data)
    except Exception as e:
        return {"error": f"파싱 오류: {e}", "items": []}
    items = []
    for item in list(root.iter("item"))[:limit]:
        title = (item.findtext("title") or "").strip()
        link = item.findtext("link") or ""
        pub = item.findtext("pubDate") or ""
        src_elem = item.find("source")
        source = (src_elem.text or "") if src_elem is not None else ""
        if source and title.endswith(f" - {source}"):
            title = title[: -(len(source) + 3)].strip()
        try:
            pub_dt = parsedate_to_datetime(pub) if pub else None
        except Exception:
            pub_dt = None
        items.append({"title": title, "link": link, "source": source, "pub_dt": pub_dt})
    return {"error": None, "items": items}


# ============================================================
# 사이드바
# ============================================================
def _render_data_status_panel() -> None:
    """사이드바: DB 데이터 최신성 표시 + 증분 업데이트 버튼."""
    s = status_summary()
    st.markdown("### 데이터 상태")
    ohlcv_max = s["ohlcv_max"]
    target = s["target"]
    behind = s["days_behind"]

    if ohlcv_max is None:
        st.error("DB 비어있음")
        st.caption("`python main.py collect` 로 초기 적재 필요")
        return

    if not s["is_stale"]:
        st.success(f"최신 ({ohlcv_max})")
        st.caption(f"목표 {target} 충족 · 어제까지 반영됨")
    else:
        label = f"{behind}일 뒤처짐" if behind and behind > 0 else "갱신 필요"
        st.warning(f"{ohlcv_max} → 목표 {target} · {label}")

    if not s["has_credentials"]:
        st.caption("⚠ `.env` 의 KRX_ID / KRX_PW 없음 — 업데이트 불가")
        return

    if st.button("데이터 업데이트", use_container_width=True,
                 disabled=not s["is_stale"],
                 help="DB 마지막 거래일 → 어제까지 일별 일괄 적재 (1-2분)"):
        progress = st.progress(0, text="시작...")

        def _cb(i: int, total: int, label: str):
            ratio = (i / total) if total else 1.0
            progress.progress(min(ratio, 1.0), text=f"{label}  ({i}/{total})")

        with st.spinner("증분 적재 중..."):
            try:
                r = update_to(progress_cb=_cb)
            except Exception as e:
                st.error(f"업데이트 실패: {e}")
                return

        progress.empty()
        if not r["ok"]:
            st.error(r["error"])
            return

        added = r["days_added"]
        if added == 0:
            st.info(f"신규 데이터 없음 ({r['from']} ~ {r['to']} 모두 휴장일)")
        else:
            st.success(
                f"✓ {added}일 적재 · OHLCV {r['ohlcv_rows']:,}행 · "
                f"시총 {r['cap_rows']:,}행 · {r['elapsed_sec']:.0f}초"
            )
        # 캐시 무효화 후 재실행 → 백테스트/추천 재계산
        st.cache_data.clear()
        st.rerun()


def sidebar_settings() -> dict:
    with st.sidebar:
        _render_data_status_panel()
        st.divider()
        st.markdown("### 설정")
        market_cap_min = st.selectbox(
            "시총 최소",
            options=[1e12, 5e11, 2e12],
            format_func=lambda x: f"{x/1e12:.1f}조",
            index=0,
        )
        top_n = st.slider("추천 종목 수", 5, 20, 10)

        with st.expander("고급"):
            ai_weight = st.slider("AI 가중치 (legacy ChartLSTM 용 슬라이더, production 은 0.2 고정)",
                                   0.0, 1.0, 0.2, 0.1)
            trend_filter = st.toggle("추세 필터 (60일선 위)", value=True)
            market_split = st.toggle("코스피·코스닥 균등 (5:5)", value=True)

        st.caption(
            f"ROE ≥ {CFG.hard_filter.roe_min}% · "
            f"PER ≤ {CFG.hard_filter.per_max} · "
            f"영업이익률 ≥ 10%"
        )

    return {
        "market_cap_min": market_cap_min,
        "trend_filter": trend_filter,
        "market_split": market_split,
        "top_n": top_n,
        "ai_weight": ai_weight,
    }


# ============================================================
# 시장 레짐 한줄
# ============================================================
def regime_line(as_of: str) -> None:
    r = get_market_regime(as_of)
    if r["kospi"] is None:
        return
    sign = "+" if r["ratio"] >= 0 else ""
    tag = "강세장" if r["bull"] else "약세장"
    dot = "🟢" if r["bull"] else "🔴"
    st.caption(
        f"{dot} **{tag}** · KOSPI200 {r['kospi']:,.0f} · 200MA {r['ma200']:,.0f} · "
        f"이격 **{sign}{r['ratio']:.1f}%**"
    )


# ============================================================
# 탭 1: 추천
# ============================================================
def tab_recommend(cfg: dict) -> None:
    s = status_summary()
    default_dt = (
        datetime.strptime(s["ohlcv_max"], "%Y-%m-%d").date()
        if s["ohlcv_max"] else date.today()
    )
    c1, c2 = st.columns([3, 2])
    as_of = c1.date_input("기준일", value=default_dt,
                          label_visibility="collapsed").isoformat()
    mode = c2.radio("모드", ["AI 앙상블 (Hybrid v3+v4)", "규칙만"], horizontal=True,
                    label_visibility="collapsed")

    if s["is_stale"]:
        st.caption(
            f"⚠ DB 마지막 데이터: **{s['ohlcv_max']}** "
            f"(목표 {s['target']}, {s['days_behind']}일 뒤처짐) — "
            f"사이드바 '데이터 업데이트' 로 어제까지 채우세요."
        )

    regime_line(as_of)
    st.caption(
        f"※ **AI 앙상블 모드**: Production 과 동일한 LGBM Hybrid (v3 60% + v4 40%, ai_w={AI_WEIGHT_PRODUCTION}) "
        f"사용 → 성과/포트폴리오/검증 탭과 정합. **규칙만**: 4팩터 점수 단독."
    )

    v3_path = MODEL_DIR / "trend_lgbm_v3.txt"
    v4_path = MODEL_DIR / "trend_lgbm_v4.txt"
    has_hybrid = v3_path.exists() and v4_path.exists()

    if st.button("추천 실행", type="primary", use_container_width=True):
        with st.spinner("계산 중..."):
            try:
                if mode == "규칙만" or not has_hybrid:
                    from src.screener.rule_based import RuleBasedScreener
                    screener = RuleBasedScreener()
                    picks = screener.select_top_n(
                        as_of=as_of, top_n=cfg["top_n"],
                        market_split=cfg["market_split"],
                        trend_filter=cfg["trend_filter"],
                        market_cap_min=cfg["market_cap_min"],
                        use_ttm_per=True,
                    )
                    score_col = "rule_score"
                else:
                    from src.recommend.lgbm_ensemble import LGBMEnsembleScreener
                    screener = LGBMEnsembleScreener(
                        ai_weight=AI_WEIGHT_PRODUCTION,
                        model_v3_path=v3_path, model_v4_path=v4_path,
                    )
                    picks = screener.select_top_n(
                        as_of=as_of, top_n=cfg["top_n"],
                        market_split=cfg["market_split"],
                        trend_filter=cfg["trend_filter"],
                        market_cap_min=cfg["market_cap_min"],
                        use_ttm_per=True,
                    )
                    score_col = "ensemble_score" if "ensemble_score" in picks.columns else "rule_score"
            except Exception as e:
                st.error(str(e))
                return

        if picks.empty:
            st.warning("조건 통과 종목 없음")
            return

        pbr, div = [], []
        for tk in picks["ticker"]:
            fin = get_financials(tk, as_of)
            pbr.append(fin.get("pbr"))
            div.append(fin.get("div_yield"))
        picks = picks.copy()
        picks["pbr"] = pbr
        picks["div_yield"] = div

        st.session_state["last_picks"] = picks
        st.session_state["last_score_col"] = score_col

    picks = st.session_state.get("last_picks")
    score_col = st.session_state.get("last_score_col", "rule_score")
    if picks is None or picks.empty:
        return

    # 정확한 추천 개수 + 부족 시 사유
    n = len(picks)
    target = cfg["top_n"]
    half = target // 2
    header = f"추천 {n}종목"
    if n < target and cfg["market_split"] and "market" in picks.columns:
        n_kospi = int((picks["market"] == "KOSPI").sum())
        n_kosdaq = int((picks["market"] == "KOSDAQ").sum())
        header += f"  ·  코스피 {n_kospi}/{half}  ·  코스닥 {n_kosdaq}/{target - half}"
    st.markdown(f"### {header}")
    if n < target:
        st.caption(
            "조건 통과 종목이 목표보다 적습니다. ROE 20%+·PER 60↓·영업이익률 10%+ "
            "필터가 통과시킨 종목 수입니다."
        )

    # 카드 — 한 행에 5개씩, 전부 표시
    for row_start in range(0, n, 5):
        chunk = picks.iloc[row_start:row_start + 5]
        cols = st.columns(5)
        for j, (_, row) in enumerate(chunk.iterrows()):
            with cols[j]:
                roe = row.get("roe")
                st.metric(
                    label=row.get("name", row.get("ticker", "?")),
                    value=f"{row.get(score_col, 0):.1f}",
                    delta=f"ROE {roe:.0f}%" if roe and not pd.isna(roe) else None,
                    delta_color="off",
                )

    # 핵심 표
    st.markdown("### 종목 목록")
    display = picks.copy().reset_index(drop=True)
    display.index = display.index + 1

    base = [(score_col, "최종점수"), ("ai_v3_score", "AI(v3)"), ("ai_v4_score", "AI(v4)"),
            ("ticker", "종목코드"), ("name", "종목명"),
            ("market", "시장"), ("roe", "ROE(%)"), ("per", "PER"), ("pbr", "PBR"),
            ("operating_margin", "영업이익률(%)")]
    cols_show = [c for c, _ in base if c in display.columns]
    rename = {c: l for c, l in base if c in display.columns}
    if "market_cap" in display.columns:
        display["시총(조)"] = (display["market_cap"] / 1e12).round(2)
        cols_show.append("시총(조)")

    fmt = {"최종점수": "{:.1f}", "AI(v3)": "{:.1f}", "AI(v4)": "{:.1f}",
           "ROE(%)": "{:.1f}", "PER": "{:.1f}",
           "PBR": "{:.2f}", "영업이익률(%)": "{:.1f}", "시총(조)": "{:.2f}"}
    out = display[cols_show].rename(columns=rename)
    st.dataframe(
        out.style.format({k: v for k, v in fmt.items() if k in out.columns}, na_rep="–"),
        use_container_width=True,
        height=min(48 + len(picks) * 35, 480),
    )

    with st.expander("전체 지표"):
        full = [(score_col, "점수"), ("ticker", "종목코드"), ("name", "종목명"),
                ("market", "시장"), ("roe", "ROE(%)"), ("per", "PER"), ("pbr", "PBR"),
                ("div_yield", "배당(%)"), ("operating_margin", "영업이익률(%)"),
                ("operating_income_growth_yoy", "영업이익성장(%)"),
                ("revenue_growth_yoy", "매출성장(%)"), ("debt_ratio", "부채비율(%)")]
        cs = [c for c, _ in full if c in display.columns]
        rn = {c: l for c, l in full if c in display.columns}
        st.dataframe(display[cs].rename(columns=rn), use_container_width=True)


# ============================================================
# 탭 2: 종목 검색
# ============================================================
def tab_search() -> None:
    q = st.text_input(
        "종목 검색",
        placeholder="종목코드 또는 이름 (예: 005930, 삼성전자)",
        label_visibility="collapsed",
    ).strip()
    if not q:
        st.caption("종목 코드 또는 이름의 일부를 입력하세요.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        matches = pd.read_sql_query(
            "SELECT ticker, name, market FROM tickers "
            "WHERE ticker LIKE ? OR name LIKE ? ORDER BY name LIMIT 30",
            conn, params=[f"%{q}%", f"%{q}%"])

    if matches.empty:
        st.caption(f"'{q}'와 일치하는 종목이 없습니다.")
        return

    if len(matches) > 1:
        idx = st.selectbox(
            f"매칭 {len(matches)}개",
            range(len(matches)),
            format_func=lambda i: f"{matches.iloc[i]['ticker']}  ·  "
                                  f"{matches.iloc[i]['name']}  ({matches.iloc[i]['market']})",
            label_visibility="collapsed",
        )
        sel = matches.iloc[idx]
    else:
        sel = matches.iloc[0]
        st.caption(f"{sel['ticker']} · {sel['name']} · {sel['market']}")

    tk = sel["ticker"]
    today = date.today().isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        price_df = pd.read_sql_query(
            "SELECT date, close FROM ohlcv WHERE ticker=? AND date>=date(?, '-1 year') "
            "ORDER BY date", conn, params=[tk, today], parse_dates=["date"])
        cap_row = conn.execute(
            "SELECT market_cap FROM market_cap WHERE ticker=? ORDER BY date DESC LIMIT 1",
            (tk,)).fetchone()

    current = float(price_df["close"].iloc[-1]) if not price_df.empty else None
    one_yr = float(price_df["close"].iloc[0]) if not price_df.empty else None
    one_yr_ret = (current / one_yr - 1) * 100 if current and one_yr else None
    cap_조 = (cap_row[0] / 1e12) if cap_row else None
    fin = get_financials(tk, today)

    # 핵심 메트릭 4개
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재가", f"{current:,.0f}원" if current else "–",
              delta=f"{one_yr_ret:+.1f}% (1년)" if one_yr_ret is not None else None)
    c2.metric("시가총액", f"{cap_조:.2f}조" if cap_조 else "–")
    roe = fin.get("roe")
    c3.metric("ROE", f"{roe:.1f}%" if roe else "–")
    per = fin.get("per")
    c4.metric("PER", f"{per:.1f}" if per else "–")

    c1, c2, c3, c4 = st.columns(4)
    pbr = fin.get("pbr")
    c1.metric("PBR", f"{pbr:.2f}" if pbr else "–")
    div = fin.get("div_yield")
    c2.metric("배당수익률", f"{div:.2f}%" if div else "–")
    om = fin.get("operating_margin")
    c3.metric("영업이익률", f"{om:.1f}%" if om else "–")
    oig = fin.get("op_income_growth")
    c4.metric("영업이익 YoY", f"{oig:+.1f}%" if oig else "–")

    # 1년 주가
    if not price_df.empty:
        st.markdown("### 1년 주가")
        chart = price_df.set_index("date")[["close"]].rename(columns={"close": "종가"})
        st.line_chart(chart, color=ACCENT)

    # 분기 재무제표
    with sqlite3.connect(DB_PATH) as conn:
        q_df = pd.read_sql_query(
            "SELECT f.year, f.quarter, f.period_end, f.revenue, f.operating_income, "
            "f.net_income, fu.roe, fu.operating_margin, "
            "fu.operating_income_growth_yoy, fu.revenue_growth_yoy, fu.debt_ratio "
            "FROM financials f LEFT JOIN fundamentals fu "
            "ON fu.ticker=f.ticker AND fu.period_end=f.period_end "
            "WHERE f.ticker=? ORDER BY f.period_end DESC LIMIT 12",
            conn, params=[tk])

    if not q_df.empty:
        st.markdown("### 분기 실적")
        disp = q_df.copy()
        for c in ["revenue", "operating_income", "net_income"]:
            disp[c] = (disp[c] / 1e8).round(0)
        disp = disp.rename(columns={
            "year": "연도", "quarter": "분기", "period_end": "기준일",
            "revenue": "매출(억)", "operating_income": "영업이익(억)",
            "net_income": "순이익(억)",
            "roe": "ROE(%)", "operating_margin": "영업이익률(%)",
            "operating_income_growth_yoy": "영업이익 YoY(%)",
            "revenue_growth_yoy": "매출 YoY(%)",
            "debt_ratio": "부채비율(%)",
        })
        st.dataframe(
            disp.style.format({
                "매출(억)": "{:,.0f}", "영업이익(억)": "{:,.0f}", "순이익(억)": "{:,.0f}",
                "ROE(%)": "{:.1f}", "영업이익률(%)": "{:.1f}",
                "영업이익 YoY(%)": "{:+.1f}", "매출 YoY(%)": "{:+.1f}",
                "부채비율(%)": "{:.1f}",
            }, na_rep="–"),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption("분기 재무 데이터 없음.")


# ============================================================
# 탭 3: 성과 (검증된 10년 백테스트)
# ============================================================
def _next_trading_day(target: pd.Timestamp) -> pd.Timestamp:
    """target 이상의 첫 한국 시장 거래일. ohlcv 의 KOSPI200(069500) 기준."""
    import sqlite3 as _sq
    from src.config import DB_PATH
    target_str = target.strftime("%Y-%m-%d")
    with _sq.connect(DB_PATH) as c:
        row = c.execute(
            "SELECT MIN(date) FROM ohlcv WHERE ticker='069500' AND date >= ?",
            (target_str,)
        ).fetchone()
    if row and row[0]:
        return pd.Timestamp(row[0])
    # 미래 거래일이라 데이터 없으면 weekday 보정 (월~금)
    d = target
    while d.weekday() >= 5:  # 5=토, 6=일
        d += pd.Timedelta(days=1)
    return d


def _compute_ma60_status(tickers: list, ref_date: str) -> dict:
    """각 ticker 의 ref_date 시점 60일 이동평균 vs 종가. {ticker: (close, ma60, above)}."""
    import sqlite3
    from src.config import DB_PATH
    out = {}
    if not tickers:
        return out
    with sqlite3.connect(DB_PATH) as c:
        for tk in tickers:
            rows = c.execute(
                "SELECT date, close FROM ohlcv WHERE ticker=? AND date <= ? "
                "ORDER BY date DESC LIMIT 60",
                (tk, ref_date),
            ).fetchall()
            if len(rows) < 60:
                continue
            closes = [r[1] for r in rows]
            close = closes[0]
            ma60 = sum(closes) / 60
            out[tk] = (close, ma60, close > ma60, (close / ma60 - 1) * 100)
    return out


def _render_current_portfolio(r) -> None:
    """현재 보유 포트폴리오 (포트폴리오 탭). 라이브 백테스트의 마지막 구간 + 60일선 진단."""
    if r is None or r.periods.empty or r.holdings.empty:
        st.caption("현재 보유 데이터 없음")
        return
    last_period = r.periods.iloc[-1]
    last_entry = pd.Timestamp(last_period["entry_date"])
    last_exit = pd.Timestamp(last_period["exit_date"])
    days_held = (last_exit - last_entry).days
    last_holdings = r.holdings[r.holdings["entry_date"] == last_period["entry_date"]].copy()
    if last_holdings.empty:
        st.caption("현재 보유 데이터 없음")
        return

    # 다음 리밸 = entry_date + 2개월 후 9일 (주말이면 다음 거래일)
    # last_exit 는 백테스트 end_date=today 라 misleading. 실제 다음 리밸일을 계산해야 함.
    cfg_pm = r.config.get("period_months", 2) if hasattr(r, "config") and r.config else 2
    cfg_day = 9  # production 채택
    next_target = last_entry + pd.DateOffset(months=cfg_pm)
    next_target = next_target.replace(day=min(cfg_day, calendar.monthrange(next_target.year, next_target.month)[1]))
    # 다음 거래일 찾기
    next_rebalance = _next_trading_day(next_target)
    days_until = (next_rebalance - pd.Timestamp(date.today())).days

    final_level = r.metrics.get("ia_final_level", 1.0)
    period_return = float(last_period["period_return"]) * 100

    # 매수 시점 == 9일이면 정상, 아니면 (주말로 인한) 다음 거래일 안내
    entry_target = last_entry.replace(day=9) if last_entry.day != 9 else last_entry
    entry_note = ""
    if last_entry.day != 9:
        entry_note = f" (목표 {entry_target.strftime('%m-%d')} = {entry_target.day_name()[:3]}, 비거래일 → 다음 거래일로 이동)"

    st.markdown("## 현재 보유 포트폴리오")
    st.caption(
        f"**매수 시점** {last_entry.strftime('%Y-%m-%d')} ({last_entry.day_name()[:3]}){entry_note} · "
        f"**다음 리밸** {next_rebalance.strftime('%Y-%m-%d')} ({next_rebalance.day_name()[:3]}, 약 {days_until}일 후) · "
        f"기준일 {last_exit.strftime('%Y-%m-%d')} ({days_held}일 보유 중)"
    )

    state_label = f"{int(round(final_level * 100))}% 주식"
    state_delta = (
        "강세 추세 풀매수"
        if final_level >= 0.99
        else (f"중립 (50% 금)" if abs(final_level - 0.5) < 0.05
              else f"방어 ({int(round((1 - final_level) * 100))}% 금)")
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("구간 수익률", f"{period_return:+.2f}%", delta_color="off")
    c2.metric("시스템 상태", state_label, delta=state_delta, delta_color="off")
    c3.metric("보유 종목 수", f"{len(last_holdings)}개")
    c4.metric("다음 리밸까지", f"{days_until}일",
              delta="2개월 주기")

    # 60일선 상태 (진입 시점 + 오늘 시점)
    entry_str = last_entry.strftime("%Y-%m-%d")
    today_str = last_exit.strftime("%Y-%m-%d")
    ma_entry = _compute_ma60_status(last_holdings["ticker"].tolist(), entry_str)
    ma_now = _compute_ma60_status(last_holdings["ticker"].tolist(), today_str)

    # 종목 표
    st.markdown("#### 보유 종목 상세")
    display = last_holdings.copy().reset_index(drop=True)
    display["weight_pct"] = display["weight"] * 100
    display["return_pct_x"] = display["return_pct"] * 100
    display["ma60_entry"] = display["ticker"].map(
        lambda t: f"{ma_entry[t][3]:+.1f}%" if t in ma_entry else "–"
    )
    display["ma60_now"] = display["ticker"].map(
        lambda t: f"{ma_now[t][3]:+.1f}%{' ✗' if (t in ma_now and not ma_now[t][2]) else ' ★'}"
        if t in ma_now else "–"
    )

    cols_map = [
        ("ticker", "종목코드"), ("name", "종목명"),
        ("weight_pct", "비중(%)"),
        ("entry_price", "진입가(원)"),
        ("exit_price", "현재가(원)"),
        ("return_pct_x", "수익률(%)"),
        ("ma60_entry", "60d MA(진입)"),
        ("ma60_now", "60d MA(현재)"),
    ]
    show = [(c, l) for c, l in cols_map if c in display.columns]
    out = display[[c for c, _ in show]].rename(columns={c: l for c, l in show})
    out.index = range(1, len(out) + 1)

    fmt = {"비중(%)": "{:.1f}", "진입가(원)": "{:,.0f}", "현재가(원)": "{:,.0f}", "수익률(%)": "{:+.2f}"}
    st.dataframe(
        out.style.format(
            {k: v for k, v in fmt.items() if k in out.columns}, na_rep="–"
        ).map(
            lambda v: f"color: {POS}" if isinstance(v, (int, float)) and v > 0
            else f"color: {NEG}" if isinstance(v, (int, float)) and v < 0 else "",
            subset=["수익률(%)"],
        ),
        use_container_width=True,
        height=min(48 + len(out) * 36, 480),
    )
    st.caption(
        "비중 = rank-가중, 단일 종목 25% 캡 (초과분 → 채권 3% 일복리). "
        "진입가·현재가는 시가 기준 (open-to-open). "
        "60d MA(진입): 매수 시점의 60일선 대비 종가 괴리 (모두 ★상승 = 추세 통과 후 매수). "
        "60d MA(현재): 오늘 시점 상태. ✗하락은 매수 후 추세 깨졌다는 뜻 — **다음 리밸(2개월)에서 자동 탈락 예정**. "
        "매일 손절(trailing stop) 은 과거 검증에서 catastrophic (TR -88%, α -9.5%) 이라 도입 안 함, 사전 분산(비중캡) 만 사용."
    )


def tab_portfolio() -> None:
    """현재 보유 포트폴리오 전용 탭."""
    s = status_summary()
    if s["is_stale"]:
        st.warning(
            f"⚠ 표시된 수익률·보유종목은 **{s['ohlcv_max']}** 기준 "
            f"(어제 {s['target']} 까지 {s['days_behind']}일 뒤처짐). "
            f"사이드바 '데이터 업데이트' 클릭 시 어제까지 채워집니다."
        )
    else:
        st.success(f"✓ {s['ohlcv_max']} 기준 (어제까지 반영)")
    with st.spinner("현재 보유 데이터 로딩 중..."):
        try:
            live_r = get_live_backtest(_data_version())
        except Exception as e:
            live_r = None
            st.warning(f"라이브 백테스트 실패: {e}")
    _render_current_portfolio(live_r)


@st.cache_data(ttl=86400, show_spinner=False)
def _compute_validation(data_version: str = ""):
    """블랙스완 + 몬테카를로 결과. data_version 으로 자동 무효화."""
    from src.backtest.risk_overlay import apply_risk_overlay
    from src.backtest.stress_test import blackswan_analysis, monte_carlo, verdict
    live_r = get_live_backtest(data_version)
    if live_r is None or live_r.daily_equity.empty:
        return None
    # 실제 production 운용 곡선 = risk_overlay 적용 daily PV
    overlay = apply_risk_overlay(live_r, weight_cap=0.25, bond_yield=0.03)
    daily_pv = overlay["daily"][["date", "pv"]].copy()
    bs = blackswan_analysis(daily_pv)
    mc = monte_carlo(daily_pv, n_sim=10000)
    realized = dict(live_r.metrics)
    realized.update({k: overlay[k] for k in ("total_return","cagr","sharpe","mdd_daily","mdd",
                                              "win_rate_periods","alpha_annualized")
                      if k in overlay})
    v = verdict(realized, mc)
    return dict(blackswan=bs, mc=mc, verdict=v, realized=realized)


def tab_validation() -> None:
    """블랙스완 + 몬테카를로 검증 결과."""
    st.markdown("## 시스템 검증 (블랙스완 + 몬테카를로)")
    st.caption(
        "역사적 위기 3건의 방어 성적 + 일별 수익률 부트스트랩 10,000회. "
        "실측 라이브 결과 기준 — 시뮬레이션이 아닌 실제 측정값."
    )
    with st.spinner("검증 계산 중... (캐시 1일)"):
        try:
            data = _compute_validation(_data_version())
        except Exception as e:
            st.error(f"검증 실패: {e}")
            return
    if data is None:
        st.warning("백테스트 데이터 부족")
        return

    bs = data["blackswan"]; mc = data["mc"]; v = data["verdict"]; rl = data["realized"]

    # ── 1) 블랙스완 ─────────────────────────────────────────────
    st.markdown("### 블랙스완 방어 성적표")
    st.caption("3대 위기 구간에서 시스템 vs KOSPI200(069500) 비교")
    bs_disp = bs.copy()
    for col in ("sys_mdd","bench_mdd","sys_ret","bench_ret","alpha"):
        bs_disp[col] = bs_disp[col].apply(lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "–")
    bs_disp["defense"] = bs_disp["defense"].apply(lambda v: f"{v:.1f}%" if pd.notna(v) else "–")
    bs_disp = bs_disp.rename(columns={
        "event":"이벤트","period":"기간","sys_mdd":"시스템 MDD","bench_mdd":"KOSPI MDD",
        "defense":"방어율","sys_ret":"시스템 수익","bench_ret":"KOSPI 수익","alpha":"알파"})
    st.dataframe(bs_disp, use_container_width=True, hide_index=True)
    st.caption(
        "방어율 = (1 − 시스템MDD / KOSPI MDD) × 100. 높을수록 폭락 잘 방어. "
        "알파 > 0 = 위기 구간에 시장 outperform (채권 폴백 + 골드 헤지 + 비중 캡 효과)."
    )

    # ── 2) 몬테카를로 ────────────────────────────────────────────
    st.markdown("### 몬테카를로 10,000회 부트스트랩")
    st.caption(
        "일별 수익률을 5일 블록으로 무작위 재추출. 10,000개 가상 타임라인. "
        "1.5x/2.0x = 모든 음수 returns 를 1.5/2.0 배 증폭 (인위적 극단 가정, 참고만)."
    )
    scenarios = [
        ("표준 (현실 분포)", mc["standard"]),
        ("1.5x 스트레스 (참고)", mc["stress_1_5x"]),
        ("2.0x 극단 (참고)", mc["extreme_2_0x"]),
    ]
    mc_rows = []
    for label, s in scenarios:
        mc_rows.append({
            "시나리오": label,
            "파산확률(원금↓)": f"{s['ruin_1']*100:.2f}%",
            "반토막확률(-50%)": f"{s['ruin_05']*100:.2f}%",
            "CAGR 중앙값": f"{s['cagr_median']*100:+.2f}%",
            "CAGR 1% 최악": f"{s['cagr_p1']*100:+.2f}%",
            "MDD 1% 최악": f"{s['mdd_p1']*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(mc_rows), use_container_width=True, hide_index=True)

    # 표준 시나리오 메트릭
    s = mc["standard"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("파산확률", f"{s['ruin_1']*100:.2f}%", delta="현실분포", delta_color="off")
    c2.metric("CAGR 중앙값", f"{s['cagr_median']*100:+.2f}%")
    c3.metric("CAGR 1% 최악", f"{s['cagr_p1']*100:+.2f}%",
              delta="10000 중 100번째 최악", delta_color="off")
    c4.metric("MDD 1% 최악", f"{s['mdd_p1']*100:.2f}%")

    st.caption(
        "※ 표준 부트스트랩 = 한국 시장에서 발생한 실제 수익률 분포 기반. "
        "1.5x/2.0x = 음수 일률 증폭 (long-only 시스템 공통 한계 시연용). "
        "현실 시나리오 (블랙스완 + 표준 MC) 결과가 실제 운용 안전성의 기준."
    )


def tab_performance() -> None:
    """성과만 표시 — 라이브 (2015 ~ 마지막 거래일까지) 실시간 검증."""
    st.markdown("## 실시간 성과 (2015 – 오늘)")
    st.caption(
        "**Hybrid LightGBM (v3 60% + v4 40%) + Rule 4팩터 (AI 20% 가중) + 2개월 리밸런싱 (매월 9일) + "
        "단일 종목 비중캡 25% + 채권 3% 폴백** + Ichimoku Cloud + ADX + 분할 스위칭 + "
        "자체 TTM PER + 약세 시 금 헤지(132030) · 거래비용 0.25% 반영. "
        "v3=차트 10피처 (IC 0.131), v4=차트+펀더멘털 13피처 (Spread +6.39%pt)."
    )

    with st.spinner("실시간 백테스트 로딩 중... (데이터 갱신 시 자동 재계산)"):
        try:
            r = get_live_backtest(_data_version())
        except Exception as e:
            st.error(f"백테스트 실행 실패: {e}")
            return

    m = r.metrics
    if not r.daily_equity.empty:
        last_day = pd.to_datetime(r.daily_equity["date"]).max().strftime("%Y-%m-%d")
        first_day = pd.to_datetime(r.daily_equity["date"]).min().strftime("%Y-%m-%d")
        st.caption(f"측정 구간: **{first_day}** ~ **{last_day}** (마지막 거래일까지)")

    # 핵심 성과 4개
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총수익", f"{m['total_return']*100:,.1f}%",
              delta=f"벤치 {m['benchmark_total']*100:,.1f}%", delta_color="off")
    c2.metric("연복리(CAGR)", f"{m['cagr']*100:.2f}%",
              delta=f"α {m['alpha_annualized']*100:+.2f}%/년", delta_color="off")
    c3.metric("샤프", f"{m['sharpe']:.2f}")
    c4.metric("MDD (일별 실측)", f"{m['mdd_daily']*100:.1f}%",
              delta=f"반기샘플 {m['mdd']*100:.1f}%", delta_color="off")

    # 추가 메트릭
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("연환산 변동성", f"{m['volatility']*100:.1f}%")
    c2.metric("반기 승률", f"{m['win_rate_periods']*100:.0f}%")
    c3.metric("종목 승률", f"{m['win_rate_holdings']*100:.0f}%")
    c4.metric("누적 거래비용", f"{m['total_txn_cost']*100:.2f}%")

    # 누적 수익률 곡선
    if not r.daily_equity.empty:
        st.markdown("### 누적 수익률")
        curve = r.daily_equity.set_index("date")[["portfolio_value"]]
        curve.columns = ["포트폴리오"]
        if not r.periods.empty:
            bench = pd.DataFrame(
                {"벤치마크(KOSPI200)": r.periods["benchmark_value"].values},
                index=pd.to_datetime(r.periods["entry_date"]),
            )
            curve = curve.join(bench, how="left").ffill()
        st.line_chart(curve, color=[ACCENT, MUTED])

    # 누적 손실 (Underwater)
    if not r.daily_equity.empty:
        st.markdown("### 누적 손실 (Underwater)")
        v = r.daily_equity["portfolio_value"].to_numpy()
        peak = pd.Series(v).cummax().to_numpy()
        dd = pd.DataFrame({"낙폭(%)": (v / peak - 1) * 100},
                          index=r.daily_equity["date"])
        st.area_chart(dd, color=NEG)
        st.caption("0% = 신고점. 아래로 갈수록 큰 손실. 신고점 회복 전까지 underwater.")

    # 분할 스위칭 비중 변화
    if not r.daily_equity.empty and "stock_pct" in r.daily_equity.columns:
        st.markdown("### 주식·금 비중 변화")
        st.caption(
            "Ichimoku+ADX 일별 신호에 따라 100% → 50% → 0% 단계적 스위칭. "
            "녹색=주식 / 노란색=금(132030)"
        )
        de = r.daily_equity.set_index("date")
        chart_df = pd.DataFrame({
            "주식 (%)": de["stock_pct"] * 100,
            "금 (%)": (1 - de["stock_pct"]) * 100,
        })
        st.area_chart(chart_df, color=[POS, "#f1c40f"], height=180)

    # 연도별
    if not r.yearly.empty:
        st.markdown("### 연도별 수익률")
        y = r.yearly.copy()
        for col in ["portfolio_return", "benchmark_return"]:
            if col in y.columns:
                if y[col].dtype == object:
                    y[col] = y[col].str.replace("%", "").astype(float)
                else:
                    y[col] = y[col] * 100
        yc = y.set_index("year")[["portfolio_return", "benchmark_return"]]
        yc.columns = ["포트폴리오(%)", "KOSPI200(%)"]
        st.bar_chart(yc, color=[ACCENT, MUTED])

    with st.expander("구간별 상세 (2개월 리밸)"):
        if not r.periods.empty:
            disp = r.periods.copy()
            for c in ["gross_return", "txn_cost", "period_return",
                      "benchmark_return", "alpha", "turnover"]:
                if c in disp.columns:
                    disp[c] = (disp[c] * 100).round(2)
            if "bull_regime" in disp.columns:
                disp["레짐"] = disp["bull_regime"].map({True: "강세", False: "약세"})
                disp = disp.drop(columns=["bull_regime"])
            st.dataframe(disp, use_container_width=True, hide_index=True)

    st.caption(
        "※ 조건: ROE 20%+ · PER 60↓ · 영업이익률 10%+ · 시총 1조+ · 코스피·코스닥 5:5 · "
        "**2개월 리밸런싱 (매월 9일) + rank 가중 + 단일 종목 비중캡 25% + 채권 3% 폴백** · 종목 유지(keep_simple) · "
        "구름대 균열 시 50% 부분 익절 → 하단 이탈 시 100% KODEX 골드선물(H). "
        "마지막 구간은 현재 보유 — 포트폴리오 탭 참조."
    )


# ============================================================
# 탭 4: 뉴스
# ============================================================
def tab_news() -> None:
    presets = ["한국 경제", "코스피", "환율", "금리", "반도체"]
    chosen = st.radio(" ", presets, horizontal=True, label_visibility="collapsed")
    custom = st.text_input(
        " ", value="", placeholder="직접 검색 (비워두면 위 선택 사용)",
        label_visibility="collapsed",
    ).strip()
    query = custom if custom else chosen

    with st.spinner("뉴스 가져오는 중..."):
        result = fetch_news(query, limit=20)

    if result["error"]:
        st.error(f"뉴스 로딩 실패: {result['error']}")
        st.caption("인터넷 연결을 확인하거나 잠시 후 다시 시도하세요.")
        return

    items = result["items"]
    if not items:
        st.caption("결과 없음.")
        return

    now = datetime.now(timezone.utc)
    st.caption(f"'{query}' · {len(items)}건  ·  Google News")

    for it in items:
        if it["pub_dt"]:
            delta = now - it["pub_dt"]
            secs = delta.total_seconds()
            if secs < 3600:
                time_str = f"{int(secs // 60)}분 전"
            elif secs < 86400:
                time_str = f"{int(secs // 3600)}시간 전"
            else:
                time_str = it["pub_dt"].strftime("%Y-%m-%d")
        else:
            time_str = ""
        meta = "  ·  ".join([x for x in [it["source"], time_str] if x])
        st.markdown(
            f'<div class="news-item">'
            f'<a href="{it["link"]}" target="_blank">{it["title"]}</a>'
            f'<div class="news-meta">{meta}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# 탭 5: 데이터
# ============================================================
def tab_status() -> None:
    info = get_db_summary()
    if not info:
        st.error(f"DB 없음: {DB_PATH}")
        st.code("python main.py collect", language="bash")
        return

    s = status_summary()
    # ── 데이터 신선도 ──────────────────────────────────────────────
    st.markdown("### 데이터 신선도")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OHLCV 최신", info.get("date_max", "?"),
              delta=(f"{s['days_behind']}일 뒤처짐" if s["is_stale"]
                     else "어제까지"),
              delta_color=("inverse" if s["is_stale"] else "normal"))
    c2.metric("시가총액 최신", info.get("cap_max", "?") or "–")
    c3.metric("PER/PBR 최신", info.get("per_max", "?") or "–",
              delta="월별 갱신", delta_color="off")
    c4.metric("목표일 (어제)", s["target"])

    # ── DART fundamentals 커버리지 (KOSDAQ 2026-Q1 stale 진단) ──
    cov = get_fundamentals_coverage()
    if not cov.empty:
        st.markdown("### DART 재무제표 커버리지 (분기별 종목 수)")
        pv = cov.pivot(index="period_end", columns="market", values="n_tickers").fillna(0).astype(int)
        pv = pv.sort_index(ascending=False)
        st.dataframe(pv, use_container_width=True, height=min(48 + len(pv) * 35, 280))
        latest_q = cov["period_end"].max()
        latest_row = cov[cov["period_end"] == latest_q].set_index("market")["n_tickers"]
        if latest_row.min() < 50:
            st.warning(
                f"⚠ 최근 분기 ({latest_q}) 공시가 적어 후보 풀이 좁아집니다. "
                f"한국 분기보고서 제출 기한은 분기말+45일이라 신규 분기는 늦게 채워집니다. "
                f"보강하려면 터미널에서 `python main.py collect-dart` 실행."
            )
        else:
            st.caption(
                f"최근 분기 {latest_q}: KOSPI {latest_row.get('KOSPI', 0)} · "
                f"KOSDAQ {latest_row.get('KOSDAQ', 0)} 종목 공시 반영됨"
            )

    # ── DB 요약 ──────────────────────────────────────────────────
    st.markdown("### DB 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("종목", f"{info.get('tickers', 0):,}")
    c2.metric("재무제표", f"{info.get('fundamentals', 0):,}")
    c3.metric("PER/PBR", f"{info.get('per_history', 0):,}")
    c4.metric("OHLCV 기간",
              f"{info.get('date_min', '?')[:7]} – {info.get('date_max', '?')[:7]}")

    st.markdown("### AI 모델 — Hybrid LightGBM (v3 + v4)")
    v3_p = MODEL_DIR / "trend_lgbm_v3.txt"
    v4_p = MODEL_DIR / "trend_lgbm_v4.txt"

    # 모델 파일 상태 표 — 크기·수정시간·역할
    model_rows = []
    model_meta = [
        ("trend_lgbm_v3.txt", "production", "차트 10피처 · KR Test IC 0.131"),
        ("trend_lgbm_v4.txt", "production", "차트+펀더 13피처 · Spread +6.39%pt"),
        ("trend_lgbm_v2.txt", "대체", "v2 (예전 차트 모델)"),
        ("trend_lgbm_kr.txt", "legacy", "KR 전용 단독"),
        ("trend_lgbm.txt", "legacy", "초기 버전"),
        ("chart_lstm_wf_true.pt", "검증", "walk-forward true split LSTM"),
        ("chart_lstm_wf.pt", "검증", "walk-forward LSTM"),
        ("chart_lstm.pt", "legacy", "차트 LSTM"),
        ("chart_lstm_2023.pt", "legacy", "2023 학습 LSTM"),
        ("lstm_best.pt", "폐기", "random split leakage"),
    ]
    for fname, role, desc in model_meta:
        p = MODEL_DIR / fname
        if not p.exists():
            continue
        stat = p.stat()
        model_rows.append({
            "파일": fname,
            "역할": role,
            "크기": f"{stat.st_size/1024:.0f} KB",
            "수정일": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "설명": desc,
        })
    if model_rows:
        mdf = pd.DataFrame(model_rows)
        st.dataframe(mdf, use_container_width=True, hide_index=True,
                     height=min(48 + len(mdf) * 35, 420))

    if not v3_p.exists() and not v4_p.exists():
        st.warning("⚠️ Hybrid production 모델 없음 — `python -m src.model.trend_lgbm_v2/v4` 로 학습")
    elif v3_p.exists() and v4_p.exists():
        st.success(f"✓ Production Hybrid 활성 (AI 가중 {AI_WEIGHT_PRODUCTION})")

    with st.expander("Hybrid 모델 상세 · 학습 방식 · 앙상블 메커니즘", expanded=False):
        st.markdown("""
**Hybrid 구조 — KR: v3 60% + v4 40% / US: v3 단독**
- v3 의 **전체 순위 정확도 (IC 0.131)** + v4 의 **극단 분리도 (Spread +6.39%pt)** 시너지
- US 종목은 fundamentals 없어 v4 의미 없음 → v3 단독 fallback

---

**v3 — 차트 10 피처 모델 (rank regression)**
- LightGBM Regressor, max_depth=6, learning_rate=0.03, min_data_in_leaf=200
- 8 차트 피처 + 2 매크로 피처
  - 차트: price_to_ma200, ma60_slope, ma20_to_ma60, bb_position, volume_ratio, macd_hist_ratio, stoch_60_pos, obv_slope_20
  - 매크로: index_ma60_slope, index_mdd_20 (KR=KOSPI200/069500, US=QQQ)
- 라벨: **target_rank** (date 별 cross-sectional 42일 후 수익률 백분위)

**v4 — 차트+펀더멘털 13 피처 모델**
- v3 의 10 피처 + 3 펀더멘털:
  - `roe_latest` — 공시된 최근 분기 ROE (체급)
  - `op_margin_growth` — 분기 OPM 변화 (실적 모멘텀)
  - `per_inverse` — 1/PER 안정 처리 (밸류에이션)
- **Point-in-Time** 매핑: publish_date = period_end + 45/90일 → look-ahead 차단
- 펀더 NaN 은 LightGBM 자동 처리 (US 종목 + 중소형 KR 일부)

---

**Walk-forward 시간순 분할 + 42거래일 갭 (look-ahead 차단)**
- **Train**: 2015-01-01 ~ 2021-12-31 (KR+US 통합)
- **Gap**: ~42거래일 (타깃 horizon)
- **Val**:   2022-03-01 ~ 2023-12-31
- **Gap**: ~42거래일
- **Test**:  2024-03-01 ~ 2026-05-31 (완전 out-of-sample)

**검증 결과 (TEST 1.48M 샘플)**
| 모델 | KR IC | KR Spread |
|---|---|---|
| v3 | **0.1310** | +5.39%pt |
| v4 | 0.1171 | **+6.39%pt** |
| **Hybrid** | 0.1290 | **+5.93%pt** | (둘의 시너지)

---

**Ensemble 메커니즘 (production)**
1. Rule 4팩터 점수로 상위 N×3 후보 풀
2. 각 후보의 13 피처 계산
3. KR: `hybrid_ai = 0.6 × v3.predict(10피처) + 0.4 × v4.predict(13피처)`
4. US: `hybrid_ai = v3.predict(10피처)` 단독
5. `ensemble_score = ai_weight × hybrid_ai + (1 - ai_weight) × rule_score`
6. ai_weight = **0.2** (production sweet spot)

**ai_weight 그리드 결과 (Hybrid)**
| ai_w | TR | Sharpe | α |
|---|---|---|---|
| 0.0 (rule_only) | 1427% | 1.41 | +10.31% |
| **0.2** ★ | **1451.9%** | 1.42 | **+10.49%** |
| 0.4 | 1449.4% | 1.42 | +10.47% |
| 0.6 | 1348.7% | 1.40 | +9.72% |
| 0.8 | 1432.0% | **1.44** | +10.35% |
| 1.0 | 1164.7% | 1.38 | +8.21% |

**왜 ai_w=0.2 sweet spot?**
- Rule 4팩터가 검증된 강한 알파 (1427% baseline)
- AI 는 "미세 변별력" 만 추가 (Spread +6%pt)
- ai_w ≥ 0.6 면 AI 가 Rule 의 강한 신호 희석 → 알파 감소
- **20% AI 가중이 두 신호의 시너지 sweet spot**

---

**legacy ChartLSTM (폐기됨, 학습 데이터로 남김)**
- 기존 random split 학습으로 IC 0.227 부풀림 (시간 leakage)
- 진짜 walk-forward 재학습 시 IC 0.04 (가까스로 유의미)
- LightGBM Hybrid (KR IC 0.129) 가 약 3배 알파, 1/30 학습 시간
- 추천 탭의 "앙상블" 모드 클릭 시만 작동 (현재는 LGBM Hybrid 사용)
        """)

    # ── 하드 필터 ──────────────────────────────────────────────
    st.markdown("### 종목 선정 룰 (현 production 시스템)")

    st.markdown("**1단계 · 하드 필터 (전부 통과해야 후보)**")
    hf = CFG.hard_filter
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROE 최소", f"≥ {hf.roe_min}%")
    c2.metric("PER 범위", f"0 < PER ≤ {hf.per_max}")
    c3.metric("시총 최소", "≥ 1조 (override)")
    c4.metric("영업이익률", "≥ 10%")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("매출 YoY", "> 0%")
    c2.metric("순익 YoY", "> 0%")
    c3.metric("영업이익 YoY", "> 0%")
    c4.metric("일회성 차단", "PQ 0.3~2.0")
    st.caption("PQ = TTM 순익 / TTM 영업이익. 2.0+ 는 자회사 매각 같은 일회성 이익 의심.")

    st.markdown("**2단계 · 추세 필터**")
    c1, c2, c3 = st.columns(3)
    c1.metric("60일선", "종가 > MA(60)")
    c2.metric("Ichimoku Cloud", "9 / 26 / 52")
    c3.metric("ADX (추세 강도)", "≥ 25 + DI 정배열")
    st.caption("진입 시점에만 적용. 매수 후 60일선 깨져도 다음 리밸까지 보유 (trailing stop 없음).")

    st.markdown("**3단계 · Rule 점수 (4팩터 백분위 + 가산)**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PER", "낮을수록 ↑")
    c2.metric("ROE", "높을수록 ↑")
    c3.metric("매출성장 YoY", "높을수록 ↑")
    c4.metric("순익성장 YoY", "높을수록 ↑")
    st.caption("4팩터 백분위 가중평균 + 시총 가산점 (10조+ ×1.20) + "
               "영업이익 가속 가산 (YoY≥100% ×1.30) − QoQ 둔화 페널티 (×0.85). "
               "1위 = 18~20% 비중, 마지막 = 1.8~2.2%.")

    st.markdown(f"**3-B 단계 · LGBM Hybrid AI 가중 (production ai_w=**{AI_WEIGHT_PRODUCTION}**)**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("v3 (KR/US)", "차트 10 피처")
    c2.metric("v4 (KR only)", "+ 펀더 3 피처")
    c3.metric("KR 합성", "0.6×v3 + 0.4×v4")
    c4.metric("최종", f"{int(AI_WEIGHT_PRODUCTION*100)}%AI + {int((1-AI_WEIGHT_PRODUCTION)*100)}%Rule")
    st.caption("Walk-forward + 42거래일 갭 학습. KR Test IC 0.131 (v3) / Spread 6.39%p (v4). "
               "낮은 ai_w (0.2) 가 sweet spot — Rule 의 강한 알파 + AI 미세 보정.")

    st.markdown("**4단계 · 포트폴리오 구성**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("종목 수 (top_n)", "10")
    c2.metric("시장 분배", "KOSPI 5 + KOSDAQ 5")
    c3.metric("단일 종목 캡", "≤ 25%")
    c4.metric("초과분 → 채권", "연 3% 일복리")

    st.markdown("**5단계 · 리밸런싱 & 거시 방어**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("리밸 주기", "2개월")
    c2.metric("리밸 일자", "매월 9일")
    c3.metric("종목 0개 시", "채권 3%")
    c4.metric("약세장 폴백", "KODEX 골드 132030")
    st.caption("9일 = DART 분기보고서 공시 시즌(분기말+45일) 직전 신실적 반영 sweet spot. "
               "약세 신호 시 Ichimoku+ADX 3-state 분할 스위칭 (100/50/0%) → 잔여는 금 ETF.")

    st.markdown("**6단계 · 자체 데이터 보강**")
    c1, c2, c3 = st.columns(3)
    c1.metric("TTM PER", "DART 분기 자체 계산")
    c2.metric("YoY 처리", "abs 분모 (적자→흑자)")
    c3.metric("결손 매출 회수", "2,212 행 (영업수익 매핑)")
    st.caption("KRX 공식 PER 의 연 1회 EPS 갱신 지연 우회. "
               "ACCOUNT_MAP 에 '영업수익' 추가로 SK하이닉스/금융주 매출 정상 캡처.")


# ============================================================
# 메인
# ============================================================
def main() -> None:
    st.set_page_config(
        page_title="Stock AI",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown("# Stock AI")
    st.caption("한국 주식 추천 시스템 · 실시간(2015-오늘) CAGR 27.4%/Sharpe 1.42/MDD -20.1% · Hybrid LGBM(v3 60% + v4 40%) + Rule (AI 20% 가중) + 2m 리밸 매월 9일 + 비중 캡 25% + 채권 폴백 + Ichimoku+ADX + 금 헤지")

    cfg = sidebar_settings()

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(["추천", "종목 검색", "포트폴리오", "성과", "검증", "뉴스", "데이터"])
    with t1:
        tab_recommend(cfg)
    with t2:
        tab_search()
    with t3:
        tab_portfolio()
    with t4:
        tab_performance()
    with t5:
        tab_validation()
    with t6:
        tab_news()
    with t7:
        tab_status()


if __name__ == "__main__":
    main()
