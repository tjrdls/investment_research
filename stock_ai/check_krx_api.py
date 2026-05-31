"""KRX OpenAPI 활용신청 '승인' 상태 확인기
==========================================
KRX 포털이 승인 상태를 잘 안 보여주므로, 인증키로 실제 API 를 호출해
각 엔드포인트가 **승인됐는지**(200) / 대기중인지(401) 직접 확인한다.

사용법:
    python check_krx_api.py            # 가장 최근 평일 기준 확인
    python check_krx_api.py 20260528   # 특정 날짜(basDd) 기준 확인

판정:
    ✅ 승인됨   — 200 응답 (rows 0 이어도 휴장일일 뿐, 승인은 된 것)
    ⏳ 승인대기 — 401 Unauthorized API Call
    ❌ 경로오류 — 404 (엔드포인트 경로 틀림)
    ⚠️ 기타     — 그 외 상태/네트워크 오류
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import requests  # noqa: E402

# 로더와 동일한 base/필드 상수 재사용 (import 만 — DB 생성 등 부작용 없음)
try:
    from src.config import CFG
    from src.data import krx_openapi_loader as K
    BASE = CFG.data.krx_api_base.rstrip("/")
    FIELDS = [K.F_TICKER, K.F_NAME, K.F_OPEN, K.F_HIGH, K.F_LOW,
              K.F_CLOSE, K.F_VOLUME, K.F_MKTCAP, K.F_SHARES]
except Exception:  # 단독 실행 폴백
    BASE = "http://data-dbg.krx.co.kr/svc/apis"
    FIELDS = ["ISU_CD", "ISU_NM", "TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC",
              "TDD_CLSPRC", "ACC_TRDVOL", "MKTCAP", "LIST_SHRS"]

# (group, endpoint, 라벨, 필수여부)
ENDPOINTS = [
    ("sto", "stk_bydd_trd", "유가증권(KOSPI) 일별매매정보", True),
    ("sto", "ksq_bydd_trd", "코스닥 일별매매정보", True),
    ("idx", "kospi_dd_trd", "KOSPI 지수 일별 (선택)", False),
    ("idx", "kosdaq_dd_trd", "KOSDAQ 지수 일별 (선택)", False),
]


def recent_weekday(anchor: datetime | None = None) -> str:
    d = (anchor or datetime.now()) - timedelta(days=1)
    while d.weekday() >= 5:  # 토/일 스킵
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


def check_one(key: str, group: str, endpoint: str, bas_dd: str) -> dict:
    url = f"{BASE}/{group}/{endpoint}"
    try:
        r = requests.get(url, headers={"AUTH_KEY": key, "Accept": "application/json"},
                         params={"basDd": bas_dd}, timeout=30)
    except Exception as e:  # noqa: BLE001
        return {"status": "ERR", "code": None, "rows": 0, "msg": f"{type(e).__name__}: {e}", "sample": None}

    if r.status_code == 200:
        try:
            block = r.json().get("OutBlock_1") or r.json().get("outBlock_1") or []
        except Exception:
            block = []
        sample = block[0] if block else None
        return {"status": "OK", "code": 200, "rows": len(block), "msg": "승인됨", "sample": sample}
    if r.status_code == 401:
        return {"status": "WAIT", "code": 401, "rows": 0, "msg": "승인 대기 (Unauthorized)", "sample": None}
    if r.status_code == 404:
        return {"status": "PATH", "code": 404, "rows": 0, "msg": "엔드포인트 경로 오류", "sample": None}
    return {"status": "ERR", "code": r.status_code, "rows": 0, "msg": r.text[:120], "sample": None}


ICON = {"OK": "✅", "WAIT": "⏳", "PATH": "❌", "ERR": "⚠️"}


def main() -> int:
    bas_dd = sys.argv[1] if len(sys.argv) > 1 else recent_weekday()
    key = os.getenv("KRX_AUTH_KEY", "").strip()

    print("=" * 64)
    print(f"  KRX OpenAPI 승인 상태 확인   (기준일 basDd={bas_dd})")
    print("=" * 64)
    if not key:
        print("⚠️  KRX_AUTH_KEY 가 .env 에 없습니다. 먼저 인증키를 넣으세요.")
        return 1
    print(f"인증키: 로드됨 ({len(key)}자)\n")

    results = []
    for group, endpoint, label, required in ENDPOINTS:
        res = check_one(key, group, endpoint, bas_dd)
        results.append((endpoint, required, res))
        tag = "[필수]" if required else "[선택]"
        extra = f" · {res['rows']}행" if res["status"] == "OK" else ""
        print(f"  {ICON[res['status']]} {tag} {label}")
        print(f"        {group}/{endpoint} → {res['code']} {res['msg']}{extra}")

    # 승인된 엔드포인트에서 실제 응답 필드명 확인 (매핑 점검용)
    ok_sample = next((r["sample"] for _, _, r in results if r["status"] == "OK" and r["sample"]), None)
    if ok_sample:
        actual = list(ok_sample.keys())
        print("\n  실제 응답 필드명:")
        print("   ", actual)
        missing = [f for f in FIELDS if f not in actual]
        if missing:
            print(f"  ⚠️ 로더 매핑과 불일치 — 누락 필드: {missing}")
            print("     → src/data/krx_openapi_loader.py 의 F_* 상수를 위 실제 필드명에 맞게 수정 필요.")
        else:
            print("  ✅ 로더 매핑(F_*) 과 일치 — 코드 수정 불필요.")

    # 최종 판정
    required_states = [r["status"] for _, req, r in results if req]
    print("\n" + "-" * 64)
    if all(s == "OK" for s in required_states):
        print("🎉 필수 API 전부 승인됨! 이제 데이터 수집 가능:")
        print("     python main.py collect --start 2015-01-01")
        return 0
    if any(s == "WAIT" for s in required_states):
        print("⏳ 아직 승인 대기 중. openapi.krx.co.kr MyPage 에서 활용신청 상태 확인.")
        print("   (필수: sto/stk_bydd_trd, sto/ksq_bydd_trd — 보통 1일 내 승인)")
        return 2
    print("⚠️ 예상치 못한 상태 — 위 메시지를 확인하세요.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
