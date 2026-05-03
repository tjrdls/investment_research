# -*- coding: utf-8 -*-
"""
역할: OpenDART API를 사용하여 재무제표 데이터를 수집하는 모듈.
매출, 영업이익, 순이익, 부채비율 등의 주요 재무 지표를 가져와 저장한다.
"""

import logging
import requests
import pandas as pd
import os
import time
import zipfile
import io
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DART_API_KEY = os.getenv("DART_API_KEY", "")
if not DART_API_KEY:
    logger.warning("DART_API_KEY가 설정되지 않았습니다. 재무 데이터 수집이 실패합니다.")


def get_corp_code_map():
    """
    DART 전체 기업코드 목록 다운로드
    
    :return: dict {stock_code: {"corp_code": ..., "name": ...}}
    """
    logger.info("기업코드 목록 다운로드 중...")
    
    try:
        r = requests.get(
            "https://opendart.fss.or.kr/api/corpCode.xml",
            params={"crtfc_key": DART_API_KEY},
            timeout=30
        )
        
        z = zipfile.ZipFile(io.BytesIO(r.content))
        xml = z.read("CORPCODE.xml").decode("utf-8")
        root = ET.fromstring(xml)
        
        mapping = {}
        for item in root.findall("list"):
            corp_code = item.findtext("corp_code", "")
            stock_code = item.findtext("stock_code", "").strip()
            corp_name = item.findtext("corp_name", "")
            
            if stock_code:
                mapping[stock_code] = {
                    "corp_code": corp_code,
                    "name": corp_name
                }
        
        logger.info("✅ %d 개 종목 로드", len(mapping))
        return mapping
    
    except Exception as e:
        logger.error("기업코드 로드 실패: %s", e)
        return {}


def dart_api_call(endpoint, params):
    """
    DART API 호출 (공통 함수)
    
    :param endpoint: API 엔드포인트
    :param params: 파라미터 dict
    :return: JSON response
    """
    params["crtfc_key"] = DART_API_KEY
    
    try:
        r = requests.get(
            "https://opendart.fss.or.kr/api/{}".format(endpoint),
            params=params,
            timeout=10
        )
        return r.json()
    except requests.RequestException as e:
        return {"status": "999", "message": str(e)}


def get_financial_statements(corp_code, year, reprt_code="11011"):
    """
    재무제표 수집 (DART fnlttSinglAcntAll)
    
    :param corp_code: 기업코드
    :param year: 사업연도 (예: 2024)
    :param reprt_code: 보고서 코드 (11011=사업보고서, 11012=반기, 11013=1분기, 11014=3분기)
    :return: dict with key financial metrics
    """
    data = dart_api_call("fnlttSinglAcntAll.json", {
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt_code,
        "fs_div": "CFS"
    })
    
    if data.get("status") != "000":
        return {}
    
    # 주요 계정 항목 찾기
    items = data.get("list", [])
    
    revenue_keys = ["매출액", "수익(매출액)", "영업수익", "매출"]
    op_inc_keys = ["영업이익", "영업이익(손실)"]
    net_inc_keys = [
        "당기순이익", "분기순이익", "반기순이익",
        "당기순이익(손실)", "분기순이익(손실)", "반기순이익(손실)"
    ]
    equity_keys = ["자본총계", "총자본"]
    debt_keys = ["부채총계"]
    
    def find_value(key_list, item_list):
        """주요 항목값 찾기"""
        for key in key_list:
            for item in item_list:
                nm = item.get("account_nm", "").strip()
                if nm == key:
                    amt = (item.get("thstrm_amount", "0") or "0").replace(",", "")
                    try:
                        return float(amt)
                    except (ValueError, TypeError):
                        pass
        return 0.0
    
    result = {
        "revenue": find_value(revenue_keys, items),
        "operating_income": find_value(op_inc_keys, items),
        "net_income": find_value(net_inc_keys, items),
        "total_equity": find_value(equity_keys, items),
        "total_debt": find_value(debt_keys, items),
    }
    
    return result


def get_stock_count(corp_code, year, reprt_code="11011"):
    """
    발행주식수 수집
    
    :param corp_code: 기업코드
    :param year: 사업연도
    :param reprt_code: 보고서 코드
    :return: 발행주식수 또는 None
    """
    data = dart_api_call("stockTotqySttus.json", {
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt_code,
    })
    
    if data.get("status") != "000":
        return None
    
    for item in data.get("list", []):
        if "보통주" in item.get("se", ""):
            cnt = (item.get("distb_stock_co", "0") or "0").replace(",", "")
            try:
                return float(cnt)
            except (ValueError, TypeError):
                pass
    
    return None


def collect_financial_data(stock_code, corp_code, year_range=2):
    """
    주식 종목의 재무 데이터를 수집
    
    :param stock_code: 주식 코드 (예: "005930")
    :param corp_code: 기업 코드 (DART)
    :param year_range: 몇 년치 데이터를 수집할지
    :return: DataFrame with financial metrics
    """
    logger.info("📊 재무제표 데이터 수집 중 (%s)...", stock_code)
    
    records = []
    
    try:
        from datetime import datetime
        current_year = datetime.now().year
        
        for year in range(current_year - year_range, current_year + 1):
            for reprt_code, label in [("11011", "연간"), ("11012", "반기")]:
                fin = get_financial_statements(corp_code, year, reprt_code)
                
                if not fin or fin.get("revenue", 0) == 0:
                    continue
                
                shares = get_stock_count(corp_code, year, reprt_code)
                time.sleep(0.3)
                
                # TTM 지표 계산
                eps = (fin["net_income"] / shares) if shares and shares > 0 else 0
                bps = (fin["total_equity"] / shares) if shares and shares > 0 else 0
                roe = (fin["net_income"] / fin["total_equity"] * 100) if fin["total_equity"] > 0 else 0
                debt_ratio = (fin["total_debt"] / fin["total_equity"] * 100) if fin["total_equity"] > 0 else 0
                
                records.append({
                    "year": year,
                    "report_type": label,
                    "revenue": fin["revenue"],
                    "operating_income": fin["operating_income"],
                    "net_income": fin["net_income"],
                    "total_equity": fin["total_equity"],
                    "total_debt": fin["total_debt"],
                    "shares": shares or 0,
                    "eps": eps,
                    "bps": bps,
                    "roe": roe,
                    "debt_ratio": debt_ratio
                })
                
                time.sleep(0.3)
        
        if not records:
            logger.warning("재무제표 데이터 없음: %s", stock_code)
            return pd.DataFrame()

        df = pd.DataFrame(records)

        os.makedirs("data/downloads", exist_ok=True)
        filename = "data/downloads/{}_financial.csv".format(stock_code)
        df.to_csv(filename, index=False)

        logger.info("💾 재무제표 저장 완료 (%d 기간)", len(records))
        return df

    except Exception as e:
        logger.error("❌ 재무제표 수집 실패: %s", e)
        return pd.DataFrame()


if __name__ == "__main__":
    # 테스트
    corp_map = get_corp_code_map()
    
    if "005930" in corp_map:
        corp_code = corp_map["005930"]["corp_code"]
        data = collect_financial_data("005930", corp_code)
        print(data)