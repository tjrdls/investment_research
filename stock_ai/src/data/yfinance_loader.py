"""
미국 주식 OHLCV 수집기.
yfinance 사용. S&P500 + Nasdaq100 = 약 500종목.
한국 데이터와 같은 ohlcv 테이블에 저장. market 컬럼으로 NYSE/NASDAQ 구분.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DB_PATH

logger = logging.getLogger(__name__)


class YFinanceLoader:
    """yfinance로 미국 주식 OHLCV 수집."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        try:
            import yfinance as yf  # noqa: F401
        except ImportError:
            raise ImportError(
                "yfinance 미설치. 설치 명령:\n"
                "  pip install yfinance"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ----- 종목 리스트 (정적) -----
    @staticmethod
    def get_universe(
        sp500: bool = True,
        nasdaq100: bool = True,
    ) -> list[tuple[str, str, str]]:
        """
        대상 종목 리스트 반환: [(ticker, name, market), ...]
        고정 리스트 사용 (Wikipedia 차단 우회).
        """
        all_rows: list[tuple[str, str, str]] = []

        if sp500:
            # S&P500 상위 100개 + 주요 종목 (전체 500개는 너무 많으니 샘플)
            sp500_tickers = [(tk, tk, 'SP500') for tk in [
                "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
                "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
                "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK",
                "AMP", "AME", "AMGN", "APH", "ADI", "ANSS", "AON", "APA", "AAPL", "AMAT",
                "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK", "ADP",
                "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX", "BDX", "WRB",
                "BRK-B", "BBY", "BIO", "TECH", "BIIB", "BLK", "BX", "BK", "BA", "BKNG",
                "BWA", "BXP", "BSX", "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR", "BG",
                "CHRW", "CDNS", "CZR", "CPT", "CPB", "COF", "CAH", "KMX", "CCL", "CARR",
                "CAT", "CBOE", "CBRE", "CDW", "CE", "COR", "CNC", "CNP", "CF", "CRL",
                "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO",
                "C", "CFG", "CLX", "CME", "CMS", "KO", "CTSH", "CL", "CMCSA", "CMA",
                "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT", "GLW", "CTVA", "CSGP",
                "COST", "CTRA", "CCI", "CSX", "CMI", "CVS", "DHR", "DRI", "DVA", "DE",
                "DAL", "DVN", "DXCM", "FANG", "DLR", "DFS", "DG", "DLTR", "D", "DPZ",
                "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "EMN", "ETN", "EBAY", "ECL",
                "EIX", "EW", "EA", "ELV", "LLY", "EMR", "ENPH", "ETR", "EOG", "EPAM",
                "EQT", "EFX", "EQIX", "EQR", "ESS", "EL", "ETSY", "EG", "EVRG", "ES",
                "EXC", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT",
                "FDX", "FIS", "FITB", "FSLR", "FE", "FI", "FLT", "F", "FTNT", "FTV",
                "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GE", "GEHC", "GEN", "GNRC",
                "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL", "GS", "HAL", "HIG",
                "HAS", "HCA", "HSIC", "HSY", "HES", "HPE", "HLT", "HOLX", "HD", "HON",
                "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX",
                "IDXX", "ITW", "ILMN", "INCY", "IR", "INTC", "ICE", "IFF", "IP", "IPG",
                "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J",
                "JNJ", "JCI", "JPM", "JNPR", "K", "KVUE", "KDP", "KEY", "KEYS", "KMB",
                "KIM", "KMI", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX", "LW", "LVS",
                "LDOS", "LEN", "LIN", "LYV", "LKQ", "LMT", "L", "LOW", "LULU", "LYB",
                "MTB", "MRO", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA", "MTCH",
                "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP",
                "MU", "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST",
                "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NWL", "NEM",
                "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH",
                "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON",
                "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PANW", "PARA", "PH", "PAYX", "PAYC",
                "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW", "PXD", "PNC",
                "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG", "PTC",
                "PSA", "PHM", "QRVO", "PWR", "QCOM", "DGX", "RL", "RJF", "RTX", "O",
                "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "ROK", "ROL", "ROP", "ROST",
                "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", "SEE", "SRE", "NOW", "SHW",
                "SPG", "SWKS", "SJM", "SNA", "SO", "LUV", "SWK", "SBUX", "STT", "STLD",
                "STE", "SYK", "SMCI", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR",
                "TRGP", "TGT", "TEL", "TDY", "TFX", "TER", "TSLA", "TXN", "TXT", "TMO",
                "TJX", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN", "USB",
                "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", "UNH", "UHS", "VLO", "VTR",
                "VLTO", "VRSN", "VRSK", "VZ", "VRTX", "VFC", "VTRS", "V", "VMC", "WAB",
                "WBA", "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST",
                "WDC", "WY", "WHR", "WMB", "WTW", "GWW", "WYNN", "XEL", "XYL", "YUM",
                "ZBRA", "ZBH", "ZTS", "ZION",
            ]]
            all_rows.extend(sp500_tickers)
            logger.info("S&P500: %d종목 (샘플)", len(sp500_tickers))

        if nasdaq100:
            # Nasdaq100 주요 종목
            nasdaq_tickers = [
                ("QQQ", "Invesco QQQ Trust", "NASDAQ100"),
                ("XBI", "SPDR S&P Biotech ETF", "NASDAQ100"),
                ("ASML", "ASML Holding", "NASDAQ100"),
                ("LRCX", "Lam Research", "NASDAQ100"),
                ("MSTR", "MicroStrategy", "NASDAQ100"),
                ("MRVL", "Marvell Technology", "NASDAQ100"),
                ("CRWD", "CrowdStrike", "NASDAQ100"),
                ("SNPS", "Synopsys", "NASDAQ100"),
                ("CDNS", "Cadence Design", "NASDAQ100"),
                ("ABNB", "Airbnb", "NASDAQ100"),
                ("MRNA", "Moderna", "NASDAQ100"),
                ("BKNG", "Booking Holdings", "NASDAQ100"),
                ("COST", "Costco", "NASDAQ100"),
                ("AMGX", "Amgen", "NASDAQ100"),
                ("ADBE", "Adobe", "NASDAQ100"),
                ("AVGO", "Broadcom", "NASDAQ100"),
                ("INTU", "Intuit", "NASDAQ100"),
                ("OKTA", "Okta", "NASDAQ100"),
                ("ZM", "Zoom Video Communications", "NASDAQ100"),
                ("SHOP", "Shopify", "NASDAQ100"),
            ]
            for tk, name, market in nasdaq_tickers:
                if tk not in {r[0] for r in all_rows}:
                    all_rows.append((tk, name, market))
            logger.info("Nasdaq100: %d종목 (샘플)", len(nasdaq_tickers))

        return all_rows

    # ----- OHLCV 수집 -----
    def collect_one(
        self, ticker: str, name: str, market: str,
        start: str = "2015-01-01",
        end: Optional[str] = None,
    ) -> int:
        """
        한 종목의 OHLCV를 yfinance에서 받아 DB에 저장.
        이미 받은 날짜는 자동 스킵 (INSERT OR IGNORE).
        """
        import yfinance as yf

        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df is None or df.empty:
                return 0
        except Exception as e:
            logger.warning("%s 수집 실패: %s", ticker, e)
            return 0

        # 컬럼 정리 (yfinance multi-index 처리)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df["ticker"] = ticker

        rows = df[["ticker", "date", "open", "high", "low", "close", "volume"]].values.tolist()

        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO tickers (ticker, name, market) VALUES (?, ?, ?)",
                (ticker, name, market),
            )
            conn.executemany("""
                INSERT OR IGNORE INTO ohlcv
                (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
            return conn.total_changes

    def collect_all(
        self, start: str = "2015-01-01",
        end: Optional[str] = None,
        sp500: bool = True, nasdaq100: bool = True,
    ) -> dict:
        """
        S&P500 + Nasdaq100 전 종목 수집.
        반환: {"total": N, "ok": N, "fail": N, "rows": N, "elapsed_sec": N}
        """
        universe = self.get_universe(sp500=sp500, nasdaq100=nasdaq100)
        logger.info("미국 주식 수집 시작: %d종목", len(universe))

        start_time = time.time()
        ok = fail = total_rows = 0

        for i, (tk, name, market) in enumerate(universe, 1):
            try:
                n = self.collect_one(tk, name, market, start=start, end=end)
                total_rows += n
                ok += 1
            except Exception as e:
                fail += 1
                logger.warning("%s 실패: %s", tk, e)

            if i % 20 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(universe) - i) / rate
                logger.info(
                    "[%d/%d] 성공 %d / 실패 %d / 행 %d / 속도 %.1f종목/초 / ETA %d초",
                    i, len(universe), ok, fail, total_rows, rate, eta,
                )

        elapsed = time.time() - start_time
        return {
            "total": len(universe),
            "ok": ok, "fail": fail,
            "rows": total_rows,
            "elapsed_sec": elapsed,
        }
