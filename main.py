# -*- coding: utf-8 -*-
"""
역할: 투자 분석 시스템의 메인 진입점.
전체 파이프라인을 실행하여 사용자의 분석 요청을 처리한다.
"""

from pipeline.prediction_pipeline import run_analysis, run_batch_analysis
import sys


def main():
    """메인 실행"""
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("  💼 AI 주식 투자 분석 시스템")
    print("╚" + "═" * 58 + "╝\n")
    
    print("  분석 모드 선택:")
    print("  1. 단일 종목 분석")
    print("  2. 상위 5개 종목 일괄 분석")
    print()
    
    choice = input("  선택 (1 또는 2): ").strip()
    
    if choice == "1":
        stock_code = input("  종목 코드 입력 (예: 005930): ").strip()
        if not stock_code:
            stock_code = "005930"
        
        print("\n  🔍 {} 분석을 시작합니다...".format(stock_code))
        result = run_analysis(stock_code)
        
        if result:
            print("\n  ✅ 분석 완료!")
            print("  상태: {}".format(result.get("status")))
            return result
    
    elif choice == "2":
        print("\n  🔍 상위 5개 종목 분석을 시작합니다...")
        results = run_batch_analysis(top_n=5)
        print("\n  ✅ 분석 완료! ({} 종목)".format(len(results)))
        return results
    
    else:
        print("  ❌ 잘못된 선택입니다.")
        return None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  ⚠️  프로그램 중단됨")
        sys.exit(0)
    except Exception as e:
        print("\n  ❌ 오류 발생: {}".format(str(e)))
        sys.exit(1)
