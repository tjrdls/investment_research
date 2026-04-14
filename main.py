# 역할: 투자 분석 시스템의 메인 진입점. 전체 파이프라인을 실행하여 사용자의 분석 요청을 처리한다.
# 이 파일은 시스템의 시작점으로, prediction_pipeline을 호출하여 데이터 수집부터 최종 결과 출력까지의 전체 흐름을 조율한다.

from pipeline.prediction_pipeline import run_analysis

if __name__ == "__main__":
    # 예시: 삼성전자 분석
    stock_symbol = "005930.KS"  # 삼성전자
    print("🎯 분석 대상: 삼성전자 ({})".format(stock_symbol))
    result = run_analysis(stock_symbol)
    print("\n🎉 분석 완료!")