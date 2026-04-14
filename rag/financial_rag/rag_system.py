# 역할: 재무제표 데이터를 문서화하여 벡터 데이터베이스에 저장하고, 검색 기능을 제공하는 RAG 모듈.
# 재무 데이터를 텍스트로 변환하여 저장하고, 쿼리에 따라 관련 정보를 검색한다.

import chromadb
from chromadb.utils import embedding_functions

def search_financial_rag(financial_data, query):
    """
    RAG 검색 수행.
    :param financial_data: dict of financial metrics
    :param query: search query
    :return: relevant info
    """
    print("   🔍 RAG 검색 중...")
    # Chroma 클라이언트
    client = chromadb.PersistentClient(path="./rag/financial_rag/db")

    # 컬렉션 생성
    collection = client.get_or_create_collection(name="financial_data")

    # 데이터 추가 (간단하게)
    doc = f"매출: {financial_data['매출']}, 영업이익: {financial_data['영업이익']}, 순이익: {financial_data['순이익']}, 부채비율: {financial_data['부채비율']}"
    collection.add(documents=[doc], ids=["financial_doc"])

    # 검색
    results = collection.query(query_texts=[query], n_results=1)
    print("   ✅ 검색 완료")
    return results['documents'][0][0] if results['documents'] else "정보 없음"

if __name__ == "__main__":
    # 테스트
    data = {"매출": 100000, "영업이익": 20000, "순이익": 15000, "부채비율": 30.5}
    result = search_financial_rag(data, "재무 상태")
    print(result)