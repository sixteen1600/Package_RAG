import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.hybrid_retriever import AdvancedHybridRetriever

def main():
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Please set it in your .env file."
        )
    
    docs = [
        Document(
            page_content=(
                "台積公司首次提出水資源正效益長期目標，"
                "預計民國129年達成率100%。"     
            ),
            metadata={"source":"sample_water_management"},
        ),
        
        Document(
            page_content=(
                "台積公司民國113年廢棄物回收率達97%，"
                "並將9,400公噸廢棄物純化再生。"
            ),
            metadata={"source":"sample_resource_circulation"},
        ),
        
        Document(
            page_content=(
                "台積公司民國113年全年研發總支出為63億5,500萬美元。"    
            ),
            metadata={"source":"sample_innovation"},
        ),        
    ]
    
    test_index_path = PROJECT_ROOT / "data" / "vector_db_test"
    
    retriever = AdvancedHybridRetriever(vector_db_path=str(test_index_path))
    retriever.build_index(docs)
    
    query = "台積電的廢棄物回收率是多少？"
    results = retriever.retrieve(query)
    
    print("\nQuery:")
    print(query)
    
    print("\nRetrieved Results:")
    for index, doc in enumerate(results, start=1):
        print(f"\nRank {index}")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content}")
        
    assert any("97%" in doc.page_content for doc in results),(
        "Test failed: expected retrieved results to contain waste recycling rate 97%."
    )
    
    print("\nMinimal reproducibility test passed")

if __name__ == "__main__":
    main()