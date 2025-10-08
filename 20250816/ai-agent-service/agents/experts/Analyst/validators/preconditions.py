# cx_agent/validators/preconditions.py
from typing import Tuple, Dict, Any

def check(state: Dict[str, Any], tool_name: str) -> Tuple[bool, str]:
    """
    특정 Tool을 실행하기 전에, 필요한 데이터가 AgentState에 준비되어 있는지 검증
    """
    print(f"--- Precondition Check for Tool: {tool_name} ---")
    
    cx_insights = state.get("cx_insights", {})

    # [핵심 추가] 1. 클러스터링을 위한 선행 조건 검증
    if tool_name == "run_clustering":
        retrieved_data = state.get("retrieved_data_summary", {})
        documents = retrieved_data.get("top_documents_sample")
        if not documents:
            error_message = "클러스터링을 실행하려면, 'retrieved_data_summary' 데이터가 먼저 제공되어야 합니다."
            print(f"Check Failed: {error_message}")
            return False, error_message

    # 2. SNA 또는 LDA 분석을 위한 선행 조건 검증
    elif tool_name in ["run_sna", "run_lda"]:
        clustering_results = cx_insights.get("clustering")
        if not clustering_results:
            error_message = "SNA 또는 LDA 분석을 실행하려면, '클러스터링'이 먼저 수행되어야 합니다."
            print(f"Check Failed: {error_message}")
            return False, error_message

    # 3. 기회 점수 계산을 위한 선행 조건 검증
    elif tool_name == "calculate_scores":
        lda_results = cx_insights.get("all_lda_results")
        if not lda_results:
            error_message = "기회 점수를 계산하려면, 'LDA 토픽 모델링'이 먼저 수행되어야 합니다."
            print(f"Check Failed: {error_message}")
            return False, error_message

    print("Check Passed: All preconditions are met.")
    return True, ""