# agents/experts/creator/validators/preconditions.py
from typing import Tuple, Dict, Any

def check(state: Dict[str, Any], tool_name: str) -> Tuple[bool, str]:
    """
    Creator 부서의 특정 Tool을 실행하기 전에, 필요한 데이터가 AgentState에 준비되어 있는지 검증합니다.
    
    반환값:
        - (True, ""): 모든 조건 충족, 작업 진행 가능
        - (False, "실패 이유"): 조건 불충분, 작업 불가
    """
    print(f"--- Creator Precondition Check for Tool: {tool_name} ---")
    
    # --- 페르소나 관련 작업의 선행 조건 ---
    if tool_name in ["create_personas", "modify_personas"]:
        if not state.get("cx_insights"):
            error_message = "페르소나 작업을 위한 CX 분석 결과가 없습니다."
            print(f"Check Failed: {error_message}")
            return False, error_message

    # --- 서비스 아이디어 관련 작업의 선행 조건 ---
    elif tool_name in ["create_service_ideas", "modify_service_ideas"]:
        if not state.get("personas") or not state.get("cx_insights"):
            error_message = "서비스 아이디어 작업을 위한 페르소나 또는 CX 분석 결과가 없습니다."
            print(f"Check Failed: {error_message}")
            return False, error_message

    # --- 데이터 기획안 관련 작업의 선행 조건 ---
    elif tool_name in ["create_data_plan", "modify_data_plan"]:
        if not state.get("service_ideas"):
            error_message = "데이터 기획안 작업을 위한 서비스 아이디어 정보가 없습니다."
            print(f"Check Failed: {error_message}")
            return False, error_message

    # --- 최종 보고서 관련 작업의 선행 조건 ---
    elif tool_name in ["create_final_document", "modify_final_document"]:
        required_keys = ["personas", "service_ideas", "data_plan"]
        if not all(key in state and state[key] for key in required_keys):
            error_message = "최종 보고서 작성을 위한 페르소나, 서비스 아이디어, 또는 데이터 기획안 정보가 부족합니다."
            print(f"Check Failed: {error_message}")
            return False, error_message
            
    # 모든 검증을 통과
    print("Check Passed: All preconditions are met.")
    return True, ""