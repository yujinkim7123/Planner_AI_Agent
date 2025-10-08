import json
from typing import Dict, Any
from agents.common.graph_state import AgentState
from prompts.master_planner_prompts import build_master_planner_prompt
from models.llm_gateway import call_llm


def get_completed_tasks(state: AgentState) -> list:
  
    # 완료된 작업을 동적으로 탐색
    return [key for key, value in state.items() if value is not None]


def run_master_planner_agent(state: AgentState) -> dict:
    """
    MCP는 LLM을 호출하여 다음에 실행할 에이전트를 결정
    """
    print("\n--- MCP (Master Planner): Reviewing project status and deciding next step... ---")
    
    # 현재 상태에서 완료된 작업 요약
    completed_tasks = get_completed_tasks(state)
    prompt = build_master_planner_prompt(state['user_request'], completed_tasks)

    try:
        # LLM 호출 및 응답 파싱
        response_json = json.loads(call_llm(prompt, model="gpt-4o", temperature=0.2))
        next_action = response_json.get("next_action", "error")
    except Exception as e:
        print(f"Error: Failed to parse LLM response in MCP. Reason: {e}")
        return {"next_action": "error", "reason": str(e)}


    if next_action == "error":
        return {"next_action": "error", "reason": "MCP failed to determine the next action."}


    # MCP는 다음에 실행할 에이전트를 결정만 합니다.
    print(f"MCP's decision: Next action is '{next_action}'")
    return {"next_action": next_action}