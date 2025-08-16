# agents/experts/creator/agent.py
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Optional, Any

from agents.experts.Creater.planner import planner as creator_planner
from agents.experts.Creater.registry import get_task_info
from agents.common.graph_state import AgentState

# 내부 상태 정의
class CreatorAgentState(TypedDict):
    main_state: AgentState
    plan_list: List[Dict]
    current_plan: Optional[Dict]
    error_message: Optional[str]
    current_observation: Optional[str] 


def planner_node(state: CreatorAgentState) -> dict:
    """1. 기획팀: 사용자의 의도를 분석하여 실행 계획 '목록'을 수립합니다."""
    print("--- Creator Agent: 1. Planning ---")
    try:
        result = creator_planner.create_plan_list(state['main_state'])
        # 성공 시 observation 추가
        result["current_observation"] = "계획 수립이 완료되었습니다."
        return result
    except Exception as e:
        error_message = f"Planner 노드 실행 중 예측하지 못한 오류가 발생했습니다: {e}"
        print(error_message)
        return {"error_message": error_message, "current_observation": "계획 수립 중 오류 발생."}

def tool_executor_node(state: CreatorAgentState) -> dict:
    """3. 실행팀: 현재 계획에 따라 단일 Tool을 실행하고 검증합니다."""
    plan = state.get('current_plan', {})
    domain = plan.get("domain")
    action = plan.get("action")
    main_state = state['main_state']

    print(f"--- Creator Agent: 3. Executing Tool [{action}_{domain}] ---")
    try:
        task_info = get_task_info(domain, action)
        tool_to_run = task_info["tool"]
        validator = task_info["validator"]
        payload_key = task_info["payload_key"]
        
        params = task_info["params_builder"](main_state, plan.get("parameters", {}))
        
        raw_result = tool_to_run(**params)
        validated_result = validator(raw_result)
        
        is_failed = validated_result is None or (isinstance(validated_result, list) and not validated_result)
        if is_failed:
            raise ValueError(f"결과물이 품질 기준을 통과하지 못했거나, 유효한 결과가 없습니다.")

        main_state[payload_key] = validated_result
        observation = f"'{domain}/{action}' 작업 및 검증이 성공적으로 완료되었습니다."

        return {"main_state": main_state, "current_observation": observation}
        
    except Exception as e:
        observation = f"'{domain}/{action}' 실행 중 오류 발생: {e}"
        print(f"Error during execution: {observation}")
        return {
            "main_state": main_state, 
            "error_message": str(e),
            "current_observation": f"작업 중단: {observation}"
        }
        

def plan_router(state: CreatorAgentState) -> str:

    if state.get("error_message"):
        return "error_handler"
    
    plan_list = state.get("plan_list", [])
    if not plan_list:
        print("--- Creator Agent: All plans completed. ---")
        return "finish"
    else:
        next_plan = plan_list.pop(0)
        print(f"--- Creator Agent: 2. Next plan is [{next_plan.get('action')}_{next_plan.get('domain')}] ---")
        state['plan_list'] = plan_list
        state['current_plan'] = next_plan
        return "tool_executor"

def error_handler_node(state: CreatorAgentState) -> dict:
    """에러 처리"""
    error_message = state.get("error_message", "Unknown error")
    last_observation = state.get("current_observation", "알 수 없는 오류 지점")
    print(f"Creator Agent Error Handler: {error_message}")
    return {
        "current_observation": f"작업 중단: {last_observation}"
    }

# --- 최종 워크플로우 조립 ---
creator_workflow = StateGraph(CreatorAgentState)
creator_workflow.add_node("planner", planner_node)
creator_workflow.add_node("tool_executor", tool_executor_node)
creator_workflow.add_node("error_handler", error_handler_node)

creator_workflow.set_entry_point("planner")

creator_workflow.add_conditional_edges(
    "planner",
    lambda s: "error_handler" if s.get("error_message") else plan_router(s),
    {
        "error_handler": "error_handler",
        "tool_executor": "tool_executor",
        "finish": END
    }
)

creator_workflow.add_conditional_edges(
    "tool_executor",
    lambda s: "error_handler" if s.get("error_message") else plan_router(s),
    {
        "error_handler": "error_handler",
        "tool_executor": "tool_executor",
        "finish": END
    }
)

creator_workflow.add_edge("error_handler", END)

creator_app = creator_workflow.compile()

def run_creator_agent(state: AgentState) -> dict:
    print("--- Expert Department Deployed: Creator Agent (LangGraph Engine) ---")
    
    initial_internal_state = {"main_state": state}
    final_internal_state = creator_app.invoke(initial_internal_state)
    
    if final_internal_state.get("error_message"):
        return {
            "next_action": "error",
            "reason": final_internal_state["error_message"],
            "updated_state": final_internal_state["main_state"]
        }

    observation = "Creator 부서의 모든 작업이 완료되었습니다."
    return {
        "next_action": "success",
        "updated_state": final_internal_state["main_state"],
        "current_observation": observation
    }