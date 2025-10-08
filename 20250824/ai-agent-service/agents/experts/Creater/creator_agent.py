# agents/experts/Creater/creator_agent.py
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Optional, Any

from agents.experts.Creater.planner import planner as creator_planner
from agents.experts.Creater.registry.registry import get_task_info
from agents.common.graph_state import AgentState

# 내부 상태 정의
class CreatorAgentState(TypedDict):
    main_state: AgentState
    plan_list: List[Dict]
    current_plan: Optional[Dict]  # 💡 이 필드를 제대로 활용
    error_message: Optional[str]
    current_observation: Optional[str] 


def planner_node(state: CreatorAgentState) -> dict:
    """1. 기획팀: 사용자의 의도를 분석하여 실행 계획 '목록'을 수립합니다."""
    print("--- Creator Agent: 1. Planning ---")
    try:
        result = creator_planner.create_plan_list(state['main_state'])
        if result.get("error_message"):
            return {
                "error_message": result["error_message"],
                "current_observation": "계획 수립 중 오류 발생."
            }
        
        # 성공 시 observation 추가
        result["current_observation"] = "계획 수립이 완료되었습니다."
        return result
    except Exception as e:
        error_message = f"Planner 노드 실행 중 예측하지 못한 오류가 발생했습니다: {e}"
        print(error_message)
        return {
            "error_message": error_message, 
            "current_observation": "계획 수립 중 오류 발생."
        }


def plan_selector_node(state: CreatorAgentState) -> dict:
    """2. 계획 선택팀: plan_list에서 다음 계획을 꺼내서 current_plan에 설정합니다."""
    plan_list = state.get("plan_list", [])
    
    if not plan_list:
        return {
            "current_plan": None,
            "current_observation": "모든 계획이 완료되었습니다."
        }
    
    # 다음 계획을 꺼냄
    next_plan = plan_list.pop(0)
    domain = next_plan.get("domain")
    action = next_plan.get("action")
    
    print(f"--- Creator Agent: 2. Next plan is [{action}_{domain}] ---")
    print(f"DEBUG - Selected plan: {next_plan}")
    
    return {
        "plan_list": plan_list,  # 업데이트된 리스트
        "current_plan": next_plan,  # 현재 실행할 계획
        "current_observation": f"다음 계획 선택됨: {action}_{domain}"
    }


def tool_executor_node(state: CreatorAgentState) -> dict:
    """3. 실행팀: current_plan에 따라 Tool을 실행하고 검증합니다."""
    plan = state.get('current_plan')
    
    if not plan:
        return {
            "error_message": "실행할 계획이 없습니다.",
            "current_observation": "실행 계획 없음."
        }
    
    domain = plan.get("domain")
    action = plan.get("action")
    main_state = state['main_state']

    print(f"--- Creator Agent: 3. Executing Tool [{action}_{domain}] ---")
    
    try:
        if not domain or not action:
            raise ValueError(f"계획에서 domain 또는 action을 찾을 수 없습니다. Plan: {plan}")
        
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

        return {
            "main_state": main_state,
            "current_plan": None,  # 실행 완료 후 초기화
            "current_observation": observation
        }
        
    except Exception as e:
        observation = f"'{domain}/{action}' 실행 중 오류 발생: {e}"
        print(f"Error during execution: {observation}")
        return {
            "main_state": main_state,
            "current_plan": None,  # 에러 발생 시에도 초기화
            "error_message": str(e),
            "current_observation": f"작업 중단: {observation}"
        }


def plan_router(state: CreatorAgentState) -> str:
    """라우터: 다음 행동을 결정합니다."""
    # 에러가 있으면 에러 핸들러로
    if state.get("error_message"):
        return "error_handler"
    
    # plan_list가 비어있으면 종료
    plan_list = state.get("plan_list", [])
    if not plan_list:
        print("--- Creator Agent: All plans completed. ---")
        return "finish"
    
    # 아직 계획이 남아있으면 다음 계획 선택으로
    return "plan_selector"


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

# 노드 추가
creator_workflow.add_node("planner", planner_node)
creator_workflow.add_node("plan_selector", plan_selector_node)
creator_workflow.add_node("tool_executor", tool_executor_node)
creator_workflow.add_node("error_handler", error_handler_node)

# 시작점 설정
creator_workflow.set_entry_point("planner")

# planner → plan_selector 또는 error_handler
creator_workflow.add_conditional_edges(
    "planner",
    lambda s: "error_handler" if s.get("error_message") else "plan_selector",
    {
        "error_handler": "error_handler",
        "plan_selector": "plan_selector"
    }
)

# current_plan이 설정되었는지로 판단
creator_workflow.add_conditional_edges(
    "plan_selector",
    lambda s: "finish" if s.get("current_plan") is None else "tool_executor",
    {
        "tool_executor": "tool_executor",
        "finish": END
    }
)

# tool_executor → plan_router를 통해 다음 단계 결정
creator_workflow.add_conditional_edges(
    "tool_executor",
    plan_router,
    {
        "error_handler": "error_handler",
        "plan_selector": "plan_selector",
        "finish": END
    }
)

# error_handler → 종료
creator_workflow.add_edge("error_handler", END)

creator_app = creator_workflow.compile()


def run_creator_agent(state: AgentState) -> dict:
    """Creator Agent 실행 (외부 인터페이스)"""
    print("--- Expert Department Deployed: Creator Agent (LangGraph Engine) ---")
    
    initial_internal_state = {
        "main_state": state,
        "plan_list": [],
        "current_plan": None,
        "error_message": None,
        "current_observation": None
    }
    
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