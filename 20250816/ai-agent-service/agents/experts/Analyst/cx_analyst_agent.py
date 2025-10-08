# cx_agent/agent.py
import json
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END

from agents.experts.Analyst.planner import planner
from agents.experts.Analyst.validators import preconditions
from agents.experts.Analyst.registry import registry
from agents.experts.Analyst.state.schemas import AgentState


class CXAgentState(TypedDict):
    main_state: AgentState
    plan_list: List[Dict]           # 전체 계획 목록
    current_plan: Optional[Dict]    # 현재 실행 중인 계획
    error_message: Optional[str]
    current_observation: Optional[str]


def summarizer_node(state: CXAgentState) -> dict:
    print("\n--- CX Analyst: Executing Final Summary Step ---")
    
    main_state = state['main_state']
    
    # 이미 요약이 있다면 중복 실행 방지
    if main_state.get("cx_insights", {}).get("summary"):
        print("Summary already exists. Skipping.")
        return {}

    try:
        # 'create_summary' 작업의 모든 정보
        tool_name = "create_summary"
        task_info = registry.get_task_info(tool_name)
        tool_to_run = task_info["tool"]
        validator = task_info["validator"]
        params_builder = task_info["params_builder"]
        payload_key = task_info["payload_key"]

        # 파라미터를 준비하고 도구를 실행
        params = params_builder(main_state)
        result = tool_to_run(**params)

        #결과물을 검증
        validated_result = validator(result)
        if validated_result is None:
            raise ValueError("생성된 요약문이 품질 기준을 통과하지 못했습니다.")

        #상태(state)를 업데이트
        if "cx_insights" not in main_state:
            main_state["cx_insights"] = {}
        main_state["cx_insights"][payload_key] = validated_result
        
        return {
            "main_state": main_state,
            "current_observation": "최종 분석 요약문 생성이 완료되었습니다."
        }
        
    except Exception as e:
        error_message = f"Summarizer node failed: {e}"
        print(error_message)
        return {"error_message": error_message, 
                "current_observation": "최종 분석 요약문 생성 중 오류 발생."}


def planner_node(state: CXAgentState) -> dict:
    """1. 계획 수립: 사용자의 요청을 분석하여 전체 실행 계획 '목록'을 수립합니다."""
    print("\n--- CX Analyst: 1. Planning ---")
    try:
        result = analyst_planner.create_plan_list(
            user_request=state['main_state'].get('user_request', ''),
            main_state=state['main_state']
        )
        if "error_message" in result:
             raise ValueError(result["error_message"])

        result["current_observation"] = "전체 작업 계획 수립이 완료되었습니다."
        return result
    except Exception as e:
        error_message = f"Planner 노드 실행 중 오류가 발생했습니다: {e}"
        print(error_message)
        return {"error_message": error_message, "current_observation": "계획 수립 중 오류 발생."}



def tool_executor_node(state: CXAgentState) -> dict:
  
    plan = state.get('current_plan', {})
    tool_name = plan.get("action")
    main_state = state['main_state']

    print(f"--- CX Agent: 3. Executing Tool [{tool_name}] ---")

    try:
        task_info = registry.get_task_info(tool_name)
        tool_to_run = task_info["tool"]
        validator = task_info["validator"]
        params_builder = task_info["params_builder"]
        payload_key = task_info["payload_key"]

        params = params_builder(main_state)
        result = tool_to_run(**params)
        validated_result = validator(result)

        if validated_result is None:
            raise ValueError(f"'{tool_name}'의 결과물이 품질 기준을 통과하지 못했습니다.")

        if "cx_insights" not in main_state:
            main_state["cx_insights"] = {}
        main_state["cx_insights"][payload_key] = validated_result
        observation = f"'{tool_name}' 작업이 성공적으로 완료되었습니다."

        return {"main_state": main_state, "current_observation": observation, "error_message": None}

    except Exception as e:
        observation = f"'{tool_name}' 실행 중 오류 발생: {e}"
        print(f"Error during execution: {observation}")
        return {
            "main_state": main_state,
            "error_message": str(e),
            "current_observation": f"작업 중단: {observation}"
        }

def plan_router(state: CXAgentState) -> str:
    if state.get("error_message"):
        return "error_handler"

    plan_list = state.get("plan_list", [])
    if not plan_list:
        print("--- CX Agent: All plans completed. Moving to summary. ---")
        # 모든 분석 도구 실행이 완료되면 summarizer로 이동
        return "summarizer"
    else:
        next_plan = plan_list.pop(0)
        action = next_plan.get("action")
        
        # 'finish'는 명시적인 도구가 아니므로, 발견 시 요약 단계로 이동
        if action == "finish":
             print("--- CX Agent: 'finish' action found. Moving to summary. ---")
             return "summarizer"

        print(f"--- CX Agent: 2. Next plan is [{action}] ---")
        state['plan_list'] = plan_list
        state['current_plan'] = next_plan
        return "tool_executor"

def error_handler_node(state: CXAgentState) -> dict:
    """에러 처리"""
    error_message = state.get("error_message", "Unknown error")
    last_log = state['agent_scratchpad'][-1] if state['agent_scratchpad'] else "알 수 없는 오류"
    print(f"Error Handler: {error_message}")
    return {
        "current_observation": f"작업 중단: {last_log}"
    }


cx_workflow = StateGraph(CXAgentState)
cx_workflow.add_node("planner", planner_node)
cx_workflow.add_node("tool_executor", tool_executor_node)
cx_workflow.add_node("summarizer", summarizer_node)
cx_workflow.add_node("error_handler", error_handler_node)

cx_workflow.set_entry_point("planner")

# Planner 실행 후 plan_router로 이동
cx_workflow.add_conditional_edges(
    "planner",
    lambda s: "error_handler" if s.get("error_message") else plan_router(s),
    {
        "error_handler": "error_handler",
        "tool_executor": "tool_executor",
        "summarizer": "summarizer" # 계획이 처음부터 비어있거나 finish만 있을 경우
    }
)

# Tool 실행 후 다시 plan_router로 돌아가 다음 계획을 확인
cx_workflow.add_conditional_edges(
    "tool_executor",
    lambda s: "error_handler" if s.get("error_message") else plan_router(s),
    {
        "error_handler": "error_handler",
        "tool_executor": "tool_executor",
        "summarizer": "summarizer"
    }
)

cx_workflow.add_edge("summarizer", END)
cx_workflow.add_edge("error_handler", END)

cx_analyst_app = cx_workflow.compile()

# (MCP)에서 호출할 cx agent 함수
def run_cx_analyst_agent(state: AgentState) -> dict:
    print("--- Expert Department Deployed: CX Analyst (LangGraph Engine) ---")

    initial_internal_state = {
        "main_state": state,
        "plan_list": [],
        "current_plan": None,
        "error_message": None
    }

    final_internal_state = cx_analyst_app.invoke(initial_internal_state)

    if final_internal_state.get("error_message"):
        return {
            "next_action": "error",
            "reason": final_internal_state["error_message"],
            "updated_state": final_internal_state["main_state"]
        }

    observation = "모든 CX 분석 작업이 완료되었습니다."
    final_cx_insights = final_internal_state["main_state"].get("cx_insights", {})

    return {
        "next_action": "success",
        "updated_state": final_internal_state["main_state"],
        "cx_insights": final_cx_insights,
        "topics": final_cx_insights.get("scores"),
        "current_observation": observation
    }