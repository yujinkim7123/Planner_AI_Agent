# cx_agent/agent.py
import json
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END

from agents.experts.Analyst.planner import planner
from agents.experts.Analyst.validators import preconditions
from agents.experts.Analyst.registry import registry
from agents.common.graph_state import AgentState


class CXAgentState(TypedDict):
    main_state: AgentState
    plan_list: List[Dict]           # 전체 계획 목록
    current_plan: Optional[Dict]    # 현재 실행 중인 계획
    error_message: Optional[str]
    current_observation: Optional[str]


def summarizer_node(state: CXAgentState) -> dict:
    print("\n--- CX Analyst: Executing Final Summary Step ---")
    
    main_state = state['main_state']
    
    
    if main_state.get("cx_insights", {}).get("summary"):
        print("Summary already exists. Skipping.")
        return {}

    try:
    
        tool_name = "create_summary"
        task_info = registry.get_task_info(tool_name)
        tool_to_run = task_info["tool"]
        validator = task_info["validator"]
        params_builder = task_info["params_builder"]
        payload_key = task_info["payload_key"]

        
        params = params_builder(main_state)
        result = tool_to_run(**params)

       
        validated_result = validator(result)
        if validated_result is None:
            raise ValueError("생성된 요약문이 품질 기준을 통과하지 못했습니다.")

      
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
  
    print("\n--- CX Analyst: 1. Planning ---")
    try:
        main_state = state.get('main_state')
        if not main_state:
            raise ValueError("main_state가 제공되지 않았습니다.")

        result = planner.create_plan_list(
            user_request=main_state.get('user_request', ''),
            main_state=main_state
        )
        print(f"Planner result: {result}")

        if result is None:
            raise ValueError("Planner가 유효한 계획을 생성하지 못했습니다.")
        if "error_message" in result and result["error_message"]:
             raise ValueError(result["error_message"])

        return {
            "main_state": main_state,
            "plan_list": result.get("plan_list", []),
            "current_observation": "전체 작업 계획 수립이 완료되었습니다."
        }
    except Exception as e:
        error_message = f"Planner 노드 실행 중 오류가 발생했습니다: {e}"
        print(error_message)
        return {
            "main_state": state.get('main_state'), 
            "error_message": error_message, 
            "current_observation": "계획 수립 중 오류 발생."
        }
    

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

        if main_state.get("cx_insights") is None:
            main_state["cx_insights"] = {}

        if "cx_insights" not in main_state:
            main_state["cx_insights"] = {}
            
        main_state["cx_insights"][payload_key] = validated_result
        observation = f"'{tool_name}' 작업이 성공적으로 완료되었습니다."

        return {"main_state": main_state, "current_observation": observation, "error_message": None,}

    except Exception as e:
        observation = f"'{tool_name}' 실행 중 오류 발생: {e}"
        print(f"Error during execution: {observation}")
        return {
            "main_state": main_state,
            "error_message": str(e),
            "current_observation": f"작업 중단: {observation}"
        }

def plan_router(state: CXAgentState) -> str:
    """plan_preparer가 준비한 current_plan을 보고 다음 목적지를 결정합니다."""
    print("--- CX Agent: Routing based on current plan ---")
    
    if state.get("error_message"):
        return "error_handler"

    current_plan = state.get("current_plan")

    # 실행할 계획이 없거나 'finish' 액션이면 요약 단계로 이동
    if not current_plan or (isinstance(current_plan, dict) and current_plan.get("action") == "finish"):
        return "summarizer"
    
    # 실행할 계획이 있으면 tool_executor로 이동
    return "tool_executor"
    
# cx_analyst_agent.py 파일에 아래 함수를 추가합니다.

def plan_preparer_node(state: CXAgentState) -> dict:
    """계획 목록에서 다음 계획을 꺼내 current_plan으로 설정하고,
       변경사항을 명시적으로 반환하여 상태를 업데이트합니다."""
    print("--- CX Agent: 2. Preparing next plan ---")
    plan_list = state.get("plan_list", [])
    
    # 더 이상 실행할 계획이 없으면 빈 딕셔너리를 반환하여 상태 변경 없음을 알림
    if not plan_list:
        return {"current_plan": None}

    next_plan = plan_list.pop(0)
    
    # 업데이트할 상태를 명시적으로 반환
    return {
        "plan_list": plan_list,
        "current_plan": next_plan
    }

def error_handler_node(state: CXAgentState) -> dict:
    """에러를 처리하고 최종 관찰 메시지를 설정합니다."""
    print("--- CX Agent Error Handler ---")
    
    error_message = state.get("error_message", "알 수 없는 오류가 발생했습니다.")
    failed_plan_action = state.get("current_plan", {}).get('action', '알 수 없는 작업')
    final_observation = f"'{failed_plan_action}' 작업 중 오류 발생: {error_message}"
    print(f"Error during plan [{failed_plan_action}]: {error_message}")
    return {
        "current_observation": final_observation
    }

cx_workflow = StateGraph(CXAgentState)
cx_workflow.add_node("planner", planner_node)
cx_workflow.add_node("plan_preparer", plan_preparer_node)
cx_workflow.add_node("tool_executor", tool_executor_node)
cx_workflow.add_node("summarizer", summarizer_node)
cx_workflow.add_node("error_handler", error_handler_node)

cx_workflow.set_entry_point("planner")

cx_workflow.add_edge("planner", "plan_preparer")

cx_workflow.add_conditional_edges(
    "plan_preparer",
    plan_router,
    {
        "tool_executor": "tool_executor",
        "summarizer": "summarizer",
        "error_handler": "error_handler"
    }
)

# Tool 실행 후 다시 plan_router로 돌아가 다음 계획을 확인
cx_workflow.add_conditional_edges(
    "tool_executor",
    lambda s: "error_handler" if s.get("error_message") else "plan_preparer",
    {
        "plan_preparer": "plan_preparer", 
        "error_handler": "error_handler"
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