from langgraph.graph import StateGraph, END
from agents.common.graph_state import AgentState

# MCP와 각 전문가 부서의 에이전트를 가져옵니다.
from agents.mcp.master_planner_agent import run_master_planner_agent
from agents.experts.Analyst.cx_analyst_agent import run_cx_analyst_agent
from agents.experts.Creater.creator_agent import run_persona_agent
from agents.experts.Creater.creator_agent import run_service_idea_agent
from agents.experts.Analyst.cx_analyst_agent import run_data_plan_agent
from agents.experts.Analyst.cx_analyst_agent import run_final_document_agent

def create_agent_workflow():
    """
    최상위 워크플로우
    """
    graph = StateGraph(AgentState)

    # MCP와 각 전문가 부서의 에이전트를 그래프에 등록합니다.
    graph.add_node("master_planner", run_master_planner_agent)
    graph.add_node("cx_analyst_team", run_cx_analyst_agent)
    graph.add_node("persona_team", run_persona_agent)
    graph.add_node("service_idea_team", run_service_idea_agent)
    graph.add_node("data_plan_team", run_data_plan_agent)
    graph.add_node("final_document_team", run_final_document_agent)

    # 워크플로우의 시작점은 MCP
    graph.set_entry_point("master_planner")

    # MCP가 다음에 어떤 부서를 호출할지 결정
    graph.add_conditional_edges(
        "master_planner",
        lambda state: state["next_action"],
        {
            "cx_analyst_team": "cx_analyst_team",
            "persona_team": "persona_team",
            "service_idea_team": "service_idea_team",
            "data_plan_team": "data_plan_team",
            "final_document_team": "final_document_team",
            "error": "error_handler",  # 에러 처리 추가
            "finish": END
        }
    )

    # 각 부서가 완료되면 다시 MCP로 돌아갑니다.
    for team in ["cx_analyst_team", "persona_team", "service_idea_team", "data_plan_team", "final_document_team"]:
        graph.add_edge(team, "master_planner")

    # 에러 처리 노드 추가
    graph.add_node("error_handler", error_handler_node)
    graph.add_edge("error_handler", END)

    # 최종적으로 완성된 AI 에이전트 '조직'을 컴파일
    return graph.compile()

def error_handler_node(state: AgentState) -> dict:
    """
    에러 처리 노드: 에러 발생 시 워크플로우를 종료하거나 기본 동작 수행
    """
    error_message = state.get("error_message", "Unknown error occurred.")
    print(f"Workflow terminated due to error: {error_message}")
    return {"current_observation": f"Workflow terminated: {error_message}"}