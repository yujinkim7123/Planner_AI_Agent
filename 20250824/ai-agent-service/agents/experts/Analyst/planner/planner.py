import json
from typing import Dict, Any, List

# 필요한 모듈 가져오기
from prompts.cx_analyst_planner_prompt import build_cx_analyst_prompt 
from agents.experts.Analyst.registry.registry import TOOL_REGISTRY, get_tools_description
from agents.experts.Analyst.validators.preconditions import check as precondition_check
from models.llm_gateway import call_llm

def create_plan_list(user_request: str, main_state: Dict[str, Any], agent_scratchpad: str = "", current_observation: str="") -> Dict[str, Any]:
    print("--- CX Agent Planner: Creating execution plan list... ---")

    tools_description = get_tools_description()
    insights_summary = (main_state.get("cx_insights") or {}).get("summary", "아직 요약된 내용이 없습니다.")

    # 프롬프트 생성
    prompt = build_cx_analyst_prompt(
        user_request=user_request,
        tools_description=tools_description,
        agent_scratchpad=agent_scratchpad,
        insights_summary=insights_summary,
        current_observation=current_observation
    )

    #LLM 호출
    llm_response = call_llm(prompt, model="gpt-4o", temperature=0.2)

    try:
        plan_list = llm_response.get("plan_list", [])

        if not isinstance(plan_list, list):
            raise ValueError("LLM 응답에서 'plan_list'를 찾을 수 없거나 리스트 형식이 아닙니다.")

        simulated_state = main_state.copy()
        
        for plan in plan_list:
            if "action" not in plan:
                raise ValueError(f"계획에 필수 키('action')가 없습니다: {plan}")

            tool_name = plan["action"]
            if tool_name != "finish" and tool_name not in TOOL_REGISTRY:
                raise ValueError(f"'{tool_name}'은(는) 레지스트리에 정의되지 않은 작업입니다.")

            # 사전 조건 검증을 '현재 시점'이 아닌 '가상 시점'의 상태로 수행
            if tool_name != "finish":
                # 1. 현재까지의 가상 상태로 실행 가능한지 확인
                is_runnable, reason = precondition_check(simulated_state, tool_name)
                if not is_runnable:
                    # 논리적 순서가 틀렸음을 명확히 알려주는 오류 메시지
                    raise ValueError(f"계획의 논리적 순서 오류 ({tool_name}): {reason}")
                
                # 2. 검증 통과 시, 이 도구가 실행된 후의 상태를 가상으로 업데이트
                #    (실제 도구를 실행하지 않고, 결과물이 생겼다고 가정만 함)
                if 'cx_insights' not in simulated_state:
                    simulated_state['cx_insights'] = {}
                
                # 각 도구의 실행 결과를 가상으로 simulated_state에 추가합니다.
                if tool_name == 'run_clustering':
                    simulated_state['cx_insights']['clustering'] = True
                elif tool_name == 'run_lda':
                    simulated_state['cx_insights']['lda'] = True
                elif tool_name == 'run_sna':
                    simulated_state['cx_insights']['sna'] = True
                elif tool_name == 'calculate_opportunity_scores':
                    simulated_state['cx_insights']['scores'] = True
                
        print(f"Plan validation passed. Plan list: {plan_list}")
        return {"plan_list": plan_list}
    except (json.JSONDecodeError, ValueError) as e:
        error_message = f"계획 수립 또는 검증 실패: {e}"
        print(f"Error: {error_message}")
        return {"error_message": error_message}