import json
from typing import Dict, Any

# 필요한 모듈 가져오기
from prompts.cx_analyst_planner_prompt import build_cx_analyst_prompt 
from agents.experts.Analyst.registry.registry import get_tools_description, TOOL_REGISTRY
from agents.experts.Analyst.validators.preconditions import check as precondition_check
from models.llm_gateway import call_llm

def create_plan(user_request: str, agent_scratchpad: str = "", current_observation: str="", main_state: Dict[str, Any] = None) -> Dict[str, Any]:
    print("--- CX Agent Planner: Creating next action plan... ---")

    tools_description = get_tools_description()
    insights_summary = main_state.get("cx_insights", {}).get("summary", "아직 요약된 내용이 없습니다.")

    # 프롬프트 생성
    prompt = build_cx_analyst_prompt(
        user_request=user_request,
        tools_description=tools_description,
        agent_scratchpad=agent_scratchpad,
        insights_summary=insights_summary, # <--- 요약본 전달
        current_observation=current_observation
    )

    #LLM 호출
    llm_response_str = call_llm(prompt, model="gpt-4o-turbo", temperature=0.2)

    try:
        #LLM 응답 파싱
        action_json_str = llm_response_str if isinstance(llm_response_str, str) else json.dumps(llm_response_str)

        # "Action:" 키워드가 있으면 그 뒤만 추출
        if "Action:" in action_json_str:
            action_json_str = action_json_str.split("Action:")[1].strip()

        # 마크다운 블록 제거
        if action_json_str.startswith("```json"):
            action_json_str = action_json_str[7:]
            if action_json_str.endswith("```"):
                action_json_str = action_json_str[:-3]
            action_json_str = action_json_str.strip()

        action_data = json.loads(action_json_str)

        #필수 키 검증
        if "action" not in action_data or "action_input" not in action_data:
            raise ValueError("LLM 응답에 'action' 또는 'action_input' 키가 없습니다.")

        # 도구 이름 검증 (TOOL_REGISTRY 활용)
        tool_name = action_data["action"]
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"'{tool_name}'은(는) 레지스트리에 정의되지 않은 작업입니다.")

        #사전 조건 검증
        if main_state is not None:
            tool_name = action_data["action"]
            if tool_name != "finish": # finish는 검증할 필요 없음
                is_runnable, reason = precondition_check(main_state, tool_name)
                if not is_runnable:
                    print(f"Precondition Check Failed: {reason}")
                    # 단순 에러가 아닌, 재계획을 위한 관찰 결과로 반환
                    return {"action": "error", "reason": f"사전 조건 불충족: {reason}"}

        print(f"Plan created: {action_data}")
        return action_data

    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        # 실패 시 비상 계획 반환
        print(f"Error: Failed to parse LLM action in planner. Reason: {e}")
        return {"action": "error", "reason": str(e)}