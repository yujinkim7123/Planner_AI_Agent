import json
from typing import Dict, Any, List

from prompts.creator_planner_prompt import build_creator_planner_prompt
from models.llm_gateway import call_llm

from agents.experts.Creater.registry.registry import TOOL_REGISTRY
from agents.experts.Creater.validators.preconditions import check as precondition_check

def create_plan_list(state: Dict[str, Any]) -> Dict[str, Any]:
   
    print("--- Creator Agent Planner: Creating and Validating execution plan list... ---")
    user_request = state.get('user_request', '')

    prompt = build_creator_planner_prompt(user_request)
    llm_response = call_llm(prompt, model="gpt-4o-mini", temperature=0.0)

    try:
        # LLM 응답에서 JSON 리스트 파싱
        action_json_str = llm_response.strip()
        if action_json_str.startswith("```json"):
            action_json_str = action_json_str[7:]
            if action_json_str.endswith("```"):
                action_json_str = action_json_str[:-3]
        
        plan_list = json.loads(action_json_str.strip())

        if not isinstance(plan_list, list):
            raise ValueError("LLM 응답이 리스트 형식이 아닙니다.")

   
        for plan in plan_list:
           
            if "domain" not in plan or "action" not in plan:
                raise ValueError(f"계획에 필수 키('domain', 'action')가 없습니다: {plan}")

            domain = plan["domain"]
            action = plan["action"]

            
            if (domain, action) not in TOOL_REGISTRY:
                raise ValueError(f"'{domain}/{action}'은(는) 레지스트리에 정의되지 않은 작업입니다.")

            
            tool_name_for_check = f"{action}_{domain}s"
            is_runnable, reason = precondition_check(state, tool_name_for_check)
            if not is_runnable:
                raise ValueError(f"사전 조건 불충족 ({domain}/{action}): {reason}")
        
        print(f"Plan validation passed. Plan list: {plan_list}")
        return {"plan_list": plan_list}

    except (json.JSONDecodeError, ValueError) as e:
        error_message = f"계획 수립 또는 검증 실패: {e}"
        print(f"Error: {error_message}")
        return {"error_message": error_message}