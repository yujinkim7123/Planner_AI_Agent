# agents/experts/Creater/planner/planner.py

import json
from typing import Dict, Any, List

from prompts.creator_planner_prompt import build_creator_planner_prompt
from models.llm_gateway import call_llm

from agents.experts.Creater.registry.registry import TOOL_REGISTRY
from agents.experts.Creater.validators.preconditions import check as precondition_check

# 도메인별 합리적인 최대값 설정
DOMAIN_MAX_LIMITS = {
    "persona": 10,
    "service_idea": 20,
    "data_plan": 1,
    "final_document": 1
}

def create_plan_list(state: Dict[str, Any]) -> Dict[str, Any]:
   
    print("--- Creator Agent Planner: Creating and Validating execution plan list... ---")
    user_request = state.get('user_request', '')

    prompt = build_creator_planner_prompt(user_request)
    
    # 🔥 수정: expect_json=True로 변경하여 dict로 받음
    llm_response = call_llm(prompt, model="gpt-4o-mini", temperature=0.0, expect_json=True)

    try:
        # 🔥 LLM 응답이 dict인지 확인
        if not isinstance(llm_response, dict):
            raise ValueError(f"LLM 응답이 dict 형식이 아닙니다. 받은 타입: {type(llm_response)}")
        
        # 🔥 에러 체크
        if "error" in llm_response:
            raise ValueError(f"LLM 호출 실패: {llm_response.get('error')}")
        
        # 🔥 plan_list 추출 (여러 가능한 키 이름 시도)
        plan_list = None
        for key in ["plan_list", "plans", "actions"]:
            if key in llm_response:
                plan_list = llm_response[key]
                break
        
        # 만약 최상위가 바로 리스트라면
        if plan_list is None and isinstance(llm_response, list):
            plan_list = llm_response
        
        if plan_list is None:
            raise ValueError(f"LLM 응답에서 계획 리스트를 찾을 수 없습니다. 응답 키: {llm_response.keys()}")

        if not isinstance(plan_list, list):
            raise ValueError(f"계획 리스트가 list 형식이 아닙니다. 받은 타입: {type(plan_list)}")

        # 계획 검증 및 수량 제한
        validated_plans = []
        for plan in plan_list:
            # 필수 키 확인
            if "domain" not in plan or "action" not in plan:
                raise ValueError(f"계획에 필수 키('domain', 'action')가 없습니다: {plan}")

            domain = plan["domain"]
            action = plan["action"]

            # 레지스트리 확인
            if (domain, action) not in TOOL_REGISTRY:
                raise ValueError(f"'{domain}/{action}'은(는) 레지스트리에 정의되지 않은 작업입니다.")

            # 🔥 수량 제한 검증 (create 액션에 대해서만)
            if action == "create" and "parameters" in plan:
                params = plan["parameters"]
                
                # 도메인별로 수량 파라미터 이름이 다를 수 있음
                quantity_keys = {
                    "persona": "num_personas",
                    "service_idea": "num_ideas_per_persona",
                }
                
                quantity_key = quantity_keys.get(domain)
                
                if quantity_key and quantity_key in params:
                    requested_num = params[quantity_key]
                    max_limit = DOMAIN_MAX_LIMITS.get(domain, 10)
                    
                    if requested_num > max_limit:
                        print(f"⚠️  WARNING: Requested {requested_num} {domain}(s), but limiting to {max_limit}")
                        params[quantity_key] = max_limit
                        plan["parameters"] = params
                    elif requested_num <= 0:
                        print(f"⚠️  WARNING: Requested {requested_num} {domain}(s), using default value 3")
                        params[quantity_key] = 3
                        plan["parameters"] = params

            # 사전 조건 확인
            tool_name_for_check = f"{action}_{domain}s"
            is_runnable, reason = precondition_check(state, tool_name_for_check)
            if not is_runnable:
                raise ValueError(f"사전 조건 불충족 ({domain}/{action}): {reason}")
            
            validated_plans.append(plan)
        
        print(f"Plan validation passed. Plan list: {validated_plans}")
        return {"plan_list": validated_plans}

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        error_message = f"계획 수립 또는 검증 실패: {e}"
        print(f"Error: {error_message}")
        print(f"LLM Response: {llm_response}")
        return {"error_message": error_message}