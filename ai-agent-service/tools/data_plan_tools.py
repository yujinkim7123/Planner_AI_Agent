# tools/data_plan_tools.py
import json
import re
from typing import Any, Dict, List
from tools.common.utils import call_llm


from tools.prompts.data_plan_prompts import (
    build_create_data_plan_prompt,
    build_modify_data_plan_prompt
)


def create_data_plan_tool(
    service_idea: Dict[str, Any],
    product_context: Dict[str, Any],
    model: str = "gpt-4o",
    temperature: float = 0.3
) -> Dict[str, Any]:
    print(f"--- TOOL: 데이터 기획안 생성 실행 ---")
    
    
    prompt = build_create_data_plan_prompt(
        service_idea=service_idea,
        product_context=product_context
    )
    
  
    llm_response = call_llm(prompt, model=model, temperature=temperature)
    
   
    return {
        "data_plan": llm_response.get("data_plan", {}),
        "meta": llm_response.get("meta", {}),
    }

def modify_data_plan_tool(
    existing_plan: Dict[str, Any],
    modification_request: str,
    service_idea: Dict[str, Any],
    model: str = "gpt-4o",
    temperature: float = 0.4
) -> Dict[str, Any]:
    """기존 데이터 기획안 수정을 위한 프롬프트를 만들고 LLM을 호출합니다."""
    print(f"--- TOOL: 데이터 기획안 수정 실행 ---")
   
   
    prompt = build_modify_data_plan_prompt(
        existing_plan=existing_plan,
        modification_request=modification_request,
        service_idea=service_idea
    )

   
    llm_response = call_llm(prompt, model=model, temperature=temperature)

  
    return {
        "data_plan": llm_response.get("data_plan", {}),
        "meta": llm_response.get("meta", {}),
    }