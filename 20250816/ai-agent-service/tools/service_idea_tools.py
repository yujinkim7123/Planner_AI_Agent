# tools/service_idea_tools.py
import json
import re
from typing import Any, Dict, List


from tools.prompts.service_idea_prompts import (
    build_create_service_idea_prompt,
    build_modify_service_idea_prompt
)

from tools.common.utils import call_llm

def create_service_ideas_tool(
    persona: Dict[str, Any], 
    cx_insights: Dict[str, Any],
    product_context: Dict[str, Any], 
    num_ideas: int = 2,
    model: str = "gpt-4o",
    temperature: float = 0.7
) -> Dict[str, Any]:
    print(f"--- TOOL: 서비스 아이디어 {num_ideas}개 생성 실행 ---")
    
    prompt = build_create_service_idea_prompt(
        persona=persona,
        cx_insights=cx_insights,
        product_context=product_context,
        num_ideas=num_ideas
    )
    
    llm_response = call_llm(prompt, model=model, temperature=temperature)
    
    return {
        "service_ideas": llm_response.get("service_ideas", []),
        "meta": llm_response.get("meta", {}),
    }

def modify_service_ideas_tool(
    existing_ideas: List[Dict[str, Any]],
    modification_request: str,
    persona: Dict[str, Any],
    cx_insights: Dict[str, Any],
    model: str = "gpt-4o",
    temperature: float = 0.4
) -> Dict[str, Any]:
    print(f"--- TOOL: 서비스 아이디어 수정 실행 ---")
   
    prompt = build_modify_service_idea_prompt(
        existing_ideas=existing_ideas,
        modification_request=modification_request,
        persona=persona,
        cx_insights=cx_insights
    )

    llm_response = call_llm(prompt, model=model, temperature=temperature)

    return {
        "service_ideas": llm_response.get("service_ideas", []),
        "meta": llm_response.get("meta", {}),
    }