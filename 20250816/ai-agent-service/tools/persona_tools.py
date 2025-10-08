# tools/persona_tools.py
import json
import re
from typing import Any, Dict, List

# 페르소나 '설계도'를 가져옵니다.
from tools.prompts.persona_prompts import (
    build_create_persona_prompt,
    build_modify_persona_prompt
)

from tools.common.utils import call_llm

def create_personas_tool(
    analysis_artifacts: Dict[str, Any],
    web_results_sample: List[Dict[str, Any]],
    num_personas: int,
    user_request: str,
    model: str = "gpt-4o",
    temperature: float = 0.3
) -> Dict[str, Any]:
    print(f"--- TOOL: {num_personas}개 페르소나 생성 실행 ---")
    
    prompt = build_create_persona_prompt(
        analysis_artifacts=analysis_artifacts,
        web_results_sample=web_results_sample,
        num_personas=num_personas,
        user_request=user_request
    )
    
    llm_response = call_llm(prompt, model=model, temperature=temperature)
    
    return {
        "personas": llm_response.get("personas", []),
        "meta": llm_response.get("meta", {}),
        "recommendation_message": llm_response.get("recommendation_message")
    }

def modify_personas_tool(
    existing_personas: List[Dict[str, Any]],
    modification_request: str,
    analysis_artifacts: Dict[str, Any],
    web_results_sample: List[Dict[str, Any]],
    model: str = "gpt-4o",
    temperature: float = 0.4
) -> Dict[str, Any]:
    
    print(f"--- TOOL: 페르소나 수정 실행 ---")
   
    prompt = build_modify_persona_prompt(
        existing_personas=existing_personas,
        modification_request=modification_request,
        analysis_artifacts=analysis_artifacts,
        web_results_sample=web_results_sample
    )

    llm_response = call_llm(prompt, model=model, temperature=temperature)

    return {
        "personas": llm_response.get("personas", []),
        "meta": llm_response.get("meta", {}),
        "recommendation_message": llm_response.get("recommendation_message")
    }