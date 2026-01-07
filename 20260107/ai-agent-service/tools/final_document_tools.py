# tools/final_document_tools.py
import json
from typing import Any, Dict, List


from prompts.final_document_prompts import (
    build_create_final_document_prompt,
    build_modify_final_document_prompt
)

from models.llm_gateway import call_llm

def create_final_document_tool(
    persona: Dict[str, Any],
    service_idea: Dict[str, Any],
    data_plan: Dict[str, Any],
    model: str = "gpt-4o",
    temperature: float = 0.2
) -> Dict[str, Any]:
  
    print(f"--- TOOL: 최종 보고서 생성 실행 ---")
    
    prompt = build_create_final_document_prompt(
        persona=persona,
        service_idea=service_idea,
        data_plan=data_plan
    )
    
    llm_response = call_llm(prompt, model=model, temperature=temperature)
    
    return {
        "final_document": llm_response.get("final_document", {}),
        "meta": llm_response.get("meta", {}),
    }

def modify_final_document_tool(
    existing_document: Dict[str, Any],
    modification_request: str,
    persona: Dict[str, Any],
    service_idea: Dict[str, Any],
    data_plan: Dict[str, Any],
    model: str = "gpt-4o",
    temperature: float = 0.3
) -> Dict[str, Any]:
    
    print(f"--- TOOL: 최종 보고서 수정 실행 ---")

    prompt = build_modify_final_document_prompt(
        existing_document=existing_document,
        modification_request=modification_request,
        persona=persona,
        service_idea=service_idea,
        data_plan=data_plan
    )

    llm_response = call_llm(prompt, model=model, temperature=temperature)

    return {
        "final_document": llm_response.get("final_document", {}),
        "meta": llm_response.get("meta", {}),
    }