from typing import List, Dict, Any, Optional

from models.llm_gateway import call_llm
from prompts.summary_prompts import build_summary_prompt

def create_analysis_summary(topics: List[Dict[str, Any]]) -> Optional[str]:

    if not topics:
        return None

    print("\n--- (Service) Generating analysis summary... ---")
    
    # 1. 분리된 프롬프트 빌더를 사용하여 프롬프트 생성
    summary_prompt = build_summary_prompt(topics)
    
    # 2. LLM 호출하여 요약 생성
    try:
        summary_text = call_llm(
            summary_prompt, 
            model="claude-3-5-sonnet-20240620", 
            temperature=0.1
        )
        print(f"Summary created successfully: {summary_text}")
        return summary_text
    except Exception as e:
        print(f"Error during summary creation: {e}")
        return None