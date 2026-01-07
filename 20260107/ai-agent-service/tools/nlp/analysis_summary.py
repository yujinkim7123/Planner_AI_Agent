from typing import List, Dict, Any, Optional

from models.llm_gateway import call_llm
from prompts.summary_prompts import build_summary_prompt

def run(topics: List[Dict[str, Any]]) -> Optional[str]:

    if not topics:
        return None

    print("\n--- (Service) Generating analysis summary... ---")
    summary_prompt = build_summary_prompt(topics)

    try:
        summary_text = call_llm(
            summary_prompt, 
            model="gpt-4o", 
            temperature=0.1,
            expect_json=False  # 요약은 단순 텍스트 응답이므로 expect_json을 False로 설정
        )
        print(f"Summary created successfully: {summary_text}")
        return summary_text
    except Exception as e:
        print(f"Error during summary creation: {e}")
        return None