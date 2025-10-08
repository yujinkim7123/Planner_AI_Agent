# models/llm_gateway.py
import json
import re
import os
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

def _extract_json(text: str) -> dict:
    """JSON 객체만 추출"""
    fenced_match = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match: text = fenced_match.group(1)
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        try: return json.loads(brace_match.group(0))
        except json.JSONDecodeError as e: raise ValueError("LLM 응답에서 유효한 JSON을 파싱하는 데 실패했습니다.") from e
    raise ValueError("LLM 응답에서 JSON 객체를 찾을 수 없습니다.")

def call_llm(prompt: str, model: str, temperature: float = 0.2) -> dict:
    """
    모델 이름을 분석하여, 해당하는 실제 상용 LLM API를 호출하는 통합 게이트웨이
    """
    print(f"--- Calling Real LLM via Gateway (Model: {model}) ---")
    
    llm = None
    try:
        if model.startswith("gpt-"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key: raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
            llm = ChatOpenAI(
                model=model, temperature=temperature, openai_api_key=api_key,
                model_kwargs={'response_format': {"type": "json_object"}}
            )
        
        elif model.startswith("gemini-"):
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key: raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다.")
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

        elif model.startswith("claude-"):
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key: raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
            llm = ChatAnthropic(model=model, temperature=temperature, anthropic_api_key=api_key)
        
        else:
            raise ValueError(f"지원하지 않는 모델입니다: {model}")

        # 2. 선택된 LLM을 호출하고 응답을 받습니다.
        response = llm.invoke(prompt)
        raw_response_content = response.content

        # 3. 응답을 표준 JSON 형식으로 '통일'하여 반환합니다.
        if model.startswith("gpt-") and 'response_format' in llm.model_kwargs:
             return json.loads(raw_response_content)
        else:
             return _extract_json(raw_response_content)

    except Exception as e:
        print(f"LLM({model}) 호출 중 심각한 오류 발생: {e}")
        return {"error": str(e)}