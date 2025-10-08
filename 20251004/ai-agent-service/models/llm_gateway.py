# models/llm_gateway.py
import json
import re
import os
from typing import Dict, Union, Any # Union을 추가하여 여러 타입을 반환할 수 있도록 명시
import time

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

def _extract_json(text: str) -> dict:
    """JSON 객체만 추출"""
    # ... (이 함수는 수정할 필요 없습니다)
    fenced_match = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match: text = fenced_match.group(1)
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        try: return json.loads(brace_match.group(0))
        except json.JSONDecodeError as e: raise ValueError("LLM 응답에서 유효한 JSON을 파싱하는 데 실패했습니다.") from e
    raise ValueError("LLM 응답에서 JSON 객체를 찾을 수 없습니다.")

def call_llm(prompt: str, model: str, temperature: float = 0.2, *, expect_json: bool = True) -> Union[Dict[str, Any], str]:
    """
    LLM을 호출하고, expect_json 값에 따라 JSON(dict) 또는 일반 텍스트(str)를 반환합니다.
    """
    print(f"--- Calling LLM (Model: {model}, Expects JSON: {expect_json}) ---")
    
    llm = None
    try:
        if model.startswith("gpt-"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key: raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
            
            # 1. JSON을 기대할 때만 JSON 모드 옵션을 추가
            model_kwargs = {}
            if expect_json:
                model_kwargs['response_format'] = {"type": "json_object"}
            
            llm = ChatOpenAI(
                model=model, temperature=temperature, openai_api_key=api_key,
                model_kwargs=model_kwargs
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
        
        response = llm.invoke(prompt)
        raw_response_content = response.content

        # 2. expect_json 값에 따라 반환 형식을 결정
        if expect_json:
            # JSON 추출 시도 (재시도 로직은 이전처럼 유용할 수 있습니다)
            return _extract_json(raw_response_content)
        else:
            # JSON이 필요 없으면, 받은 텍스트를 그대로 반환
            return raw_response_content

    except Exception as e:
        print(f"LLM({model}) 호출 중 심각한 오류 발생: {e}")
        # 3. 에러 발생 시 반환 형식도 분기 처리
        if expect_json:
            return {"error": str(e)}
        else:
            return f"오류가 발생했습니다: {e}"