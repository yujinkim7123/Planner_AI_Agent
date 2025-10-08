# tools/prompts/service_idea_prompts.py
import json
from typing import Dict, Any, List

PROMPT_VERSION = "service_idea.v1.1.0"

def build_create_service_idea_prompt(
    persona: Dict[str, Any], 
    cx_insights: Dict[str, Any],
    product_context: Dict[str, Any], 
    num_ideas: int
) -> str:
    """신규 서비스 아이디어 생성을 위한 전체 LLM 프롬프트를 구성합니다."""

    prompt = f"""
    (prompt_version: {PROMPT_VERSION})
    당신은 LG전자의 신사업 기획을 총괄하는 최고의 서비스 전략가입니다.
    고객 데이터에 기반하여, 기존의 틀을 깨는 혁신적이면서도 실현 가능한 서비스 아이디어를 만드는 데 특화되어 있습니다.

    ### 1. 분석 대상 페르소나 정보
    {json.dumps(persona, ensure_ascii=False, indent=2)}

    ### 2. 핵심 CX 분석 결과 (기회 점수 상위 토픽 등)
    {json.dumps(cx_insights, ensure_ascii=False, indent=2)}

    ### 3. 기존 제품 및 기능 정보
    {json.dumps(product_context, ensure_ascii=False, indent=2)}

    ---
    ### 최종 지시사항
    1.  위 페르소나의 **'Pain Points'**를 명확하고 직접적으로 해결하는 **새로운 서비스 아이디어 {num_ideas}개**를 제안해주세요.
    2.  서비스의 핵심 기능이 **어떤 제품 또는 센서 데이터**를 어떻게 활용하는지 구체적으로 설명해야 합니다.
    3.  서비스가 미래에 어떻게 성장하고 확장될 수 있는지 **'서비스 확장성(Service Scalability)'**을 반드시 포함해주세요.
    4.  결과는 반드시 아래의 JSON 형식으로만 반환해주세요.

    ```json
    {{
    "service_ideas": [
        {{
        "service_name": "AI 육아 위생 컨설턴트",
        "description": "페르소나의 아이 연령과 건강 상태에 맞춰, 의류, 장난감 등의 최적 살균 주기를 알려주고 가전을 자동으로 제어해주는 구독형 서비스입니다.",
        "solved_pain_points": ["살균 기능의 실제 효과를 눈으로 확인할 수 없어 불안하다", "매번 옷을 삶는 것은 번거롭다"],
        "service_scalability": "초기에는 ThinQ 앱 기능으로 제공하고, 추후 영유아 건강 데이터를 연동한 프리미엄 유료 모델로 확장할 수 있습니다."
        }}
    ],
    "meta": {{ "prompt_version": "{PROMPT_VERSION}" }}
    }}"""
    return prompt

def build_modify_service_idea_prompt(
existing_ideas: List[Dict[str, Any]],
modification_request: str,
persona: Dict[str, Any],
cx_insights: Dict[str, Any]
) -> str:

    prompt = f"""
    (prompt_version: {PROMPT_VERSION})
    당신은 최고의 서비스 전략가입니다. '기존 서비스 아이디어'를 '사용자 수정 요청'에 맞게 수정해주세요.
    수정의 기반이 되는 페르소나 및 분석 데이터도 참고하세요.

    1. 수정 대상이 되는 '기존 서비스 아이디어' (AS-IS)
    {json.dumps(existing_ideas, ensure_ascii=False, indent=2)}

    2. 사용자의 '수정 요청사항' (TO-BE)
    "{modification_request}"

    3. 참고용 원본 데이터 (CONTEXT)
    기반 페르소나 정보: {json.dumps(persona, ensure_ascii=False)}

    핵심 CX 분석 결과: {json.dumps(cx_insights, ensure_ascii=False)}

    최종 지시사항
    1.'기존 서비스 아이디어'를 기반으로, '수정 요청사항'을 반영하여 아이디어를 '재창조'해주세요.

    2. 수정 요청이 없는 부분은 기존 아이디어의 내용을 유지해야 합니다.

    3. 결과는 반드시 아래의 JSON 형식으로만 반환해주세요.
      ```json
    {{
    "service_ideas": [
        {{
        "service_name": "AI 육아 위생 컨설턴트",
        "description": "페르소나의 아이 연령과 건강 상태에 맞춰, 의류, 장난감 등의 최적 살균 주기를 알려주고 가전을 자동으로 제어해주는 구독형 서비스입니다.",
        "solved_pain_points": ["살균 기능의 실제 효과를 눈으로 확인할 수 없어 불안하다", "매번 옷을 삶는 것은 번거롭다"],
        "service_scalability": "초기에는 ThinQ 앱 기능으로 제공하고, 추후 영유아 건강 데이터를 연동한 프리미엄 유료 모델로 확장할 수 있습니다."
        }}
    ],
    "meta": {{ "prompt_version": "{PROMPT_VERSION}" }}
    }}

    """
    return prompt