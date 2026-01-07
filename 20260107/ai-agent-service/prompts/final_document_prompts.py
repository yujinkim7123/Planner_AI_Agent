# tools/prompts/final_document_prompts.py
import json
from typing import Dict, Any, List

PROMPT_VERSION = "final_doc.v1.0.0"

def build_create_final_document_prompt(
    persona: Dict[str, Any],
    service_idea: Dict[str, Any],
    data_plan: Dict[str, Any]
) -> str:
        
    prompt = f"""
    (prompt_version: {PROMPT_VERSION})
    당신은 신규 서비스의 핵심 가치와 성과 지표를 정의하는 최고의 비즈니스 전략가입니다.
    아래에 제공된 페르소나, 서비스 아이디어, 데이터 기획안 정보를 종합적으로 분석하여, 최종 보고서의 DX(Digital Transformation) 파트를 완성하고 고객 감동 목표를 설정해주세요.

    ### 1. 핵심 타겟 고객 (페르소나)
    {json.dumps(persona, ensure_ascii=False, indent=2)}

    ### 2. 핵심 서비스 아이디어
    {json.dumps(service_idea, ensure_ascii=False, indent=2)}

    ### 3. 핵심 데이터 기획안
    {json.dumps(data_plan, ensure_ascii=False, indent=2)}

    ---
    ### 최종 지시사항
    위 모든 정보를 종합하여, 최종 보고서를 아래 JSON 형식에 맞춰 완성해주세요.
    특히, `dx` 섹션의 각 항목에 대한 구체적인 아이디어를 2개 이상 생성하여 반환해야 합니다.

    ```json
    {{
    "final_document": {{
        "title": "유첨. {service_idea.get('service_name', '')} 최종 보고서",
        "customer_delight_goal": "사용자의 마음을 사로잡을 수 있는 감동적인 목표 슬로건",
        "cx": {{
        "target_definition": {{
            "description": "{persona.get('title', '')} ({persona.get('demographics', '')})",
            "quote": "{persona.get('motivating_quote', '')}",
            "market_info": "대한민국 전체 가구의 핵심 니즈를 공략하는 주요 타겟 고객층"
        }},
        "core_experience": {{
            "title": "우리가 만드는 고객가치는?",
            "care": "{service_idea.get('description', '')}",
            "customization": {json.dumps(service_idea.get('solved_pain_points', []))},
            "servitization": "{service_idea.get('service_scalability', '')}"
        }}
        }},
        "performance": {{
        "concept": {{
            "find": "살균된 가습을 안심하고 이용할 수 있는 경험",
            "unique": [
            "아이디어 1: 기존 센서 데이터를 활용한 새로운 가치 제안",
            "아이디어 2: 신규 추천 센서를 통한 차별화된 경험 제공"
            ]
        }}
        }},
        "dx": {{
        "trigger": {{
            "title": "CX 기획 Data 기반 발굴",
            "items": [
            "CX 기획을 위한 데이터 기반 발굴 아이디어 1 (예: 기존 제품 사용 데이터 분석을 통한 잠재 니즈 파악)",
            "CX 기획을 위한 데이터 기반 발굴 아이디어 2 (예: VOC, 리뷰 데이터 분석을 통한 페인 포인트 구체화)"
            ]
        }},
        "accelerator": {{
            "title": "CX 구현 솔루션 제공",
            "up_contents_service": [
            "UP-Contents 서비스 아이디어 1 (예: 맞춤형 가습 모드 추천)",
            "UP-Contents 서비스 아이디어 2 (예: 소모품 교체 주기 알림 및 자동 주문)"
            ],
            "data_driven_experience": [
            "데이터 기반 경험 제공 아이디어 1 (예: 실내 공기질 데이터와 연동한 자동 운전 모드)",
            "데이터 기반 경험 제공 아이디어 2 (예: 사용자 수면 패턴 분석을 통한 야간 모드 최적화)"
            ]
        }},
        "tracker": {{
            "title": "CX검증 Data 기반 고객경험 모니터링",
            "items": [
            "CX 검증을 위한 핵심 지표 1 (예: UXD 기반 월 사용 시간 분석)",
            "CX 검증을 위한 핵심 지표 2 (예: 특정 기능(맞춤 모드) 사용 빈도 및 만족도 조사)"
            ]
        }}
        }}
    }}
    }}"""
    return prompt


def build_modify_final_document_prompt(
existing_document: Dict[str, Any],
modification_request: str,
persona: Dict[str, Any],
service_idea: Dict[str, Any],
data_plan: Dict[str, Any]
) -> str:

    prompt = f"""
    (prompt_version: {PROMPT_VERSION})
    당신은 최고의 비즈니스 전략가입니다. '기존 최종 보고서'를 '사용자 수정 요청'에 맞게 수정해주세요.

    1. 수정 대상이 되는 '기존 최종 보고서' (AS-IS)
    {json.dumps(existing_document, ensure_ascii=False, indent=2)}

    2. 사용자의 '수정 요청사항' (TO-BE)
    "{modification_request}"

    3. 참고용 원본 데이터 (CONTEXT)
    페르소나: {json.dumps(persona, ensure_ascii=False)}

    서비스 아이디어: {json.dumps(service_idea, ensure_ascii=False)}

    데이터 기획안: {json.dumps(data_plan, ensure_ascii=False)}

    최종 지시사항
    1.'기존 최종 보고서'를 기반으로, '사용자 수정 요청'을 반영하여 보고서를 '재창조'해주세요.

    2. 수정 시에는 '참고용 원본 데이터'를 반드시 확인하여, 요청사항이 데이터와 일치하는지 검증해야 합니다.

    3. 수정 요청이 없는 부분은 기존 보고서의 내용을 유지해야 합니다.

    4. 최종 결과물은 아래 JSON 형식에 맞춰 작성해주세요.

        ```json
    {{
    "final_document": {{
        "title": "유첨. {service_idea.get('service_name', '')} 최종 보고서",
        "customer_delight_goal": "사용자의 마음을 사로잡을 수 있는 감동적인 목표 슬로건",
        "cx": {{
        "target_definition": {{
            "description": "{persona.get('title', '')} ({persona.get('demographics', '')})",
            "quote": "{persona.get('motivating_quote', '')}",
            "market_info": "대한민국 전체 가구의 핵심 니즈를 공략하는 주요 타겟 고객층"
        }},
        "core_experience": {{
            "title": "우리가 만드는 고객가치는?",
            "care": "{service_idea.get('description', '')}",
            "customization": {json.dumps(service_idea.get('solved_pain_points', []))},
            "servitization": "{service_idea.get('service_scalability', '')}"
        }}
        }},
        "performance": {{
        "concept": {{
            "find": "살균된 가습을 안심하고 이용할 수 있는 경험",
            "unique": [
            "아이디어 1: 기존 센서 데이터를 활용한 새로운 가치 제안",
            "아이디어 2: 신규 추천 센서를 통한 차별화된 경험 제공"
            ]
        }}
        }},
        "dx": {{
        "trigger": {{
            "title": "CX 기획 Data 기반 발굴",
            "items": [
            "CX 기획을 위한 데이터 기반 발굴 아이디어 1 (예: 기존 제품 사용 데이터 분석을 통한 잠재 니즈 파악)",
            "CX 기획을 위한 데이터 기반 발굴 아이디어 2 (예: VOC, 리뷰 데이터 분석을 통한 페인 포인트 구체화)"
            ]
        }},
        "accelerator": {{
            "title": "CX 구현 솔루션 제공",
            "up_contents_service": [
            "UP-Contents 서비스 아이디어 1 (예: 맞춤형 가습 모드 추천)",
            "UP-Contents 서비스 아이디어 2 (예: 소모품 교체 주기 알림 및 자동 주문)"
            ],
            "data_driven_experience": [
            "데이터 기반 경험 제공 아이디어 1 (예: 실내 공기질 데이터와 연동한 자동 운전 모드)",
            "데이터 기반 경험 제공 아이디어 2 (예: 사용자 수면 패턴 분석을 통한 야간 모드 최적화)"
            ]
        }},
        "tracker": {{
            "title": "CX검증 Data 기반 고객경험 모니터링",
            "items": [
            "CX 검증을 위한 핵심 지표 1 (예: UXD 기반 월 사용 시간 분석)",
            "CX 검증을 위한 핵심 지표 2 (예: 특정 기능(맞춤 모드) 사용 빈도 및 만족도 조사)"
            ]
        }}
        }}
    }}
    }}
    """
    return prompt