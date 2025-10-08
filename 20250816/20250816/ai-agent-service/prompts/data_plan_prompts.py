import json
from typing import Dict, Any, List

PROMPT_VERSION = "data_plan.v1.0.0"

def build_create_data_plan_prompt(
    service_idea: Dict[str, Any],
    product_context: Dict[str, Any]
) -> str:

  prompt = f"""
  당신은 신규 서비스에 대한 '데이터 기반 기능'을 기획하는 최고의 데이터 전략가(Data Strategist)입니다.
  주어진 서비스 아이디어와 관련 데이터를 바탕으로, 사용자의 경험을 획기적으로 개선할 수 있는 데이터 활용 아이디어를 구체적으로 제안해주세요.

  ### 1. 기획 대상 서비스 아이디어
  {json.dumps(service_idea, ensure_ascii=False, indent=2)}

  ### 2. 관련 제품/센서 데이터 컨텍스트
  {json.dumps(product_context, ensure_ascii=False, indent=2)}

  ---
  ### 최종 지시사항
  아래 네 가지 관점에 따라, 각 항목별로 구체적인 아이디어를 2개 이상 제시하여 상세한 데이터 기획안을 완성해주세요.
  각 아이디어는 사용자가 어떤 새로운 가치를 얻게 되는지 명확히 설명해야 합니다.

  결과는 반드시 아래의 JSON 형식으로만 반환해주세요.

  ```json
  {{
    "data_plan": {{
      "service_name": "{service_idea.get('service_name', '')}",
      "data_driven_features": [
        {{
          "idea_name": "기존 데이터 활용 아이디어 이름 (예: AI 최적 세척)",
          "description": "어떤 기존 데이터를 어떻게 가공하여 사용자에게 어떤 새로운 기능을 제공할 것인지 설명합니다.",
          "required_data": ["활용할 기존 데이터 필드 목록"]
        }}
      ],
      "inferred_insights": [
        {{
          "idea_name": "센서 데이터 기반 추론 인사이트 (예: 가족 식사 패턴 분석)",
          "description": "어떤 기존 센서 데이터들을 조합하고 분석하여 어떤 새로운 인사이트(정보)를 추론해낼 것인지 설명합니다.",
          "required_sensors": ["조합할 기존 센서 목록"]
        }}
      ],
      "new_data_sources": [
        {{
          "source_type": "신규 센서 또는 외부 데이터 (예: 신규 센서)",
          "source_name": "추천할 센서 또는 외부 데이터의 이름 (예: 식기 오염도 센서)",
          "collectable_data": "해당 소스를 통해 수집 가능한 데이터에 대한 설명입니다.",
          "value_proposition": "이 새로운 데이터를 서비스와 결합했을 때 창출되는 사용자 가치를 설명합니다."
        }}
      ]
    }}
  }}
  ```
  """
  return prompt


def build_modify_data_plan_prompt(
existing_plan: Dict[str, Any],
modification_request: str,
service_idea: Dict[str, Any]
) -> str:
   
  prompt = f"""
  (prompt_version: {PROMPT_VERSION})
  당신은 최고의 데이터 전략가입니다. '기존 데이터 기획안'을 '사용자 수정 요청'에 맞게 수정해주세요.

  1. 수정 대상이 되는 '기존 데이터 기획안' (AS-IS)
  {json.dumps(existing_plan, ensure_ascii=False, indent=2)}

  2. 사용자의 '수정 요청사항' (TO-BE)
  "{modification_request}"

  3. 참고용 원본 서비스 아이디어 (CONTEXT)
  {json.dumps(service_idea, ensure_ascii=False, indent=2)}

  최종 지시사항
  '기존 데이터 기획안'을 기반으로, '수정 요청사항'을 반영하여 기획안을 '재창조'해주세요.
  수정 요청이 없는 부분은 기존 기획안의 내용을 유지해야 합니다.
  결과는 반드시 기존과 동일한 JSON 형식으로 반환해야 합니다.
  """
  return prompt