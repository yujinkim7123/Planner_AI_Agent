# tools/prompts/persona_prompts.py
import json
from typing import Dict, Any, List

PROMPT_VERSION = "persona.v1.1.0"

def build_create_persona_prompt(
    analysis_artifacts: Dict[str, Any],
    web_results_sample: List[Dict[str, Any]],
    num_personas: int,
    user_request: str
) -> str:
    
  # 분석 결과와 웹 샘플을 문자열로 변환
  analysis_summary = json.dumps(analysis_artifacts, ensure_ascii=False, indent=2)
  raw_texts_sample_str = "\n- ".join([d.get('original_text', '') for d in web_results_sample])

  prompt = f"""
  (prompt_version: {PROMPT_VERSION})
  당신은 소비자 데이터 분석 결과를 해석하여, 생생하고 데이터에 기반한 고객 페르소나를 도출하는 전문 UX 리서처입니다.
  주어진 정보를 바탕으로 단계별로 생각하여(Think step-by-step) 요청받은 과업을 수행하세요.

  ### 1. (핵심) CX 분석 결과 요약
  {analysis_summary}

  ### 2. (참고) 고객 발화 원문 (샘플)
  - {raw_texts_sample_str}

  ### 3. (필수) 사용자 지시사항
  "{user_request}"

  ---
  ### 최종 지시사항
  - 위 모든 정보를 종합적으로 해석하여, 각 페르소나의 인구 통계 정보, 핵심 행동, 니즈와 목표, 페인 포인트를 구체적으로 추론해주세요.
  - **서로 다른 핵심적인 특징과 동기를 가진 {num_personas}명의 페르소나**를 생성해주세요.
  - 반드시 '사용자 지시사항'을 최우선으로 반영해야 합니다.
  - 결과는 반드시 아래의 JSON 형식으로만 반환해주세요. 다른 설명은 절대 추가하지 마세요.
  - 각 페르소나를 생성할 때, 위 'CX 분석 결과 요약'에서 가장 큰 영감을 준 '토픽 ID'를 1~2개 찾아서 `source_topic_ids` 필드에 포함해주세요.
  ```json
  {{
    "personas": [
      {{
        "name": "박서준 (가명)",
        "role": "꼼꼼한 위생관리맘",
        "demographics": "30대 후반, 맞벌이, 7세 아이 엄마",
        "behavioral_traits": [ "아이 옷은 반드시 살균 기능으로 관리", "가전제품 구매 전 온라인 후기를 30개 이상 비교 분석" ],
        "needs_and_goals": [ "가족의 건강을 유해세균으로부터 지키고 싶다", "반복적인 가사 노동 시간을 줄이고 싶다" ],
        "pain_points": [ "매번 옷을 삶는 것은 번거롭고 옷감이 상할까 걱정된다", "살균 기능의 실제 효과를 눈으로 확인할 수 없어 불안하다" ],
        "motivating_quote": "아이가 쓰는 건데, 조금 비싸더라도 확실한 걸로 사야 마음이 놓여요."
        "source_topic_ids": ["0-1", "0-3"]
      }}
    ],
    "meta": {{
      "prompt_version": "{PROMPT_VERSION}"
    }},
    "recommendation_message": "페르소나 생성이 완료되었습니다."
  }}"""
  return prompt


def build_modify_persona_prompt(
    existing_personas: List[Dict[str, Any]],
    modification_request: str,
    analysis_artifacts: Dict[str, Any], # 💡 [핵심] 생성의 근거가 되었던 원본 분석 데이터를 함께 전달받습니다.
    web_results_sample: List[Dict[str, Any]]
) -> str:

  existing_personas_str = json.dumps(existing_personas, ensure_ascii=False, indent=2)
  analysis_summary = json.dumps(analysis_artifacts, ensure_ascii=False, indent=2)
  raw_texts_sample_str = "\n- ".join([d.get('original_text', '') for d in web_results_sample])

  prompt = f"""
  (prompt_version: {PROMPT_VERSION})
  당신은 데이터 기반 페르소나를 사용자의 피드백을 반영하여 개선하는 최고의 UX 리서처입니다.

  ### 1. 수정 대상이 되는 '기존 페르소나' (AS-IS)
  {existing_personas_str}

  ### 2. 사용자의 '수정 요청사항' (TO-BE)
  "{modification_request}"

  ### 3. 페르소나 생성의 근거가 되었던 '원본 데이터' (CONTEXT)
  - CX 분석 결과: {analysis_summary}
  - 고객 발화 원문 샘플: 
  - {raw_texts_sample_str}

  ---
  ### 최종 지시사항
  1.  '기존 페르소나'를 기반으로, '수정 요청사항'을 반영하여 페르소나를 '재창조'해주세요.
  2.  이때, 수정 요청이 **'원본 데이터'와 일치하는지 반드시 검증**하고, 데이터에 근거하여 요청을 구체화해야 합니다.
  3.  수정 요청이 없는 부분은 '기존 페르소나'의 내용을 최대한 유지해야 합니다.
  4.  결과는 반드시 기존과 동일한 JSON 형식으로 반환해야 합니다.

  ```json
  {{
    "personas": [
      {{
        "name": "박서준 (가명)",
        "role": "꼼꼼한 위생관리맘",
        "demographics": "30대 후반, 맞벌이, 7세 아이 엄마",
        "behavioral_traits": [ "아이 옷은 반드시 살균 기능으로 관리", "가전제품 구매 전 온라인 후기를 30개 이상 비교 분석" ],
        "needs_and_goals": [ "가족의 건강을 유해세균으로부터 지키고 싶다", "반복적인 가사 노동 시간을 줄이고 싶다" ],
        "pain_points": [ "매번 옷을 삶는 것은 번거롭고 옷감이 상할까 걱정된다", "살균 기능의 실제 효과를 눈으로 확인할 수 없어 불안하다" ],
        "motivating_quote": "아이가 쓰는 건데, 조금 비싸더라도 확실한 걸로 사야 마음이 놓여요."
      }}
    ],
    "meta": {{
      "prompt_version": "{PROMPT_VERSION}"
    }},
    "recommendation_message": "페르소나 수정 요청이 반영되었습니다."
  }}
  """
  return prompt