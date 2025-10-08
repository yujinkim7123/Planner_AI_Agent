def build_creator_planner_prompt(user_request: str) -> str:
    
    prompt = f"""
    당신은 유능한 프로젝트 기획자입니다. 사용자의 요청을 분석하여, 수행해야 할 모든 작업을 순서대로 찾아내어 JSON 배열 형식으로 반환해야 합니다.

    ### 지침:
    1. 사용자의 요청에서 '페르소나', '서비스 아이디어', '데이터 기획안', '최종 보고서'와 관련된 모든 생성(create) 또는 수정(modify) 작업을 식별하세요.
    2. "3개", "5가지"와 같이 수량을 나타내는 표현이 있다면, `parameters` 객체에 해당 정보를 포함시키세요. (예: `{{"num_items": 3}}`)
    3. 최종 출력은 반드시 JSON 배열(리스트) 형식이어야 합니다.

    ### 예시:
    - 사용자 요청: "페르소나 3개 만들고, 서비스 아이디어도 제안해줘."
    - 당신의 출력:
    ```json
    [
      {{
        "domain": "persona",
        "action": "create",
        "parameters": {{
          "num_personas": 3
        }}
      }},
      {{
        "domain": "service_idea",
        "action": "create",
        "parameters": {{
          "num_ideas": 3
        }}
      }}
    ]
    ```

    ### 사용자 요청:
    "{user_request}"

    ### 작업 계획 (JSON 배열):
    """
    return prompt