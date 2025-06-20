
# agents/creator.py

import json
from .utils import get_openai_client

def _get_cdp_llm_results(prompt: str) -> dict:
    """주어진 프롬프트로 LLM을 호출하여 C-D-P의 일부 항목을 생성합니다."""
    client = get_openai_client()
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"C-D-P LLM 호출 중 오류: {e}")
        return {}


def _assemble_cdp_json(workspace: dict, llm_results: dict) -> dict:
    """워크스페이스 데이터와 LLM 결과를 조합하여 최종 C-D-P JSON을 조립합니다."""
    artifacts = workspace.get("artifacts", {})
    persona = artifacts.get("selected_persona")
    service_idea = artifacts.get("selected_service_idea")
    data_plan = artifacts.get("selected_data_plan_for_service")

    # 조립에 필요한 데이터가 하나라도 없으면 오류 반환
    if not all([persona, service_idea, data_plan]):
        return {"error": "C-D-P 정의서 조립에 필요한 정보(페르소나, 서비스, 데이터 기획안)가 부족합니다."}

    return {
        "title": f"유첨. {service_idea.get('service_name', '')} C-D-P 정의서",
        "customer_delight_goal": llm_results.get("customer_delight_goal", "정의된 고객 감동 목표 없음"),
        "cx": {
            "target_definition": {
                "description": f"{persona.get('title', '')} ({persona.get('demographics', '')})",
                "quote": persona.get('motivating_quote', ''),
                "market_info": "대한민국 전체 가구의 핵심 니즈를 공략하는 주요 타겟 고객층"
            },
            "core_experience": {
                "title": "우리가 만드는 고객가치는?",
                "care": service_idea.get('description', ''),
                "customization": service_idea.get('solved_pain_points', []),
                "servitization": service_idea.get('service_scalability', '')
            }
        },
        "performance": {
            "concept": {
                "find": "살균된 가습을 안심하고 이용할 수 있는 경험",
                "unique": [
                    item.get("idea", "") + ": " + item.get("details", "") for item in data_plan.get("new_data_from_sensors", [])
                ] + [
                    item.get("sensor_name", "") + ": " + item.get("collectable_data", "") for item in data_plan.get("new_sensor_recommendation", [])
                ]
            },
            "competitiveness": { "lump_sum_sales": "...", "subscription_sales": "...", "revenue": "..." },
            "customer_value_graph": "고객가치 그래프 (시간에 따른 가치 변화, 예: '23.12, '24.1...)"
        },
        "dx": {
            "trigger": { "title": "CX 기획 Data 기반 발굴", "items": [...] },
            "accelerator": { "title": "CX 구현 솔루션 제공", "up_contents_service": [...], "data_driven_experience": [...] },
            "tracker": {
                "title": "CX검증 Data 기반 고객경험 모니터링",
                "items": llm_results.get("dx_tracker_items", [])
            }
        }
    }

def create_cdp_definition(workspace: dict, data_plan_service_name: str):
    """(신규 생성) 페르소나, 서비스 아이디어, 데이터 기획안을 종합하여 최종 C-D-P 정의서를 생성합니다."""
    print("✅ [Creator Agent] Running C-D-P Definition Generation...")
    artifacts = workspace.get("artifacts", {})
    all_data_plans = artifacts.get("data_plan_for_service", [])
    data_plan = next((p for p in all_data_plans if p.get("service_name") == data_plan_service_name), None)

    if not data_plan:
        return {"error": f"'{data_plan_service_name}' 이름의 데이터 기획안을 찾을 수 없습니다."}
        
    # all_service_ideas = artifacts.get("service_ideas", {}).get("service_ideas", [])
    service_idea = artifacts.get("selected_service_idea")
    # if not service_idea:
    #     return {"error": f"'{data_plan_service_name}' 이름의 서비스 아이디어를 찾을 수 없습니다."}

    persona = artifacts.get("selected_persona")
    # service_idea = artifacts.get("selected_service_idea")
    # data_plan = artifacts.get("selected_data_plan_for_service")

    workspace["artifacts"]["selected_data_plan"] = data_plan
    
    if not all([persona, service_idea, data_plan]):
        return {"error": "C-D-P 정의서 생성을 위한 정보(페르소나, 서비스, 데이터 기획안)가 부족합니다."}

    # 생성용 프롬프트
    prompt = f"""
    당신은 신규 서비스의 핵심 가치와 성과 지표를 정의하는 최고의 비즈니스 전략가입니다.
    아래에 제공된 정보를 종합적으로 분석하여, `customer_delight_goal`과 `dx_tracker_items` 두 가지 항목을 생성해주세요.

    ### 1. 페르소나 정보: {json.dumps(persona, ensure_ascii=False, indent=2)}
    ### 2. 서비스 아이디어 정보: {json.dumps(service_idea, ensure_ascii=False, indent=2)}
    ### 3. 데이터 기획안 정보: {json.dumps(data_plan, ensure_ascii=False, indent=2)}

    결과는 반드시 아래 JSON 형식으로만 반환해주세요.
    ```json
    {{
      "customer_delight_goal": "생성된 고객 감동 목표 슬로건",
      "dx_tracker_items": ["제안된 첫 번째 추적 지표", "제안된 두 번째 추적 지표"]
    }}
    ```
    """
    llm_results = _get_cdp_llm_results(prompt)
    if not llm_results:
        return {"error": "C-D-P 정의서의 일부 항목 생성 중 LLM 오류 발생"}

    cdp_definition = _assemble_cdp_json(workspace, llm_results)
    if "error" in cdp_definition:
        return cdp_definition

    if "error" not in cdp_definition:
        workspace["artifacts"]["cdp_definition"] = [
            c for c in workspace["artifacts"].get("cdp_definition", []) 
            if c.get("title") != cdp_definition.get("title")
        ]
        workspace["artifacts"]["cdp_definition"].append(cdp_definition)
        workspace["artifacts"]["selected_cdp_definition"] = cdp_definition
    return {"cdp_definition": cdp_definition}




def modify_cdp_definition(workspace: dict, modification_request: str):
    """(수정) 기존 C-D-P 정의서를 사용자의 요청에 따라 수정합니다."""
    print(f"✅ [Creator Agent] Running C-D-P Definition Modification...")
    artifacts = workspace.get("artifacts", {})
    existing_cdp = artifacts.get("selected_cdp_definition")
    if not existing_cdp:
        return {"error": "수정할 C-D-P 정의서가 없습니다."}

    # 수정용 프롬프트
    prompt = f"""
    당신은 최고의 비즈니스 전략가입니다. 단계별로 생각하여(Think step-by-step), '기존 정의서'의 내용을 '사용자 수정 요청'에 맞게 수정해주세요.
    수정은 `customer_delight_goal`과 `dx_tracker_items` 두 항목에 대해서만 이루어집니다.

    ### 기존 정의서 일부
    - customer_delight_goal: {existing_cdp.get('customer_delight_goal')}
    - dx_tracker_items: {existing_cdp.get('dx', {}).get('tracker', {}).get('items')}

    ### 사용자 수정 요청
    "{modification_request}"

    ### 지시사항
    사용자 수정 요청을 반영하여 `customer_delight_goal`과 `dx_tracker_items` 두 항목을 다시 생성해주세요.

    결과는 반드시 아래 JSON 형식으로만 반환해주세요.
    ```json
    {{
      "customer_delight_goal": "수정된 고객 감동 목표 슬로건",
      "dx_tracker_items": ["수정된 첫 번째 추적 지표", "수정된 두 번째 추적 지표"]
    }}
    ```
    """
    llm_results = _get_cdp_llm_results(prompt)
    if not llm_results:
        return {"error": "C-D-P 정의서의 일부 항목 수정 중 LLM 오류 발생"}

    cdp_definition = _assemble_cdp_json(workspace, llm_results)
    if "error" in cdp_definition:
        return cdp_definition

    modified_cdp = _assemble_cdp_json(workspace, llm_results)

    if "error" not in modified_cdp:
        workspace["artifacts"]["cdp_definition"] = [
            c for c in workspace["artifacts"].get("cdp_definition", []) 
            if c.get("title") != modified_cdp.get("title")
        ]
        workspace["artifacts"]["cdp_definition"].append(modified_cdp)
        workspace["artifacts"]["selected_cdp_definition"] = modified_cdp

    return {"cdp_definition": cdp_definition}