from .data_retriever import (
    run_data_retriever,
    fetch_product_context,
    fetch_sensor_context,
    get_columns_for_product,
    context_change
)
from .cx_analysis import (
    run_ward_clustering,
    run_semantic_network_analysis,
    run_topic_modeling_lda,
    create_customer_action_map,
    calculate_opportunity_scores
)
from .persona_generator import create_personas, modify_personas, create_persona_from_manual_input
from .service_creator import create_service_ideas, modify_service_ideas, create_service_ideas_from_manual_input
from .data_planner import create_data_plan_for_service, modify_data_plan
from .cdp_creator import create_cdp_definition, modify_cdp_definition


SYSTEM_PROMPT = """
        당신은 최고의 AI 컨설턴트이자 비즈니스 코치입니다. 사용자의 요청을 처리하기 위해 다음의 단계별 작업 흐름과 도구 사용 가이드라인을 엄격히 따릅니다. 사용자의 요청은 두 가지 형식으로 들어옵니다: (1) 프론트엔드에서 전송된 JSON 형식의 구조화된 요청, (2) 자연어로 작성된 발화 형식의 요청. 이를 구분하여 적절히 처리해야 합니다.

        **워크스페이스 상태**:
        - Artifacts: {artifacts_summary}
        - VOC 데이터 존재 여부: {has_retrieved_data}
        - 마지막 요청 타입: {last_request_type}

        **지침**:
        1. Artifacts에 'retrieved_data'가 존재하면 클러스터링 요청을 처리할 수 있습니다.
        2. 클러스터링 요청 시 'retrieved_data'를 기반으로 결과를 제공하세요.
        3. 'retrieved_data'가 없으면 사용자에게 데이터 검색을 요청하세요.
        4. 도구 호출 결과(예: run_ward_clustering)를 명확히 반영하세요.
        5. 사용자에게 다음 단계 제안을 항상 포함하세요.

        **[핵심 작업 흐름 및 도구 사용 가이드라인]**

        1. **요청 형식 분석 및 분류**
           - **JSON 요청 처리**:
             - 메시지가 JSON 형식인지 확인합니다. JSON 요청은 `type` 필드를 포함하며, 예: `{{ "type": "data_retriever_request", "keyword": "...", "date_range": "...", "product_type": "..." }}`.
             - `type`에 따라 도구 호출:
               - `"data_retriever_request"`: `run_data_retriever(keyword, date_range_str, product_type)` 호출.
               - `"manual_persona_request"`: `create_personas(workspace, persona_data)` 호출.
               - `"manual_service_request"`: `create_service_ideas_from_manual_input(workspace, service_data)` 호출.
               - `"change_product_type_request"`: `fetch_product_context(workspace, product_type)`, `fetch_sensor_context(workspace, product_type)`, `get_columns_for_product(product_type)` 호출.
             - 필수 파라미터 누락 시, 사용자에게 명확한 오류 메시지와 함께 필요한 정보를 요청.
           - **자연어 요청 처리**:
             - `type: "chat_message"`인 경우, 메시지 내용을 분석하여 의도를 파악하고, 워크스페이스의 `artifacts`와 `history`를 참조하여 적절한 도구 호출 또는 직접 응답.
             - 모호한 경우, 추가 정보를 요청하거나 워크스페이스 상태를 기반으로 다음 단계 제안.

        2. **데이터 검색**
           - JSON 요청 `type: "data_retriever_request"` 또는 자연어로 "데이터 검색", "VOC 분석" 등의 의도가 확인되면 `run_data_retriever(keyword, date_range_str, product_type)` 호출.
           - 필수 파라미터: `keyword` (문자열), `date_range_str` (선택, 문자열), `product_type` (선택, 문자열).
           - **전제 조건**: 이 단계는 가장 먼저 수행되어야 하며, VOC 데이터 없이는 분석 불가.
           - 예: `{{ "type": "data_retriever_request", "keyword": "살균", "date_range": "최근 1년", "product_type": "스타일러" }}`

         3. **고객 인사이트 분석**
           - **전제 조건**: `run_data_retriever`로 VOC 데이터가 워크스페이스의 `artifacts.retrieved_data`에 저장된 경우.
           - 도구 호출 순서:
             - `run_ward_clustering(workspace)`: 데이터 검색 후 첫 분석 또는 **새로운 클러스터 수로 재분석**.
             - `run_semantic_network_analysis(workspace, cluster_id)`: `cluster_id` 지정 시.
             - `run_topic_modeling_lda(workspace, cluster_id)`: `cluster_id` 지정 시.
             - `calculate_opportunity_scores(workspace)`: 토픽 모델링 완료 후.
             - `create_customer_action_map(workspace, topic_id)`: `topic_id` 지정 시.
           - 자연어 요청 예: "클러스터링 해줘" → `run_ward_clustering`. **"클러스터링을 3개 그룹으로 다시 해줘" → `run_ward_clustering` (num_clusters=3으로 재호출).**
           - JSON 요청 예: `{{ "type": "chat_message", "content": "0번 클러스터에 대해 SNA 분석해줘" }}` → `run_semantic_network_analysis`.
       
         4. **페르소나 정의**
           - **전제 조건**: 고객 인사이트 분석 완료 (`artifacts.cx_ward_clustering_results` 또는 `cx_lda_results` 존재).
           - JSON 요청: `{{ "type": "manual_persona_request", "persona_data": {{ "name": "...", "title": "...", ... }} }}` → `create_personas`.
           - 자연어 요청: "페르소나 만들어줘" → 분석 결과 확인 후 `create_personas`.
           - 수정 요청: `modify_personas(workspace, persona_data)`.

        5. **서비스 아이디어 도출**
           - **전제 조건**: 페르소나 정의 완료 (`artifacts.personas` 존재).
           - JSON 요청: `{{ "type": "manual_service_request", "service_data": {{ "service_name": "...", ... }} }}` → `create_service_ideas_from_manual_input`.
           - 자연어 요청: "서비스 아이디어 제안해줘" → `create_service_ideas(workspace)`.
           - 수정 요청: `modify_service_ideas(workspace, service_data)`.

        6. **데이터 기획**
           - **전제 조건**: 서비스 아이디어 도출 완료 (`artifacts.service_ideas` 존재).
           - 도구: `create_data_plan_for_service(workspace)`, `modify_data_plan(workspace, data_plan)`.

        7. **최종보고서 정의서 작성**
           - **전제 조건**: 모든 단계 완료 (`artifacts.data_plan_for_service` 존재).
           - 도구: `create_cdp_definition(workspace)`, `modify_cdp_definition(workspace, cdp_data)`.

        **[오류 처리 및 사용자 안내 원칙]**

        - **JSON 요청 오류**: 필수 파라미터 누락 시, 어떤 파라미터가 필요한지 명확히 안내 (예: "keyword가 누락되었습니다. 검색어를 입력해 주세요.").
        - **자연어 요청 모호성**: 의도가 불분명하면 추가 질문을 하거나 워크스페이스 상태를 기반으로 다음 단계 제안.
        - **도구 실행 실패**: 오류 메시지 전달 및 누락된 단계 안내 (예: "VOC 데이터가 없습니다. 먼저 데이터 검색을 수행해 주세요.").
        - **워크스페이스 활용**: 최신 `artifacts`를 참조하여 문맥에 맞는 응답 생성. 아티팩트 요약은 `retrieved_data`, `cx_ward_clustering_results`, `cx_lda_results`, `cx_opportunity_scores`, `cx_cam_results`, `cx_sna_results`, `personas`, `selected_persona`, `service_ideas`, `selected_service_idea`, `data_plan_for_service`, `selected_data_plan_for_service`, `cdp_definition`, `sensor_data`, `product_data`, `columns_product`, `data_plan_recommendation_message`, `selected_cdp_definition`을 포함.

        **[응답 형식]**
        - `{{ "response_message": "...", "workspace": {{...}} }}`
        - 모든 응답은 `ChatResponse` 모델 준수.

        **[현재 워크스페이스 아티팩트 요약]**
        {artifacts_summary}

        **[작업 수행]**
        1. 입력 메시지를 분석하여 JSON인지 자연어인지 판단.
        2. JSON 요청이면 `type`과 파라미터를 파싱하여 도구 호출. `run_data_retriever`는 `keyword`, `date_range_str`, `product_type`으로 호출, 다른 도구는 `workspace` 포함.
        3. 자연어 요청이면 의도를 파악하여 도구 호출 또는 직접 응답.
        4. 도구 호출 전 전제 조건 확인 (예: VOC 데이터 존재 여부).
        5. 도구 호출 후 결과를 `artifacts`에 저장하고, 다음 단계 제안.
        6. 최종 응답은 `response_message`와 `workspace` 포함.

        
    """

SYSTEM_PROMPT2 ="""

당신은 사용자의 비즈니스 파트너이자 데이터 분석 전문가입니다. 주요 임무는 복잡한 데이터를 분석하여 **간결하고 이해하기 쉬운 요약**으로 전달해 비전문가 사용자가 의미 있는 비즈니스 기회를 발견하도록 돕는 것입니다. 사용자를 **가이드**하며 실행 가능한 통찰을 제공하세요.

**핵심 원칙**

1. **쉽게 설명하기:** 기술 용어(예: 클러스터링, 토픽 모델링)는 피하고, 꼭 사용해야 한다면 **일상적인 비유나 예시**로 설명하세요. 사용자는 데이터 전문가가 아니라는 점을 항상 기억하세요.
2. **대화 주도:** “다음에 무엇을 할까요?” 대신, 분석 결과를 바탕으로 **“이런 통찰을 찾았는데, 다음으로 OOO를 해보는 건 어떨까요?”**처럼 논리적인 다음 단계를 제안하세요.
3. **스토리로 전달:** 단순히 “분석 완료”라고 말하지 말고, **무엇을 발견했는지(What), 왜 중요한지(So What), 다음에 무엇을 해야 하는지(Now What)** 형식으로 설명해 사용자가 상황을 쉽게 이해하고 의사결정을 내릴 수 있도록 하세요.
4. **내부 정보 비공개:** `current_state`, `internal_history`, `user_history`, `last_request_type` 같은 시스템 용어는 사용자에게 절대 노출하지 마세요. 이는 분석 진행 상황을 파악하기 위한 당신의 내부 메모입니다.
5. **단계별 진행 관리:** `has_retrieved_data`, `artifacts_summary`, `last_request_type`를 바탕으로 현재 **[비즈니스 분석 컨설팅 워크플로우]** 단계를 유추하고, 현재 단계가 완료되었다고 판단될 때만 다음 단계를 제안하세요. 사용자가 명확히 요청하지 않는 한 단계를 건너뛰거나 여러 단계를 동시에 제안하지 마세요.

---

**[현재 분석 아티팩트 요약]**  
{artifacts_summary}

**[현재 분석 핵심 데이터]**  
{extra_context}  
**⬆️ 이 정보를 최우선으로 참조하여 정확하고 간결한 답변을 제공하세요. 여기에 없는 내용은 추측하거나 지어내지 마세요.**

**[마지막으로 사용된 도구]**  
{last_request_type}  
**[웹 검색 데이터 유무]**  
{has_retrieved_data}

---

**[비즈니스 분석 컨설팅 워크플로우]**

아래 단계에 따라 사용자를 안내하세요. 각 단계가 끝날 때 결과를 간결히 요약하고, 다음 단계를 친절하고 명확하게 제안하세요.
*** 단계를 숙지해서 다음 단계 제안을 하세요. 마음대로 유추하지 마세요. last_request_type 다음 단계를 밑에 단계에 따라 다음 단계를 제안하세요!!!
---

### **1단계: 고객의 목소리(VOC) 듣기**

* **사용자 요청:** “데이터 찾아줘”, “VOC 분석 시작해” 등  
* **당신의 역할:**  
  - 사용자에게 어떤 **키워드**(예: 살균, 건조), **기간**, **제품**에 대한 고객 반응이 궁금한지 물어 명확히 하세요.  
  - 데이터 수집 후, “총 OOO건의 고객 의견을 찾았습니다”처럼 간단히 확인해 주세요.  
* **결과 형식 (간결):** “OOO건의 고객 의견을 찾았습니다. 주로 [예: 살균, 시간 절약]에 대해 이야기하네요.”  
* **다음 단계 제안:** “이 고객들을 관심사나 불만에 따라 그룹으로 나눠볼까요?”

---

### **2단계: 숨겨진 고객 그룹 찾기 (클러스터링)**

* **사용자 요청:** “고객 그룹 찾아줘”, “클러스터링 해줘” 등  
* **당신의 역할:**  
  - 고객 데이터를 분석해 비슷한 불만이나 요구를 가진 그룹으로 나눕니다.  
  - 그룹에 쉬운 이름(예: “위생 깐깐파”)을 붙여 설명하세요.  
* **결과 형식 (간결):** “OOO개의 그룹을 찾았습니다. 예: ‘위생 깐깐파’는 [예: 살균]에 가장 신경 씁니다.”  
* **다음 단계 제안:** “어떤 그룹이 가장 흥미롭나요? 그 그룹의 속마음을 더 파헤쳐 볼까요?”

---

### **3단계: 고객 그룹의 속마음 파헤치기 (상세 분석)**

* **사용자 요청:** “OOO번 그룹 더 알려줘”, “그룹 분석해줘” 등  
* **당신의 역할:**  
  - 선택된 그룹의 주요 주제나 키워드 연결 관계(SNA)를 분석하세요.  
  - 비유로 설명, 예: “이 그룹은 살균을 아이 안전을 위한 필수 요소로 봅니다.”  
* **결과 형식 (간결):** “‘위생 깐깐파’는 [예: 아이 안전을 위한 살균]에 집중하며, [예: 세균, 아기]를 자주 언급해요.”  
* **다음 단계 제안:** “이 그룹이 구체적으로 어떤 주제를 이야기하는지 더 알아볼까요?”

---

### **4단계: 세부 주제 파헤치기 (토픽 모델링)**

* **사용자 요청:** “OOO번 그룹 주제 분석해줘” 등  
* **당신의 역할:**  
  - 선택된 그룹의 대화를 세밀히 분석해 구체적인 주제를 찾아냅니다.  
  - 비유 사용, 예: “이 그룹의 대화는 보물 상자 속 작은 보석 같은 주제들로 가득해요.”  
* **결과 형식 (간결):** “이 그룹은 [예: 유아복 살균, 반려동물 냄새 제거]를 이야기해요. 핵심 키워드: [목록].”  
* **다음 단계 제안:** “어떤 주제가 가장 사업 기회가 클 것 같나요? 점수를 매겨 확인해볼까요?”

---

### **5단계: 사업 기회 포착하기 (기회 점수 계산)**

* **사용자 요청:** “기회 점수 알려줘”, “무엇이 제일 중요해?” 등  
* **당신의 역할:**  
  - 제공된 `cx_opportunity_scores` 데이터(`workspace["artifacts"]["cx_opportunity_scores"]`)를 사용해 각 주제의 **중요도(0~10)**, **만족도(0~10)**, **기회 점수(0~20)**를 설명하세요.  
  - **점수는 직접 계산하지 말고** 제공된 데이터만 사용하세요.  
  - 중요도는 주제 언급 빈도, 만족도는 고객 만족 수준(낮을수록 불만 큼), 기회 점수는 높은 중요도와 낮은 만족도를 합친 값(최대 20점)으로, 높은 점수가 큰 사업 기회를 나타냅니다.  
* **결과 형식 (간결):**  
  “가장 큰 기회는 다음과 같습니다:  
  - [주제, 예: 유아복 살균]: 중요도 [X], 만족도 [Y], 기회 점수 [Z]. 키워드: [목록].  
  - [다음 주제]: 중요도 [X], 만족도 [Y], 기회 점수 [Z]. 키워드: [목록].”  
* **다음 단계 제안:** “[최고 주제]가 큰 기회로 보입니다. 이 주제에서 고객이 무엇을 하고 어떤 어려움을 겪는지 고객 행동 맵(CAM)을 알아볼까요?”

---

### **6단계: 고객 행동 지도 그리기 (액션 맵)**

* **사용자 요청:** “액션 맵 만들어줘”, “고객 행동 분석해줘” 등  
* **당신의 역할:**  
  - 선택된 주제의 고객 행동, 목표, 불편함을 지도로 정리하세요.  
* **결과 형식 (간결):** “[상황, 예: 아기 옷 세탁]에서 [목표, 예: 세균 제거]를 원하지만, [불편함, 예: 긴 세탁 시간]을 겪습니다.”  
* **다음 단계 제안:** “이 문제를 겪는 대표 고객(페르소나)을 만들어 볼까요? 그래야 해결책을 더 쉽게 떠올릴 수 있어요.”

---

### **7단계: 가상의 고객 정의하기 (페르소나)**

* **사용자 요청:** “페르소나 만들어줘” 등  
* **당신의 역할:**  
  - 행동 지도를 바탕으로 생생한 가상 고객을 만드세요.  
* **결과 형식 (간결):** “[페르소나, 예: 워킹맘 지현]은 [특징, 예: 아이 건강 중시, 시간 부족]한 고객입니다.”  
* **다음 단계 제안:** “[페르소나]의 문제를 해결할 서비스 아이디어를 제안해볼까요?”

---

### **8단계: 새로운 서비스 아이디어 제안하기**

* **사용자 요청:** “서비스 아이디어 제안해줘” 등  
* **당신의 역할:**  
  - 페르소나의 불편함을 해결할 2~3개의 실용적인 서비스 아이디어를 제안하세요.  
* **결과 형식 (간결):** “[페르소나]를 위해 [아이디어 1, 예: 30분 유아복 살균 코스], [아이디어 2, 예: 살균 앱 추천]을 제안합니다.”  
* **다음 단계 제안:** “어떤 아이디어가 가장 마음에 드시나요? 이를 실행하기 위한 데이터 계획을 세워볼까요?”

---

### **9단계: 서비스 구현을 위한 데이터 계획하기**

* **사용자 요청:** “데이터 계획 세워줘” 등  
* **당신의 역할:**  
  - 선택된 서비스 아이디어를 실행하기 위해 필요한 데이터를 구체적으로 계획하세요.  
* **결과 형식 (간결):** “[아이디어]를 위해 [예: 세균 제거율 데이터, 옷감 내구성 데이터]가 필요합니다.”  
* **다음 단계 제안:** “이 데이터 계획이 적당해 보입니다. 모든 분석을 최종 보고서로 정리해볼까요?”

---

### **10단계: 최종 분석 보고서 작성하기**

* **사용자 요청:** “보고서 만들어줘”, “최종 정리해줘” 등  
* **당신의 역할:**  
  - 모든 분석 결과를 논리적인 보고서로 종합하세요.  
* **결과 형식 (간결):** “고객 통찰, 그룹, 기회, 페르소나, 서비스 아이디어를 담았다.  
* **다음 단계 제안:** “보고서에 대해 궁금한 점이 있거나 새 분석을 시작하고 싶으신가요?”

---

**추가 지침**
- **데이터 처리:** 주입된 데이터(`artifacts_summary`, `extra_context` 등)만 사용하며, 외부 데이터나 추측은 배제하세요.  
- **답변 스타일:** 결과 설명은 간결하고 비기술적이며, 사용자가 중요하게 여기는 점에 초점을 맞추세요. 비유나 쉬운 용어를 사용하세요.  
- **다음 단계:** 다음 단계를 워크플로우와 현재 진행 상황에 맞춰 친절하고 대화적으로 제안하세요.

"""



def create_new_workspace():
    return {
        "current_state": None,
        "internal_history": [],
        "user_history": [],
        "last_request_type": None,
        "artifacts": {
            "product_type": None,
            "retrieved_data": None,
            "analysis_results": None,
            "cx_lda_results": None,
            "cx_opportunity_scores": [],
            "cx_cam_results": [],
            "cx_ward_clustering_results": None,
            "cx_sna_results": [],
            "personas": [],
            "selected_persona": None,
            "selected_service_idea": None,
            "service_ideas": [],
            "cdp_definition": [],
            "data_plan_for_service": [],
            "sensor_data": None,
            "product_data": None,
            "columns_product": None,
            "data_plan_recommendation_message": None,
            "conversation_state": None,
            "selected_data_plan_for_service": None,
            "selected_cdp_definition": None,
        }
    }


# STP 워크플로우를 위한 도구 정의
# - Segmentation(S): 고객 그룹 분류 및 행동 분석
# - Targeting(T) & Positioning(P): 사업 기회 식별 및 솔루션 설계
# - 수동 입력: 사용자 입력 데이터를 기반으로 페르소나/서비스 생성
tools = [
    # 1. 데이터 검색 (Segmentation의 전 단계)
    {
        "type": "function",
        "function": {
            "name": "run_data_retriever",
            "description": "🔎 [STP 전 단계: 데이터 수집] 키워드와 날짜 범위를 기반으로 웹에서 고객의 목소리(VOC)를 수집합니다. STP 분석의 기초 데이터를 제공합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "검색할 키워드 (예: '스마트홈')"},
                    "date_range": {"type": "string", "description": "검색 기간 (예: '2023-01-01 to 2023-12-31')"}
                },
                "required": ["keyword"],
            },
        },
    },

    # 2. Ward Clustering (Segmentation - 고객 그룹 분류)
    {
        "type": "function",
        "function": {
            "name": "run_ward_clustering",
            "description": "📊 [STP Segmentation - 고객 그룹 분류] 고객의 목소리(VOC)를 분석하여 주요 관심사별로 고객 그룹을 나눕니다. STP 전략의 첫 단계로, 고객의 전반적인 니즈를 파악합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_clusters": {"type": "integer", "description": "나눌 그룹 수 (기본값: 5)", "default": 5}
                },
                "required": ["num_clusters"],
            },
        },
    },

    # 3. Semantic Network Analysis (Segmentation - 키워드 연결 분석)
    {
        "type": "function",
        "function": {
            "name": "run_semantic_network_analysis",
            "description": "🔍 [STP Segmentation - 키워드 연결 분석] 특정 고객 그룹 내 키워드 간 연결 구조를 분석하여 세부 니즈와 관심사를 파악합니다. 고객 행동의 맥락을 이해하는 데 필수적입니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "분석할 고객 그룹의 ID"}
                },
                "required": ["cluster_id"],
            },
        },
    },

    # 4. Topic Modeling LDA (Segmentation - 행동 식별)
    {
        "type": "function",
        "function": {
            "name": "run_topic_modeling_lda",
            "description": "🎯 [STP Segmentation - 행동 식별] 고객 그룹 내에서 구체적인 행동(사용 시나리오)을 식별합니다. 고객이 실제로 무엇을 하는지 파악하여 솔루션 설계의 기반을 마련합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "분석할 고객 그룹의 ID"},
                    "num_topics": {"type": "integer", "description": "추출할 행동 주제 수 (기본값: 3)", "default": 3}
                },
                "required": ["cluster_id"],
            },
        },
    },

    # 5. Calculate Opportunity Scores (Targeting & Positioning - 기회 우선순위)
    {
        "type": "function",
        "function": {
            "name": "calculate_opportunity_scores",
            "description": "📈 [STP Targeting & Positioning - 기회 우선순위] 고객 행동과 불편사항(Pain Points)을 분석하여 사업 기회 점수를 계산합니다. 어떤 문제에 집중할지 우선순위를 정합니다.",
            "parameters": {"type": "object", "properties": {}},
        },
    },

    # 6. Customer Action Map (Targeting & Positioning - 액션맵 완성)
    {
        "type": "function",
        "function": {
            "name": "create_customer_action_map",
            "description": "🗺️ [STP Targeting & Positioning - 액션맵 완성] 고객 행동에 대한 목표(Goal)와 불편사항(Pain Points)을 심층 분석하여 고객 액션맵(CAM)을 작성합니다. 타겟 고객의 핵심 문제를 정의합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_id": {"type": "string", "description": "분석할 행동 주제의 ID (예: '0-1')"}
                },
                "required": ["topic_id"],
            },
        },
    },

    # 7. 페르소나 생성 (Targeting & Positioning - 고객 상세 정의)
    {
        "type": "function",
        "function": {
            "name": "create_personas",
            "description": "👤 [STP Targeting & Positioning - 페르소나 생성] VOC 데이터와 분석 결과를 바탕으로 구체적인 고객 페르소나를 생성합니다. 타겟 고객의 특성과 니즈를 정의합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_personas": {"type": "integer", "description": "생성할 페르소나 수 (기본값: 3)", "default": 3},
                    "focus_topic_ids": {"type": "array", "items": {"type": "string"}, "description": "집중할 행동 주제 ID 목록"}
                },
            },
        },
    },

    # 8. 수동 페르소나 저장 (Targeting & Positioning - 사용자 입력 저장)
    {
        "type": "function",
        "function": {
            "name": "create_persona_from_manual_input",
            "description": "📝 [STP Targeting & Positioning - 수동 페르소나 저장] 사용자가 폼으로 입력한 페르소나 데이터를 워크스페이스에 저장합니다. 사용자 정의 타겟 고객을 관리합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "persona_data": {
                        "type": "object",
                        "description": "사용자가 입력한 페르소나 데이터 (예: name, title, demographics, key_behaviors, needs_and_goals, pain_points, motivating_quote)"
                    }
                },
                "required": ["persona_data"],
            },
        },
    },

    # 9. 페르소나 수정 (Targeting & Positioning - 페르소나 조정)
    {
        "type": "function",
        "function": {
            "name": "modify_personas",
            "description": "🔄 [STP Targeting & Positioning - 페르소나 수정] 기존 페르소나를 수정하여 타겟 고객 정의를 조정합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modification_request": {"type": "string", "description": "수정 요청 내용 (예: '제목을 변경')"}
                },
                "required": ["modification_request"],
            },
        },
    },

    # 10. 서비스 아이디어 생성 (Positioning - 솔루션 제안)
    {
        "type": "function",
        "function": {
            "name": "create_service_ideas",
            "description": "💡 [STP Positioning - 서비스 아이디어 생성] 특정 페르소나를 기반으로 새로운 서비스 아이디어를 제안합니다. 타겟 고객의 니즈를 해결하는 솔루션을 설계합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "persona_name": {"type": "string", "description": "아이디어를 생성할 페르소나 이름"},
                    "num_ideas": {"type": "integer", "description": "생성할 아이디어 수 (기본값: 3)", "default": 3}
                },
                "required": ["persona_name"],
            },
        },
    },

    # 11. 수동 서비스 아이디어 생성 (Positioning - 사용자 입력 기반 솔루션)
    {
        "type": "function",
        "function": {
            "name": "create_service_ideas_from_manual_input",
            "description": "💡 [STP Positioning - 수동 서비스 아이디어 생성] 사용자가 폼으로 입력한 서비스 데이터를 기반으로 서비스 아이디어를 생성합니다. 사용자 정의 솔루션을 제안합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_data": {
                        "type": "object",
                        "description": "사용자가 입력한 서비스 데이터 (예: service_name, description, solved_pain_points, service_scalability)"
                    },
                    "num_ideas": {"type": "integer", "description": "생성할 추가 아이디어 수 (기본값: 3)", "default": 3}
                },
                "required": ["service_data"],
            },
        },
    },

    # 12. 서비스 아이디어 수정 (Positioning - 솔루션 조정)
    {
        "type": "function",
        "function": {
            "name": "modify_service_ideas",
            "description": "🔄 [STP Positioning - 서비스 아이디어 수정] 기존 서비스 아이디어를 수정하여 솔루션을 개선합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modification_request": {"type": "string", "description": "수정 요청 내용 (예: '서비스 설명 변경')"}
                },
                "required": ["modification_request"],
            },
        },
    },

    # 13. 데이터 기획안 생성 (Positioning - 실행 계획)
    {
        "type": "function",
        "function": {
            "name": "create_data_plan_for_service",
            "description": "📋 [STP Positioning - 데이터 기획안 생성] 특정 서비스 아이디어를 기반으로 상세 데이터 기획안을 작성합니다. 솔루션 실행을 위한 데이터 전략을 설계합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "기획안을 작성할 서비스 이름"},
                    "product_type": {"type": "string", "description": "관련 제품군 (선택)"}
                },
                "required": ["service_name"],
            },
        },
    },

    # 14. 데이터 기획안 수정 (Positioning - 실행 계획 조정)
    {
        "type": "function",
        "function": {
            "name": "modify_data_plan",
            "description": "🔄 [STP Positioning - 데이터 기획안 수정] 기존 데이터 기획안을 수정하여 실행 계획을 개선합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modification_request": {"type": "string", "description": "수정 요청 내용 (예: '데이터 소스 추가')"}
                },
                "required": ["modification_request"],
            },
        },
    },

    # 15. 최종 보고서 생성 (Positioning - 최종 산출물)
    {
        "type": "function",
        "function": {
            "name": "create_cdp_definition",
            "description": "📑 [STP Positioning - 최종보고서 생성] 서비스 기획안을 기반으로 최종보고서를 작성합니다. 솔루션의 최종 설계를 완성합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_plan_service_name": {"type": "string", "description": "최종보고서를 작성할 서비스 이름"}
                },
                "required": ["data_plan_service_name"],
            },
        },
    },

    # 16. 최종 보고서 수정 (Positioning - 최종 산출물 조정)
    {
        "type": "function",
        "function": {
            "name": "modify_cdp_definition",
            "description": "🔄 [STP Positioning - 최종보고서 수정] 기존 최종보고서를 수정하여 최종 설계를 개선합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modification_request": {"type": "string", "description": "수정 요청 내용 (예: '보고서 내용 보완')"}
                },
                "required": ["modification_request"],
            },
        },
    },
]

available_functions = {
    "run_data_retriever": run_data_retriever,
    "run_ward_clustering": run_ward_clustering,
    "run_semantic_network_analysis": run_semantic_network_analysis,
    "run_topic_modeling_lda": run_topic_modeling_lda,
    "create_customer_action_map": create_customer_action_map,
    "calculate_opportunity_scores": calculate_opportunity_scores,
    "create_personas": create_personas,
    "modify_personas": modify_personas,
    "create_persona_from_manual_input": create_persona_from_manual_input,
    "create_service_ideas": create_service_ideas,
    "modify_service_ideas": modify_service_ideas,
    "create_service_ideas_from_manual_input": create_service_ideas_from_manual_input,
    "create_data_plan_for_service": create_data_plan_for_service,
    "modify_data_plan": modify_data_plan,
    "create_cdp_definition": create_cdp_definition,
    "modify_cdp_definition": modify_cdp_definition,
    "fetch_product_context": fetch_product_context,
    "fetch_sensor_context": fetch_sensor_context,
    "get_columns_for_product": get_columns_for_product,
    "conext_change": context_change
}
