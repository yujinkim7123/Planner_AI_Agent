import os
from dotenv import load_dotenv
import json
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # 1. 이 줄을 추가합니다.



# --- 1. .env 파일 로드 및 초기화 ---
load_dotenv()

# --- 2. [수정] 명시적 import로 변경 ---
from agents.utils import (
    get_embedding_models, get_qdrant_client, get_openai_client, 
    get_sentiment_analyzer, parse_natural_date
)
from agents import (
    run_data_retriever,
    run_ward_clustering,
    run_semantic_network_analysis,
    run_topic_modeling_lda,
    create_customer_action_map,
    calculate_opportunity_scores,
    create_personas,
    create_personas, 
    modify_personas,
    create_data_plan_for_service,
    modify_data_plan,
    create_service_ideas, 
    modify_service_ideas, 
    create_cdp_definition, 
    modify_cdp_definition,
    fetch_product_context,
    fetch_sensor_context,
    get_columns_for_product,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    [수정됨] FastAPI 앱의 시작 시, .env 파일을 가장 먼저 로드하도록 수정합니다.
    """
    # --- 서버가 시작될 때 실행될 코드 ---
    print("\n" + "="*50)
    print("🚀 [Lifespan] 서버 시작 프로세스에 진입합니다...")
    
    # 1. ❗️가장 먼저 .env 파일을 로드합니다.
    load_dotenv()
    print("   - .env 파일 로드를 시도했습니다.")

    # 2. ❗️(디버깅용) 키가 제대로 로드되었는지 즉시 확인합니다.
    api_key_check = os.getenv("OPENAI_API_KEY")
    print(f"   - 로드된 OPENAI_API_KEY: {api_key_check[:5]}..." if api_key_check else "   - ❗️ ERROR: 키를 찾을 수 없습니다!")
    
    # 3. 모델과 클라이언트를 초기화합니다.
    get_embedding_models()
    get_qdrant_client()
    get_openai_client() # 이제 인자 없이 호출합니다.
    get_sentiment_analyzer()
    print("✅ [Lifespan] 모든 준비 완료. 요청을 받을 수 있습니다.")
    print("="*50 + "\n")

    yield
    
    # --- 서버가 종료될 때 실행될 코드 ---
    print("\n" + "="*50)
    print(" gracefully [Lifespan] 서버를 종료합니다.")
    print("="*50)

PERSONA_INPUT_GUIDE = """
---
**[페르소나 입력 추천 가이드]**
* **핵심 설명 (Who):** "저는 `[30대 직장인]`입니다."
* **목표와 니즈 (Goal):** "주로 `[간편한 저녁 식사를 원]합니다."`
* **가장 큰 불편함 (Pain Point):** "**가장 불편한 점은** `[퇴근 후 요리할 에너지가 없는 것]`**입니다.**"
* **(선택) 제품 연계 (Product):** "`[디오스 냉장고]`와 연계하고 싶습니다."
---
"""

SERVICE_INPUT_GUIDE = """
---
**[서비스 아이디어 수동 입력 가이드]**
* **서비스 이름 (What):** "제가 생각한 서비스는 `[서비스 이름]`입니다."
* **핵심 기능 (How):** "이 서비스는 `[사용자에게 제공하는 핵심 기능]`을 합니다."
* **해결 문제 (Why):** "이를 통해 `[사용자의 어떤 불편함이나 니즈를 해결]`할 수 있습니다."
* **(선택) 연관 제품 (Product):** "`[디오스 냉장고]`와 연계하면 좋을 것 같아요."
---
"""

# [3. SYSTEM_PROMPT 수정]
SYSTEM_PROMPT = f"""
당신은 사용자가 더 나은 제품과 서비스를 기획할 수 있도록 돕는 전문 AI 어시스턴트입니다.
당신은 '데이터 검색', '페르소나 생성', '서비스 아이디어 생성' 등 다양한 분석 도구를 사용할 수 있습니다.

# 핵심 행동 지침
1.  **항상 사용자의 최종 목표를 파악하고, 목표 달성에 가장 적합한 다음 단계를 제안해주세요.**

2.  **모호함 해결 규칙:**
    - 사용자의 수정 요청이 모호할 경우(예: "설명 문구를 수정해줘"), 절대 추측해서 도구를 호출하지 마세요.
    - 먼저, 아래에 주어질 `[현재 대화 초점]`과 `[생성된 아티팩트 목록]` 정보를 핵심 힌트로 사용하세요.
    - 그럼에도 불구하고 여전히 명확하지 않다면, 반드시 사용자에게 선택지를 제시하며 되물어봐야 합니다.
    - 예시: "어떤 항목의 설명을 수정할까요? (1. 페르소나, 2. 서비스 아이디어, 3. 데이터 기획안)"

3.  **과거 분석 수정 규칙 (매우 중요!):**
    - 사용자의 요청이 이전에 완료된 분석 단계를 수정하려는 의도(예: SNA 분석 후 '클러스터 개수를 2개로 바꿔줘')로 보일 경우, 순차적인 다음 단계를 제안하지 말고 **반드시 이전 분석 도구를 다시 호출해야 합니다.**
    - `[생성된 아티팩트 목록]`을 참고하여 사용자의 의도에 가장 적합한 도구를 다시 찾아보세요.

4.  **대화 시작 및 컨텍스트 없는 요청 처리:**
    - 사용자의 첫 메시지가 "아이디어 구상부터 시작"과 같거나, 어떤 페르소나/서비스 기반인지 명확하지 않은 요청에는 도구를 바로 호출하지 말고, 사용자에게 선택지를 제시하며 안내해주세요.

[현재 대화 초점]: {{dynamic_context}}

---
{PERSONA_INPUT_GUIDE}
{SERVICE_INPUT_GUIDE}
"""

# FastAPI 앱 객체를 생성합니다.
app = FastAPI(lifespan=lifespan, title="기획자 AI Agent MCP 서버")

origins = [
    "http://localhost:3001", # 프론트엔드 서버의 주소
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # 모든 메소드 허용
    allow_headers=["*"], # 모든 헤더 허용
)


SESSIONS = {}
tools = [
    # 2. Ward Clustering (Segmentation의 'S' - 숲 파악)
    {
        "type": "function",
        "function": {
            "name": "run_ward_clustering",
            "description": "📊 **[STS Segmentation - S (Segmentation) 1단계: 고객 그룹 분류 (숲 파악)]** 전체 고객의 목소리(VOC)에서 나타나는 거시적인 주제나 관심사 그룹을 발견합니다. 고객 대화의 '숲'을 먼저 파악하는 과정입니다. **이 단계는 STP 전략 수립을 위한 첫걸음이며, 고객의 니즈를 폭넓게 파악하지 않으면 비효율적인 마케팅으로 이어질 수 있습니다.**",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_clusters": {"type": "integer", "description": "나눌 그룹의 개수 (기본값: 5)", "default": 5}
                },
                "required": ["num_clusters"],
            },
        },
    },
    
    # 3. Semantic Network Analysis (Segmentation의 'S' - 나무 파악)
    {
        "type": "function",
        "function": {
            "name": "run_semantic_network_analysis",
            "description": "🔍 **[STS Segmentation - S (Segmentation) 2단계: 고객 생각 연결 구조 분석 (나무 파악)]** 특정 주제 그룹 내부의 핵심 키워드 간의 연결 구조를 분석합니다. 이를 통해 고객의 생각이 어떤 세부적인 개념들로 구성되어 있는지, 즉 '나무'들을 자세히 들여다봅니다. **이 단계를 통해 세그먼트의 구체적인 니즈를 파악하지 못하면, 추상적인 전략에 머물러 실행력이 떨어집니다.**",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "분석할 고객 그룹의 ID 번호"}
                },
                "required": ["cluster_id"],
            },
        },
    },
    
    # 4. Topic Modeling LDA (Segmentation의 'S' - 행동 식별)
    {
        "type": "function",
        "function": {
            "name": "run_topic_modeling_lda",
            "description": "🎯 **[STS Segmentation - S (Segmentation) 3단계: 고객 행동 식별 (액션 파악)]** 고객의 목소리에서 구체적인 '고객 행동(Customer Action)' 또는 '사용 시나리오'를 식별합니다. 고객들이 실제로 무엇을 '하는지'에 대한 주제들을 찾아냅니다. **이 단계를 통해 고객의 실제 행동을 파악하지 못하면, 고객의 문제 상황에 딱 맞는 솔루션 기획이 어려워집니다.**",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "토픽을 분석할 고객 그룹의 ID 번호"},
                    "num_topics": {"type": "integer", "description": "추출할 토픽의 개수 (기본값: 3)", "default": 3}
                },
                "required": ["cluster_id"],
            },
        },
    },
    
    # 5. Calculate Opportunity Scores (Targeting의 'T' & Positioning의 'P' - 사업 기회 우선순위)
    # CAM보다 먼저 오도록 순서 변경 및 설명 업데이트
    {
        "type": "function",
        "function": {
            "name": "calculate_opportunity_scores",
            "description": "📈 **[STS Targeting & Positioning - T/P 1단계: 사업 기회 점수 계산]** 도출된 모든 '고객 행동'과 'Pain Point'에 대해, 언급량(중요도)과 고객 만족도(감성)를 종합하여 사업적 '기회 점수(Opportunity Score)'를 계산합니다. 어떤 문제에 집중해야 할지 정량적으로 우선순위를 결정합니다. **이 단계를 통해 리소스 투입의 우선순위를 정량적으로 확보하지 못하면, 어떤 Pain Point에 집중할지 모호해져 STP 전략 실행의 효율성이 떨어집니다.**", # 설명 수정 및 강화
            "parameters": {"type": "object", "properties": {}},
        },
    },

    # 6. Customer Action Map (Targeting의 'T' & Positioning의 'P' - 고통과 목표 심층 분석, 이제 최종 단계)
    # Opportunity Scores 다음에 오도록 순서 변경 및 설명 업데이트
    {
        "type": "function",
        "function": {
            "name": "create_customer_action_map",
            "description": "🗺️ **[STS Targeting & Positioning - T/P 2단계: 고객 액션맵(CAM) 완성 (최종 분석 단계)]** 식별된 '고객 행동(Action)'에 대해, 고객이 궁극적으로 원하는 'Goal'과 그 과정에서 겪는 'Pain Point'를 심층적으로 분석하여 고객 액션맵(CAM)을 완성합니다. 고객의 숨은 의도와 불편함을 파악하는 핵심 단계입니다. **이 단계는 STP 중 타겟 고객의 '진짜 문제'를 정의하고 포지셔닝할 '가치'를 발굴하는 데 필수적이며, 기회 점수를 통해 우선순위가 높은 행동에 대해 더욱 깊이 있는 이해를 돕습니다.**", # 설명 수정 및 강화
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_id": {"type": "string", "description": "분석할 토픽(Action)의 ID 번호 (예: '0-1')"}
                },
                "required": ["topic_id"],
            },
        },
    },

     {"type": "function", "function": {
        "name": "create_personas",
        "description": "✅ (신규 생성) CX 분석 결과나 VOC 데이터를 바탕으로 완전히 새로운 고객 페르소나를 생성합니다.",
        "parameters": {"type": "object", "properties": {
            "num_personas": {"type": "integer", "default": 3},
            "focus_topic_ids": {"type": "array", "items": {"type": "string"}}
        }}
    }},
    {"type": "function", "function": {
        "name": "modify_personas",
        "description": "🔄 (수정) 이미 생성된 페르소나에 대해 '제목을 바꿔줘' 등 수정사항을 반영합니다.",
        "parameters": {"type": "object", "properties": { "modification_request": {"type": "string"} }, "required": ["modification_request"]}
    }},

    # --- 서비스 아이디어 도구 ---
    {"type": "function", "function": {
        "name": "create_service_ideas",
        "description": "✅ (신규 생성) 특정 페르소나를 기반으로 새로운 서비스 아이디어를 생성합니다.",
        "parameters": {"type": "object", "properties": {
            "persona_name": {"type": "string"},
            "num_ideas": {"type": "integer", "default": 3}
        }, "required": ["persona_name"]}
    }},
    {"type": "function", "function": {
        "name": "modify_service_ideas",
        "description": "🔄 (수정) 이미 생성된 서비스 아이디어에 대해 수정사항을 반영합니다.",
        "parameters": {"type": "object", "properties": { "modification_request": {"type": "string"} }, "required": ["modification_request"]}
    }},
    # 폼 입력을 위한 함수도 도구로 등록
    {"type": "function", "function": {
        "name": "create_service_ideas_from_manual_input",
        "description": "사용자가 폼(Form)으로 페르소나 정보를 직접 입력하여 서비스 아이디어를 생성할 때 사용합니다.",
        "parameters": { "type": "object", "properties": {
            "persona_data": { "type": "object", "description": "사용자가 폼에 입력한 페르소나 데이터"},
            "num_ideas": { "type": "integer", "default": 3}
        }, "required": ["persona_data"]}
    }},
    
    # --- 데이터 기획 도구 ---
    {"type": "function", "function": {
        "name": "create_data_plan_for_service",
        "description": "✅ (신규 생성) 특정 서비스 아이디어를 기반으로 상세 데이터 기획안을 생성합니다.",
        "parameters": {"type": "object", "properties": {
            "service_name": {"type": "string"},
            "product_type": {"type": "string"}
        }, "required": ["service_name"]}
    }},
    {"type": "function", "function": {
        "name": "modify_data_plan",
        "description": "🔄 (수정) 이미 생성된 데이터 기획안에 대해 수정사항을 반영합니다.",
        "parameters": {"type": "object", "properties": { "modification_request": {"type": "string"} }, "required": ["modification_request"]}
    }},

    # --- C-D-P 정의서 도구 ---
    {"type": "function", "function": {
        "name": "create_cdp_definition",
        "description": "📑 (신규 생성) 특정 서비스 이름의 기획안을 지정하여 최종 C-D-P 정의서를 생성합니다.",
        "parameters": {"type": "object", "properties": {
            "data_plan_service_name": {"type": "string"}
        }, "required": ["data_plan_service_name"]}
    }},
    {"type": "function", "function": {
        "name": "modify_cdp_definition",
        "description": "🔄 (수정) 이미 생성된 C-D-P 정의서에 대해 수정사항을 반영합니다.",
        "parameters": {"type": "object", "properties": { "modification_request": {"type": "string"} }, "required": ["modification_request"]}
    }}

]

LG_PRODUCT_KEYWORDS = [
    "스타일러", "트롬", "휘센", "퓨리케어", "디오스", "그램", 
    "올레드", "코드제로", "틔운", "시네빔", "울트라기어"
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
    "create_service_ideas": create_service_ideas,
    "modify_service_ideas": modify_service_ideas,
    "create_data_plan_for_service": create_data_plan_for_service,
    "modify_data_plan": modify_data_plan,
    "create_cdp_definition": create_cdp_definition,
    "modify_cdp_definition": modify_cdp_definition,
}




# 루트 URL("/")로 GET 요청이 오면 이 함수를 실행합니다.
@app.get("/")
def read_root():
    # JSON 형식으로 메시지를 반환합니다.
    return {"message": "MCP 서버가 성공적으로 실행되었습니다."}

# --- 워크스페이스 생성 함수 ---
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response_message: str
    workspace: dict

def create_new_workspace():
    """새로운 세션의 워크스페이스 뼈대를 생성합니다."""
    return {
        "conversation_state": None,
        "pending_action": None, 
        "history": [],
        "artifacts": {
            "product_type":None,
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
            "service_ideas": None,
            "cdp_definition": [],
            "data_plan_for_service": [],
            "sensor_data" : None,
            "product_data": None,
            "columns_product": None,
            "data_plan_recommendation_message":None,
            "conversation_state": None,
            "selected_data_plan_for_service":None,
            "selected_cdp_definition": None,
        }
    }

MODEL_NAME = "gpt-4o-mini"

def summarize_and_reset_history(workspace: dict, completed_tool_name: str, result_artifact: dict):
    """
    주요 작업(섹션)이 끝났을 때, history를 요약된 시스템 메시지로 교체하여
    LLM의 컨텍스트를 정리하고 다음 단계에 집중할 수 있도록 합니다.
    """
    print(f"🌀 Context Reset: '{completed_tool_name}' 작업 완료 후 히스토리를 리셋합니다.")
    
    # 다음 단계에 필요한 최소한의 요약 정보를 생성합니다.
    summary_text = f"이전 단계 작업인 '{completed_tool_name}'이 성공적으로 완료되었습니다."
    
    workspace['history'] = [
        {"role": "system", "content": summary_text}
    ]
    
    return workspace


def interpret_and_suggest_next_step(tool_name: str, result_artifact: dict, workspace: dict) -> str:
    client = get_openai_client()
    prompt = f"""당신은 데이터 분석 결과를 비전문가인 기획자에게 아주 쉽게 설명해주는 친절한 CX 분석 컨설턴트입니다. 
    분석 단계의 '비즈니스적 의미'를 먼저 설명하고, 
    기술 용어는 최소화하여 대화해주세요. 방금 '{tool_name}' 분석을 마쳤습니다.
    가장 중요한건 항상 단계별로 생각하여(Think step-by-step) 결과를 반환해주세요.
    """
    
    if tool_name.startswith("modify_"):
        return "✅ 요청하신 수정사항이 반영되었습니다. 워크스페이스에서 변경된 내용을 확인해보세요. 추가 작업이 필요하시면 말씀해주세요."

    if tool_name == "run_ward_clustering":
        num_clusters = result_artifact.get("cx_ward_clustering_results", {}).get("num_clusters", "N/A")
        cluster_summaries = result_artifact.get("cx_ward_clustering_results", {}).get("cluster_summaries", {})

        summary_text = ""
        for cluster_id, summary in cluster_summaries.items():
            keywords_preview = ', '.join(summary.get('keywords', [])[:3]) # 상위 3개 키워드만 표시
            summary_text += f"\n- {cluster_id}번 그룹: '{keywords_preview}...' 등"

        workspace["artifacts"]["cx_ward_clustering_results"] = result_artifact.get("cx_ward_clustering_results")

        prompt += f"""
         [지시사항]
         1. 첫 번째 S(Segmentation) 단계인 **'고객 관심사 그룹 분석(Ward Clustering)'**이 완료되었음을 알려주세요.
         2. 고객들의 목소리가 **{num_clusters}개의 큰 주제 그룹**으로 나뉘었음을 설명하고, 각 그룹의 특징(대표 키워드)을 1문장으로 간략히 요약해주세요:
         {summary_text}
         3. 이제 두 가지 선택지가 있음을 명확히 안내해주세요.
            - **옵션 1 (분석 심화):** 특정 그룹 내부를 더 깊이 들여다보는 **'의미 연결망 분석(SNA)'** 진행. (예: `1번 그룹 SNA 분석해줘`)
            - **옵션 2 (분석 수정):** 현재 그룹 분류가 만족스럽지 않다면, **클러스터 개수를 바꿔 다시 분석** 진행. (예: `클러스터 3개로 다시 분석해줘`)
         4. 사용자에게 위 두 가지 옵션 중 하나를 선택하도록 자연스럽게 질문하며 대화를 마무리해주세요. 
         1 ~4번까지의 이 모든 내용을 3~4 문장으로 요약해야 합니다.
         """

    elif tool_name == "run_semantic_network_analysis":
        cluster_id = result_artifact.get("cx_sna_results", {}).get("cluster_id")
        micro_segments = result_artifact.get("cx_sna_results", {}).get("micro_segments", [])

        core_keywords_preview = ', '.join([seg.get('core_keyword', '') for seg in micro_segments[:3]])

        workspace["artifacts"]["cx_sna_results"] = result_artifact.get("cx_sna_results")

        prompt += f"""
        [지시사항]
        1. **{cluster_id}번 그룹**에 대한 **'의미 연결망 분석(SNA)'**이 완료되었음을 알려주세요.
        2. 이 그룹 고객들의 생각은 '{core_keywords_preview}...' 등의 핵심 개념들을 중심으로 연결되어 있음을 설명해주세요.
        3. 이제 다음 S(Segmentation) 단계로, 이 그룹 고객들이 실제로 어떤 **'행동(Customer Action)'**을 하는지 파악하는 **'토픽 모델링(LDA)'**을 진행할 차례임을 설명해주세요.
        4. 이 단계를 건너뛰면 추상적인 니즈에 머물러 구체적인 제품/서비스 기획이 어렵다는 점을 언급하여 필요성을 강조해주세요.
        5. 사용자에게 "이 그룹의 고객 행동을 분석해볼까요?" 라고 물으며, 다음 행동을 명확히 제시해주세요. (예: `{cluster_id}번 그룹 LDA 분석` 또는 `다른 그룹 SNA 분석해줘`)
         1 ~5번까지의 이 모든 내용을 3문장 내외로 안내해주세요.
        """

    elif tool_name == "run_topic_modeling_lda":
        # result_artifact는 {"success": True, "message": ..., "newly_identified_topics_preview": [...] } 형태
        cluster_id_from_topic = result_artifact.get("newly_identified_topics_preview", [{}])[0].get("topic_id", "").split('-')[0] if result_artifact.get("newly_identified_topics_preview") else "N/A"
        topics_preview = result_artifact.get("newly_identified_topics_preview", [])
        if "cx_lda_results" in result_artifact and isinstance(result_artifact["cx_lda_results"], dict):
            workspace["artifacts"]["cx_lda_results"] = result_artifact["cx_lda_results"]

        topics_summary = ""
        for topic in topics_preview:
            topics_summary += f"\n- 토픽 {topic.get('topic_id')}: '{', '.join(topic.get('action_keywords', [])[:3])}...' 등의 행동"

        prompt += f"""
        [지시사항]
        1. **{cluster_id_from_topic}번 그룹**의 고객들이 보이는 주요 '행동(Customer Action)'들을 **'토픽 모델링(LDA)'**을 통해 식별했음을 알려주세요.
        2. 식별된 주요 행동들은 다음과 같습니다:{topics_summary}
        3. 이 정보는 이제 우리가 어떤 고객(Target)에게 집중하고, 어떤 문제를 해결할지 정량적으로 우선순위를 정하는 데 중요합니다. 다음 단계로, 모든 고객 행동과 Pain Point들을 종합하여 사업적 **'기회 점수(Opportunity Score)'**를 계산합니다.
        4. 이 단계를 건너뛰면 어디에 집중해야 할지 명확한 근거 없이 결정하게 되어 STP 전략 수립에 어려움을 겪을 수 있다는 점을 언급하여 필요성을 강조해주세요.
        5. 사용자에게 `기회 점수 계산해줘` 라고 명확히 다음 행동을 제시해주세요.
        1 ~5번까지의 이 모든 내용을 3문장 내외로 안내해주세요.
        """

    elif tool_name == "calculate_opportunity_scores":
        # calculate_opportunity_scores의 반환값은 {"cx_opportunity_scores": scores}
        opportunity_scores_list = result_artifact.get("cx_opportunity_scores", [])

        top_3_opportunities = ""
        if opportunity_scores_list:
            for i, score_item in enumerate(opportunity_scores_list[:3]):
                action_keywords = score_item.get("action_keywords", [])
                score = score_item.get("opportunity_score", 0)
                top_3_opportunities += f"\n- {i+1}순위: '{', '.join(action_keywords[:2])}...' (점수: {score})"
        else:
            top_3_opportunities = "\n- (아직 도출된 기회 영역이 없습니다.)"

        prompt += f"""
        [지시사항]
        1. '기회 점수' 계산 결과, 가장 점수가 높은 **상위 3개의 기회 영역(토픽)**은 다음과 같습니다:
        {top_3_opportunities}
        2. 이 점수는 STP 중 타겟 고객에게 어떤 문제(Pain Point)를 해결해줄 것인지(Positioning) 정량적으로 우선순위를 정하는 데 매우 중요한 근거가 됩니다.
        3. 이제 이 결과를 바탕으로, 가장 중요한 기회 영역에 대한 **'고객 액션맵(CAM) 분석'**을 진행하여 고객의 목표와 불편함을 심층적으로 파악할 차례입니다. 이는 우리가 어떤 Pain Point에 집중할지 최종적으로 결정하는 핵심 단계입니다.
        4. 이 단계를 건너뛰면 우선순위만 확인하고 실제 고객의 고통을 해결하기 위한 구체적인 전략을 세우기 어렵다는 점을 강조해주세요.
        5. 사용자에게 "어떤 행동(토픽 ID)에 대한 CAM 분석을 할까요?" 라고 물으며, 예를 들어 `1-0번 토픽 CAM 분석해줘`와 같이 다음 행동을 명확히 제시해주세요.
         1 ~5번까지의 이 모든 내용을 3문장 내외로 안내해주세요.
        """

    elif tool_name == "create_customer_action_map":
        # create_customer_action_map의 반환값은 {"cx_cam_results": existing_cams} (전체 누적 리스트)
        # 여기서는 방금 생성된 CAM 하나를 대상으로 설명해야 합니다.
        last_cam_result = workspace.get("artifacts", {}).get("cx_cam_results", [])[-1] if \
                          workspace.get("artifacts", {}).get("cx_cam_results") else {}
        action_name = last_cam_result.get("action_name", "N/A")
        pain_points_preview = ', '.join(last_cam_result.get("pain_points", [])[:2])

        prompt += f"""
        [지시사항]
        1. '{action_name}' 행동에 대한 **'고객 액션맵(CAM) 분석'**이 완료되었음을 알려주세요.
        2. 이 분석을 통해 핵심적인 Pain Point와 Goal이 명확히 파악되었습니다. (주요 Pain Point 예시: '{pain_points_preview}...')
        3. 모든 **STS 세그멘테이션 분석**이 성공적으로 완료되었습니다! 고객들의 다양한 관심사, 행동, 그리고 그들의 고통까지 깊이 이해할 수 있었네요.
        4. 이제 이 분석 결과를 바탕으로 '**핵심 고객 페르소나를 생성**'하여 전략을 구체화할 수 있습니다. 예를 들어, `페르소나 3명 생성해줘` 또는 `1-0 토픽 중심으로 페르소나 만들어줘` 와 같이 요청하여 다음 단계를 진행해보세요.
        5. 또는 다른 토픽(행동)에 대한 '고객 액션맵(CAM)'을 다시 생성하여 상세한 고객의 목표와 불편함을 더 깊이 이해할 수도 있습니다. (예: `1-0번 토픽 CAM 분석해줘`)
         1 ~5번까지의 이 모든 내용을 3문장 내외로 안내해주세요.
        """
    elif tool_name == "create_personas":
        personas = result_artifact.get("personas_result", {}).get("personas", [])

        if not personas:
            return "페르소나 생성에 실패했거나, 생성된 페르소나가 없습니다. 다시 시도해주세요."
        
        num_personas = len(personas)
        # 페르소나 목록을 보여주기 위한 문자열 생성
        persona_list_str = "\n".join(
            [f"* **{p.get('name')} ({p.get('title')})**" for p in personas]
        )
        
        # 첫 번째 페르소나 이름을 예시에 사용
        example_persona_name = personas[0].get('name')
        workspace["conversation_state"] = "PERSONA_COMPLETED"
        # 사용자에게 보여줄 최종 메시지
        return f"""✅ 페르소나 생성이 완료되었습니다.\n\n
          분석 결과를 바탕으로 다음과 같은 {num_personas}명의 핵심 고객 페르소나를 도출했습니다.\n\n

            {persona_list_str}

            이제 이 중 한 명을 선택하여 맞춤형 서비스 아이디어를 구체화해볼까요?\n\n
            예를 들어, "{example_persona_name} 페르소나를 위한 서비스 아이디어 3개 제안해줘" 와 같이 요청해주세요.\n\n

            💡수정을 원하시면, "페르소나 제목을 좀 더 전문적으로 바꿔줘" 와 같이 구체적인 수정사항을 알려주세요.
            """
   
    elif tool_name == "create_service_ideas" or tool_name == "create_service_ideas_from_manual_input":
        service_ideas = result_artifact.get("service_ideas_result", {}).get("service_ideas", [])
        if service_ideas:
            num_ideas = len(service_ideas)
            persona_type = "직접 입력해주신 페르소나를 기반으로" if "manual_input" in tool_name else "분석 결과를 바탕으로"
            response = f"""✅ 좋습니다! {persona_type} {num_ideas}개의 새로운 서비스 아이디어를 생성했습니다.\n\n
                        워크스페이스에서 상세 내용을 확인해보세요."\n\n
                        이제 마음에 드는 아이디어를 바탕으로 구체적인 "데이터 기획"을 시작해볼까요? 
                        데이터 기획 수행 시 서비스에 필요한 데이터를 기획해 드립니다.\n\n
                        예를 들어, `'[서비스 이름] 데이터 기획안 만들어줘'` 와 같이 요청해보세요."\n\n
                        💡 수정을 원하시면, "서비스 설명에 기술적인 내용을 보강해줘" 와 같이 구체적인 수정사항을 알려주세요."""
        
            workspace["conversation_state"] = "SERVICE_IDEA_COMPLETED"
            return response
        else:
            return "서비스 아이디어 생성에 실패했거나, 생성된 아이디어가 없습니다."
       

    # [8. 데이터 기획 에이전트 완료 후 안내 메시지 추가]
    elif tool_name == "create_data_plan_for_service":
        plan = result_artifact.get("data_plan_result", {}).get("data_plan", {})
        if plan:
            service_name = plan.get('service_name', '해당 서비스')
            workspace["conversation_state"] = "DATA_PLAN_COMPLETED"
            return f"""✅ **'{service_name}'**에 대한 데이터 기획안 생성이 완료되었습니다. \n\n
                    워크스페이스의 'data_plan_for_service' 항목에서 상세 내용을 확인해보세요.\n\n
                    이제 모든 준비가 끝났습니다!\n\n
                    마지막으로, 지금까지의 모든 내용을 종합하여 최종 산출물인 **'C-D-P 정의서'**를 만들어볼까요?\n\n
                    `"C-D-P 정의서 만들어줘"` 라고 요청해주세요.\n\n
                    💡 수정을 원하시면, "외부 데이터 활용 방안을 더 구체적으로 제안해줘" 와 같이 구체적인 수정사항을 알려주세요.
                    """
        else:
            return "데이터 기획안 생성에 실패했거나, 생성된 기획안이 없습니다."

    elif tool_name == "create_cdp_definition":
        workspace["conversation_state"] = "CDP_DEFINITION_COMPLETED"
        return """
        📑 **모든 기획 과정이 완료되었습니다!**
        최종 산출물인 C-D-P 정의서가 성공적으로 생성되었습니다. \n\n
        워크스페이스의 'cdp_definition' 항목에서 상세 내용을 확인해보세요.\n\n
        이 문서를 바탕으로 실제 제품 및 서비스 개발을 시작할 수 있습니다. 수고하셨습니다!\n\n
        💡 수정을 원하시면, "고객 감동 목표 문구를 수정해줘" 와 같이 구체적인 수정사항을 알려주세요.\n\n
        """

    else:
        return f"✅ '{tool_name}' 작업이 완료되었습니다."
    

    if prompt:
        llm_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7
        )
        return llm_response.choices[0].message.content.strip()
    else:
        # 도구 호출이 아니거나, 정의되지 않은 도구 이름일 경우
        return f"✅ '{tool_name}' 작업이 완료되었습니다."


# main.py의 handle_chat 함수를 아래 내용으로 전체 교체해주세요. (최종 완결판)
@app.post("/chat", response_model=ChatResponse)
def handle_chat(request: ChatRequest):
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    if session_id not in SESSIONS: SESSIONS[session_id] = create_new_workspace()
    
    workspace = SESSIONS[session_id].copy()
    
    response_to_user = "알 수 없는 오류가 발생했습니다." # 기본값 설정

    try:
        parsed_message = None
        try:
            parsed_message = json.loads(request.message)
        except json.JSONDecodeError:
            pass # JSON이 아니면 다음 LLM 처리 로직으로 넘어갑니다.

        if parsed_message and parsed_message.get("type") == "data_retriever_request":
            keyword = parsed_message.get("keyword")
            date_range_str = parsed_message.get("date_range")
            product_type = parsed_message.get("product_type")

            if not keyword:
                raise ValueError("키워드가 누락되었습니다. 다시 시도해주세요.")

            # data_retriever.py의 `run_data_retriever` 함수가 직접 받을 인자들
            result_artifact = run_data_retriever(
                keyword=keyword,
                date_range_str=date_range_str, # 이 인자는 data_retriever에서 parse_natural_date로 처리됩니다.
                product_type=product_type if product_type != '지정 안함' else None
            )

            # 결과 확인 로직 (기존 '검문소' 유지)
            retrieved_data = result_artifact.get("retrieved_data", {})
            if not retrieved_data.get("web_results") and not retrieved_data.get("product_results"):
                response_to_user = "해당 조건에 맞는 데이터를 찾을 수 없었습니다. 다른 키워드나 기간으로 다시 시도해보시겠어요?"
            else:
                # 데이터 검색 성공 시 사용자에게 알림 및 다음 단계 제안 (간소화)
                num_web_results = len(retrieved_data.get('web_results', []))
                num_product_results = len(artifacts.get('product_results', []))
                num_sensor_samples = len(artifacts.get('sensor_data_samples', []))
                num_columns_samples = len(artifacts.get('columns_samples', []))
   

                response_to_user = f"데이터 검색이 완료되었습니다. 웹 문서 {num_web_results}개, 제품 기능 문서 {num_product_results}개, 센서 데이터 샘플 {num_sensor_samples}개, {num_columns_samples} 슈퍼셋 데이터를 찾았습니다."
                response_to_user += "\n이제 이 데이터를 기반으로 본격적으로 'STS 세그멘테이션 분석'단계를 시작할 수 있습니다. 예를 들어, `2개 그룹으로 고객을 분류해줘 혹은 STS 분석 시작해줘` 와 같이 요청해주세요."
                
                # Artifacts 업데이트
                if "error" not in result_artifact and result_artifact:
                    workspace["artifacts"] = {**workspace["artifacts"], **result_artifact}
            
            # 사용자 메시지와 어시스턴트 응답을 history에 추가
            # 폼 입력 메시지는 "사용자 입력 폼을 통해 데이터가 검색되었습니다." 와 같은 형태로 간소화
            workspace["history"].append({"role": "user", "content": f"사용자 입력 폼을 통해 데이터가 검색되었습니다. (키워드: {keyword}, 기간: {date_range_str}, 제품군: {product_type})"})
            workspace["history"].append({"role": "assistant", "content": response_to_user})
            
            SESSIONS[session_id] = workspace
            return ChatResponse(session_id=session_id, response_message=response_to_user, workspace=workspace)
        
        elif parsed_message and parsed_message.get("type") == "manual_persona_request":
            print("✅ Handling manual persona form input request...")
            persona_data = parsed_message.get("persona_data")
            if not persona_data:
                raise ValueError("페르소나 폼 데이터(persona_data)가 누락되었습니다.")

            # 페르소나를 workspace에 추가하고 선택된 것으로 지정
            workspace["artifacts"]["personas"].append(persona_data)
            workspace["artifacts"]["selected_persona"] = persona_data
            
            response_to_user = f"입력해주신 '{persona_data.get('title')}' 페르소나가 성공적으로 추가되었습니다. 이제 이 페르소나를 기반으로 서비스 아이디어를 생성할 수 있습니다."
            
            workspace["history"].append({"role": "user", "content": "사용자 폼을 통해 페르소나 정보가 입력되었습니다."})
            workspace["history"].append({"role": "assistant", "content": response_to_user})
            
            SESSIONS[session_id] = workspace
            return ChatResponse(session_id=session_id, response_message=response_to_user, workspace=workspace)

        # 📌 [신규] 서비스 아이디어 폼 입력 요청 처리 경로
        elif parsed_message and parsed_message.get("type") == "manual_service_request":
            print("✅ Handling manual service idea form input request...")
            service_data = parsed_message.get("service_data")
            if not service_data:
                raise ValueError("서비스 아이디어 폼 데이터(service_data)가 누락되었습니다.")
            
            # 서비스 아이디어를 workspace에 추가하고 선택된 것으로 지정
            if not workspace["artifacts"].get("service_ideas"):
                workspace["artifacts"]["service_ideas"] = {"service_ideas": []}
            workspace["artifacts"]["service_ideas"]["service_ideas"].append(service_data)
            workspace["artifacts"]["selected_service_idea"] = service_data
            
            response_to_user = f"""
            입력해주신 '{service_data.get('service_name')}' 서비스 아이디어가 성공적으로 추가되었습니다. 
            이제  데이터 기획 수행 시 서비스에 필요한 데이터를 기획할 수 있습니다. 
            """
            
            workspace["history"].append({"role": "user", "content": "사용자 폼을 통해 서비스 아이디어 정보가 입력되었습니다."})
            workspace["history"].append({"role": "assistant", "content": response_to_user})
            
            SESSIONS[session_id] = workspace
            return ChatResponse(session_id=session_id, response_message=response_to_user, workspace=workspace)

        elif parsed_message and parsed_message.get("type") == "change_product_type_request":
            new_product_type = parsed_message.get("product_type")
            if not new_product_type or new_product_type == "지정 안함":
                # 제품군 선택을 해제하는 경우
                workspace = create_new_workspace() # 워크스페이스 완전 리셋
                response_to_user = "✅ 제품군 선택이 해제되었습니다. 모든 작업 환경이 초기화되었습니다."
            else:
                # 새로운 제품군으로 변경하는 경우
                print(f"🔄 [Workspace Reset] 제품군 변경 요청: '{new_product_type}'")
                workspace = create_new_workspace() # 워크스페이스 완전 리셋
                
                # 1. 새 제품군 정보 저장
                workspace["artifacts"]["product_type"] = new_product_type
                
                # 2. 새 제품군에 대한 데이터 미리 조회
                print(f"  - 새 제품군 '{new_product_type}'에 대한 데이터 미리 조회 중...")
                # 키워드는 제품군 이름 자체를 사용하여 일반적인 정보를 가져옵니다.
                product_docs = fetch_product_context(product_type=new_product_type)
                sensor_columns = fetch_sensor_context(product_type=new_product_type)
                device_columns = get_columns_for_product(product_type=new_product_type)
                
                # 3. 조회된 데이터를 새 워크스페이스에 저장
                workspace["artifacts"]["product_data"] = product_docs
                workspace["artifacts"]["sensor_data"] = sensor_columns
                workspace["artifacts"]["columns_product"] = device_columns
                
                response_to_user = f"✅ 제품군이 '{new_product_type}'(으)로 변경되었습니다. 관련 데이터가 업데이트되었으며, 기존 작업 내용은 모두 초기화되었습니다."

            # 공통 로직: 히스토리 기록 및 세션 저장 후 반환
            workspace["history"].append({"role": "user", "content": f"제품군을 '{new_product_type}'(으)로 변경했습니다."})
            workspace["history"].append({"role": "assistant", "content": response_to_user})
            SESSIONS[session_id] = workspace
            return ChatResponse(session_id=session_id, response_message=response_to_user, workspace=workspace)

        # 일반 텍스트 메시지 또는 처리되지 않은 JSON 메시지의 경우

        state = workspace.get("conversation_state")
        artifacts = workspace.get("artifacts", {})
        
        # 1-1. 마지막으로 완료된 주요 작업 상태 요약
        state_guidance_text = "아직 주요 작업이 완료되지 않았습니다."
        if state:
            state_map = {
                "PERSONA_COMPLETED": "페르소나 생성",
                "SERVICE_IDEA_COMPLETED": "서비스 아이디어 도출",
                "DATA_PLAN_COMPLETED": "데이터 기획안 작성",
                "CDP_DEFINITION_COMPLETED": "C-D-P 정의서 완성"
            }
            state_guidance_text = f"바로 이전에 '{state_map.get(state, state)}' 단계를 완료했습니다."

        # 1-2. 현재까지 생성된 분석 결과물(아티팩트) 목록 생성
        generated_artifacts = []
        if artifacts.get("retrieved_data"): generated_artifacts.append("VOC 데이터")
        if artifacts.get("cx_ward_clustering_results"): generated_artifacts.append("고객 그룹 (Ward Clustering)")
        if artifacts.get("cx_sna_results"): generated_artifacts.append("의미 연결망 분석 (SNA)")
        if artifacts.get("cx_lda_results"): generated_artifacts.append("고객 행동 토픽 (LDA)")
        if artifacts.get("cx_opportunity_scores"): generated_artifacts.append("기회 점수")
        if artifacts.get("cx_cam_results"): generated_artifacts.append("고객 액션맵 (CAM)")
        if artifacts.get("personas"): generated_artifacts.append("페르소나")
        if artifacts.get("service_ideas"): generated_artifacts.append("서비스 아이디어")
        if artifacts.get("data_plan_for_service"): generated_artifacts.append("데이터 기획안")
        if artifacts.get("cdp_definition"): generated_artifacts.append("C-D-P 정의서")

        artifacts_summary_text = f"생성된 아티팩트 목록: [{', '.join(generated_artifacts) if generated_artifacts else '없음'}]"

        # 1-3. 최종적으로 프롬프트에 주입할 동적 컨텍스트 조합
        dynamic_context_for_prompt = f"{state_guidance_text}\n{artifacts_summary_text}"
        
        # 동적 시스템 프롬프트 생성
        dynamic_system_prompt = SYSTEM_PROMPT.format(dynamic_context=dynamic_context_for_prompt)
        user_text_content = parsed_message.get('content') if parsed_message and 'content' in parsed_message else request.message
        workspace["history"].append({"role": "user", "content": user_text_content})
        messages_for_api = [{"role": "system", "content": dynamic_system_prompt}] + workspace["history"] 


        client = get_openai_client()
        first_response = client.chat.completions.create(model=MODEL_NAME, messages=messages_for_api, tools=tools, tool_choice="auto")
        response_message = first_response.choices[0].message
        
        llm_response_content = response_message.content if response_message.content is not None else ""
        workspace["history"].append(response_message.model_dump()) # LLM의 원본 응답 저장

        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            
            function_to_call = available_functions[function_name]
            result_artifact = function_to_call(workspace=workspace, **function_args) # 다른 도구 호출
            
            tool_content = result_artifact if (result_artifact and "error" not in result_artifact) else {"error": str(result_artifact.get("error", "Unknown error"))}
            if "error" not in result_artifact and result_artifact:
                workspace["artifacts"] = {**workspace["artifacts"], **result_artifact}

            workspace["history"].append({"role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": json.dumps(tool_content, ensure_ascii=False, default=str)})
            
            temp_response_to_user = interpret_and_suggest_next_step(function_name, result_artifact, workspace)
            response_to_user = temp_response_to_user if temp_response_to_user is not None else "요청을 처리했습니다."
            workspace["history"].append({"role": "assistant", "content": response_to_user})
        else:
            response_to_user = llm_response_content if llm_response_content else "어떤 도움을 드릴까요?"

    except Exception as e:
        import traceback; traceback.print_exc()
        response_to_user = f"요청 처리 중 심각한 오류가 발생했습니다: {e}"
        workspace["history"].append({"role": "assistant", "content": response_to_user})
    
    SESSIONS[session_id] = workspace
    return ChatResponse(session_id=session_id, response_message=response_to_user, workspace=workspace)
