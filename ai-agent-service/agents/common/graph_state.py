# agents/graph_state.py
from typing import List, Dict, TypedDict, Optional

class AgentState(TypedDict):
    """
    LangGraph 워크플로우 전체에서 공유되는 상태 객체
    최종 ERD 설계를 반영
    """
    # 1. 프로젝트 기본 정보 (최초 입력)
    project_id: int
    user_request: str
    retrieved_data_summary: Dict   # 클라이언트에서 받은 데이터 요약
    
    #사용자의 세부 분석 요청 옵션
    analysis_options: Optional[Dict] 

    #CXAnalystAgent의 결과물
    cx_insights: Optional[Dict]      # 클러스터링, SNA, LDA 등 모든 분석 결과가 담긴 '종합 보고서'
    topics: Optional[List[Dict]]     # 기회 점수 순으로 정렬된 '핵심 토픽 목록'
    insights_summary: Optional[str]  # 분석 결과 요약 (예: "고객의 주요 불만은 가격과 품질입니다.")

    # 3. 결과물 제작 전문가 (CreatorAgent)의 결과물
    personas: Optional[List[Dict]]
    service_ideas: Optional[List[Dict]]
    data_plan: Optional[Dict]
    final_document: Optional[Dict]

    # 워크플로우 제어를 위한 상태
    current_observation: str
    next_action: Optional[str]