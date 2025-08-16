# validators/lda_validator.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class Position2D(BaseModel):
    x: float
    y: float

class TopicModel(BaseModel):
    topic_id: str
    action_keywords: List[str] = Field(..., min_items=1)
    position_2d: Optional[Position2D] = None

class LDAResultModel(BaseModel):
    status: str = Field(..., pattern=r"LDA complete")
    cluster_id: int
    num_topics: int
    topics_summary_list: List[TopicModel]
    _temp_data: Dict[str, Any] # 다음 단계(기회 점수 계산)에 필요한 데이터

def validate_lda_result(result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        validated = LDAResultModel(**result_data)
        return validated.dict()
    except Exception as e:
        print(f"LDA 결과 검증 실패: {e}")
        return None