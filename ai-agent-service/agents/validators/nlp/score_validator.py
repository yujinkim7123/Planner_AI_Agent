# validators/score_validator.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class OpportunityScoreItem(BaseModel):
    topic_id: str
    action_keywords: List[str]
    importance: float
    satisfaction: float
    opportunity_score: float

def validate_scores_result(result_data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(result_data, list):
        print(f"--- 기회 점수 결과 검증 경고 ---\n입력 데이터가 리스트가 아닙니다: {result_data}\n-------------------------")
        return None
        
    validated_scores = []
    try:
        for item in result_data:
            validated_item = OpportunityScoreItem(**item)
            validated_scores.append(validated_item.dict())
        return validated_scores
    except Exception as e:
        print(f"기회 점수 결과 검증 실패: {e}")
        return None