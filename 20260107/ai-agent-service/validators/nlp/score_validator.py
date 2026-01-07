# validators/score_validator.py
import math
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

class OpportunityScoreItem(BaseModel):
    topic_id: str
    action_keywords: List[str]
    importance: float
    satisfaction: float
    opportunity_score: float

    @field_validator('importance', 'satisfaction')
    @classmethod
    def check_score_range(cls, v: float) -> float:
        """[논리 검증 1] 중요도와 만족도 점수가 0과 10 사이인지 확인합니다."""
        if not (0.0 <= v <= 10.0):
            raise ValueError(f"Score '{v}' is not within the valid range of 0-10.")
        return v

    @model_validator(mode='after')
    def check_opportunity_score_formula(self) -> 'OpportunityScoreItem':
        """[논리 검증 2] 기회 점수가 공식과 일치하는지 확인합니다."""
        expected_score = self.importance + (10.0 - self.satisfaction)
        # 부동소수점 오차를 감안하여 거의 같은지(isclose) 비교합니다.
        if not math.isclose(self.opportunity_score, expected_score, rel_tol=1e-5):
            raise ValueError(
                f"Opportunity score {self.opportunity_score} does not match "
                f"the formula result {expected_score}."
            )
        return self

def validate_scores_result(result_data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(result_data, list):
        print(f"--- 기회 점수 결과 검증 경고 ---\n입력 데이터가 리스트가 아닙니다.\n-------------------------")
        return None
        
    validated_scores = []
    try:
        for item in result_data:
            validated_item = OpportunityScoreItem(**item)
            validated_scores.append(validated_item.model_dump())
        
        # --- [논리 검증 3] 리스트가 기회 점수 기준으로 내림차순 정렬되었는지 확인 ---
        for i in range(len(validated_scores) - 1):
            if validated_scores[i]['opportunity_score'] < validated_scores[i+1]['opportunity_score']:
                raise ValueError("The list is not sorted by opportunity_score in descending order.")

        return validated_scores
    except (ValidationError, ValueError, TypeError) as e:
        print(f"--- 기회 점수 결과 검증 실패 (구조적 또는 논리적 오류) ---\n상세 내용: {e}\n-------------------------")
        return None