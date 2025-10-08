# tools/validators/service_idea_validator.py
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError

class ServiceIdeaModel(BaseModel):

    service_name: str = Field(..., min_length=1, description="서비스의 이름 (빈 값 불가)")
    description: str = Field(..., min_length=1, description="서비스에 대한 상세 설명 (빈 값 불가)")
    solved_pain_points: List[str] = Field(..., min_items=1, description="해결하는 핵심 불편함 목록 (최소 1개 이상)")
    service_scalability: str = Field(..., min_length=1, description="서비스의 미래 확장성 계획 (빈 값 불가)")

def validate_service_ideas(ideas_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    
    validated_ideas = []
    if not isinstance(ideas_data, list):
        print(f"--- 서비스 아이디어 유효성 검사 경고 ---")
        print(f"입력 데이터가 리스트가 아닙니다: {ideas_data}")
        print("---------------------------------")
        return []

    for idea_data in ideas_data:
        try:
            validated_idea = ServiceIdeaModel(**idea_data)
            validated_ideas.append(validated_idea.dict())
        except ValidationError as e:
            print(f"--- 서비스 아이디어 유효성 검사 실패 ---")
            print(f"오류 발생 데이터: {idea_data}")
            print(f"상세 내용: {e}")
            print("---------------------------------")
            continue
            
    return validated_ideas