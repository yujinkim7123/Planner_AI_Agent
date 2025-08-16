# tools/validators/persona_validator.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError, validator

# 10년차 개발자의 조언 💡:
# Pydantic 모델을 사용하는 것은 LLM의 출력을 '신뢰'하는 것이 아니라 '검증'하겠다는 의미입니다.
# 이것이 바로 안정적인 AI 시스템을 만드는 가장 중요한 첫걸음입니다.

class PersonaModel(BaseModel):

    name: str = Field(..., min_length=1, description="페르소나의 이름 (빈 값 불가)")
    role: str = Field(..., min_length=1, description="페르소나의 역할을 나타내는 한 줄 요약 (빈 값 불가)")
    demographics: str = Field(..., min_length=1, description="인구 통계 정보 (빈 값 불가)")
    

    behavioral_traits: List[str] = Field(default_factory=list, description="주요 행동 특성 목록")
    
    needs_and_goals: List[str] = Field(default_factory=list, description="핵심 니즈와 목표 목록")
    pain_points: List[str] = Field(default_factory=list, description="핵심 불편함 목록")
    motivating_quote: Optional[str] = Field(None, description="페르소나를 대표하는 인용구")

    @validator("behavioral_traits", "needs_and_goals", "pain_points", pre=True)
    def ensure_list_if_none(cls, v):
        if v is None:
            return []
        return v

def validate_personas(personas_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    
    validated_personas = []
    for p_data in personas_data:
        try:
            # Pydantic 모델로 데이터 유효성 검사 시도
            validated_persona = PersonaModel(**p_data)
            # 유효성 검사를 통과하면, 다시 딕셔너리 형태로 변환하여 리스트에 추가
            validated_personas.append(validated_persona.dict())
        except ValidationError as e:
            # 유효성 검사 실패 시, 어떤 페르소나에서 어떤 오류가 났는지 상세히 로그를 남깁니다.
            print(f"--- 페르소나 유효성 검사 실패 ---")
            print(f"오류 발생 페르소나 데이터: {p_data}")
            print(f"상세 오류 내용: {e}")
            print("---------------------------------")
            # 실패한 페르소나는 결과에 포함시키지 않습니다.
            continue
            
    return validated_personas