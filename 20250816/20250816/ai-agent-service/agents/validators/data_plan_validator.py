# tools/validators/data_plan_validator.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

class DataDrivenFeatureItem(BaseModel):
    idea_name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    required_data: List[str] = Field(..., min_items=1)

class InferredInsightItem(BaseModel):
    idea_name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    required_sensors: List[str] = Field(..., min_items=1)

class NewDataSourceItem(BaseModel):
    source_type: str = Field(..., min_length=1)
    source_name: str = Field(..., min_length=1)
    collectable_data: str = Field(..., min_length=1)
    value_proposition: str = Field(..., min_length=1)


class DataPlanModel(BaseModel):

    service_name: str = Field(..., min_length=1)
    
    # [핵심 수정] 필드명을 최신 프롬프트와 동일하게 변경합니다.
    data_driven_features: List[DataDrivenFeatureItem]
    inferred_insights: List[InferredInsightItem]
    new_data_sources: List[NewDataSourceItem]


def validate_data_plan(data_plan_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
 
    if not isinstance(data_plan_data, dict):
        print(f"--- 데이터 기획안 유효성 검사 경고 ---")
        print(f"입력 데이터가 딕셔너리가 아닙니다: {data_plan_data}")
        print("---------------------------------")
        return None
    
    try:
        validated_plan = DataPlanModel(**data_plan_data)
        return validated_plan.dict()
    except ValidationError as e:
        print(f"--- 데이터 기획안 유효성 검사 실패 ---")
        print(f"오류 발생 데이터: {data_plan_data}")
        print(f"상세 내용: {e}")
        print("---------------------------------")
        return None