# tools/validators/final_document_validator.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError


class CXTargetDefinition(BaseModel):
    description: str
    quote: str
    market_info: str

class CXCoreExperience(BaseModel):
    title: str
    care: str
    customization: List[str]
    servitization: str

class CXModel(BaseModel):
    target_definition: CXTargetDefinition
    core_experience: CXCoreExperience

class PerformanceConcept(BaseModel):
    find: str
    unique: List[str]

class PerformanceModel(BaseModel):
    concept: PerformanceConcept
    competitiveness: Optional[Dict[str, Any]] = None
    customer_value_graph: Optional[str] = None

class DXTrigger(BaseModel):
    title: str
    items: List[str]

class DXAccelerator(BaseModel):
    title: str
    up_contents_service: List[str]
    data_driven_experience: List[str]

class DXTracker(BaseModel):
    title: str
    items: List[str]

class DXModel(BaseModel):
    trigger: DXTrigger
    accelerator: DXAccelerator
    tracker: DXTracker

class FinalDocumentModel(BaseModel):
 
    title: str = Field(..., min_length=1)
    customer_delight_goal: str = Field(..., min_length=1)
    cx: CXModel
    performance: PerformanceModel
    dx: DXModel

def validate_final_document(doc_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
   
    if not isinstance(doc_data, dict):
        print(f"--- 최종 보고서 유효성 검사 경고 ---\n입력 데이터가 딕셔너리가 아닙니다: {doc_data}\n-------------------------")
        return None
    
    try:
        validated_doc = FinalDocumentModel(**doc_data)
        return validated_doc.dict()
    except ValidationError as e:
        print(f"--- 최종 보고서 유효성 검사 실패 ---\n오류 발생 데이터: {doc_data}\n상세 내용: {e}\n-------------------------")
        return None