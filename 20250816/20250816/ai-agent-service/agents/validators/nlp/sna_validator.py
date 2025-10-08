# validators/sna_validator.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class NodeModel(BaseModel):
    id: str
    community: int

class LinkModel(BaseModel):
    source: str
    target: str
    weight: float

class GraphDataModel(BaseModel):
    nodes: List[NodeModel]
    links: List[LinkModel]

class SNAResultModel(BaseModel):
    status: str = Field(..., pattern=r"SNA complete")
    cluster_id: int
    graph_data: GraphDataModel

def validate_sna_result(result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        validated = SNAResultModel(**result_data)
        return validated.dict()
    except Exception as e:
        print(f"SNA 결과 검증 실패: {e}")
        return None