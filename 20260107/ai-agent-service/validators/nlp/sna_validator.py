# validators/sna_validator.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ValidationError, model_validator

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

    @model_validator(mode='after')
    def check_graph_integrity(self) -> 'GraphDataModel':
        """그래프 데이터의 논리적 무결성을 검증합니다."""
        # 검증을 빠르게 하기 위해 모든 노드 ID를 Set으로 만듭니다.
        node_ids = {node.id for node in self.nodes}

        for link in self.links:
            # [논리 검증 1] 링크에 연결된 노드가 실제 노드 목록에 존재하는지 확인
            if link.source not in node_ids:
                raise ValueError(f"Link source '{link.source}' not found in nodes.")
            if link.target not in node_ids:
                raise ValueError(f"Link target '{link.target}' not found in nodes.")
        return self    
    
#무관계인 노드도 사용자가 확인할 필요있음 검증에서 제외
""" 
           # [논리 검증 2] 노드가 자기 자신을 가리키지 않는지 확인
            if link.source == link.target:
                raise ValueError(f"Self-loop detected in node '{link.source}'.")
                
            # [논리 검증 3] 링크의 가중치가 0보다 큰지 확인
            if link.weight <= 0:
                raise ValueError(f"Link weight for ({link.source}-{link.target}) is not positive.")

        return self"""

class SNAResultModel(BaseModel):
    status: str = Field(..., pattern=r"SNA complete")
    cluster_id: int
    graph_data: GraphDataModel

def validate_sna_result(result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """SNA 결과 데이터의 구조, 타입, 논리적 일관성을 검증합니다."""
    try:
        validated = SNAResultModel(**result_data)
        return validated.model_dump()
    except (ValidationError, ValueError, TypeError) as e:  # 논리적 오류(ValueError)도 처리
        print(f"--- SNA 결과 검증 실패 ---")
        print(f"상세 오류 내용: {e}")
        print("--------------------------")
        return None