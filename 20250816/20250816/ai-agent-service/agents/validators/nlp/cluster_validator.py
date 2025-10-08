# validators/cluster_validator.py
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ClusteringResultModel(BaseModel):
    status: str = Field(..., pattern=r"Clustering complete")
    num_clusters: int
    cluster_summaries: Dict[str, Any]
    visual_data: Dict[str, Any]
    _temp_data: Dict[str, Any]

def validate_clustering_result(result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        validated = ClusteringResultModel(**result_data)
        return validated.dict()
    except Exception as e:
        print(f"클러스터링 결과 검증 실패: {e}")
        return None