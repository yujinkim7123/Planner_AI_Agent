# validators/cluster_validator.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, model_validator

# --- 상세 구조 검증을 위한 내부 모델 정의 ---
class ClusterSummaryItem(BaseModel):
    keywords: List[str]
    num_docs: int
    description: Optional[str] = None # description은 선택사항으로 처리

class VisualDataModel(BaseModel):
    reduced_features_2d: List[List[float]]
    cluster_labels: List[int]

class TempDataModel(BaseModel):
    tfidf_matrix: List[Any]
    feature_names: List[str]
    cluster_labels: List[int]

# --- 메인 검증 모델 ---
class ClusteringResultModel(BaseModel):
    status: str = Field(..., pattern=r"Clustering complete")
    num_clusters: int
    cluster_labels: List[int]
    cluster_summaries: Dict[str, ClusterSummaryItem] # 상세 모델 적용
    visual_data: VisualDataModel # 상세 모델 적용
    temp_data: TempDataModel # 상세 모델 적용

    @model_validator(mode='after')
    def check_logical_consistency(self) -> 'ClusteringResultModel':
        # 1. num_clusters와 summaries 개수 일치 여부 (기존 검증)
        if len(self.cluster_summaries) != self.num_clusters:
            raise ValueError(f"num_clusters({self.num_clusters})와 summaries 개수({len(self.cluster_summaries)}) 불일치")

        # 2. cluster_labels 데이터 일관성 (기존 검증)
        if not (self.cluster_labels == self.visual_data.cluster_labels == self.temp_data.cluster_labels):
            raise ValueError("cluster_labels 데이터가 구조 내에서 일관되지 않음")

        # 3. [추가] 문서 수 총합 일치 여부 검증
        total_docs_in_summaries = sum(summary.num_docs for summary in self.cluster_summaries.values())
        if total_docs_in_summaries != len(self.cluster_labels):
            raise ValueError(f"summaries 내 문서 총합({total_docs_in_summaries})과 실제 문서 수({len(self.cluster_labels)}) 불일치")

        # 4. [추가] cluster_labels 값의 유효 범위 검증
        valid_labels = set(range(self.num_clusters))
        for label in self.cluster_labels:
            if label not in valid_labels:
                raise ValueError(f"유효하지 않은 label '{label}' 발견. 0과 {self.num_clusters-1} 사이여야 함.")
        
        return self

def validate_clustering_result(result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        validated = ClusteringResultModel(**result_data)
        return validated.model_dump()
    except Exception as e:
        print(f"클러스터링 결과 검증 실패: {e}")
        return None