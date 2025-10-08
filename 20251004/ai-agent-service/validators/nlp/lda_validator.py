# tools/validators/lda_validator.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ValidationError, model_validator
import math 

class LdaVisualDataModel(BaseModel):
    topic_positions_2d: List[List[float]]
    doc_topic_dist: List[List[float]]

class TopicModel(BaseModel):
    topic_id: str
    action_keywords: List[str] = Field(..., min_length=1)

class TempDataModel(BaseModel):
    document_indices_in_corpus: List[int]
    doc_primary_topic: List[int]

class LDAResultModel(BaseModel):
    status: str = Field(..., pattern=r"LDA complete")
    cluster_id: int
    num_topics: int
    topics_summary_list: List[TopicModel]
    visual_data: LdaVisualDataModel 
    temp_data: TempDataModel        

    @model_validator(mode='after')
    def check_logical_consistency(self) -> 'LDAResultModel':

        if self.num_topics != len(self.topics_summary_list):
            raise ValueError(f"num_topics({self.num_topics})와 topics_summary_list의 길이({len(self.topics_summary_list)})가 일치하지 않습니다.")

        topic_ids = [topic.topic_id for topic in self.topics_summary_list]
        if len(set(topic_ids)) != len(topic_ids):
            raise ValueError("topics_summary_list 내에 중복된 topic_id가 존재합니다.")
        for topic_id in topic_ids:
            if not topic_id.startswith(f"{self.cluster_id}-"):
                raise ValueError(f"topic_id '{topic_id}'가 cluster_id '{self.cluster_id}'로 시작하지 않습니다.")

        num_docs_in_cluster = len(self.temp_data.document_indices_in_corpus)

        if self.num_topics >= 2 and len(self.visual_data.topic_positions_2d) != self.num_topics:
            raise ValueError(f"topic_positions_2d의 길이({len(self.visual_data.topic_positions_2d)})가 num_topics({self.num_topics})와 일치하지 않습니다.")
        if len(self.visual_data.doc_topic_dist) != num_docs_in_cluster:
            raise ValueError(f"doc_topic_dist의 행 수({len(self.visual_data.doc_topic_dist)})가 문서 수({num_docs_in_cluster})와 일치하지 않습니다.")
        for i, dist in enumerate(self.visual_data.doc_topic_dist):
            if len(dist) != self.num_topics:
                raise ValueError(f"doc_topic_dist의 {i}번째 행의 열 수({len(dist)})가 num_topics({self.num_topics})와 일치하지 않습니다.")
            
            if not math.isclose(sum(dist), 1.0, rel_tol=1e-5):
                raise ValueError(f"doc_topic_dist의 {i}번째 행의 합이 1이 아닙니다: {sum(dist)}")

      
        if len(self.temp_data.doc_primary_topic) != num_docs_in_cluster:
            raise ValueError(f"doc_primary_topic의 길이({len(self.temp_data.doc_primary_topic)})가 문서 수({num_docs_in_cluster})와 일치하지 않습니다.")
        valid_labels = set(range(self.num_topics))
        for topic_idx in self.temp_data.doc_primary_topic:
            if topic_idx not in valid_labels:
                raise ValueError(f"doc_primary_topic에 유효하지 않은 토픽 번호({topic_idx})가 있습니다.")

        return self

def validate_lda_result(result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
 
    try:
        validated = LDAResultModel(**result_data)
        return validated.model_dump()
    except (ValidationError, ValueError, TypeError) as e:
        print("--- LDA 결과 검증 실패 ---")
        print(f"상세 오류 내용: {e}")
        print("--------------------------")
        return None