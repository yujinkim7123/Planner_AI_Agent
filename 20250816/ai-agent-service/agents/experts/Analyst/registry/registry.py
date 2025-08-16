# agents/experts/cx_agent/registry/registry.py
from typing import Dict, Any, Callable

# --- CX 분석 부서가 사용하는 모든 '기술(Tools)'과 '품질 기준서(Validators)'를 가져옵니다 ---
from tools.nlp import cluster, sna, lda, score
from validators.nlp import cluster_validator, sna_validator, lda_validator, score_validator
from tools.nlp import create_analysis_summary
from validators.nlp import summary_validator

def _build_summary_params(state: Dict[str, Any]) -> Dict[str, Any]:
    """요약 생성에 필요한 재료(topics)를 준비합니다."""
    # 기회 점수 계산 결과가 'scores'라는 키로 저장되어 있다고 가정
    topics = state.get("cx_insights", {}).get("scores")
    if not topics:
        raise ValueError("요약 생성을 위한 기회 점수(topics) 데이터가 없습니다.")
    return {"topics": topics}

def _build_clustering_params(state: Dict[str, Any]) -> Dict[str, Any]:
    retrieved_data = state.get("retrieved_data_summary", {})
    documents = [d.get('sentence_nouns', '') for d in retrieved_data.get("top_documents_sample", [])]
    if not documents:
        raise ValueError("클러스터링을 위한 문서 데이터가 없습니다.")
    return {
        "documents": documents,
        "num_clusters": state.get("analysis_options", {}).get("num_clusters", 5)
    }

def _build_sna_params(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state.get("cx_insights", {}).get("clustering"):
        raise ValueError("SNA를 위한 클러스터링 데이터가 없습니다.")
    return {
        "temp_data": state["cx_insights"]["clustering"]["_temp_data"],
        # TODO: 사용자가 여러 클러스터 중 어떤 것을 분석할지 선택하는 로직 필요
        "cluster_id": state.get("analysis_options", {}).get("target_cluster_id_for_sna", 0)
    }

def _build_lda_params(state: Dict[str, Any]) -> Dict[str, Any]:
    """LDA에 필요한 재료를 준비합니다."""
    if not state.get("cx_insights", {}).get("clustering"):
        raise ValueError("LDA를 위한 클러스터링 데이터가 없습니다.")
    return {
        "temp_data": state["cx_insights"]["clustering"]["_temp_data"],
        "cluster_id": state.get("analysis_options", {}).get("target_cluster_id_for_lda", 0),
        "num_topics": state.get("analysis_options", {}).get("num_topics_per_cluster", 3)
    }

def _build_scores_params(state: Dict[str, Any]) -> Dict[str, Any]:
    """기회 점수 계산에 필요한 재료를 준비합니다."""
    if not state.get("cx_insights", {}).get("lda"):
        raise ValueError("기회 점수 계산을 위한 LDA 데이터가 없습니다.")
    return {
        "original_documents": state.get("retrieved_data_summary", {}).get("top_documents_sample", []),
        "lda_results": state["cx_insights"]["lda"]
    }

TOOL_REGISTRY = {
    "run_clustering": {
        "tool": cluster.run_clustering,
        "validator": cluster_validator.validate_clustering_result,
        "payload_key": "clustering",
        "params_builder": _build_clustering_params
    },
    "run_sna": {
        "tool": sna.run_sna,
        "validator": sna_validator.validate_sna_result,
        "payload_key": "sna",
        "params_builder": _build_sna_params
    },
    "run_lda": {
        "tool": lda.run_lda,
        "validator": lda_validator.validate_lda_result,
        "payload_key": "lda",
        "params_builder": _build_lda_params
    },
    "calculate_scores": {
        "tool": score.calculate_opportunity_scores,
        "validator": score_validator.validate_scores_result,
        "payload_key": "scores",
        "params_builder": _build_scores_params
    },
   "create_summary": {
        "tool": create_analysis_summary.run,
        "validator": summary_validator.validate_summary_result,
        "payload_key": "summary",
        "params_builder": _build_summary_params
    }
}

def get_task_info(tool_name: str) -> Dict[str, Any]:
    """Tool 이름에 맞는 작업 정보를 레지스트리에서 찾아 반환합니다."""
    task_info = TOOL_REGISTRY.get(tool_name)
    if not task_info:
        raise ValueError(f"'{tool_name}'에 해당하는 작업을 Registry에서 찾을 수 없습니다.")
    return task_info

def get_tools_description() -> str:
    """LLM Planner에게 Agent가 사용할 수 있는 도구 목록과 설명을 알려줍니다."""
    return "\n".join([f"- {name}: {info['tool'].__doc__}" for name, info in TOOL_REGISTRY.items()])
```