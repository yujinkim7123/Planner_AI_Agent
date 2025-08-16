# agents/experts/creator/registry/registry.py
from typing import Dict, Any, Callable

# '품질 기준서(Validators)'
from tools.persona_tools import create_personas_tool, modify_personas_tool
from validators.persona_validator import validate_personas

from tools.service_idea_tools import create_service_ideas_tool, modify_service_ideas_tool
from validators.service_idea_validator import validate_service_ideas

from tools.data_plan_tools import create_data_plan_tool, modify_data_plan_tool
from validators.data_plan_validator import validate_data_plan

from tools.final_document_tools import create_final_document_tool, modify_final_document_tool
from validators.final_document_validator import validate_final_document


def _build_persona_create_params(state: Dict[str, Any], plan_params: Dict[str, Any]) -> Dict[str, Any]:
    """페르소나 생성에 필요한 재료를 준비합니다."""
    return {
        "analysis_artifacts": state.get("cx_insights", {}),
        "web_results_sample": state.get("retrieved_data_summary", {}).get("top_documents_sample", []),
        "user_request": state.get("user_request", ""),
        "num_personas": plan_params.get("num_personas", 3)
    }

def _build_persona_modify_params(state: Dict[str, Any], plan_params: Dict[str, Any]) -> Dict[str, Any]:
    """페르소나 수정에 필요한 재료를 준비합니다."""
    return {
        "existing_personas": state.get("personas", []),
        "modification_request": state.get("user_request"),
        "analysis_artifacts": state.get("cx_insights", {}),
        "web_results_sample": state.get("retrieved_data_summary", {}).get("top_documents_sample", [])
    }


def _build_service_idea_create_params(state: Dict[str, Any], plan_params: Dict[str, Any]) -> Dict[str, Any]:
    """서비스 아이디어 생성에 필요한 재료를 준비합니다."""
    if not state.get("personas"): 
        raise ValueError("서비스 아이디어 생성을 위한 페르소나 데이터가 없습니다.")
    return {
        "personas": state.get("personas", []),
        "cx_insights": state.get("cx_insights", {}),
        "product_context": {"product_type": state.get("product_type")},
        "num_ideas_per_persona": plan_params.get("num_ideas", 3)
    }

def _build_service_idea_modify_params(state: Dict[str, Any], plan_params: Dict[str, Any]) -> Dict[str, Any]:
    """서비스 아이디어 수정에 필요한 재료를 준비합니다."""
    if not state.get("personas"): 
        raise ValueError("서비스 아이디어 수정을 위한 페르소나 데이터가 없습니다.")
    return {
        "existing_ideas": state.get("service_ideas", []),
        "modification_request": state.get("user_request"),
        "persona": state.get("personas", [{}])[0], 
        "cx_insights": state.get("cx_insights", {})
    }


def _build_data_plan_create_params(state: Dict[str, Any], plan_params: Dict[str, Any]) -> Dict[str, Any]:
    """데이터 기획안 생성에 필요한 재료를 준비합니다."""
    if not state.get("service_ideas"): raise ValueError("데이터 기획안 생성을 위한 서비스 아이디어 데이터가 없습니다.")
    return {
        "service_idea": state.get("service_ideas", [{}])[0],
        "product_context": {"product_type": state.get("product_type")}
    }

def _build_data_plan_modify_params(state: Dict[str, Any], plan_params: Dict[str, Any]) -> Dict[str, Any]:
    """데이터 기획안 수정에 필요한 재료를 준비합니다."""
    if not state.get("service_ideas"): 
        raise ValueError("데이터 기획안 수정을 위한 서비스 아이디어 데이터가 없습니다.")
    return {
        "existing_plan": state.get("data_plan", {}),
        "modification_request": state.get("user_request"),
        "service_idea": state.get("service_ideas", [{}])[0]
    }


def _build_final_document_create_params(state: Dict[str, Any], plan_params: Dict[str, Any]) -> Dict[str, Any]:
    """최종 보고서 생성에 필요한 재료를 준비합니다."""
    required = ["personas", "service_ideas", "data_plan"]
    if not all(key in state and state[key] for key in required):
        raise ValueError("최종 보고서 생성을 위한 페르소나, 서비스 아이디어, 또는 데이터 기획안 정보가 부족합니다.")
    return {
        "persona": state.get("personas", [{}])[0],
        "service_idea": state.get("service_ideas", [{}])[0],
        "data_plan": state.get("data_plan", {})
    }

def _build_final_document_modify_params(state: Dict[str, Any], plan_params: Dict[str, Any]) -> Dict[str, Any]:
    """최종 보고서 수정에 필요한 재료를 준비합니다."""
    required = ["personas", "service_ideas", "data_plan"]
    if not all(key in state and state[key] for key in required):
        raise ValueError("최종 보고서 수정을 위한 필수 정보가 부족합니다.")
    return {
        "existing_document": state.get("final_document", {}),
        "modification_request": state.get("user_request"),
        "persona": state.get("personas", [{}])[0],
        "service_idea": state.get("service_ideas", [{}])[0],
        "data_plan": state.get("data_plan", {})
    }


TOOL_REGISTRY = {
    ("persona", "create"): {
        "tool": create_personas_tool, "validator": validate_personas,
        "payload_key": "personas", "params_builder": _build_persona_create_params
    },
    ("persona", "modify"): {
        "tool": modify_personas_tool, "validator": validate_personas,
        "payload_key": "personas", "params_builder": _build_persona_modify_params
    },
    ("service_idea", "create"): {
        "tool": create_service_ideas_tool, "validator": validate_service_ideas,
        "payload_key": "service_ideas", "params_builder": _build_service_idea_create_params
    },
    ("service_idea", "modify"): {
        "tool": modify_service_ideas_tool, "validator": validate_service_ideas,
        "payload_key": "service_ideas", "params_builder": _build_service_idea_modify_params
    },
    ("data_plan", "create"): {
        "tool": create_data_plan_tool, "validator": validate_data_plan,
        "payload_key": "data_plan", "params_builder": _build_data_plan_create_params
    },
    ("data_plan", "modify"): {
        "tool": modify_data_plan_tool, "validator": validate_data_plan,
        "payload_key": "data_plan", "params_builder": _build_data_plan_modify_params
    },
    ("final_document", "create"): {
        "tool": create_final_document_tool, "validator": validate_final_document,
        "payload_key": "final_document", "params_builder": _build_final_document_create_params
    },
    ("final_document", "modify"): {
        "tool": modify_final_document_tool, "validator": validate_final_document,
        "payload_key": "final_document", "params_builder": _build_final_document_modify_params
    }
}

def get_task_info(domain: str, action: str) -> Dict[str, Any]:
    """도메인과 액션에 맞는 작업 정보를 레지스트리에서 찾아 반환합니다."""
    task_info = TOOL_REGISTRY.get((domain, action))
    if not task_info:
        raise ValueError(f"'{domain}/{action}'에 해당하는 작업을 Registry에서 찾을 수 없습니다.")
    return task_info