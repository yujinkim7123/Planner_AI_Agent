# agents/tool_invoker.py
from typing import Dict, Tuple
import asyncio
import json
from ..common.utils import  MODEL_NAME, append_to_history,setup_logging
from ..tools import SYSTEM_PROMPT,available_functions,tools
from .response_generator import ResponseGenerator

class ToolInvoker:
    def __init__(self):
        self.logger = setup_logging()
        self.function_mapping = {
            "data_retriever_request": {
                "func": available_functions["run_data_retriever"],
                "required": ["keyword"]
            },
            "manual_persona_request": {
                "func": available_functions["create_persona_from_manual_input"],
                "required": ["persona_data"]
            },
            "manual_service_request": {
                "func": available_functions["create_service_ideas_from_manual_input"],
                "required": ["service_data"]
            },
            "change_product_type_request": {
                "func": available_functions["conext_change"],
                "required": ["product_type"]
            }
        }
        self.nlp_intent_to_function = {
            "데이터 검색": {"func": "run_data_retriever", "required": ["keyword"]},
            "클러스터링": {"func": "run_ward_clustering", "required": []},
            "토픽 모델링": {"func": "run_topic_modeling_lda", "required": ["cluster_id"]},
            "기회 점수": {"func": "calculate_opportunity_scores", "required": []},
            "고객 행동 맵": {"func": "create_customer_action_map", "required": ["topic_id"]},
            "페르소나": {"func": "create_personas", "required": []},
            "서비스 아이디어": {"func": "create_service_ideas", "required": ["persona_name"]},
            "데이터 기획안": {"func": "create_data_plan_for_service", "required": ["service_name"]},
            "최종 보고서 정의서": {"func": "create_cdp_definition", "required": ["data_plan_service_name"]}
        }

    async def invoke_tool(self, message_dict: Dict, workspace: Dict, session_id: str) -> Tuple[str, Dict]:
        """
        JSON 요청에 따라 적절한 도구를 호출합니다.
        """
        message_type = message_dict.get("type")
        workspace["last_request_type"] = message_type

        if message_type not in self.function_mapping:
            return f"🚨 오류: 알 수 없는 요청 유형: {message_type}", workspace

        func_info = self.function_mapping[message_type]
        function_name = func_info["func"].__name__
        required_params = func_info["required"]
        function_args = {
            k if k != "date_range" else "date_range_str": message_dict.get(k)
            for k in ["keyword", "date_range", "product_type", "persona_data", "service_data"]
            if message_dict.get(k)
        }

        missing_params = [k for k in required_params if k not in function_args or not function_args[k]]
        if missing_params:
            return f"⚠️ 요청에 필수 파라미터가 누락되었습니다: {', '.join(missing_params)}", workspace

        try:
            self.logger.debug(f"{function_name} 호출, 인자: {function_args}")
            result_artifact = await asyncio.to_thread(func_info["func"], workspace=workspace, **function_args)
            if "error" in result_artifact:
                return f"⚠️ {function_name} 실행 실패: {result_artifact['error']}", workspace
            return f"{function_name} 작업이 완료되었습니다.", workspace
        except Exception as e:
            self.logger.error(f"{function_name} 실행 중 오류: {e}")
            return f"🚨 {function_name} 실행 중 오류: {e}", workspace

    async def invoke_nlp_tool(self, message_dict: Dict, workspace: Dict, session_id: str) -> Tuple[str, Dict]:
        """
        자연어 요청에 따라 ChatGPT(OpenAI LLM)가 도구와 파라미터를 선택하여 호출합니다.
        """
        #intent = message_dict.get("intent", "unknown")
        content = message_dict.get("content", "")
        #params = {k: v for k, v in message_dict.items() if k not in ["type", "intent", "content"]}

        Response_Generator = ResponseGenerator()

        try:
            response_message, tool_calls, llm_content = await Response_Generator.tools_select_llm(content, workspace)
           
            if isinstance(response_message, dict): # dict인 경우 (예: 에러 응답)
                append_to_history(workspace, response_message)
            else: # Pydantic 모델인 경우 (예: OpenAI Message 객체)
                append_to_history(workspace, response_message.model_dump(exclude_none=True))
            # --- 수정 끝 ---

            if tool_calls:
                tool_outputs_to_append = []
                collected_error_messages = []

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    workspace["last_request_type"] = function_name
                    function_to_call = available_functions.get(function_name)


                    if not function_to_call:
                        error_message = f"🚨 오류: 알 수 없는 함수 호출 시도: {function_name}"
                        collected_error_messages.append(error_message)
                        tool_outputs_to_append.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps({"error": error_message}, ensure_ascii=False)
                        })
                        continue

                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        print(f"{function_name} 호출, 인자: {function_args}")
                        self.logger.debug(f"{function_name} 호출, 인자: {function_args}")
                        result_artifact = await asyncio.to_thread(function_to_call, workspace=workspace, **function_args)
                        tool_summary_content = {
                            "tool_name": function_name,
                            "success": "error" not in result_artifact,
                            "details": Response_Generator.summarize_artifact(workspace.get("artifacts", {}))
                        }

                        if "error" in result_artifact:
                            tool_summary_content["error"] = result_artifact["error"]
                            collected_error_messages.append(f"도구 '{function_name}' 실행 실패: {result_artifact['error']}")


                        tool_outputs_to_append.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": json.dumps(tool_summary_content, ensure_ascii=False)
                            })
    
                    except Exception as e:
                        error_message = f"🚨 도구 '{function_name}' 실행 중 오류 발생: {e}"
                        collected_error_messages.append(error_message)
                        tool_outputs_to_append.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps({"error": error_message}, ensure_ascii=False)
                        })
                for output_item in tool_outputs_to_append:
                    append_to_history(workspace, output_item)
                #append_to_history(workspace, tool_outputs_to_append if len(tool_outputs_to_append) > 1 else tool_outputs_to_append[0])

                if collected_error_messages:
                    return "\n".join(collected_error_messages), workspace

                return f"{function_name} 작업이 완료되었습니다.", workspace
            else:
                # LLM이 도구를 호출하지 않은 경우, 기본 응답 반환
                #return llm_content or "어떤 도움을 드릴까요?", workspace
                return "어떤 도움을 드릴까요?", workspace
            
        except Exception as e:
                self.logger.error(f"invoke_nlp_tool 실행 중 오류: {e}")
                return f"🚨 오류: 자연어 요청 처리 중 문제가 발생했습니다: {e}", workspace
   