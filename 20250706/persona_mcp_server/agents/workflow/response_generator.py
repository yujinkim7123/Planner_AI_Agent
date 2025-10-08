# agents/response_generator.py
from typing import Dict, Tuple,Optional,List
from ..common.utils import setup_logging,get_openai_client,save_workspace_to_redis,trim_history,append_to_history,MODEL_NAME
from ..tools import SYSTEM_PROMPT,tools


class ResponseGenerator:
    def __init__(self):
        self.logger = setup_logging()

    def suggest_next_step(self, workspace: dict) -> str:
        """
        워크스페이스 상태를 기반으로 다음 단계를 설정 기반으로 제안합니다.
        """
        artifacts = workspace.get("artifacts", {})
        last_request_type = workspace.get("last_request_type", None)

        # 워크플로우 단계 정의
        workflow_steps = [
            {
                "condition": lambda a: last_request_type == "manual_service_request" and a.get("service_ideas"),
                "suggestion": "서비스 아이디어를 기반으로 데이터 기획안을 작성해 보세요. 예: {'type': 'chat_message', 'content': '데이터 기획안 만들어줘'}"
            },
            {
                "condition": lambda a: last_request_type == "manual_persona_request" and a.get("personas"),
                "suggestion": "페르소나를 기반으로 서비스 아이디어를 제안해 보세요. 예: {'type': 'chat_message', 'content': '서비스 아이디어 제안해줘'}"
            },
            {
                "condition": lambda a: last_request_type == "data_retriever_request" and a.get("retrieved_data"),
                "suggestion": "고객 그룹을 분류하기 위해 워드 클러스터링을 수행해 보세요. 예: {'type': 'chat_message', 'content': '클러스터링 해줘'}"
            },
            {
                "condition": lambda a: not a.get("retrieved_data"),
                "suggestion": "먼저 VOC 데이터를 검색해 주세요. 예: {'type': 'data_retriever_request', 'keyword': '살균', 'date_range': '최근 1년', 'product_type': '스타일러'}"
            },
            {
                "condition": lambda a: not a.get("cx_ward_clustering_results"),
                "suggestion": "고객 그룹을 분류하기 위해 워드 클러스터링을 수행해 보세요. 예: {'type': 'chat_message', 'content': '클러스터링 해줘'}"
            },
            {
                "condition": lambda a: not a.get("cx_lda_results"),
                "suggestion": "고객 행동을 식별하기 위해 토픽 모델링을 수행해 보세요. 예: {'type': 'chat_message', 'content': '0번 클러스터에 대해 토픽 모델링 해줘'}"
            },
            {
                "condition": lambda a: not a.get("cx_opportunity_scores"),
                "suggestion": "사업 기회 우선순위를 정하기 위해 기회 점수를 계산해 보세요. 예: {'type': 'chat_message', 'content': '기회 점수 계산해줘'}"
            },
            {
                "condition": lambda a: not a.get("cx_cam_results"),
                "suggestion": "고객의 목표와 Pain Point를 분석하기 위해 고객 행동 맵(CAM)을 생성해 보세요. 예: {'type': 'chat_message', 'content': '0-1 토픽에 대해 CAM 생성해줘'}"
            },
            {
                "condition": lambda a: not a.get("personas"),
                "suggestion": "고객 인사이트를 기반으로 페르소나를 생성해 보세요. 예: {'type': 'chat_message', 'content': '페르소나 만들어줘'}"
            },
            {
                "condition": lambda a: not a.get("service_ideas"),
                "suggestion": "페르소나를 기반으로 서비스 아이디어를 제안해 보세요. 예: {'type': 'chat_message', 'content': '서비스 아이디어 제안해줘'}"
            },
            {
                "condition": lambda a: not a.get("data_plan_for_service"),
                "suggestion": "서비스 아이디어를 기반으로 데이터 기획안을 작성해 보세요. 예: {'type': 'chat_message', 'content': '데이터 기획안 만들어줘'}"
            },
            {
                "condition": lambda a: not a.get("cdp_definition"),
                "suggestion": "최종 C-D-P 정의서를 작성해 보세요. 예: {'type': 'chat_message', 'content': 'C-D-P 정의서 작성해줘'}"
            },
            {
                "condition": lambda a: True,  # 기본 케이스
                "suggestion": "모든 단계를 완료했습니다. 추가 분석이나 수정이 필요하시면 요청해 주세요!"
            }
        ]

        # 첫 번째로 일치하는 단계 반환
        for step in workflow_steps:
            if step["condition"](artifacts):
                return step["suggestion"]

        return "모든 단계를 완료했습니다. 추가 분석이나 수정이 필요하시면 요청해 주세요!"

    def prepare_openai_messages(self, workspace: dict, system_message_content: str) -> list:
        """Prepare messages for OpenAI API with explicit artifacts state."""
        messages = [{"role": "system", "content": system_message_content}]
        messages.extend(workspace.get("internal_history", []))
            # Artifacts 상태를 메시지에 추가
        artifacts_summary = self.summarize_artifact(workspace.get("artifacts", {}))
        messages.append({
            "role": "system",
            "content": f"Current artifacts state: {artifacts_summary}"
        })
        return messages

    #워크스페이스에 internal_history, user_history에 메시지를 추가하고, 최대 50개로 제한한다.
    def append_to_history(self, workspace, message):
        workspace["internal_history"].append(message)
        if message["role"] in ["user", "assistant"]:
            workspace["user_history"].append(message)
        max_history_length = 50
        if len(workspace["internal_history"]) > max_history_length:
            workspace["internal_history"] = workspace["internal_history"][-max_history_length:]
        if len(workspace["user_history"]) > max_history_length:
            workspace["user_history"] = workspace["user_history"][-max_history_length:]

    def summarize_artifact(self, artifacts: dict) -> str:
        """워크스페이스의 아티팩트를 요약하여 LLM 프롬프트에 포함할 문자열을 생성합니다."""
        summary_parts = []

        if not artifacts:
            return "현재 워크스페이스에 저장된 아티팩트가 없습니다."

        for key, value in artifacts.items():
            if key == "retrieved_data" and value and value.get("web_results"):
                summary_parts.append(f"- 검색된 VOC 데이터: {len(value['web_results'])}건")
            elif key == "cx_ward_clustering_results" and value and value.get("cluster_summaries"):
                summary_parts.append(f"- 워드 클러스터링: {len(value['cluster_summaries'])}개 클러스터")
            elif key == "cx_lda_results" and value and value.get("topics"):
                summary_parts.append(f"- 토픽 모델링: {len(value['topics'])}개 토픽")
            elif key == "cx_cam_results" and value:
                summary_parts.append(f"- 고객 행동 맵: {len(value)}개 생성됨")
            elif key == "cx_opportunity_scores" and value:
                summary_parts.append(f"- 기회 점수 분석: {len(value)}개 완료됨")
            elif key == "cx_sna_results" and value:
                summary_parts.append(f"- 의미 네트워크 분석: {len(value)}개 완료됨")
            elif key == "personas" and value and isinstance(value, list):
                names = ", ".join([p.get("name", "이름 없음") for p in value])
                summary_parts.append(f"- 페르소나: {len(value)}개 ({names})")
            elif key == "selected_persona" and value and value.get("name"):
                summary_parts.append(f"- 현재 선택된 페르소나: {value['name']}")
            elif key == "service_ideas" and value and isinstance(value, list):
                names = ", ".join([s.get("service_name", "이름 없음") for s in value])
                summary_parts.append(f"- 서비스 아이디어: {len(value)}개 ({names})")
            elif key == "selected_service_idea" and value and value.get("service_name"):
                summary_parts.append(f"- 현재 선택된 서비스 아이디어: {value['service_name']}")
            elif key == "data_plan_for_service" and value and isinstance(value, list):
                names = ", ".join([p.get("service_name", "이름 없음") for p in value])
                summary_parts.append(f"- 데이터 기획안: {len(value)}개 ({names})")
            elif key == "selected_data_plan_for_service" and value and value.get("service_name"):
                summary_parts.append(f"- 현재 선택된 데이터 기획안: {value['service_name']}")
            elif key == "cdp_definition" and value:
                summary_parts.append(f"- C-D-P 정의서: {len(value)}개 생성됨")
            elif key == "sensor_data" and value:
                summary_parts.append(f"- 센서 데이터: {len(value)}건")
            elif key == "product_data" and value:
                summary_parts.append(f"- 제품 데이터: {len(value)}건")
            elif key == "columns_product" and value:
                summary_parts.append(f"- 제품 메타데이터: {len(value)}개 필드")
            elif key == "data_plan_recommendation_message" and value:
                summary_parts.append(f"- 데이터 기획 추천 메시지: 저장됨")
            elif key == "selected_cdp_definition" and value:
                summary_parts.append(f"- 현재 선택된 C-D-P 정의서: 저장됨")
            # conversation_state는 artifacts 외부에서 관리하므로 여기서 제외

        if not summary_parts:
            return "현재 워크스페이스에 저장된 아티팩트가 없습니다."

        return "현재 워크스페이스에는 다음 아티팩트가 저장되어 있습니다:\n" + "\n".join(summary_parts)
    
    async def tools_select_llm(self, content: str, workspace: Dict) -> Tuple[Dict, Optional[List], Optional[str]]:
            """
            LLM을 호출하여 도구와 파라미터를 선택합니다.
            반환: (response_message, tool_calls, llm_content)
            """
            client = get_openai_client(async_client=True)
            artifacts_summary = self.summarize_artifact(workspace)
            has_retrieved_data = bool(workspace.get("artifacts", {}).get("retrieved_data"))
            system_message_content = SYSTEM_PROMPT.format(
                artifacts_summary=artifacts_summary,
                has_retrieved_data=str(has_retrieved_data),
                last_request_type=workspace.get("last_request_type", "없음")
            )
            messages = self.prepare_openai_messages(workspace, system_message_content)
            messages.append({"role": "user", "content": content})

            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=False,
                )
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                llm_content = response_message.content
                return response_message, tool_calls, llm_content
            
            except Exception as e:
                self.logger.error(f"LLM 도구 선택 중 오류: {e}")
                error_response = {"role": "system", "content": f"🚨 LLM 도구 선택 중 오류 발생: {e}"}
                return error_response, None, None
        

    async def generate_llm_summary(self, content: str, workspace: Dict) -> str:
        """
        LLM을 사용하여 응답을 요약하거나 보완합니다.
        """
        client = get_openai_client(async_client=True)
        current_artifacts_summary = self.summarize_artifact(workspace.get("artifacts", {}))
        has_retrieved_data = bool(workspace.get("artifacts", {}).get("retrieved_data"))
        system_message_content = SYSTEM_PROMPT.format(
            artifacts_summary=current_artifacts_summary,
            has_retrieved_data=str(has_retrieved_data),
            last_request_type=workspace.get("last_request_type", "없음")
        )
        messages = self.prepare_openai_messages(workspace, system_message_content)
        messages.append({"role": "assistant", "content": content})

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=False,
            )
            return response.choices[0].message.content or content
        except Exception as e:
            self.logger.error(f"LLM 요약 생성 중 오류: {e}")
            return content  # 오류 시 원래 콘텐츠 반환

    async def generate_response(self, content: str, workspace: Dict, session_id: str, is_error: bool = False, include_next_step_in_error: bool = False, use_llm_summary: bool = False) -> Tuple[str, Dict]:
        """
        사용자 친화적 응답을 생성하고 워크스페이스를 업데이트합니다.
        성공 응답에는 suggest_next_step을 필수 포함하며, LLM 요약을 선택적으로 수행합니다.
        """
        role = "assistant" if not is_error else "system"
        final_content = content

        # LLM 요약 수행
        if use_llm_summary and not is_error:
            final_content = await self.generate_llm_summary(content, workspace)

        # 성공 응답에는 항상 다음 단계 제안 포함
        if not is_error:
            next_step = self.suggest_next_step(workspace)
            final_content += f"\n\n📌 다음 단계 제안: {next_step}"
        elif include_next_step_in_error and workspace.get("artifacts"):
            next_step = self.suggest_next_step(workspace)
            final_content += f"\n\n📌 다음 단계 제안: {next_step}"

        append_to_history(workspace, {"role": role, "content": final_content})
        workspace["internal_history"] = trim_history(workspace["internal_history"])
        workspace["user_history"] = trim_history(workspace["user_history"])
        save_workspace_to_redis(session_id, workspace)

        return final_content, workspace

    def generate_error_response(self, error_message: str, workspace: Dict, session_id: str, include_next_step: bool = False) -> Tuple[str, Dict]:
        """
        오류 응답을 생성하고 워크스페이스를 업데이트합니다.
        """
        return self.generate_response(f"⚠️ 오류: {error_message}", workspace, session_id, is_error=True, include_next_step_in_error=include_next_step)