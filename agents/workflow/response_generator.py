# agents/response_generator.py
from typing import Dict, Tuple,Optional,List, Any
from ..common.utils import setup_logging,get_openai_client,save_workspace_to_redis,trim_history,append_to_history,MODEL_NAME
from ..tools import SYSTEM_PROMPT,tools,SYSTEM_PROMPT2
import json


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
            # 데이터 수집 (1단계)
            {
                "condition": lambda a: last_request_type is None or last_request_type == "VOC_수집_실패", # 초기 상태 또는 데이터 수집 실패
                "suggestion": "먼저 VOC 데이터를 검색해 주세요. 예: {'type': 'data_retriever_request', 'keyword': '살균', 'date_range': '최근 1년', 'product_type': '스타일러'}"
            },
            {
                "condition": lambda a: last_request_type == "run_data_retriever" and a.get("retrieved_data"),
                "suggestion": "고객 그룹을 분류하기 위해 워드 클러스터링을 수행해 보세요. 예: {'type': 'chat_message', 'content': '클러스터링 해줘'}"
            },
            # 클러스터링 (2단계)
            {
                "condition": lambda a: last_request_type == "run_ward_clustering" and a.get("cx_ward_clustering_results"),
                "suggestion": "고객 행동을 식별하기 위해 토픽 모델링 또는 SNA 분석을 수행해 보세요. 예: {'type': 'chat_message', 'content': '0번 클러스터에 대해 토픽 모델링 해줘'} 또는 {'type': 'chat_message', 'content': '0번 클러스터에 대해 SNA 분석해줘'}"
            },
            # SNA 분석 (3단계)
            {
                "condition": lambda a: last_request_type == "run_semantic_network_analysis" and a.get("cx_sna_results"),
                "suggestion": "클러스터의 세부 주제를 파악하기 위해 토픽 모델링을 수행해 보세요. 예: {'type': 'chat_message', 'content': '0번 클러스터에 대해 토픽 모델링 해줘'}"
            },
            # 토픽 모델링 (4단계)
            {
                "condition": lambda a: last_request_type == "run_topic_modeling_lda" and a.get("cx_lda_results"),
                "suggestion": "사업 기회 우선순위를 정하기 위해 기회 점수를 계산해 보세요. 예: {'type': 'chat_message', 'content': '기회 점수 계산해줘'}"
            },
            # 기회 점수 계산 (5단계)
            {
                "condition": lambda a: last_request_type == "calculate_opportunity_scores" and a.get("cx_opportunity_scores"),
                "suggestion": "고객의 목표와 Pain Point를 분석하기 위해 고객 행동 맵(CAM)을 생성해 보세요. 예: {'type': 'chat_message', 'content': '0-1 토픽에 대해 CAM 생성해줘'}"
            },
            # 고객 행동 맵 (6단계)
            {
                "condition": lambda a: last_request_type == "create_customer_action_map" and a.get("cx_cam_results"),
                "suggestion": "고객 인사이트를 기반으로 페르소나를 생성해 보세요. 예: {'type': 'chat_message', 'content': '페르소나 만들어줘'}"
            },
            # 페르소나 생성 (7단계)
            {
                "condition": lambda a: (last_request_type == "create_personas" or last_request_type == "create_persona_from_manual_input") and a.get("cx_personas") and a.get("cx_personas").get("personas"),
                "suggestion": "페르소나를 기반으로 서비스 아이디어를 제안해 보세요. 예: {'type': 'chat_message', 'content': '서비스 아이디어 제안해줘'}"
            },
            # 서비스 아이디어 제안 (8단계)
            {
                "condition": lambda a: (last_request_type == "create_service_ideas" or last_request_type == "create_service_ideas_from_manual_input") and a.get("cx_service_ideas") and a.get("cx_service_ideas").get("ideas"),
                "suggestion": "서비스 아이디어를 기반으로 데이터 기획안을 작성해 보세요. 예: {'type': 'chat_message', 'content': '데이터 기획안 만들어줘'}"
            },
            # 데이터 기획 (9단계)
            {
                "condition": lambda a: last_request_type == "create_data_plan_for_service" and a.get("cx_data_plan"),
                "suggestion": "최종 보고서를 작성해 보세요. 예: {'type': 'chat_message', 'content': '최종 보고서 작성해줘'}"
            },
            # 최종 보고서 (10단계)
            {
                "condition": lambda a: last_request_type == "create_cdp_definition" and a.get("cdp_definition"),
                "suggestion": "모든 단계를 완료했습니다. 추가 분석이나 수정이 필요하시면 요청해 주세요!"
            },
            # 기타 예외 처리 또는 일반적인 다음 단계
            {
                "condition": lambda a: True, # 모든 조건에 해당하지 않을 경우
                "suggestion": "현재 워크스페이스 상태를 확인하여 어떤 도움이 필요하신지 알려주세요."
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

    def generate_system_prompt_context(self, workspace: Dict[str, Any]) -> str:
        """
        현재 workspace 상태를 기반으로 LLM 시스템 프롬프트에 포함될 추가 컨텍스트를 생성합니다.
        last_request_type을 기반으로 각 단계별 필요한 정보를 선별하여 제공합니다.
        """
        context_parts = []

        # 1. 공통 정보 (모든 단계에서 유용할 수 있는 정보)
        last_request_type = workspace.get("last_request_type", "없음")
        context_parts.append(f"마지막 요청 유형: **{last_request_type}**")

        artifacts = workspace.get("artifacts", {})

        # last_request_type에 따라 필요한 컨텍스트만 선별하여 추가
        # tool_invoker.py에 정의된 function_name 또는 message_type을 기준으로 합니다.
        if last_request_type == "run_data_retriever":
            if "retrieved_data" in artifacts and artifacts["retrieved_data"]:
                if isinstance(artifacts["retrieved_data"].get("web_results"), list):
                    num_voc = len(artifacts["retrieved_data"]["web_results"])
                    context_parts.append(f"\n현재 총 **{num_voc}건의 고객 VOC 데이터**가 준비되었습니다.")

        elif last_request_type == "run_ward_clustering":
            if "cx_ward_clustering_results" in artifacts and artifacts["cx_ward_clustering_results"]:
                if isinstance(artifacts["cx_ward_clustering_results"].get("cluster_summaries"), dict):
                    clusters = artifacts["cx_ward_clustering_results"]["cluster_summaries"]
                    context_parts.append("\n**[현재 고객 그룹 분석 결과]**")
                    context_parts.append(f"총 **{len(clusters)}개의 고객 그룹**이 발견되었습니다.")
                    for cluster_id, summary in clusters.items():
                        context_parts.append(f"- 그룹 {cluster_id}: {summary.get('description', '요약 없음')}")

        elif last_request_type == "run_semantic_network_analysis":
            if "cx_sna_results" in artifacts and artifacts["cx_sna_results"]:
                sna_results = artifacts["cx_sna_results"]
                cluster_id = sna_results.get("cluster_id")
                micro_segments = sna_results.get("micro_segments", []) if isinstance(sna_results.get("micro_segments"), list) else []
                analysis_desc = sna_results.get("analysis_description", "의미 연결망 분석 완료.")

                context_parts.append(f"\n**[클러스터 {cluster_id}에 대한 의미 연결망 분석 (SNA) 결과]**")
                context_parts.append(f"분석 요약: {analysis_desc}")
                if micro_segments:
                    context_parts.append("핵심 마이크로 세그먼트 (커뮤니티) 목록:")
                    for segment in micro_segments:
                        core_keyword = segment.get("core_keyword", "N/A")
                        keywords = ", ".join(segment.get("keywords", [])[:5]) if isinstance(segment.get("keywords"), list) else ""
                        context_parts.append(f"- 핵심 키워드: '{core_keyword}' (관련 키워드: {keywords}...)")
                else:
                    context_parts.append("발견된 마이크로 세그먼트가 없습니다.")
                if sna_results.get("graph_data"):
                    context_parts.append("의미 연결망 시각화를 위한 그래프 데이터가 생성되었습니다.")
        
        elif last_request_type == "run_topic_modeling_lda":
            if "cx_lda_results" in artifacts and artifacts["cx_lda_results"]:
                topics = artifacts["cx_lda_results"].get("topics_summary_list") # topics_summary_list 사용 유지
                if isinstance(topics, list):
                    lda_cluster_id = artifacts["cx_lda_results"].get("cluster_id", "알 수 없음")
                    context_parts.append(f"\n**[클러스터 {lda_cluster_id}의 토픽 모델링 (LDA) 결과]**")
                    context_parts.append(f"총 **{len(topics)}개의 핵심 주제**가 발견되었습니다.")
                    for i, topic in enumerate(topics):
                        keywords = ", ".join(topic.get("action_keywords", [])) if isinstance(topic.get("action_keywords"), list) else ""
                        context_parts.append(f"- 토픽 {topic.get('topic_id', i)}: {topic.get('description', '요약 없음')} (핵심 키워드: {keywords})")

        elif last_request_type == "calculate_opportunity_scores":
            if "cx_opportunity_scores" in artifacts and artifacts["cx_opportunity_scores"]:
                opportunity_scores = artifacts["cx_opportunity_scores"]
                context_parts.append("\n**[고객 사업 기회 점수 데이터]**")
                context_parts.append(json.dumps(opportunity_scores, indent=2, ensure_ascii=False))
                context_parts.append("\n이 데이터는 중요도, 만족도, 기회 점수를 포함하며, 기회 점수가 높을수록 집중해야 할 사업 기회가 큽니다.")

        elif last_request_type == "create_customer_action_map":
            if "cx_cam_results" in artifacts and isinstance(artifacts["cx_cam_results"], list) and artifacts["cx_cam_results"]:
                latest_cam = artifacts["cx_cam_results"][-1]
                context_parts.append(f"\n**[최신 고객 액션맵 (CAM) 결과 - '{latest_cam.get('action_name', '미정')}' 토픽]**")
                context_parts.append(f"  - 목표 (Goal): {', '.join(latest_cam.get('goals', []) if isinstance(latest_cam.get('goals'), list) else [])}")
                context_parts.append(f"  - 불편함 (Pain Point): {', '.join(latest_cam.get('pain_points', []) if isinstance(latest_cam.get('pain_points'), list) else [])}")
                context_parts.append(f"  - 상황 (Context): {', '.join(latest_cam.get('context', []) if isinstance(latest_cam.get('context'), list) else [])}")
                context_parts.append(f"  - 관련 사물/서비스 (Touchpoint/Artifact): {', '.join(latest_cam.get('touchpoint_artifact', []) if isinstance(latest_cam.get('touchpoint_artifact'), list) else [])}")
        
        elif last_request_type == "create_personas" or last_request_type == "create_persona_from_manual_input":
            if "cx_personas" in artifacts and artifacts["cx_personas"]:
                personas = artifacts["cx_personas"].get("personas")
                if isinstance(personas, list):
                    context_parts.append("\n**[정의된 고객 페르소나]**")
                    for i, persona in enumerate(personas):
                        context_parts.append(f"- 이름: {persona.get('name', '미정')}")
                        context_parts.append(f"  특징: {persona.get('description', '설명 없음')}")
                        context_parts.append(f"  주요 문제: {', '.join(persona.get('pain_points', []) if isinstance(persona.get('pain_points'), list) else [])}")

        elif last_request_type == "create_service_ideas" or last_request_type == "create_service_ideas_from_manual_input":
            if "cx_service_ideas" in artifacts and artifacts["cx_service_ideas"]:
                service_ideas = artifacts["cx_service_ideas"].get("ideas")
                if isinstance(service_ideas, list):
                    context_parts.append("\n**[제안된 서비스 아이디어]**")
                    for i, idea in enumerate(service_ideas):
                        context_parts.append(f"- {i+1}. {idea.get('name', '이름 없음')}: {idea.get('description', '설명 없음')}")

        elif last_request_type == "create_data_plan_for_service":
            if "cx_data_plan" in artifacts and artifacts["cx_data_plan"]:
                data_plan = artifacts["cx_data_plan"].get("plan_details")
                if isinstance(data_plan, dict):
                    context_parts.append("\n**[서비스 구현을 위한 데이터 계획]**")
                    context_parts.append(f"목표 서비스: {data_plan.get('service_name', '없음')}")
                    context_parts.append(f"필요 데이터: {', '.join(data_plan.get('required_data_types', []) if isinstance(data_plan.get('required_data_types'), list) else [])}")
                    context_parts.append(f"수집/활용 방안: {data_plan.get('collection_utilization_strategy', '없음')}")
        
        elif last_request_type == "create_cdp_definition":
            if "cdp_definition" in artifacts and artifacts["cdp_definition"]:
                # cdp_definition은 리스트일 수 있으므로 마지막 항목 참조 또는 전체 요약
                context_parts.append("\n**[최종 분석 보고서가 생성되었습니다.]**")
                # 필요에 따라 cdp_definition 내용 중 일부를 요약하여 추가

        return "\n".join(context_parts)



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
                summary_parts.append(f"- 최종보고서: {len(value)}개 생성됨")
            elif key == "sensor_data" and value:
                summary_parts.append(f"- 센서 데이터: {len(value)}건")
            elif key == "product_data" and value:
                summary_parts.append(f"- 제품 데이터: {len(value)}건")
            elif key == "columns_product" and value:
                summary_parts.append(f"- 제품 메타데이터: {len(value)}개 필드")
            elif key == "data_plan_recommendation_message" and value:
                summary_parts.append(f"- 데이터 기획 추천 메시지: 저장됨")
            elif key == "selected_cdp_definition" and value:
                summary_parts.append(f"- 현재 선택된 최종 보고서: 저장됨")
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
        extra_system_context = self.generate_system_prompt_context(workspace)
        print(extra_system_context)

        client = get_openai_client(async_client=True)
        current_artifacts_summary = self.summarize_artifact(workspace.get("artifacts", {}))
        has_retrieved_data = bool(workspace.get("artifacts", {}).get("retrieved_data"))
        system_message_content = SYSTEM_PROMPT2.format(
            artifacts_summary=current_artifacts_summary,
            has_retrieved_data=str(has_retrieved_data),
            last_request_type=workspace.get("last_request_type", "없음"),
            extra_context=extra_system_context
        )
        messages = self.prepare_openai_messages(workspace, system_message_content)
        messages.append({"role": "assistant", "content": content})

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False,
            )
            print(response)
            return response.choices[0].message.content
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

        print("afsdfhaskdfjasd")
        print(final_content)
        # 성공 응답에는 항상 다음 단계 제안 포함
        # if not is_error:
        #     next_step = self.suggest_next_step(workspace)
        #     final_content += f"\n\n📌 다음 단계 제안: {next_step}"
        # elif include_next_step_in_error and workspace.get("artifacts"):
        #     next_step = self.suggest_next_step(workspace)
        #     final_content += f"\n\n📌 다음 단계 제안: {next_step}"

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