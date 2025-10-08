# agents/request_parser.py
import json
from typing import Dict, Tuple, Optional

class RequestParser:
    def __init__(self):
        # json 폼 요청 의도 매핑
        self.valid_request_types = [
            "data_retriever_request",
            "manual_persona_request",
            "manual_service_request",
            "change_product_type_request",
            "chat_message"
        ]
        # 자연어 요청의 의도 매핑
        self.intent_mapping = {
            "데이터 검색": ["검색", "데이터", "VOC"],
            "클러스터링": ["클러스터링", "그룹 분류"],
            "토픽 모델링": (["토픽", "행동 식별"], ["cluster_id"]),
            "기회 점수": ["기회 점수", "우선순위"],
            "고객 행동 맵": (["CAM", "행동 맵"], ["topic_id"]),
            "페르소나": ["페르소나", "고객 정의"],
            "서비스 아이디어": (["서비스 아이디어", "서비스 제안"], ["persona_name"]),
            "데이터 기획안": (["데이터 기획안", "데이터 계획"], ["service_name"]),
            "최종보고서": (["C-D-P", "정의서", "보고서"], ["data_plan_service_name"])
        }

    def parse_request(self, user_message: str) -> Tuple[Dict, str, str]:
        """
        사용자 입력을 파싱하여 요청 딕셔너리, 유형, 자연어 콘텐츠로 반환합니다.
        반환: (parsed_request, request_type, natural_language_content)
        """
        try:
            # json 폼 요청
            message_dict = json.loads(user_message)
            request_type = message_dict.get("type")
            if request_type not in self.valid_request_types:
                return {"error": f"알 수 없는 요청 유형: {request_type}"}, "error", user_message
            
            natural_language_content = self.transform_to_natural_language(user_message, message_dict, request_type)

            #여기서 한번 자연어 파싱해서 params나 initent를 예측하면 저기 llm_selct에 message_dict에 들어갈때 판단하기 좋게 들어감
            if request_type == "chat_message":
                intent, params = self.analyze_nlp_intent(user_message)
                message_dict = {"type": "chat_message", "content": natural_language_content, "intent": intent}
                if params:
                    message_dict.update(params)

            print("json")
            return message_dict, request_type, natural_language_content
        
        ###json으로 답변이 와서 안오지 않음
        except json.JSONDecodeError:
            # 자연어 요청 처리
            intent, params = self.analyze_nlp_intent(user_message)
            message_dict = {"type": "chat_message", "content": user_message, "intent": intent}
            if params:
                message_dict.update(params)
            natural_language_content = message_dict.get("content")
            return message_dict, request_type, natural_language_content

    def transform_to_natural_language(self, user_message: str, message_dict: Dict, request_type: str) -> str:
        """
        JSON 요청을 자연어로 변환하여 히스토리 로그에 저장합니다.
        """
        if request_type == "data_retriever_request":
            keyword = message_dict.get("keyword", "지정되지 않음")
            date_range = message_dict.get("date_range", "지정되지 않음")
            product_type = message_dict.get("product_type", "지정되지 않음")
            return f"{product_type} 제품에 대해 '{keyword}' 키워드로 {date_range} 데이터를 검색해 주세요."
        elif request_type == "manual_persona_request":
            persona_name = message_dict.get("persona_data", {}).get("name", "지정되지 않음")
            return f"'{persona_name}' 페르소나를 생성해 주세요."
        elif request_type == "manual_service_request":
            service_name = message_dict.get("service_data", {}).get("service_name", "지정되지 않음")
            return f"'{service_name}' 서비스 아이디어를 제안해 주세요."
        elif request_type == "change_product_type_request":
            product_type = message_dict.get("product_type", "지정되지 않음")
            return f"제품군을 '{product_type}'로 변경해 주세요."
        return message_dict.get("content")


    def validate_parameters(self, message_dict: Dict, request_type: str) -> Optional[str]:
        """
        JSON 요청의 필수 파라미터를 검증합니다.
        """
        function_mapping = {
            "data_retriever_request": ["keyword"],
            "manual_persona_request": ["persona_data"],
            "manual_service_request": ["service_data"],
            "change_product_type_request": ["product_type"]
        }
        if request_type in function_mapping:
            required_params = function_mapping[request_type]
            missing_params = [k for k in required_params if k not in message_dict or not message_dict[k]]
            if missing_params:
                return f"⚠️ 요청에 필수 파라미터가 누락되었습니다: {', '.join(missing_params)}"
        return None

    def analyze_nlp_intent(self, message: str) -> Tuple[str, Dict]:
        """
        자연어 요청의 의도를 분석하여 의도와 파라미터를 추출합니다.
        반환: (intent, params)
        """
        message = message.lower()
        for intent, mapping in self.intent_mapping.items():
            keywords = mapping[0] if isinstance(mapping, tuple) else mapping
            required_params = mapping[1] if isinstance(mapping, tuple) else []
            if any(keyword in message for keyword in keywords):
                params = {}
                for param in required_params:
                    # 간단한 파라미터 추출 로직 (예: 숫자 또는 이름 추출)
                    if param in ["cluster_id", "topic_id"]:
                        import re
                        match = re.search(r'(\d+)[번|번 클러스터|번 토픽]', message)
                        if match:
                            params[param] = int(match.group(1))
                    elif param in ["persona_name", "service_name", "data_plan_service_name"]:
                        # 이름은 메시지에서 직접 추출 (예시로 단순화)
                        params[param] = message.split("'")[-1] if "'" in message else "지정되지 않음"
                return intent, params
        return "unknown", {}