#---외부라이브러리--
import os
import sys
#--웹 서버와 api 요청/응답 처리 지원--
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware # 1. 이 줄을 추가합니다.

#--데이터 모델 정의
from pydantic import BaseModel
from typing import List, Dict,Tuple

from dotenv import load_dotenv
import uuid
from pydantic import BaseModel

#--내부 모듈 함수
from agents.common.utils import (
    save_workspace_to_redis, load_workspace_from_redis,
    setup_logging, append_to_history
)
from agents.tools import create_new_workspace
from agents.workflow.request_parser import RequestParser
from agents.workflow.tool_invoker import ToolInvoker
from agents.workflow.response_generator import ResponseGenerator

# --- 파일 로드 및 초기화 ---
load_dotenv()
#-----로그 준비------
logger = setup_logging()

#--------------------- FastAPI 앱 객체 설정--------------------------------
app = FastAPI(title="기획자 AI Agent MCP 서버")

# CORS 설정 프론트엔드 통신 허용 기준
origins = [
    "http://localhost:3001", # 프론트엔드 서버의 주소
    "http://localhost:3000",
]

# X-Session-ID 헤더를 통해 세션 ID를 전달하여 워크스페이스 지속성을 유지
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-ID"]
)

#-----클래스 정의-------------------------------------
#사용자 요청 데이터 모델
class UserRequest(BaseModel):
    session_id: str | None = None
    message: str

#서버 응답 데이터 모델
class ChatResponse(BaseModel):
    response_message: str
    workspace: dict
    user_history: list
    artifacts: dict
    error: str | None = None

#internal_history의 메시지를 검증하여 tooll 메시지가 유효한 tool_call_id를 가지는지 확인합니다.---
def validate_messages(messages: List[Dict]) -> List[Dict]:
    """Validate messages to ensure 'tool' messages follow 'tool_calls'."""
    validated_messages = []
    tool_call_ids = set()
    
    for i, msg in enumerate(messages):
        if msg.get('role') == 'tool':
            if not msg.get('tool_call_id'):
                print(f"Warning: Tool message without tool_call_id at index {i}")
                continue  # Skip invalid tool message
            if msg['tool_call_id'] not in tool_call_ids:
                print(f"Warning: Tool message with invalid tool_call_id {msg['tool_call_id']} at index {i}")
                continue
        validated_messages.append(msg)

        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            for tool_call in msg['tool_calls']:
                tool_call_ids.add(tool_call['id'])
    
    return validated_messages


async def run_agent_and_get_response(user_message: str, workspace: dict, session_id: str, use_llm_summary: bool = False) -> Tuple[str, dict]:
    logger = setup_logging()
    parser = RequestParser()
    invoker = ToolInvoker()
    response_generator = ResponseGenerator()

    # 요청 파싱
    message_dict, message_type, natural_language_content = parser.parse_request(user_message)
    # 히스토리 업데이트
    if natural_language_content:
        append_to_history(workspace, {"role": "user", "content": natural_language_content})
    
    # 요청 처리
    if message_type != "chat_message":
        validation_error = parser.validate_parameters(message_dict, message_type)
        if validation_error:
            return response_generator.generate_error_response(validation_error, workspace, session_id, include_next_step=True)
        
        response_to_user, workspace = await invoker.invoke_tool(message_dict, workspace, session_id)
        print(response_to_user)
        return await response_generator.generate_response(response_to_user, workspace, session_id, use_llm_summary=True)
    else:
        response_to_user, workspace = await invoker.invoke_nlp_tool(message_dict, workspace, session_id)
        print(response_to_user)
        return await response_generator.generate_response(response_to_user, workspace, session_id, use_llm_summary=True)




@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(user_request: UserRequest, response: Response):
    print("--- 💬 /chat 엔드포인트 호출됨 ---")
    session_id = user_request.session_id or str(uuid.uuid4())
    logger.info(f"Session ID: {session_id}")

    workspace = load_workspace_from_redis(session_id)
    if not workspace or not isinstance(workspace, dict):
        workspace = create_new_workspace()
        logger.info(f"New workspace initialized for session: {session_id}")

    try:
        #일단 지금은 요약 false로 하고 테스트 중
        use_llm_summary = False
        assistant_response_content, updated_workspace = await run_agent_and_get_response(
            user_message=user_request.message,
            workspace=workspace,
            session_id=session_id,
            use_llm_summary=use_llm_summary
        )
        response.headers["X-Session-ID"] = session_id
        save_workspace_to_redis(session_id, updated_workspace)
        return {
            "response_message": assistant_response_content,
            "workspace": updated_workspace,
            "user_history": updated_workspace.get("user_history", []),
            "artifacts": updated_workspace.get("artifacts", {}),
            "error": None
        }
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        error_message = f"🚨 서버 오류: {str(e)}"
        append_to_history(workspace, {"role": "assistant", "content": error_message})
        save_workspace_to_redis(session_id, workspace)
        return {
            "response_message": error_message,
            "workspace": workspace,
            "user_history": workspace.get("user_history", []),
            "artifacts": workspace.get("artifacts", {}),
            "error": str(e)
        }