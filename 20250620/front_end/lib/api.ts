// lib/api.ts

import { useWorkspaceStore } from '@/store/workspaceStore';

const API_URL = 'http://localhost:8000/chat';

/**
 * [수정] 이 함수는 이제 상태를 직접 변경하지 않고, 서버로부터 받은 데이터 전체를 반환만 합니다.
 * @param message 서버로 보낼 JSON 형식의 메시지
 * @returns 서버로부터 받은 전체 응답 데이터 (e.g., { workspace: ..., response_message: ... })
 */
export async function sendMessageToAssistant(message: string): Promise<any> { // 반환 타입을 any 또는 구체적인 타입으로 변경
  const { sessionId } = useWorkspaceStore.getState();
  
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: sessionId,
        message: message,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || '서버에서 오류가 발생했습니다.');
    }

    // [수정] 상태를 직접 변경하는 setWorkspace, setSessionId 등을 모두 제거하고
    // 서버에서 받은 data를 그대로 반환합니다.
    const data = await response.json();
    return data;

  } catch (error) {
    console.error('API 통신 중 오류 발생:', error);
    // 오류가 발생했을 때도 일관된 처리를 위해 객체 형태로 반환할 수 있습니다.
    const errorMessage = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
    return { error: errorMessage };
  }
}