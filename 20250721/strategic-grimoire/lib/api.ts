import { useWorkspaceStore } from '@/store/workspaceStore';

const API_URL = 'http://localhost:8000/chat';

export async function sendMessageToAssistant(message: string): Promise<any> {
  const { sessionId, setSessionId } = useWorkspaceStore.getState();
  const effectiveSessionId = sessionId || crypto.randomUUID();
  if (!sessionId) {
    setSessionId(effectiveSessionId);
  }

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': effectiveSessionId,
      },
      body: JSON.stringify({
        session_id: effectiveSessionId,
        message,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || errorData.detail || '서버에서 오류가 발생했습니다.');
    }

    const data = await response.json();
    const newSessionId = response.headers.get('X-Session-ID');
    if (newSessionId && newSessionId !== effectiveSessionId) {
      setSessionId(newSessionId);
    }
    return data;
  } catch (error) {
    console.error('API 통신 중 오류 발생:', error);
    const errorMessage = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
    return { error: errorMessage, user_history: [], session_id: effectiveSessionId };
  }
}