// components/chat/ChatInterface.tsx

"use client";

import { useState, FormEvent, useEffect, useRef } from 'react';
import { useWorkspaceStore } from '@/store/workspaceStore';
// [수정] sendMessageToAssistant는 더 이상 여기서 직접 호출하지 않습니다.
// import { sendMessageToAssistant } from '@/lib/api';
import Message from './Message';

// [수정] props 타입을 정의합니다.
interface ChatInterfaceProps {
  isLoading: boolean;
  onSendMessage: (message: string) => Promise<void>;
}

export default function ChatInterface({ isLoading, onSendMessage }: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  // [수정] ChatInterface 내부의 isLoading 상태를 제거합니다.
  // const [isLoading, setIsLoading] = useState(false); 
  const { history } = useWorkspaceStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  const handleSubmit = async (e: FormEvent) => {
    console.log("handleSubmit 함수가 호출되었습니다!");
    
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input;
    setInput('');
    
    // [수정] 자체 API 호출 로직 대신 부모로부터 받은 onSendMessage 함수를 호출합니다.
    await onSendMessage(userMessage);
  };

  return (
    <div className="flex-1 flex flex-col h-full">
      <div className="flex-1 p-6 overflow-y-auto">
        {history.length === 0 ? (
          <p className="text-center text-gray-500">AI 어시스턴트와 대화를 시작하여 전략 기획을 시작하세요.</p>
        ) : (
          history
            .filter(msg => msg.role === 'user' || msg.role === 'assistant')
            .map((msg, index) => {
              if (msg.role === 'assistant' && msg.tool_calls) {
                return null;
              }
              return <Message key={index} role={msg.role} content={msg.content || ""} />
            })
        )}
        
        {/* [수정] isLoading을 props에서 직접 사용합니다. */}
        {isLoading && (
          <div className="flex justify-start">
             <div className="p-3 rounded-lg bg-gray-200 dark:bg-gray-700 text-sm flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                <span>AI가 생각 중입니다...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="border-t border-gray-200 dark:border-gray-700 p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={isLoading ? "응답을 기다리는 중..." : "예: '스타일러 살균 기능에 대한 소비자 반응 찾아줘'"}
            disabled={isLoading} // props에서 받은 isLoading 사용
            className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          />
          <button 
            type="submit"
            disabled={isLoading} // props에서 받은 isLoading 사용
            className="bg-blue-600 text-white font-semibold px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed"
          >
            전송
          </button>
        </form>
      </div>
    </div>
  );
}