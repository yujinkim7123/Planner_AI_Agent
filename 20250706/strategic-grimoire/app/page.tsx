// app/page.tsx
"use client";
import { useState, useEffect } from 'react';
import { useWorkspaceStore } from '@/store/workspaceStore';
import { sendMessageToAssistant } from '@/lib/api';
import { initialState } from '@/store/workspaceStore';
import Script from 'next/script';
import ChatInterface from "@/components/chat/ChatInterface";
import ArtifactRenderer from "@/components/layout/ArtifactRenderer";
import LeftPanel from '@/components/layout/LeftPanel';
import SearchForm from '@/components/chat/SearchForm';
import PersonaInputForm, { FinalPersonaData } from '@/components/chat/PersonaInputForm';
import ServiceIdeaInputForm, { ServiceIdeaFormData } from '@/components/chat/ServiceIdeaInputForm';

type ModalView = 'none' | 'search' | 'persona_form' | 'service_idea_form';

export default function Home() {
  const { history, addMessageToHistory, setWorkspace, setSessionId, sessionId } = useWorkspaceStore();
  const [isLoading, setIsLoading] = useState(false);
  const [modalView, setModalView] = useState<ModalView>('none');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) {
      setSessionId(crypto.randomUUID());
    }
    if (history.length === 0) {
      setModalView('search');
    }
  }, [history.length, sessionId, setSessionId]);

  const handleRequest = async (apiMessage: string, userDisplayMessage?: string) => {
    if (userDisplayMessage) {
      addMessageToHistory({ role: 'user', content: userDisplayMessage });
    }
    
    setIsLoading(true);
    setError(null);
    setModalView('none');

    const data = await sendMessageToAssistant(apiMessage);
    
    if (data && !data.error) {
      console.log('Updating workspace with user_history:', data.user_history);
      setWorkspace({
        ...data.workspace,
        history: data.user_history || data.workspace.user_history || []
      });
      if (!sessionId && data.session_id) {
        setSessionId(data.session_id);
      }
    } else {
      const errorMessage = data.error || '알 수 없는 오류가 발생했습니다.';
      setError(`오류: ${errorMessage}. 세션을 재설정하려면 페이지를 새로고침하세요.`);
      addMessageToHistory({ role: 'assistant', content: `오류가 발생했습니다: ${errorMessage}` });
    }

    setIsLoading(false);
  };
  
  const handleSkipSearch = () => {
    setModalView('none');
    addMessageToHistory({
      role: 'assistant',
      content: "데이터 검색을 건너뛰셨네요! 괜찮습니다. 원하신다면 직접 페르소나나 서비스 아이디어를 생성해서 C-D-P 정의서 작성을 시작해볼까요? \n\n 좌측패널에 생성 버튼을 누르시면 되요."
    });
  };

  const handleSearch = async (searchParams: { keyword: string; dateRange: string; productType: string; }) => {
    const userDisplayMessage = `'${searchParams.keyword}'에 대한 데이터 검색을 요청합니다.`;
    const apiMessage = JSON.stringify({
      type: "data_retriever_request",
      keyword: searchParams.keyword,
      date_range: searchParams.dateRange,
      product_type: searchParams.productType || '지정 안함'
    });
    await handleRequest(apiMessage, userDisplayMessage);
  };

  const handleChatSubmit = async (chatText: string) => {
    const apiMessage = JSON.stringify({
      type: "chat_message",
      content: chatText,
    });
    await handleRequest(apiMessage, chatText);
  };

  const handlePersonaSubmit = async (personaData: FinalPersonaData) => {
    const userDisplayMessage = `새로운 페르소나 '${personaData.name}' 생성을 요청합니다.`;
    const apiMessage = JSON.stringify({
      type: "manual_persona_request",
      persona_data: personaData,
    });
    await handleRequest(apiMessage, userDisplayMessage);
  };

  const handleServiceIdeaSubmit = async (ideaData: ServiceIdeaFormData) => {
    const userDisplayMessage = `새로운 서비스 아이디어 '${ideaData.service_name}' 생성을 요청합니다.`;
    const apiMessage = JSON.stringify({
      type: "manual_service_request",
      service_data: ideaData,
    });
    await handleRequest(apiMessage, userDisplayMessage);
  };

  const handleProductChange = async (productType: string) => {
    const userDisplayMessage = `분석 대상 제품을 '${productType}'(으)로 변경합니다.`;
    const apiMessage = JSON.stringify({
      type: "change_product_type_request",
      product_type: productType
    });
    await handleRequest(apiMessage, userDisplayMessage);
  };

  const renderModal = () => {
    const modalBackdrop = (
      <div 
        className="absolute inset-0 bg-gray-900 bg-opacity-60 flex items-center justify-center z-50"
        onClick={() => {
          if (modalView === 'search') {
            handleSkipSearch();
          } else {
            setModalView('none');
          }
        }}
      >
        <div onClick={(e) => e.stopPropagation()}>
          {
            {
              'search': <SearchForm onSearch={handleSearch} isLoading={isLoading} onSkip={handleSkipSearch} />,
              'persona_form': <PersonaInputForm onSubmit={handlePersonaSubmit} onBack={() => setModalView('none')} isLoading={isLoading} />,
              'service_idea_form': <ServiceIdeaInputForm onSubmit={handleServiceIdeaSubmit} onBack={() => setModalView('none')} isLoading={isLoading} />,
              'none': null
            }[modalView]
          }
        </div>
      </div>
    );
    return modalView !== 'none' ? modalBackdrop : null;
  };

  return (
    <>
    {error && (
        <div className="fixed top-4 right-4 bg-red-500 text-white p-4 rounded">
          {error}
          <button
            onClick={() => {
              setSessionId(crypto.randomUUID());
              setWorkspace(initialState);
              setError(null);
              setModalView('search');
            }}
            className="ml-4 bg-white text-red-500 px-2 py-1 rounded"
          >
            세션 재설정
          </button>
        </div>
      )}
      <Script 
        src="https://cdn.jsdelivr.net/npm/pptxgenjs@3.12.0/dist/pptxgen.bundle.js" 
        strategy="afterInteractive"
      />
      <main className="flex h-screen w-full overflow-hidden bg-white dark:bg-gray-800">
        <aside className="w-1/5 flex-shrink-0 border-r dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 p-4 overflow-y-auto">
          <LeftPanel onSetModal={setModalView} onProductChange={handleProductChange} />
        </aside>
        <section className="flex-1 flex flex-col">
          <ChatInterface isLoading={isLoading} onSendMessage={handleChatSubmit} />
        </section>
        <aside className="w-2/5 flex-shrink-0 border-l dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 p-4 overflow-y-auto">
          <ArtifactRenderer />
        </aside>
      </main>
      {renderModal()}
    </>
  );
}