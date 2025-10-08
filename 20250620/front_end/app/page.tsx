// app/page.tsx

"use client";

import { useState, useEffect } from 'react';
import { useWorkspaceStore } from '@/store/workspaceStore';
import { sendMessageToAssistant } from '@/lib/api';

import ChatInterface from "@/components/chat/ChatInterface";
import ArtifactRenderer from "@/components/layout/ArtifactRenderer";
import LeftPanel from '@/components/layout/LeftPanel';
import SearchForm from '@/components/chat/SearchForm';
import PersonaInputForm, { FinalPersonaData } from '@/components/chat/PersonaInputForm';
import ServiceIdeaInputForm, { ServiceIdeaFormData } from '@/components/chat/ServiceIdeaInputForm';

type ModalView = 'none' | 'search' | 'persona_form' | 'service_idea_form';

export default function Home() {
  // [수정] 상태 변경에 필요한 모든 함수를 store에서 가져옵니다.
  const { history, addMessageToHistory, setWorkspace, setSessionId, sessionId } = useWorkspaceStore();
  const [isLoading, setIsLoading] = useState(false);
  const [modalView, setModalView] = useState<ModalView>('none');

  useEffect(() => {
    if (history.length === 0) {
      setModalView('search');
    }
  }, [history.length]);

  /**
   * [수정] 모든 API 호출과 상태 업데이트를 처리하는 중앙 함수
   * @param apiMessage 서버로 보낼 JSON 메시지
   * @param userDisplayMessage 사용자가 볼 메시지 (선택 사항)
   */
  const handleRequest = async (apiMessage: string, userDisplayMessage?: string) => {
    // 1. 사용자에게 보여줄 메시지가 있다면, 즉시 대화 기록에 추가 (낙관적 업데이트)
    if (userDisplayMessage) {
      addMessageToHistory({ role: 'user', content: userDisplayMessage });
    }
    
    setIsLoading(true);
    setModalView('none');

    // 2. API를 호출하고 서버로부터 전체 데이터를 받습니다.
    const data = await sendMessageToAssistant(apiMessage);
    
    // 3. 받은 데이터를 사용하여 상태를 '안전하게' 업데이트합니다.
    if (data && !data.error) {

      setWorkspace(data.workspace);
      
      if (!sessionId && data.session_id) {
        setSessionId(data.session_id);
      }
    } else {
      // API 오류 발생 시 사용자에게 알려줍니다.
      addMessageToHistory({ role: 'assistant', content: `오류가 발생했습니다: ${data.error}` });
    }

    setIsLoading(false);
  };

  const handleSkipSearch = () => {
    setModalView('none'); // 모달을 닫습니다.
    addMessageToHistory({
      role: 'assistant',
      content: "데이터 검색을 건너뛰셨네요! 괜찮습니다. 원하신다면 직접 페르소나나 서비스 아이디어를 생성해서 C-D-P 정의서 작성을 시작해볼까요? \n\n 좌측패널에 생성 버튼을 누르시면 되요."
    });
  };
  
  // 모든 핸들러가 새로운 `handleRequest` 함수를 사용하도록 수정합니다.
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
    // 채팅은 handleRequest의 userDisplayMessage 인자를 통해 낙관적 업데이트를 수행합니다.
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

  // ... 나머지 핸들러들도 위와 같은 패턴으로 수정 ...
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