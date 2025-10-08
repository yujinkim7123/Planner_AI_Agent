// components/layout/LeftPanel.tsx
"use client";

import { useWorkspaceStore } from "@/store/workspaceStore";
import { PlusCircleIcon, UserCircleIcon, LightBulbIcon, DocumentTextIcon, ClipboardDocumentListIcon, TagIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';

const PRODUCT_KEYWORDS = ["(선택 안함)", "스타일러", "트롬", "휘센", "퓨리케어", "디오스", "그램", "올레드", "코드제로", "틔운", "시네빔", "울트라기어"];

type ModalView = 'none' | 'search' | 'persona_form' | 'service_idea_form';

interface LeftPanelProps {
  onSetModal: (modal: ModalView) => void;
  onProductChange: (productType: string) => void; // 제품군 변경 핸들러
}

export default function LeftPanel({ onSetModal, onProductChange }: LeftPanelProps) {
  const { artifacts } = useWorkspaceStore();

  const selectedPersona = artifacts.selected_persona;
  const selectedServiceIdea = artifacts.selected_service_idea;
  const selectedDataPlan = artifacts.selected_data_plan_for_service?.data_plan || artifacts.selected_data_plan_for_service;
  const selectedCdpDefinition = artifacts.selected_cdp_definition;

  const StatusItem = ({ icon, label, value }: { icon: React.ReactNode, label: string, value: string | null | undefined }) => {
    if (!value) return null;
    return (
      <div className="flex items-start text-xs text-gray-600 dark:text-gray-300 mt-2">
        <div className="w-4 h-4 mr-2 flex-shrink-0">{icon}</div>
        <div>
          <span className="font-semibold">{label}:</span>
          <span className="ml-1 italic text-blue-600 dark:text-blue-400">{value}</span>
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full">
      <div className="font-bold text-lg mb-4 text-gray-800 dark:text-gray-200">🔮 전략 마법서</div>
      
      {/* --- 제품군 변경 --- */}
      <div className="mb-6">
        <label htmlFor="productTypeGlobal" className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
          제품군 컨텍스트
        </label>
        <select
          id="productTypeGlobal"
          value={artifacts.product_type || "(선택 안함)"}
          onChange={(e) => onProductChange(e.target.value)}
          className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          {PRODUCT_KEYWORDS.map(p => <option key={p} value={p}>{p}</option>)}
        </select>
        <p className="text-xs text-gray-400 mt-1">변경 시 작업 환경이 초기화됩니다.</p>
      </div>

      {/* --- 바로가기 생성 버튼 --- */}
      <div className="space-y-2">
        {/* ▼▼▼▼▼ 이 버튼 코드를 여기에 추가하세요 ▼▼▼▼▼ */}
        <button
          onClick={() => onSetModal('search')}
          className="w-full flex items-center justify-center px-3 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          <MagnifyingGlassIcon className="w-5 h-5 mr-2" />
          데이터 다시 검색
        </button>
        {/* ▲▲▲▲▲ 여기까지 추가 ▲▲▲▲▲ */}

        <button
          onClick={() => onSetModal('persona_form')}
          className="w-full flex items-center justify-center px-3 py-2 text-sm font-medium text-white bg-green-600 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
        >
          <PlusCircleIcon className="w-5 h-5 mr-2" />
          페르소나 만들기
        </button>
        <button
          onClick={() => onSetModal('service_idea_form')}
          className="w-full flex items-center justify-center px-3 py-2 text-sm font-medium text-white bg-purple-600 rounded-md hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500"
        >
          <PlusCircleIcon className="w-5 h-5 mr-2" />
          서비스 아이디어 만들기
        </button>
      </div>

      {/* --- 현재 선택된 항목 --- */}
      <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-700">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">현재 작업 컨텍스트</h3>
        <div className="mt-3 space-y-2">
            <StatusItem icon={<TagIcon />} label="선택된 제품군" value={artifacts.product_type} />
            <StatusItem icon={<UserCircleIcon />} label="선택된 페르소나" value={selectedPersona?.title} />
            <StatusItem icon={<LightBulbIcon />} label="선택된 서비스" value={selectedServiceIdea?.service_name} />
            <StatusItem icon={<ClipboardDocumentListIcon />} label="선택된 데이터 기획안" value={selectedDataPlan?.service_name} />
            <StatusItem icon={<DocumentTextIcon />} label="선택된 C-D-P" value={selectedCdpDefinition?.title.replace('유첨. ', '').replace(' C-D-P 정의서', '')} />
        </div>
      </div>
    </div>
  );
}