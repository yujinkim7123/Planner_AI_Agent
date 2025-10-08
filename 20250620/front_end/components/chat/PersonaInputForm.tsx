// components/chat/PersonaInputForm.tsx
"use client";

import { useState } from 'react';

export interface PersonaFormData {
  name: string;
  title:string;
  demographics: string;
  key_behaviors: string; // ✅ string 타입 유지
  needs_and_goals: string; // ✅ string 타입 유지
  pain_points: string; // ✅ string 타입 유지
  motivating_quote: string;
}

// 2. 폼이 '제출'할 최종 데이터의 타입 (배열 사용)
// 이 타입은 PersonaCard가 기대하는 데이터 형태와 일치합니다.
export interface FinalPersonaData {
  name: string;
  title: string;
  demographics: string;
  key_behaviors: string[]; // ✅ string[] 타입
  needs_and_goals: string[]; // ✅ string[] 타입
  pain_points: string[]; // ✅ string[] 타입
  motivating_quote: string;
}

// 3. Props 타입은 '제출'될 타입을 기준으로 설정
interface PersonaInputFormProps {
  onSubmit: (personaData: FinalPersonaData) =>  Promise<void>; // ✅ FinalPersonaData 사용
  onBack: () => void;
  isLoading: boolean;
}

export default function PersonaInputForm({ onSubmit, onBack, isLoading }: PersonaInputFormProps) {
  const [formData, setFormData] = useState<PersonaFormData>({
    name: '', title: '', demographics: '', key_behaviors: '',
    needs_and_goals: '', pain_points: '', motivating_quote: ''
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { id, value } = e.target;
    setFormData(prev => ({ ...prev, [id]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
   e.preventDefault();
    if (isLoading) return;

    // ❗️ 여기서 데이터를 변환하는 '다리' 역할이 필요합니다.
    const processedData = {
      ...formData, // 우선 기존 formData를 모두 복사하고
      // 문자열을 배열로 변환할 필드들만 덮어씁니다.
      key_behaviors: formData.key_behaviors.split('\n').filter(line => line.trim() !== ''),
      needs_and_goals: formData.needs_and_goals.split('\n').filter(line => line.trim() !== ''),
      pain_points: formData.pain_points.split('\n').filter(line => line.trim() !== ''),
    };


    onSubmit(processedData);

  };

  return (
    <div className="flex flex-col items-center justify-center h-full bg-gray-50 dark:bg-gray-900 p-8">
      <div className="w-full max-w-2xl">
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold mb-2">💡 페르소나 직접 설정</h1>
          <p className="text-md text-gray-500 dark:text-gray-400">서비스를 사용하게 될 가상의 고객을 설정해주세요.</p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4 p-6 border rounded-lg bg-white dark:bg-gray-800 shadow-xl">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label htmlFor="name" className="block text-sm font-medium mb-1">이름</label>
              <input type="text" id="name" value={formData.name} onChange={handleChange} placeholder="예: 박서준 (가명)" required className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"/>
            </div>
            <div>
              <label htmlFor="title" className="block text-sm font-medium mb-1">한 줄 요약</label>
              <input type="text" id="title" value={formData.title} onChange={handleChange} placeholder="예: 꼼꼼한 위생관리맘" required className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"/>
            </div>
             <div>
              <label htmlFor="demographics" className="block text-sm font-medium mb-1">인구통계 정보</label>
              <input type="text" id="demographics" value={formData.demographics} onChange={handleChange} placeholder="예: 30대 후반, 맞벌이, 7세 아이 엄마" required className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"/>
            </div>
          </div>
          <div>
            <label htmlFor="pain_points" className="block text-sm font-medium mb-1">핵심 불편함 (Pain Points)</label>
            <textarea id="pain_points" value={formData.pain_points} onChange={handleChange} placeholder="가장 큰 불편함이나 문제점을 한 줄에 하나씩 입력해주세요." required rows={3} className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"/>
          </div>
           <div>
            <label htmlFor="needs_and_goals" className="block text-sm font-medium mb-1">니즈와 목표 (Needs & Goals)</label>
            <textarea id="needs_and_goals" value={formData.needs_and_goals} onChange={handleChange} placeholder="이 페르소나가 원하는 것이나 목표를 한 줄에 하나씩 입력해주세요." required rows={3} className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"/>
          </div>
          <div>
            <label htmlFor="motivating_quote" className="block text-sm font-medium mb-1">동기부여 문구 (인용문)</label>
            <input type="text" id="motivating_quote" value={formData.motivating_quote} onChange={handleChange} placeholder="예: 아이가 쓰는 건데, 확실한 걸로 사야 마음이 놓여요." required className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"/>
          </div>
          <div className="flex gap-4 pt-4">
            <button type="button" onClick={onBack} className="w-1/3 bg-gray-500 text-white font-bold py-3 rounded-md hover:bg-gray-600">
              ← 뒤로가기
            </button>
            <button type="submit" disabled={isLoading} className="w-2/3 bg-green-600 text-white font-bold py-3 rounded-md hover:bg-green-700 disabled:bg-green-400 flex items-center justify-center gap-2">
              {isLoading && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>}
              {isLoading ? '아이디어 생성 중...' : '이 페르소나로 아이디어 생성'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}