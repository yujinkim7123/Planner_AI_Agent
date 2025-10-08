"use client";

import { useState } from 'react';

// 백엔드 JSON 구조에 맞춘 데이터 인터페이스
export interface ServiceIdeaFormData {
  service_name: string;
  description: string;
  solved_pain_points: string[];
  service_scalability: string;
}

interface ServiceIdeaFormProps {
  onSubmit: (ideaData: ServiceIdeaFormData) => void;
  onBack: () => void;
  isLoading: boolean;
}

export default function ServiceIdeaInputForm({ onSubmit, onBack, isLoading }: ServiceIdeaFormProps) {
  // 새로운 인터페이스에 맞게 초기 상태 설정
  const [formData, setFormData] = useState<ServiceIdeaFormData>({
    service_name: '',
    description: '',
    solved_pain_points: [''], // Pain Point 입력을 위해 초기에 빈 문자열 하나를 추가
    service_scalability: ''
  });

  // 일반적인 input, textarea 변경 핸들러
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { id, value } = e.target;
    setFormData(prev => ({ ...prev, [id]: value }));
  };

  // Pain Point 배열의 특정 인덱스 값 변경 핸들러
  const handlePainPointChange = (index: number, value: string) => {
    const newPainPoints = [...formData.solved_pain_points];
    newPainPoints.splice(index, 1, value); // splice를 사용하여 해당 인덱스의 값을 직접 교체
    setFormData(prev => ({ ...prev, solved_pain_points: newPainPoints }));
  };

  // Pain Point 입력 필드 추가 핸들러
  const addPainPoint = () => {
    setFormData(prev => ({ ...prev, solved_pain_points: [...prev.solved_pain_points, ''] }));
  };

  // Pain Point 입력 필드 삭제 핸들러
  const removePainPoint = (index: number) => {
    // 최소 1개의 입력 필드는 유지
    if (formData.solved_pain_points.length <= 1) return;
    const newPainPoints = formData.solved_pain_points.filter((_, i) => i !== index);
    setFormData(prev => ({ ...prev, solved_pain_points: newPainPoints }));
  };

  // 폼 제출 핸들러
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isLoading) return;

    // 제출 전, 비어있는 Pain Point 항목을 필터링하여 제거
    const cleanedData = {
      ...formData,
      solved_pain_points: formData.solved_pain_points.filter(p => p.trim() !== '')
    };
    
    onSubmit(cleanedData);
  };

  return (
    <div className="flex flex-col items-center justify-center h-full bg-gray-50 dark:bg-gray-900 p-8">
      {/* 가로 너비를 확보하기 위해 max-w-3xl로 수정 */}
      <div className="w-full max-w-3xl"> 
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold mb-2">💡 서비스 아이디어 직접 설정</h1>
          <p className="text-md text-gray-500 dark:text-gray-400">구상 중인 서비스 아이디어를 입력하고 데이터 기획을 시작하세요.</p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-6 p-6 border rounded-lg bg-white dark:bg-gray-800 shadow-xl">
          {/* 그리드 레이아웃을 사용하여 PC/태블릿에서 가로 배치 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="md:col-span-2"> {/* 서비스 이름은 한 줄 전체를 사용하도록 설정 */}
              <label htmlFor="service_name" className="block text-sm font-medium mb-1">서비스 이름</label>
              <input type="text" id="service_name" value={formData.service_name} onChange={handleChange} placeholder="예: AI 육아 위생 컨설턴트" required className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"/>
            </div>
          </div>

          <div>
            <label htmlFor="description" className="block text-sm font-medium mb-1">서비스 상세 설명</label>
            <textarea id="description" value={formData.description} onChange={handleChange} placeholder="이 서비스가 무엇을 하는지, 어떤 문제를 해결하는지 등을 상세히 설명해주세요." required rows={4} className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"/>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">해결하려는 Pain Point</label>
            <div className="space-y-2">
              {formData.solved_pain_points.map((point, index) => (
                <div key={index} className="flex items-center gap-2">
                  <input
                    type="text"
                    value={point}
                    onChange={(e) => handlePainPointChange(index, e.target.value)}
                    placeholder={`예: 살균 기능의 실제 효과를 눈으로 확인할 수 없어 불안함`}
                    className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                  />
                  <button
                    type="button"
                    onClick={() => removePainPoint(index)}
                    disabled={formData.solved_pain_points.length <= 1}
                    className="p-2 bg-red-500 text-white rounded-md hover:bg-red-600 disabled:bg-red-300 disabled:cursor-not-allowed flex-shrink-0"
                  >
                    삭제
                  </button>
                </div>
              ))}
            </div>
            <button type="button" onClick={addPainPoint} className="mt-2 text-sm text-purple-600 hover:text-purple-800 font-semibold">
              + Pain Point 추가하기
            </button>
          </div>

          <div>
            <label htmlFor="service_scalability" className="block text-sm font-medium mb-1">서비스 확장성</label>
            <textarea id="service_scalability" value={formData.service_scalability} onChange={handleChange} placeholder="이 서비스가 향후 어떻게 확장될 수 있는지 설명해주세요. (예: 프리미엄 유료 모델, 신규 제품 개발 연계 등)" required rows={3} className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"/>
          </div>

          <div className="flex gap-4 pt-4">
            <button type="button" onClick={onBack} className="w-1/3 bg-gray-500 text-white font-bold py-3 rounded-md hover:bg-gray-600">
              ← 뒤로가기
            </button>
            <button type="submit" disabled={isLoading} className="w-2/3 bg-purple-600 text-white font-bold py-3 rounded-md hover:bg-purple-700 disabled:bg-purple-400 flex items-center justify-center gap-2">
              {isLoading && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>}
              {isLoading ? '기획안 생성 중...' : '이 아이디어로 데이터 기획안 생성'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}