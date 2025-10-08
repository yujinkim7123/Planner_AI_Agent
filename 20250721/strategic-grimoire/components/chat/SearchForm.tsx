// components/chat/SearchForm.tsx
"use client";

import { useState } from 'react';

interface SearchFormProps {
  onSearch: (searchParams: { keyword: string; dateRange: string; productType: string; }) => void;
  isLoading: boolean;
  onSkip: () => void;
}

// 렌더링 최적화를 위해 컴포넌트 밖에 선언합니다.
const PRODUCT_KEYWORDS = ["(선택 안함 신제품 시)", "의류관리기", "청소기", "로봇청소기", "냉장고", "세탁기", "에어컨", "건조기", "노트북"];

export default function SearchForm({ onSearch, isLoading, onSkip }: SearchFormProps) {
  const [keyword, setKeyword] = useState('');
  const [dateRange, setDateRange] = useState('최근 1년');
  const [productType, setProductType] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!keyword.trim() || isLoading) return;
    onSearch({ keyword, dateRange, productType: productType === "(선택 안함)" ? "" : productType });
  };

  return (
    <div className="flex flex-col items-center justify-center h-full bg-gray-50 dark:bg-gray-900 p-8">
      <div className="w-full max-w-lg">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">🔮 전략 기획가 마법서</h1>
          <p className="text-md text-gray-500 dark:text-gray-400">데이터 기반 기획의 첫걸음, 무엇이 궁금하신가요?</p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4 p-6 border rounded-lg bg-white dark:bg-gray-800 shadow-xl">
          <div>
            <label htmlFor="keyword" className="block text-sm font-medium mb-1">핵심 키워드 / 문장</label>
            <textarea
              id="keyword"
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              placeholder="예: 의류관리기 냄새, 아기 옷 살균 방법"
              required
              className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={3}
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="dateRange" className="block text-sm font-medium mb-1">기간</label>
              <input
                type="text"
                id="dateRange"
                value={dateRange}
                onChange={(e) => setDateRange(e.target.value)}
                required
                className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label htmlFor="productType" className="block text-sm font-medium mb-1">제품군 (선택)</label>
              <select
                id="productType"
                value={productType}
                onChange={(e) => setProductType(e.target.value)}
                className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {PRODUCT_KEYWORDS.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
          </div>
        <div className="flex justify-end gap-3 pt-4">
          <button
            type="button" // form 제출을 방지
            onClick={onSkip} // 클릭 시 page.tsx로부터 받은 onSkip 함수 실행
            disabled={isLoading}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-200 rounded-md hover:bg-gray-300 dark:bg-gray-600 dark:text-gray-200 dark:hover:bg-gray-500 disabled:opacity-50"
          >
            나중에 하기
          </button>
          <button
            type="submit"
            disabled={isLoading}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed"
          >
            {isLoading ? '검색 중...' : '데이터 검색'}
          </button>
        </div>
        </form>
      </div>
    </div>
  );
}