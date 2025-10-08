// components/visualizations/OpportunityMap.tsx

import React from 'react';

interface OpportunityScoreData {
  topic_id: string;
  action_keywords: string[];
  importance: number;
  satisfaction: number;
  opportunity_score: number;
}

interface OpportunityMapProps {
  data: OpportunityScoreData[]; // 기회 점수 데이터 배열
}

const OpportunityMap: React.FC<OpportunityMapProps> = ({ data }) => {
  // 데이터가 아예 없으면 (undefined 또는 null) 컴포넌트 자체를 렌더링하지 않도록 ArtifactRenderer에서 처리하므로,
  // 여기서는 빈 배열일 경우에 대한 메시지 처리
  const hasData = data && data.length > 0;

  return (
    // 🚨 디자인 수정: 배경, 테두리, 그림자 효과를 추가하여 카드 형태로 만듦.
    <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
      <h3 className="font-semibold text-xl text-gray-800 dark:text-gray-100 mb-4">
        📈 사업 기회 점수 분석
      </h3>

      {!hasData ? (
        // 🚨 데이터가 없을 때 표시될 메시지와 디자인
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          <p className="mb-2">계산된 사업 기회 점수 결과가 없습니다.</p>
          <p>LDA 분석 및 기회 점수 계산을 다시 시도해주세요.</p>
        </div>
      ) : (
        // 데이터가 있을 때 목록 렌더링
        <ul className="space-y-4"> {/* 각 리스트 아이템 간 간격 추가 */}
          {data.map((item, index) => (
            <li 
              key={item.topic_id || index} 
              className="p-3 border border-gray-100 dark:border-gray-700 rounded-md shadow-sm bg-gray-50 dark:bg-gray-700"
            >
              <div className="flex items-center mb-1">
                <span className="text-lg font-bold text-blue-600 dark:text-blue-400 mr-2">
                  #{index + 1}
                </span>
                <h4 className="font-semibold text-gray-800 dark:text-gray-200">
                  토픽 {item.topic_id}: <span className="text-gray-600 dark:text-gray-400">{item.action_keywords.join(', ')}</span>
                </h4>
              </div>
              <p className="text-sm text-gray-700 dark:text-gray-300 ml-6">
                - **기회 점수:** <span className="font-bold text-green-600 dark:text-green-400">{item.opportunity_score.toFixed(2)}</span>
                <span className="text-gray-500"> (중요도: {item.importance}, 만족도: {item.satisfaction.toFixed(2)})</span>
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default OpportunityMap;