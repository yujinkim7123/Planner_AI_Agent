// components/layout/LdaGraph.tsx
import React, { useState, useCallback } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// 백엔드 LDA 결과의 graph_data.topics 배열의 각 항목 타입
interface TopicDataPoint {
  topic_id: number;
  x: number; // 2D 위치 X 좌표
  y: number; // 2D 위치 Y 좌표
  keywords_data: { [key: string]: number }; // 해당 토픽의 키워드와 가중치 (확률)
}

// 백엔드 LDA 결과의 graph_data 전체 타입
interface LdaGraphData {
  topics: TopicDataPoint[]; // 각 토픽의 데이터 포인트 배열
  num_topics: number; // 총 토픽 개수
}

interface LdaGraphProps {
  graphData: LdaGraphData; // graph_data 전체를 prop으로 받습니다.
}

// 커스텀 툴팁 컴포넌트 (마우스 오버 시 키워드 및 가중치 표시)
interface CustomTooltipProps {
  active?: boolean; // 툴팁 활성화 여부
  payload?: Array<{ payload: TopicDataPoint }>; // 툴팁에 표시될 데이터
  label?: string; // 툴팁 라벨 (여기서는 사용 안 함)
}


const CustomTooltip: React.FC<CustomTooltipProps> = ({ active, payload }) => {
  if (active && payload && payload.length > 0) {
    const topicData = payload[0].payload; // 첫 번째 payload의 payload 필드에 TopicDataPoint가 있습니다.
    const topicId = topicData.topic_id;
    const keywords = topicData.keywords_data; // 해당 토픽의 키워드 데이터

    return (
      <div className="bg-white border p-2 shadow-md rounded-md">
        <p className="font-semibold text-gray-800">📊 토픽 {topicId} - 주요 키워드</p>
        {keywords && Object.entries(keywords)
          .sort(([, weightA], [, weightB]) => weightB - weightA) // 가중치 내림차순 정렬
          .map(([keyword, weight]) => (
            <p key={keyword} className="text-gray-700 text-sm">
              {keyword}: {(weight * 100).toFixed(2)}% {/* 가중치를 백분율로 표시 */}
            </p>
          ))}
      </div>
    );
  }
  return null;
};

const LdaGraph: React.FC<LdaGraphProps> = ({ graphData }) => {
  // 현재 하이라이트된(선택된) 토픽 ID 상태 관리
  const [highlightedTopic, setHighlightedTopic] = useState<number | null>(null);

  // 토픽 점 클릭 시 하이라이트/선택 토글 핸들러
  const handleTopicClick = useCallback((topic: TopicDataPoint) => {
    // 이전에 선택된 토픽과 같으면 선택 해제, 다르면 새로 선택
    setHighlightedTopic(prev => (prev === topic.topic_id ? null : topic.topic_id));
  }, []);

  // 토픽별 색상 팔레트
  // 토픽 개수가 많을 때 대비하여 충분한 색상을 정의합니다.
  const colors = [
    '#4CAF50', '#2196F3', '#FFC107', '#E91E63', '#9C27B0', '#00BCD4', '#FF5722', '#607D8B', '#795548', '#FFEB3B',
    '#8BC34A', '#03A9F4', '#FF9800', '#F44336', '#673AB7', '#009688', '#FFC107', '#CDDC39', '#FF4081', '#536DFE',
    '#3F51B5', '#00E676', '#FFD180', '#FF8A80', '#B39DDB', '#4DB6AC'
  ];
  // 데이터 유효성 검사
  if (!graphData || !graphData.topics || graphData.topics.length === 0) {
    return <p>LDA 그래프 데이터를 찾을 수 없습니다.</p>;
  }

  return (
  <ResponsiveContainer width="100%" height={500}>
    {/* ← exactly one child element here */}
    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
      <CartesianGrid stroke="#000000" strokeDasharray="3 3" opacity={0.7} /> 
        <XAxis 
          type="number" 
          dataKey="x" 
          name="주성분 1" 
          unit="" 
          stroke="#444" // 축 색상
          tick={{ fill: '#444', fontSize: 12 }} // 눈금 라벨 색상
        />
        <YAxis 
          type="number" 
          dataKey="y" 
          name="주성분 2" 
          unit="" 
          stroke="#444" // 축 색상
          tick={{ fill: '#444', fontSize: 12 }} // 눈금 라벨 색상
        />
      <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
      <Legend wrapperStyle={{ color: '#444', fontSize: 13 }} />
      {graphData.topics.map(topic => (
        <Scatter
          key={topic.topic_id}
          name={`토픽 ${topic.topic_id}`}
          data={[topic]}
          fill={colors[topic.topic_id % colors.length]}
          onClick={() => handleTopicClick(topic)}
          opacity={highlightedTopic === null || highlightedTopic === topic.topic_id ? 1 : 0.4} // 덜 선택된 것은 더 투명하게
          radius={highlightedTopic === topic.topic_id ? 8 : 5}
          shape="circle"
        />
      ))}
    </ScatterChart>
  </ResponsiveContainer>
  );
};

export default LdaGraph;