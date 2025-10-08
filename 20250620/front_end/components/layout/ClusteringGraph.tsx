// components/layout/ClusteringGraph.tsx
import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// 백엔드에서 넘어오는 visual_data의 타입을 정의합니다.
interface ClusteringGraphData {
  reduced_features_2d: number[][]; // 2차원으로 축소된 각 데이터 포인트의 좌표
  cluster_labels: number[];        // 각 데이터 포인트에 할당된 클러스터 ID
}

interface ClusteringGraphProps {
  data: ClusteringGraphData;
}

const ClusteringGraph: React.FC<ClusteringGraphProps> = ({ data }) => {
  // 데이터 유효성 검사
  if (!data || !data.reduced_features_2d || !data.cluster_labels || data.reduced_features_2d.length === 0) {
    return <p>클러스터링 시각화 데이터를 찾을 수 없습니다.</p>;
  }

  // Recharts는 각 데이터 포인트가 { x: value, y: value, group: label } 형태를 갖는 것을 선호합니다.
  const chartData = data.reduced_features_2d.map((coords, index) => ({
    x: coords[0],
    y: coords[1],
    cluster: `Cluster ${data.cluster_labels[index]}`, // 클러스터 라벨을 그룹으로 사용합니다.
  }));

  // 클러스터 라벨의 고유한 값들을 추출하여 범례(Legend) 및 색상 매핑에 사용합니다.
  const uniqueClusters = Array.from(new Set(data.cluster_labels)).sort((a, b) => a - b);
  
  // 클러스터 수에 따라 다양한 색상을 미리 정의합니다.
  // 필요에 따라 더 많은 색상을 추가하거나, 색상 팔레트 라이브러리를 사용할 수 있습니다.
  const colors = [
    '#8884d8', '#82ca9d', '#ffc658', '#d0ed57', '#a4de6c', '#8dd1e1', '#b5a4de', '#dea48d', '#aedea4', '#d88884',
    '#ff7300', '#00bcd4', '#795548', '#e91e63', '#673ab7', '#3f51b5', '#4caf50', '#ff9800', '#cddc39', '#009688'
  ]; 

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart
        margin={{
          top: 20,
          right: 20,
          bottom: 20,
          left: 20,
        }}
      >
        <CartesianGrid /> {/* 배경 격자 */}
        <XAxis type="number" dataKey="x" name="주성분 1" unit="" /> {/* X축 설정 */}
        <YAxis type="number" dataKey="y" name="주성분 2" unit="" /> {/* Y축 설정 */}
        <Tooltip cursor={{ strokeDasharray: '3 3' }} /> {/* 마우스 오버 시 툴팁 표시 */}
        <Legend /> {/* 범례 표시 */}
        
        {/* 각 클러스터별로 Scatter 플롯을 그립니다. */}
        {uniqueClusters.map((clusterId, index) => (
          <Scatter
            key={`cluster-${clusterId}`} // 고유한 키
            name={`클러스터 ${clusterId}`} // 범례에 표시될 이름
            data={chartData.filter(d => d.cluster === `Cluster ${clusterId}`)} // 해당 클러스터 데이터만 필터링
            fill={colors[index % colors.length]} // 클러스터별 색상 할당
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
};

export default ClusteringGraph;