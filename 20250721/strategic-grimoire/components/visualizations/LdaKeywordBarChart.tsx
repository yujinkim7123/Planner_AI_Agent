// components/visualizations/LdaKeywordBarChart.tsx
"use client";

import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell
} from 'recharts';

interface LdaBarProps {
  /**
   * 키워드별 가중치 맵
   * 예: { "담배냄새": 0.12, "제거": 0.08, … }
   **/
  keywordsData: Record<string, number>;
  /** 보여줄 상위 N개 */
  topN?: number;
}

export default function LdaKeywordBarChart({ keywordsData, topN = 10 }: LdaBarProps) {
  // 객체 → 배열 변환 후, 가중치 내림차순, 상위 topN개
  const data = Object.entries(keywordsData)
    .map(([keyword, weight]) => ({ keyword, weight }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, topN);

  return (
    <div className="w-full h-64">
      <h4 className="font-medium text-sm mb-2 text-center">토픽 주요 키워드 가중치</h4>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={data}
          margin={{ top: 5, right: 20, bottom: 5, left: 80 }}
        >
          <XAxis type="number" hide />
          <YAxis
            dataKey="keyword"
            type="category"
            width={80}
            tick={{ fontSize: 12, fill: '#444' }}
          />
          <Tooltip
            formatter={(value: number) => [(value * 100).toFixed(1) + '%', 'Weight']}
          />
          <Bar dataKey="weight" barSize={12}>
            {data.map((entry, idx) => (
              <Cell key={idx} fill="rgba(34, 120, 220, 0.6)" />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
