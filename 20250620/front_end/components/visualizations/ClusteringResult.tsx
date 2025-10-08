// components/visualizations/ClusteringResult.tsx
"use client";

export default function ClusteringResult({ data }: { data: any }) {
  return (
    <div className="p-4 border rounded-lg bg-gray-50 dark:bg-gray-800/50">
      <h3 className="font-bold mb-2 text-md">📊 고객 관심사 그룹 분석 결과</h3>
      <p className="text-xs mb-3 text-gray-600 dark:text-gray-400">총 {data.num_clusters}개의 그룹이 발견되었습니다.</p>
      <div className="space-y-2">
        {Object.entries(data.cluster_summaries).map(([id, summary]: [string, any]) => (
          <div key={id} className="p-2 border-l-4 border-blue-500 bg-white dark:bg-gray-700 rounded-r-md">
            <p className="font-semibold text-sm">{id}번 그룹</p>
            <p className="text-xs text-gray-500 dark:text-gray-300">{summary.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}