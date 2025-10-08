// components/visualizations/TopicList.tsx
"use client";

export default function TopicList({ data }: { data: any[] }) {
  return (
    <div className="p-4 border rounded-lg bg-gray-50 dark:bg-gray-800/50">
       <h3 className="font-bold mb-2 text-md">🎯 고객 행동 토픽 목록</h3>
       <div className="space-y-2">
        {data.map((topic) => (
           <div key={topic.topic_id} className="p-2 border-l-4 border-green-500 bg-white dark:bg-gray-700 rounded-r-md">
            <p className="font-semibold text-sm">토픽 {topic.topic_id}</p>
            <p className="text-xs text-gray-500 dark:text-gray-300">
              주요 키워드: {topic.action_keywords.join(', ')}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}