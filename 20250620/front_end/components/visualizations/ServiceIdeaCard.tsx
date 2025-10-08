// components/visualizations/ServiceIdeaCard.tsx
"use client";

interface ServiceIdeaData {
  service_name: string;
  description: string;
  solved_pain_points: string[];
  service_scalability: string;
}

export default function ServiceIdeaCard({ data }: { data: ServiceIdeaData }) {
  return (
    <div className="p-4 border rounded-lg bg-white dark:bg-gray-800 shadow-md text-gray-900 dark:text-gray-100 space-y-3">
      <div className="text-center">
        <div className="text-4xl mb-2">💡</div>
        <h3 className="font-bold text-lg text-green-600 dark:text-green-400">{data.service_name}</h3>
      </div>
      
      <div>
        <h4 className="font-semibold text-sm mb-1">서비스 설명</h4>
        <p className="text-xs text-gray-700 dark:text-gray-300">{data.description}</p>
      </div>
      
      <div>
        <h4 className="font-semibold text-sm mb-1">해결되는 Pain Points</h4>
        <ul className="list-disc list-inside pl-2 text-xs space-y-1 text-gray-700 dark:text-gray-300">
          {data.solved_pain_points.map((item, index) => <li key={index}>{item}</li>)}
        </ul>
      </div>

      <div>
        <h4 className="font-semibold text-sm mb-1">서비스 확장성</h4>
        <p className="text-xs text-gray-700 dark:text-gray-300">{data.service_scalability}</p>
      </div>
    </div>
  );
}