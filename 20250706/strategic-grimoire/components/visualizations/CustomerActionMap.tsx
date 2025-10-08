// components/visualizations/CustomerActionMap.tsx
"use client";

interface CamData {
  action_name: string;
  goals: string[];
  pain_points: string[];
  context: string[];
  touchpoint_artifact: string[];
}

export default function CustomerActionMap({ data }: { data: CamData }) {
  const Section = ({ title, items, icon }: { title: string, items: string[], icon: string }) => (
    <div>
      <h4 className="font-semibold text-sm flex items-center gap-2 mb-1">
        <span>{icon}</span>
        <span>{title}</span>
      </h4>
      <ul className="list-disc list-inside pl-2 text-xs space-y-1 text-gray-700 dark:text-gray-300">
        {items.map((item, index) => <li key={index}>{item}</li>)}
      </ul>
    </div>
  );

  return (
    <div className="p-4 border rounded-lg bg-white dark:bg-gray-800 shadow-md text-gray-900 dark:text-gray-100 space-y-4">
      <h3 className="font-bold text-center text-md text-blue-600 dark:text-blue-400">
        고객 행동 분석 (CAM): {data.action_name}
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Section title="고객의 목표 (Goals)" items={data.goals} icon="🎯" />
        <Section title="핵심 불편함 (Pain Points)" items={data.pain_points} icon="😫" />
        <Section title="주요 상황 (Context)" items={data.context} icon="💡" />
        <Section title="관련 사물/서비스 (Touchpoints)" items={data.touchpoint_artifact} icon="📱" />
      </div>
    </div>
  );
}