// components/visualizations/PersonaCard.tsx
"use client";

interface PersonaData {
  name: string;
  title: string;
  demographics: string;
  key_behaviors: string[];
  needs_and_goals: string[];
  pain_points: string[];
  motivating_quote: string;
}

export default function PersonaCard({ data }: { data: PersonaData }) {
  const ListSection = ({ title, items, icon }: { title: string, items: string[], icon: string }) => (
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
      <div className="text-center">
        <div className="text-4xl mb-2">👤</div>
        <h3 className="font-bold text-lg">{data.name}</h3>
        <p className="text-sm font-semibold text-blue-600 dark:text-blue-400">{data.title}</p>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{data.demographics}</p>
      </div>
      
      <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-md">
        <p className="italic text-center text-sm">"{data.motivating_quote}"</p>
      </div>

      <div className="grid grid-cols-1 gap-4">
        <ListSection title="주요 행동" items={data.key_behaviors} icon="🏃" />
        <ListSection title="니즈와 목표" items={data.needs_and_goals} icon="🎯" />
        <ListSection title="핵심 불편함" items={data.pain_points} icon="😫" />
      </div>
    </div>
  );
}