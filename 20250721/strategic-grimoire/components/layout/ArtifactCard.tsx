// components/layout/ArtifactCard.tsx
import React, { useState, type ReactNode } from 'react';
import { ChevronUp } from 'lucide-react'; // 아이콘 라이브러리 (lucide-react 추천)

interface ArtifactCardProps {
  title: string;
  icon?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}

export default function ArtifactCard({ title, icon, children, defaultOpen = true }: ArtifactCardProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-md hover:shadow-lg transition-all duration-300">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex justify-between items-center p-4 text-left"
      >
        <div className="flex items-center space-x-3">
          {icon && <span className="text-gray-500 dark:text-gray-400">{icon}</span>}
          <h3 className="font-semibold text-lg text-gray-800 dark:text-gray-100">{title}</h3>
        </div>
        <ChevronUp
          className={`h-5 w-5 text-gray-500 transform transition-transform duration-300 ${isOpen ? 'rotate-0' : 'rotate-180'}`}
        />
      </button>
      
      {isOpen && (
        <div className="p-4 pt-0">
          <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            {children}
          </div>
        </div>
      )}
    </div>
  );
}