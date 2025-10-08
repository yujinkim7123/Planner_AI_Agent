// components/visualizations/DataPlanViewer.tsx
import React from 'react';

// 백엔드에서 오는 data_plan 구조에 맞춰 타입 정의
interface DataPlanItem {
  idea?: string;
  details?: string;
  required_data?: string[];
  required_sensors?: string[];
  sensor_name?: string;
  collectable_data?: string;
  value_proposition?: string;
  external_data_name?: string;
  integration_plan?: string;
}

interface DataPlan {
  service_name: string;
  product_data_utilization: DataPlanItem[];
  new_data_from_sensors: DataPlanItem[];
  new_sensor_recommendation: DataPlanItem[];
  external_data_integration: DataPlanItem[];
}

interface DataPlanViewerProps {
  data: DataPlan;
  recommendationMessage?: string;
}

const SectionCard: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 hover:shadow-xl transition-shadow duration-300">
    <h4 className="font-semibold text-xl text-indigo-700 dark:text-indigo-400 mb-4 border-b pb-2 border-indigo-100 dark:border-indigo-600">
      {title}
    </h4>
    <div className="space-y-4">
      {children}
    </div>
  </div>
);

const IdeaItem: React.FC<{ item: DataPlanItem }> = ({ item }) => (
  <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg border border-gray-100 dark:border-gray-600">
    <p className="font-medium text-gray-800 dark:text-gray-100 mb-1">💡 {item.idea || item.sensor_name || item.external_data_name}</p>
    <p className="text-gray-600 dark:text-gray-300 text-sm">{item.details || item.collectable_data || item.integration_plan}</p>
    {item.required_data && item.required_data.length > 0 && (
      <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
        <span className="font-semibold">필요 데이터:</span> {item.required_data.join(', ')}
      </p>
    )}
    {item.required_sensors && item.required_sensors.length > 0 && (
      <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
        <span className="font-semibold">사용 센서:</span> {item.required_sensors.join(', ')}
      </p>
    )}
    {item.value_proposition && (
      <p className="text-xs text-green-700 dark:text-green-300 mt-2">
        <span className="font-semibold">예상 가치:</span> {item.value_proposition}
      </p>
    )}
  </div>
);

const DataPlanViewer: React.FC<DataPlanViewerProps> = ({ data, recommendationMessage }) => {
  if (!data) {
    return <div className="text-center text-gray-500 dark:text-gray-400 py-8">데이터 기획안을 불러올 수 없습니다.</div>;
  }

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md mb-6">
      <h3 className="font-bold text-2xl text-gray-900 dark:text-gray-50 mb-4 border-b pb-3 border-gray-200 dark:border-gray-700">
        📊 서비스 데이터 기획안: <span className="text-indigo-600 dark:text-indigo-300">{data.service_name}</span>
      </h3>

      <div className="space-y-8 mt-6">
        {data.product_data_utilization && data.product_data_utilization.length > 0 && (
          <SectionCard title="1. 기존 제품 데이터 활용 방안">
            {data.product_data_utilization.map((item, idx) => (
              <IdeaItem key={idx} item={item} />
            ))}
          </SectionCard>
        )}

        {data.new_data_from_sensors && data.new_data_from_sensors.length > 0 && (
          <SectionCard title="2. 기존 센서 데이터 기반 신규 데이터 생성">
            {data.new_data_from_sensors.map((item, idx) => (
              <IdeaItem key={idx} item={item} />
            ))}
          </SectionCard>
        )}

        {data.new_sensor_recommendation && data.new_sensor_recommendation.length > 0 && (
          <SectionCard title="3. 신규 센서 및 데이터 추천">
            {data.new_sensor_recommendation.map((item, idx) => (
              <IdeaItem key={idx} item={item} />
            ))}
          </SectionCard>
        )}

        {data.external_data_integration && data.external_data_integration.length > 0 && (
          <SectionCard title="4. 외부 데이터 연동 및 활용">
            {data.external_data_integration.map((item, idx) => (
              <IdeaItem key={idx} item={item} />
            ))}
          </SectionCard>
        )}
      </div>

      {recommendationMessage && (
        <div className="mt-8 p-4 bg-blue-50 dark:bg-blue-900 border border-blue-200 dark:border-blue-700 rounded-lg text-blue-800 dark:text-blue-200 text-sm">
          <p className="font-medium">💡 추가 제언:</p>
          <p>{recommendationMessage}</p>
        </div>
      )}
    </div>
  );
};

export default DataPlanViewer;