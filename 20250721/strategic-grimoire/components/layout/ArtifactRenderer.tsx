// components/layout/ArtifactRenderer.tsx
"use client";


import { useWorkspaceStore } from "@/store/workspaceStore";
//import CDPSheet from "../visualizations/CDPSheet";
import OpportunityMap from "../visualizations/OpportunityMap";
import CustomerActionMap from "../visualizations/CustomerActionMap";
import PersonaCard from "../visualizations/PersonaCard";
import ServiceIdeaCard from "../visualizations/ServiceIdeaCard";
import ClusteringResult from "../visualizations/ClusteringResult";
import TopicList from "../visualizations/TopicList";
import DataViewer from "../visualizations/DataViewer";
import dynamic from 'next/dynamic';
import React, { useState } from 'react';
import LdaKeywordBarChart from '../visualizations/LdaKeywordBarChart';
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import ArtifactCard from "./ArtifactCard"; // 새로 만든 카드 컴포넌트

// Lucide 아이콘들
import { 
  Database, BarChart3, Binary, Waypoints, BrainCircuit, 
  Lightbulb, Users, FileText, ClipboardList 
} from 'lucide-react';

interface SnaResults {
  cluster_id: number;
  micro_segments: Array<{
    community_id?: number | string;
    core_keyword: string;
    keywords: string[];
    centrality_score?: number;
  }>;
  graph_data: {
    nodes: Array<{ id: string; name?: string; community?: number; centrality?: number }>;
    links: Array<{ source: string; target: string; weight?: number }>;
  };
  analysis_description?: string;
}


interface LdaTopicDataPoint {
  topic_id: number;
  x: number;
  y: number;
  keywords_data: { [key: string]: number };
}

interface LdaGraphData {
  topics: LdaTopicDataPoint[];
  num_topics: number;
}

interface LdaResults {
  cluster_id: number;
  num_topics: number;
  topics_summary_list: Array<{
    topic_id: string; 
    action_keywords: string[];
    topic_summary?: string; 
  }>;
  graph_data: LdaGraphData; 
}


const CDPSheet = dynamic(() => import('../visualizations/CDPSheet'), { ssr: false }); 
const DataPlanViewer = dynamic(() => import('../visualizations/DataPlanViewer'), { ssr: false });
const ClusteringGraph = dynamic(() => import('./ClusteringGraph'), { ssr: false });
const SnaGraph = dynamic(() => import('./SnaGraph'), { ssr: false });
const LdaGraph = dynamic(() => import('./LdaGraph'), { ssr: false });
const SnaGraph_module = dynamic(() => import('./SnaGraph_module'), { ssr: false });

export default function ArtifactRenderer() {
  const { artifacts } = useWorkspaceStore();

  const INITIAL_MICRO_SEGMENTS_COUNT = 5;
  const [showAllMicroSegments, setShowAllMicroSegments] = useState(false);
  
  const hasContent = Object.values(artifacts).some(value => {
    if (Array.isArray(value)) return value.length > 0;
    return value !== null;
  });

  const snaResults = (typeof artifacts.cx_sna_results === 'object' && artifacts.cx_sna_results !== null && !Array.isArray(artifacts.cx_sna_results))
    ? (artifacts.cx_sna_results as SnaResults)
    : null;

  const ldaResults = (typeof artifacts.cx_lda_results === 'object' && artifacts.cx_lda_results !== null && !Array.isArray(artifacts.cx_lda_results))
    ? (artifacts.cx_lda_results as LdaResults)
    : null;

  const personaMap = new Map();
  if (artifacts.personas?.length > 0) {
    artifacts.personas.forEach((p: any) => personaMap.set(p.name, p));
  }
  if (artifacts.selected_persona) {
    personaMap.set(artifacts.selected_persona.name, artifacts.selected_persona);
  }
  const displayablePersonas = Array.from(personaMap.values());

  const ideaMap = new Map();
  if (artifacts.service_ideas?.length > 0) {
    artifacts.service_ideas.forEach((idea: any) => ideaMap.set(idea.service_name, idea));
  }
  if (artifacts.selected_service_idea) {
    ideaMap.set(artifacts.selected_service_idea.service_name, artifacts.selected_service_idea);
  }
  const displayableServiceIdeas = Array.from(ideaMap.values());
  
  // 선택된 항목들을 변수로 추출
  const selectedDataPlan = artifacts.selected_data_plan_for_service;
  const selectedCdpDefinition = artifacts.selected_cdp_definition;


    return (
    <PanelGroup direction="horizontal" className="h-screen w-full bg-gray-50 dark:bg-gray-900">
      <Panel defaultSize={65} minSize={40}>
        <div className="p-4 sm:p-6 overflow-y-auto h-full">
          <div className="font-bold text-2xl mb-6 text-gray-900 dark:text-white">📜 나의 결과물</div>

          {!hasContent && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-md text-gray-400">
                분석을 진행하면<br />이곳에 결과물이 보관됩니다.
              </div>
            </div>
          )}
          
          <div className="space-y-6">
            {artifacts.retrieved_data && (
              // <ArtifactCard title="데이터 검색 결과" icon={<Database size={20} />}>
                <DataViewer 
                  retrievedData={artifacts.retrieved_data}
                  productData={artifacts.product_data}
                  sensorData={artifacts.sensor_data}
                  columnsProduct={artifacts.columns_product}
                  productType={artifacts.product_type} 
                />
              // </ArtifactCard>
            )}

            {artifacts.cx_ward_clustering_results && (
              <ArtifactCard title="고객군 클러스터링" icon={<Binary size={20} />}>
                <ClusteringResult data={artifacts.cx_ward_clustering_results} />
                {artifacts.cx_ward_clustering_results.visual_data && (
                  <div className="mt-4">
                    <h4 className="font-medium text-md mb-2">클러스터 시각화</h4>
                    <ClusteringGraph data={artifacts.cx_ward_clustering_results.visual_data} />
                  </div>
                )}
              </ArtifactCard>
            )}

            {snaResults && snaResults.graph_data && (
              <ArtifactCard title={`의미 연결망 분석 (SNA) - 클러스터 ${snaResults.cluster_id}`} icon={<Waypoints size={20} />}>
                <SnaGraph_module data={snaResults.graph_data} />
                <SnaGraph data={snaResults.graph_data} />
                {/* <SnaGraph data={snaResults.graph_data} />  // 둘 중 하나만 사용하거나 필요에 따라 조정 */}
                {/* ... (마이크로 세그먼트 렌더링 로직) ... */}
              </ArtifactCard>
            )}

            {ldaResults && ldaResults.graph_data && (
              <ArtifactCard title="토픽 모델링 (LDA)" icon={<BrainCircuit size={20} />}>
                <LdaGraph graphData={ldaResults.graph_data} />
                {ldaResults.topics_summary_list?.length > 0 && 
                  <TopicList data={ldaResults.topics_summary_list} />}
              </ArtifactCard>
            )}

            {artifacts.cx_opportunity_scores?.length > 0 && (
                <ArtifactCard title="기회 발굴 점수" icon={<BarChart3 size={20} />}>
                    <OpportunityMap data={artifacts.cx_opportunity_scores} />
                </ArtifactCard>
            )}

            {artifacts.cx_cam_results?.length > 0 && (
                <ArtifactCard title="고객 행동 지도" icon={<BarChart3 size={20} />}>
                    {artifacts.cx_cam_results.map((cam, i) => 
                        <div key={`cam-${i}`} className="mb-4">
                            <CustomerActionMap data={cam} />
                            <LdaKeywordBarChart keywordsData={cam.keywords_data} topN={7} />
                        </div>
                    )}
                </ArtifactCard>
            )}

            {displayablePersonas.length > 0 && (
              
              <ArtifactCard title="페르소나" icon={<Users size={20} />}>
                {displayablePersonas.map((p, i) => <PersonaCard key={`persona-${i}`} data={p} />)}
              </ArtifactCard>
            )}
            
            {displayableServiceIdeas.length > 0 && (
              <ArtifactCard title="서비스 아이디어" icon={<Lightbulb size={20} />}>
                {displayableServiceIdeas.map((idea, i) => <ServiceIdeaCard key={`idea-${i}`} data={idea} />)}
              </ArtifactCard>
            )}
            
            {selectedDataPlan && (
              <ArtifactCard title="데이터 기획안" icon={<FileText size={20} />} defaultOpen={false}>
                <DataPlanViewer 
                  data={selectedDataPlan.data_plan || selectedDataPlan} 
                  recommendationMessage={selectedDataPlan.recommendation_message}
                />
              </ArtifactCard>
            )}
            
            {selectedCdpDefinition && (
              <ArtifactCard title="최종 보고서" icon={<ClipboardList size={20} />} defaultOpen={false}>
                <CDPSheet data={selectedCdpDefinition} />
              </ArtifactCard>
            )}
          </div>
        </div>
      </Panel>
    </PanelGroup>
  );
}
