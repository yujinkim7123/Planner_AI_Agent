// components/layout/SnaGraph_module.tsx (파일명이 SnaGraph_module.tsx라고 가정)
"use client";

import React, { useRef, useEffect } from "react";
import ForceGraph2D, { NodeObject, LinkObject } from "react-force-graph-2d";
import { forceLink, forceManyBody, forceCenter, forceCollide } from "d3-force";

type Node = NodeObject & {
  id: string;
  community?: number;
  centrality?: number;
};
type Link = LinkObject & {
  source: string;
  target: string;
  weight?: number;
};

interface SnaGraphProps {
  data: {
    nodes: Node[];
    links: Link[];
  };
}

export default function SnaGraph({ data }: SnaGraphProps) { // SnaGraph로 컴포넌트 이름은 유지
  const fgRef = useRef<any>(null);

  useEffect(() => {
    const fg = fgRef.current!;
    if (!fg) return; // fgRef.current가 없을 경우를 대비

    // ① 사이 링크가 너무 멀어지지 않도록
    fg.d3Force(
      "link",
      forceLink()
        .id((d: any) => d.id)
        .distance(50)
        .strength(0.1)
    );
    // ② 노드 반발력 (가까울 때만 튕기도록)
    fg.d3Force(
      "charge",
      forceManyBody().strength(-40).distanceMax(200)
    );
    // ③ 노드 충돌 방지
    fg.d3Force(
      "collide",
      forceCollide().radius(6).strength(0.8)
    );
    // ④ 화면 중앙으로
    fg.d3Force("center", forceCenter());
    
    // **더 이상 sim.alpha, d3ReheatSimulation 절대 사용하지 않습니다.**

    // 🚨 그래프가 로드된 후 모든 노드가 보이도록 줌 및 패닝 조정
    // 이 코드는 시뮬레이션이 안정화된 후 (warmupTicks 이후) 실행되어야 합니다.
    // setTimeout을 사용하여 렌더링 및 초기 시뮬레이션이 완료될 시간을 줍니다.
    const timer = setTimeout(() => {
      // zoomToFit 메서드 호출. 첫 번째 인자는 애니메이션 시간 (ms), 두 번째 인자는 줌 패딩 (%).
      fg.zoomToFit(400, 20); // 400ms 동안 애니메이션, 20% 패딩
    }, 500); // 렌더링 후 500ms 지연

    return () => clearTimeout(timer); // 컴포넌트 언마운트 시 타이머 정리
    
  }, [data]); // data 변경 시에만 다시 force 설정

  // 노드 크기 함수 (중심성에 비례)
  const getNodeSize = (node: Node) => {
    // centrality가 없거나 0이면 기본 크기, 있으면 중심성에 비례하여 증가
    const baseSize = 8; // 기본 노드 반지름
    const scaleFactor = 15; // 중심성 점수 스케일링 팩터 (조절 필요)
    return baseSize + (node.centrality || 0) * scaleFactor;
  };

  // 노드 색상 함수 (커뮤니티에 따라 다르게)
  const getNodeColor = (node: Node) => {
    if (node.community !== undefined && node.community !== null) {
      // D3 스케일 컬러 팔레트를 사용하거나, 직접 색상 배열 정의
      const colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
      ];
      return colors[node.community % colors.length];
    }
    return '#ccc'; // 커뮤니티 정보가 없으면 기본 회색
  };

  return (
    <div className="w-full h-96 overflow-hidden">
      <ForceGraph2D
        ref={fgRef}
        graphData={data}
        warmupTicks={200}
        cooldownTicks={0}
        d3AlphaDecay={0.02}
        
        linkColor={() => "#888"} // 기존과 동일
        linkWidth={(l: Link) => Math.max(1, (l.weight || 0.1) * 2)} // 기존과 동일
        nodeAutoColorBy="community" // community 값으로 자동 색상 부여. 만약 getNodeColor를 사용하려면 이 줄을 제거해야 합니다.
        nodeLabel={(n: Node) => `${n.id}\ncent:${n.centrality?.toFixed(2)}`} // 기존과 동일
        
        enableNodeDrag={true} // 드래그 활성화 (기존 파일은 false였으나, SnaGraph.tsx로 가정한 파일에 맞춰 true로 변경)
        enableZoomInteraction={true}
        enablePanInteraction={true}

        // getNodeSize 및 getNodeColor 함수가 작동하도록 prop 추가 또는 조정
        nodeVal={getNodeSize} // 🚨 노드 크기 함수 적용
        nodeColor={getNodeColor} // 🚨 노드 색상 함수 적용
        
        backgroundColor="#f8f8f8" // 연한 회색 배경
      />
    </div>
  );
}