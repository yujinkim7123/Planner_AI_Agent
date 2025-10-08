import React, { useRef, useEffect, useCallback, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import * as d3 from 'd3'; // d3 라이브러리 임포트 확인

interface SnaGraphProps {
  data: {
    nodes: Array<{ id: string; name?: string; community?: number }>;
    links: Array<{ source: string; target: string; weight?: number }>;
  };
}

  const SnaGraph: React.FC<SnaGraphProps> = ({ data }) => {
  const fgRef = useRef<any>(null); // ForceGraph 인스턴스 참조
  const containerRef = useRef<HTMLDivElement>(null); // 그래프 컨테이너 div 참조
  const [dimensions, setDimensions] = React.useState({ width: 0, height: 0 }); // 컨테이너 너비/높이 상태
  const [linkVisibilityMap, setLinkVisibilityMap] = useState<Map<string, boolean>>(new Map()); // 엣지 가시성 관리 Map

  // 노드 색상을 정의하는 함수 (useCallback으로 최적화)
  const getNodeColor = useCallback((node: any) => {
    const colors = [
      '#8884d8', '#82ca9d', '#ffc658', '#d0ed57', '#a4de6c', '#8dd1e1', '#b5a4de', '#dea48d', '#aedea4', '#d88884',
      '#ff7300', '#00bcd4', '#795548', '#e91e63', '#673ab7', '#3f51b5', '#4caf50', '#ff9800', '#cddc39', '#009688'
    ];
    if (typeof node.community === 'number') {
      return colors[node.community % colors.length];
    }
    return '#cccccc'; // 기본 회색
  }, []);

  // 링크 색상을 정의하는 함수 (useCallback으로 최적화)
  const getLinkColor = useCallback((link: any) => {
    if (typeof link.weight === 'number') {
      const alpha = Math.min(1, link.weight * 3); // 가중치에 따라 투명도 조절
      return `rgba(0, 0, 0, ${alpha})`; // 검은색에 투명도 적용
    }
    return 'rgba(0, 0, 0, 0.2)'; // 기본 옅은 회색
  }, []);

  // 링크 두께를 정의하는 함수 (useCallback으로 최적화)
  const getLinkWidth = useCallback((link: any) => {
    if (typeof link.weight === 'number') {
      return Math.max(0.5, link.weight * 5); // 최소 0.5, 가중치에 따라 두께 증가
    }
    return 0.5;
  }, []);

  // 노드 클릭 시 동작하는 함수 (useCallback으로 최적화)
  const handleClick = useCallback((node: any) => {
    if (fgRef.current) {
      fgRef.current.centerAndZoom([node.x, node.y], 400); // 줌 레벨 조정
    }
  }, []);

  // 엣지 클릭 핸들러 (useCallback으로 최적화)
  const handleLinkClick = useCallback((link: any) => {
    const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
    const targetId = typeof link.target === 'object' ? link.target.id : link.target;
    const linkId = `${sourceId}-${targetId}`;

    setLinkVisibilityMap(prevMap => {
      const newMap = new Map(prevMap);
      newMap.set(linkId, !newMap.get(linkId)); // 토글
      return newMap;
    });
  }, []);

  // 엣지 가시성 함수 (useCallback으로 최적화)
  const linkVisibility = useCallback((link: any) => {
    const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
    const targetId = typeof link.target === 'object' ? link.target.id : link.target;
    const linkId = `${sourceId}-${targetId}`;
    return linkVisibilityMap.get(linkId) !== false; // Map에 없거나 true이면 보임
  }, [linkVisibilityMap]);

  // // 컨테이너 크기 감지 및 반응형 처리
  // useEffect(() => {
  //   const updateDimensions = () => {
  //     if (containerRef.current) {
  //       setDimensions({
  //         width: containerRef.current.offsetWidth,
  //         height: Math.max(containerRef.current.offsetWidth * 0.75, 400),
  //       });
  //     }
  //   };

  //   updateDimensions();
  //   window.addEventListener('resize', updateDimensions);

  //   return () => {
  //     window.removeEventListener('resize', updateDimensions);
  //   };
  // }, []);


    // 컨테이너 크기 감지 및 반응형 처리 (ResizeObserver를 사용한 최종 수정 버전)
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height: Math.max(containerRef.current.offsetWidth * 0.75, 400),
        });
      }
    };

    // containerRef가 아직 설정되지 않았으면 함수를 종료합니다.
    if (!containerRef.current) return;

    // ResizeObserver 인스턴스를 생성합니다.
    // 이 옵저버는 관찰 대상(그래프 컨테이너)의 크기가 변경될 때마다 updateDimensions 함수를 호출합니다.
    const observer = new ResizeObserver(() => {
      updateDimensions();
    });

    // 그래프 컨테이너(containerRef.current)에 대한 관찰을 시작합니다.
    observer.observe(containerRef.current);

    // 컴포넌트가 화면에서 사라질 때(unmount) 관찰을 중단하여 메모리 누수를 방지합니다.
    return () => {
      observer.disconnect();
    };
  }, []); // 의존성 배열은 비워두어 컴포넌트 마운트 시 한 번만 실행되게 합니다.

  // 데이터나 dimensions 변경 시 그래프 줌 초기화 및 물리 엔진 재시작
  useEffect(() => {
    if (fgRef.current && dimensions.width > 0 && dimensions.height > 0) {
      fgRef.current.zoomToFit(150, 200); // 패딩 150px, 200ms 애니메이션
      
      // 물리 엔진 설정 직접 조작 (ForceGraph2D의 prop 대신 ref를 통해 조작)
      // d3Force prop을 사용할 때 콜백 함수 형태로 전달할 필요가 없어집니다.
      // fgRef.current.d3Force('charge').strength(-150).distanceMax(150); // 척력
      // fgRef.current.d3Force('link').distance(60).iterations(1);      // 인력
      fgRef.current.d3Force('center', d3.forceCenter(dimensions.width / 2, dimensions.height / 2)); // 중앙 정렬
      
      fgRef.current.d3ReheatSimulation(); // 시뮬레이션 재가열
    }
    setLinkVisibilityMap(new Map()); // 숨겨진 링크 상태 초기화
  }, [data, dimensions]);

  // 데이터 유효성 검사
  if (!data || !data.nodes || !data.links || data.nodes.length === 0) {
    return <p>SNA 시각화 데이터를 찾을 수 없습니다.</p>;
  }

  return (
    <div ref={containerRef} style={{ width: '100%', height: `${dimensions.height}px`, minHeight: '300px', border: '1px solid #eee' }}>
      {dimensions.width > 0 && dimensions.height > 0 && (
        <ForceGraph2D
          ref={fgRef}
          graphData={data}
          nodeId="id"
          nodeLabel={(node: any) => node.id}
          nodeCanvasObject={getNodeCanvasObject(getNodeColor)}
          nodePointerAreaPaint={getNodePointerAreaPaint(getNodeColor)}
          
          linkWidth={getLinkWidth}
          linkColor={getLinkColor}
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={1}
          onNodeClick={handleClick}
          onLinkClick={handleLinkClick}
          linkVisibility={linkVisibility} 

          width={dimensions.width}
          height={dimensions.height}
          
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.4}
          // 🚨 d3Force prop을 사용하지 않습니다. 대신 useEffect에서 ref를 통해 직접 힘을 조작합니다.
          // d3Force 속성은 여기서 제거합니다.
        />
      )}
    </div>
  );
};

export default SnaGraph;

// 노드 캔버스 객체 그리기 함수 (컴포넌트 외부에서 정의)
const getNodeCanvasObject = (getNodeColor: (node: any) => string) => (node: any, ctx: CanvasRenderingContext2D, scale: number) => {
  const label = node.id;
  const fontSize = 12 / scale;
  ctx.font = `${fontSize}px Sans-Serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  
  ctx.fillStyle = getNodeColor(node); 
  const textWidth = ctx.measureText(label).width;
  const rectPadding = 2;
  ctx.fillRect(node.x - textWidth / 2 - rectPadding, node.y - fontSize / 2 - rectPadding, textWidth + rectPadding * 2, fontSize + rectPadding * 2);
  
  ctx.fillStyle = 'black';
  ctx.fillText(label, node.x, node.y);
};

// 노드 포인터 영역 그리기 함수 (컴포넌트 외부에서 정의)
const getNodePointerAreaPaint = (getNodeColor: (node: any) => string) => (node: any, color: string, ctx: CanvasRenderingContext2D) => {
  ctx.fillStyle = color;
  
  const label = node.id;
  const fontSize = 12;
  const rectPadding = 2;
  const textWidth = ctx.measureText(label).width;
  ctx.fillRect(node.x - textWidth / 2 - rectPadding, node.y - fontSize / 2 - rectPadding, textWidth + rectPadding * 2, fontSize + rectPadding * 2);
};