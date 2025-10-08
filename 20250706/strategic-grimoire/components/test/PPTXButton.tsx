"use client";

import React from 'react';
declare global {
  interface Window {
    PptxGenJS: any;
  }
}

// props 타입 정의
interface PPTXButtonProps {
  data: any;
}

export default function PPTXButton({ data }: PPTXButtonProps) {
  
const handleGenerate = async () => {
    // 1. window 객체에서 PptxGenJS 라이브러리를 가져옵니다.
    const PptxGenJS = window.PptxGenJS;

    // 2. 라이브러리가 로드되었는지 반드시 확인합니다.
    if (!PptxGenJS) {
      alert("PPTX 생성 라이브러리를 로드하는 중입니다. 잠시 후 다시 시도해주세요.");
      console.error("PptxGenJS library not found on window object.");
      return;
    }
    if (!data) {
      console.error("PPTX 생성을 위한 데이터가 없습니다.");
      return;
    }
  try {
      const pres = new PptxGenJS();
      // A4 용지 크기(가로)로 레이아웃을 정의합니다.
      pres.defineLayout({ name: 'A4_LANDSCAPE', width: 11.69, height: 8.27 });
      pres.layout = 'A4_LANDSCAPE';

      const slide = pres.addSlide();

      const FONT_FACE = '맑은 고딕'; // PPT에서 사용할 기본 폰트

      // --- 상단 헤더 ---
      slide.addText("유첨. 하이드로타워 C-D-P 정의서", { x: 0.5, y: 0.2, fontSize: 12, fontFace: FONT_FACE });
      slide.addText("LGE Internal Use Only", { x: 8.5, y: 0.2, w: 2.69, h: 0.3, fontSize: 9, fontFace: FONT_FACE, align: pres.AlignH.right, color: '666666' });
      slide.addShape(pres.ShapeType.line, { x: 0.5, y: 0.5, w: 10.69, h: 0, line: { color: 'CCCCCC', width: 1 } });

      // --- 고객감동목표 섹션 ---
      slide.addShape(pres.ShapeType.rect, { x: 0.5, y: 0.7, w: 2.5, h: 0.35, fill: { color: '365F91' } });
      slide.addText("고객감동목표", { x: 0.5, y: 0.7, w: 2.5, h: 0.35, align: 'center', valign: 'middle', fontSize: 12, fontFace: FONT_FACE, color: 'FFFFFF', bold: true });
      slide.addText(data.customer_delight_goal || '', { x: 3.2, y: 0.7, w: 7.99, h: 0.35, valign: 'middle', fontSize: 14, fontFace: FONT_FACE, color: 'C00000', bold: true });


      // --- 3단 컬럼 레이아웃 설정 ---
      const colWidth = 3.5;
      const colGap = 0.095;
      const startY = 1.3;
      const colHeight = 6.5;

      const startX_CX = 0.5;
      const startX_P = startX_CX + colWidth + colGap;
      const startX_DX = startX_P + colWidth + colGap;

      const titleH = 0.4;
      const titleFillColor = 'F2F2F2';
      const contentY = startY + titleH;
      const contentH = colHeight - titleH;

      // --- CX 컬럼 ---
      slide.addShape(pres.ShapeType.rect, { x: startX_CX, y: startY, w: colWidth, h: colHeight, line: { color: 'BFBFBF', width: 1 }, fill: { color: 'FFFFFF' } });
      slide.addShape(pres.ShapeType.rect, { x: startX_CX, y: startY, w: colWidth, h: titleH, fill: { color: titleFillColor } });
      slide.addText([{ text: 'CX  ', options: { bold: true } }, { text: 'CX 기획 / 구현' }], { x: startX_CX + 0.1, y: startY, w: colWidth - 0.2, h: titleH, valign: 'middle', fontSize: 11, fontFace: FONT_FACE });

      let cxCurrentY = contentY + 0.1;
      // CX - 타겟 고객
      slide.addShape(pres.ShapeType.rect, { x: startX_CX + 0.1, y: cxCurrentY, w: colWidth - 0.2, h: 0.3, fill: { color: 'FDE9E9' } });
      slide.addText('타겟 고객 정의 “우리의 고객은?”', { x: startX_CX + 0.15, y: cxCurrentY, w: colWidth - 0.3, h: 0.3, valign: 'middle', fontSize: 10, fontFace: FONT_FACE, bold: true });
      cxCurrentY += 0.35;
      slide.addText(`${data.cx?.target_definition?.description || ''}\n\n“${data.cx?.target_definition?.quote || ''}”`, { x: startX_CX + 0.15, y: cxCurrentY, w: colWidth - 0.3, h: 1.5, valign: 'top', fontSize: 9, fontFace: FONT_FACE });
      cxCurrentY += 1.55;

      // CX - 핵심 경험
      slide.addShape(pres.ShapeType.rect, { x: startX_CX + 0.1, y: cxCurrentY, w: colWidth - 0.2, h: 0.3, fill: { color: 'FDE9E9' } });
      slide.addText('핵심 경험 “우리가 만드는 고객가치는?”', { x: startX_CX + 0.15, y: cxCurrentY, w: colWidth - 0.3, h: 0.3, valign: 'middle', fontSize: 10, fontFace: FONT_FACE, bold: true });
      cxCurrentY += 0.35;
      slide.addText([
          { text: 'Care: ', options: { bold: true } }, { text: data.cx?.core_experience?.care || '' },
          { text: '\n\nCustomization:', options: { bold: true, breakLine: true } },
          ...((data.cx?.core_experience?.customization || []).map((item: string) => ({ text: `\n• ${item}` }))),
          { text: '\n\nServitization: ', options: { bold: true, breakLine: true } }, { text: data.cx?.core_experience?.servitization || '' },
      ], { x: startX_CX + 0.15, y: cxCurrentY, w: colWidth - 0.3, h: 2.5, valign: 'top', fontSize: 9, fontFace: FONT_FACE });

      // --- Performance 컬럼 ---
      slide.addShape(pres.ShapeType.rect, { x: startX_P, y: startY, w: colWidth, h: colHeight, line: { color: 'BFBFBF', width: 1 }, fill: { color: 'FFFFFF' } });
      slide.addShape(pres.ShapeType.rect, { x: startX_P, y: startY, w: colWidth, h: titleH, fill: { color: titleFillColor } });
      slide.addText([{ text: 'P  ', options: { bold: true } }, { text: 'Performance' }], { x: startX_P + 0.1, y: startY, w: colWidth - 0.2, h: titleH, valign: 'middle', fontSize: 11, fontFace: FONT_FACE });

      let pCurrentY = contentY + 0.1;
      // P - 고객가치
      slide.addShape(pres.ShapeType.rect, { x: startX_P + 0.1, y: pCurrentY, w: colWidth - 0.2, h: 0.3, fill: { color: 'FDE9E9' } });
      slide.addText('고객가치', { x: startX_P + 0.15, y: pCurrentY, w: colWidth - 0.3, h: 0.3, valign: 'middle', fontSize: 10, fontFace: FONT_FACE, bold: true });
      pCurrentY += 0.35;
      slide.addText([
          { text: 'Find: ', options: { bold: true } }, { text: data.performance?.concept?.find || '' },
          { text: '\n\nUnique:', options: { bold: true, breakLine: true } },
          ...((data.performance?.concept?.unique || []).map((item: string) => ({ text: `\n• ${item}` }))),
      ], { x: startX_P + 0.15, y: pCurrentY, w: colWidth - 0.3, h: 2.0, valign: 'top', fontSize: 9, fontFace: FONT_FACE });
      pCurrentY += 2.05;

      // P - 경영성과
      slide.addShape(pres.ShapeType.rect, { x: startX_P + 0.1, y: pCurrentY, w: colWidth - 0.2, h: 0.3, fill: { color: 'FDE9E9' } });
      slide.addText('경영성과', { x: startX_P + 0.15, y: pCurrentY, w: colWidth - 0.3, h: 0.3, valign: 'middle', fontSize: 10, fontFace: FONT_FACE, bold: true });
      pCurrentY += 0.35;
      slide.addText(
        (data.performance?.business_outcome || []).map((item: string) => ({ text: `• ${item}\n` })),
        { x: startX_P + 0.15, y: pCurrentY, w: colWidth - 0.3, h: 2.0, valign: 'top', fontSize: 9, fontFace: FONT_FACE }
      );

      // --- DX 컬럼 ---
      slide.addShape(pres.ShapeType.rect, { x: startX_DX, y: startY, w: colWidth, h: colHeight, line: { color: 'BFBFBF', width: 1 }, fill: { color: 'FFFFFF' } });
      slide.addShape(pres.ShapeType.rect, { x: startX_DX, y: startY, w: colWidth, h: titleH, fill: { color: titleFillColor } });
      slide.addText([{ text: 'DX', options: { bold: true } }], { x: startX_DX + 0.1, y: startY, w: colWidth - 0.2, h: titleH, valign: 'middle', fontSize: 11, fontFace: FONT_FACE });

      let dxCurrentY = contentY + 0.1;
      const dxSections = [
        { title: data.dx?.trigger?.title || "Trigger", items: data.dx?.trigger?.items },
        { title: data.dx?.accelerator?.title || "Accelerator", items: [
            'UP-Contents 서비스',
            ...((data.dx?.accelerator?.up_contents_service || []).map((s:string) => `  - ${s}`)),
            'Data 기반 경험',
            ...((data.dx?.accelerator?.data_driven_experience || []).map((s:string) => `  - ${s}`)),
        ]},
        { title: data.dx?.tracker?.title || "Tracker", items: data.dx?.tracker?.items },
      ];

      dxSections.forEach(section => {
        if (!section.items || section.items.length === 0) return;
        slide.addShape(pres.ShapeType.rect, { x: startX_DX + 0.1, y: dxCurrentY, w: colWidth - 0.2, h: 0.3, fill: { color: 'E0F2F7' } });
        slide.addText(section.title, { x: startX_DX + 0.15, y: dxCurrentY, w: colWidth - 0.3, h: 0.3, valign: 'middle', fontSize: 10, fontFace: FONT_FACE, bold: true });
        dxCurrentY += 0.35;
        slide.addText(
          (section.items).map((item: string) => ({ text: `• ${item}\n` })),
          { x: startX_DX + 0.15, y: dxCurrentY, w: colWidth - 0.3, h: 1.8, valign: 'top', fontSize: 9, fontFace: FONT_FACE }
        );
        dxCurrentY += 1.85;
      });

      // --- 파일 생성 ---
      pres.writeFile({ fileName: `${data.title || 'C-D-P_정의서'}.pptx` });

    } catch (error) {
      console.error("PPTX 생성 중 오류 발생:", error);
      alert("PPTX 파일을 생성하는 데 실패했습니다.");
    }
  };

  return (
    <button
      onClick={handleGenerate}
      className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded text-sm transition-colors"
    >
      📄 PPTX로 다운로드
    </button>
  );
}