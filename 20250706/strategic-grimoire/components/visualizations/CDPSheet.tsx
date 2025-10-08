// components/visualizations/CDPSheet.tsx (수정된 최종본)
//import { generateCdpPptx } from "@/lib/pptxGenerator";
"use client";
import React from "react";
import dynamic from 'next/dynamic';


const PPTXButton = dynamic(() => import('../test/PPTXButton'), { ssr: false });

export default function CDPSheet({ data }: { data: any }) {
  if (!data) return null;

  //  const handleDownload = async () => {
  //   // 이 함수 안에서 동적으로 import 합니다.
  //   const { generateCdpPptx } = await import("@/lib/pptxGenerator");
  //   generateCdpPptx(data);
  // };


  // 섹션 제목과 내용을 감싸는 작은 컴포넌트
  const Section = ({ title, children }: { title: string, children: React.ReactNode }) => (
    <div className="border border-gray-300 dark:border-gray-600">
      <h3 className="bg-gray-200 dark:bg-gray-700 p-2 font-bold text-center text-sm">{title}</h3>
      <div className="p-3 text-xs space-y-2">{children}</div>
    </div>
  );


  // 섹션 내부의 소제목과 내용을 감싸는 작은 컴포넌트
  const SubSection = ({ title, children, titleBg = "bg-gray-100 dark:bg-gray-600" }: { title: string, children: React.ReactNode, titleBg?: string }) => (
    <div>
      <h4 className={`font-semibold p-1.5 text-xs ${titleBg}`}>{title}</h4>
      <div className="p-2 space-y-1">{children}</div>
    </div>
  );



  return (
    <div className="p-4 border rounded-lg bg-white dark:bg-gray-800 shadow-lg text-gray-900 dark:text-gray-100">
          
      {/* ✅ 3. 최상단에 제목과 다운로드 버튼을 배치합니다. */}
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-center font-bold text-lg">{data.title}</h2>
         <PPTXButton data={data} />
      </div>
      <div className="text-center p-3 mb-4 bg-red-100 dark:bg-red-900/50 border border-red-300 dark:border-red-700 rounded-md">
        <p className="font-bold text-red-800 dark:text-red-200">{data.customer_delight_goal}</p>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {/* CX Column */}
        <div className="col-span-1 space-y-2">
          <Section title="CX 기획 / 구현">
            <SubSection title="타겟 고객 정의">
              <p>{data.cx?.target_definition?.description}</p>
              <p className="font-bold italic mt-1">"{data.cx?.target_definition?.quote}"</p>
            </SubSection>
            <SubSection title={data.cx?.core_experience?.title}>
              <p><strong>Care:</strong> {data.cx?.core_experience?.care}</p>
              <div>
                <strong>Customization:</strong>
                <ul className="list-disc list-inside ml-2">
                  {data.cx?.core_experience?.customization.map((item: string, i: number) => <li key={i}>{item}</li>)}
                </ul>
              </div>
              <p><strong>Servitization:</strong> {data.cx?.core_experience?.servitization}</p>
            </SubSection>
          </Section>
        </div>

        {/* Performance Column */}
        <div className="col-span-1 space-y-2">
          <Section title="Performance">
            <SubSection title="컨셉(경험구체화)">
              <p><strong>Find:</strong> {data.performance?.concept?.find}</p>
              <div>
                <strong>Unique:</strong>
                <ul className="list-disc list-inside ml-2">
                  {data.performance?.concept?.unique.map((item: string, i: number) => <li key={i}>{item}</li>)}
                </ul>
              </div>
            </SubSection>
            {/* 다른 Performance 항목들이 있다면 여기에 추가될 수 있습니다. */}
          </Section>
        </div>

        {/* DX Column */}
        <div className="col-span-1 space-y-2">
          <Section title="DX">
            <SubSection title={data.dx?.trigger?.title} titleBg="bg-blue-100 dark:bg-blue-900/50">
                <ul className="list-disc list-inside">
                  {data.dx?.trigger?.items.map((item: string, i: number) => <li key={i}>{item}</li>)}
                </ul>
            </SubSection>
            
            {/***************************************************************/}
            {/* 여기가 수정된 부분입니다. */}
            <SubSection title={data.dx?.accelerator?.title} titleBg="bg-green-100 dark:bg-green-900/50">
              <div>
                <h5 className="font-medium text-xs mb-1">UP-Contents 서비스</h5>
                <ul className="list-disc list-inside ml-2">
                  {data.dx?.accelerator?.up_contents_service?.map((item: string, i: number) => <li key={i}>{item}</li>)}
                </ul>
              </div>
              <div className="mt-2">
                <h5 className="font-medium text-xs mb-1">Data기반 경험</h5>
                 <ul className="list-disc list-inside ml-2">
                  {data.dx?.accelerator?.data_driven_experience?.map((item: string, i: number) => <li key={i}>{item}</li>)}
                </ul>
              </div>
            </SubSection>
            {/***************************************************************/}

            <SubSection title={data.dx?.tracker?.title} titleBg="bg-yellow-100 dark:bg-yellow-900/50">
               <ul className="list-disc list-inside">
                  {data.dx?.tracker?.items.map((item: string, i: number) => <li key={i}>{item}</li>)}
                </ul>
            </SubSection>
          </Section>
        </div>
      </div>
    </div>
  );
}