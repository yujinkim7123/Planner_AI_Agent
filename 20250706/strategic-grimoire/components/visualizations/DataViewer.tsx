import React, { useState, useEffect } from 'react';
import {
    Card,
    Title,
    Badge,
    Table,
    TableHead,
    TableRow,
    TableHeaderCell,
    TableBody,
    TableCell,
    Text
} from '@tremor/react';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/20/solid';

interface DataViewerProps {
    retrievedData: any;
    productData: any;
    sensorData: any;
    columnsProduct: any;
    productType: string | null;
}

export default function DataViewer({ retrievedData, productData, sensorData, columnsProduct, productType }: DataViewerProps) {
    // Prepare data
    const webResults = retrievedData?.web_results || [];
    const productResultsRaw = Array.isArray(productData) ? productData : [];
    const productResults = productResultsRaw.map((item: any) => ({
        content: item.page_content || item.content || item.text || '',
        fullContent: item.content || item.page_content || item.text || '',
        source: item.metadata?.source || item.source || item.source_url || 'N/A'
    }));

    const sensorColumns = Array.isArray(sensorData)
        ? (sensorData as any[]).map(d => ({
            Sensor: d.Sensor,
            Description: d.Description || '설명 없음'
        })).filter(d => d.Sensor)
        : [];

    const deviceColumns = Array.isArray(columnsProduct)
        ? (columnsProduct as string[]).map(col => ({
            Name: col,
            Description: '설명 없음'
        }))
        : Object.keys(columnsProduct || {}).map(key => ({
            Name: key,
            Description: (columnsProduct[key]?.description || '설명 없음')
        }));

    // Tabs config
    const tabs = [
        { key: 'web', label: '웹(VOC)', count: webResults.length, data: webResults },
        { key: 'product', label: '제품 문서', count: productResults.length, data: productResults },
        { key: 'device', label: '디바이스 컬럼', count: deviceColumns.length, data: deviceColumns },
        { key: 'sensor', label: '센서 컬럼', count: sensorColumns.length, data: sensorColumns }
    ].filter(tab => tab.count > 0);

    const [selectedTab, setSelectedTab] = useState(tabs[0]?.key || '');
    const [isTableVisible, setIsTableVisible] = useState(false);

    const MAX_TABLE_HEIGHT = '160px';

    if (!tabs.length) return null;

    const renderTableContent = () => {
        switch (selectedTab) {
            case 'web':
                return (
                    <Table className="w-full text-xs border-collapse rounded-md overflow-hidden">
                        <TableHead>
                            <TableRow className="bg-gray-100 dark:bg-gray-700">
                                <TableHeaderCell className="py-2 px-3 text-left font-semibold text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-600">요약</TableHeaderCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {webResults.map((item: any, idx: number) => (
                                <TableRow key={idx} className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750">
                                    <TableCell className="py-1.5 px-3 text-gray-600 dark:text-gray-400">{item.text}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                );
            case 'product':
                return (
                    <Table className="w-full text-xs border-collapse rounded-md overflow-hidden">
                        <TableHead>
                            <TableRow className="bg-gray-100 dark:bg-gray-700">
                                <TableHeaderCell className="py-2 px-3 text-left font-semibold text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-600">문서 요약</TableHeaderCell>
                                <TableHeaderCell className="py-2 px-3 text-left font-semibold text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-600">내용</TableHeaderCell>
                                <TableHeaderCell className="py-2 px-3 text-left font-semibold text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-600">출처</TableHeaderCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {productResults.map((item: any, idx: number) => (
                                <TableRow key={idx} className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750">
                                    <TableCell className="py-1.5 px-3 truncate max-w-[100px] text-gray-800 dark:text-gray-200">
                                        {item.content.length > 80 ? item.content.substring(0, 80) + '...' : item.content}
                                    </TableCell>
                                    <TableCell className="py-1.5 px-3 max-w-[150px] text-gray-800 dark:text-gray-200 overflow-hidden text-ellipsis whitespace-nowrap">
                                        {item.fullContent.length > 120 ? item.fullContent.substring(0, 120) + '...' : item.fullContent}
                                    </TableCell>
                                    <TableCell className="py-1.5 px-3 text-gray-600 dark:text-gray-400 max-w-[80px] truncate">{item.source}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                );
            case 'device':
                return (
                    <Table className="w-full text-xs border-collapse rounded-md overflow-hidden">
                        <TableHead>
                            <TableRow className="bg-gray-100 dark:bg-gray-700">
                                <TableHeaderCell className="py-2 px-3 text-left font-semibold text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-600">데이터 컬럼</TableHeaderCell>
                                <TableHeaderCell className="py-2 px-3 text-left font-semibold text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-600">설명</TableHeaderCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {deviceColumns.map((col, i) => (
                                <TableRow key={i} className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750">
                                    <TableCell className="py-1.5 px-3 text-gray-800 dark:text-gray-200">{col.Name}</TableCell>
                                    <TableCell className="py-1.5 px-3 text-gray-600 dark:text-gray-400 max-w-sm truncate">{col.Description}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                );
            case 'sensor':
                return (
                    <Table className="w-full text-xs border-collapse rounded-md overflow-hidden">
                        <TableHead>
                            <TableRow className="bg-gray-100 dark:bg-gray-700">
                                <TableHeaderCell className="py-2 px-3 text-left font-semibold text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-600">센서 컬럼</TableHeaderCell>
                                <TableHeaderCell className="py-2 px-3 text-left font-semibold text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-600">설명</TableHeaderCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {sensorColumns.map((col, i) => (
                                <TableRow key={i} className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750">
                                    <TableCell className="py-1.5 px-3 text-gray-800 dark:text-gray-200">{col.Sensor}</TableCell>
                                    <TableCell className="py-1.5 px-3 text-gray-600 dark:text-gray-400 max-w-sm truncate">{col.Description}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                );
            default:
                return null;
        }
    };

    return (
        <Card className="p-3 mb-4 bg-white dark:bg-gray-800 shadow-md border border-gray-200 dark:border-gray-700 rounded-lg max-w-lg mx-auto">
            <div className="flex justify-between items-center mb-3">
                <Title className="text-xl font-bold text-gray-900 dark:text-gray-100">데이터 검색 결과</Title>
                {/* 더보기 버튼을 Title 옆으로 이동 */}
                <button
                    onClick={() => setIsTableVisible(!isTableVisible)}
                    className="flex items-center justify-center py-1.5 px-3 bg-blue-500 hover:bg-blue-600 text-white font-semibold rounded-md transition-colors duration-200 ease-in-out shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-75 text-sm"
                >
                    {isTableVisible ? (
                        <>
                            <ChevronUpIcon className="h-4 w-4 mr-1.5" />
                            숨기기
                        </>
                    ) : (
                        <>
                            <ChevronDownIcon className="h-4 w-4 mr-1.5" />
                            상세 보기
                        </>
                    )}
                </button>
            </div>

            {/* Product Type Badge (가독성 향상을 위해 text-white 추가) */}
            {productType && <Badge color="blue" size="sm" className="py-0.5 px-2 rounded-full text-xs font-medium mb-3 text-white dark:text-gray-900">{productType}</Badge>}


            {/* Top navigation tabs */}
            <nav className="flex space-x-1.5 overflow-x-auto border-b-2 border-blue-50 dark:border-blue-950 pb-1.5 mb-3">
                {tabs.map(tab => (
                    <button
                        key={tab.key}
                        className={`flex items-center px-3 py-1.5 text-sm font-medium rounded-t-md transition-colors duration-200 ease-in-out focus:outline-none
                            ${selectedTab === tab.key
                                ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-200 border-b-2 border-blue-500 dark:border-blue-400'
                                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700'}`}
                        onClick={() => setSelectedTab(tab.key)}
                    >
                        {tab.label} <Badge color="blue" size="xs" className="ml-1.5 bg-blue-200 dark:bg-blue-800 text-blue-800 dark:text-blue-100">{tab.count}</Badge>
                    </button>
                ))}
            </nav>

            {/* Content Area - 조건부 렌더링 및 스크롤 */}
            {isTableVisible && (
                <div className="overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-md shadow-inner" style={{ maxHeight: MAX_TABLE_HEIGHT }}>
                    {renderTableContent()}
                    {tabs.find(tab => tab.key === selectedTab)?.data.length === 0 && (
                        <div className="p-3 text-center text-gray-500 dark:text-gray-400 text-sm">
                            해당 탭에 표시할 데이터가 없습니다.
                        </div>
                    )}
                </div>
            )}
        </Card>
    );
}