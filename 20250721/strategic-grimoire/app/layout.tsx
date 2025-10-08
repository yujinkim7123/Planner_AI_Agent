// app/layout.tsx

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "전략 기획가 마법서",
  description: "데이터 기반 전략 기획 AI 어시스턴트",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko" className="dark">
      <body className={`${inter.className} bg-gray-50 dark:bg-gray-900`}>
        {children}
      </body>
    </html>
  );
}