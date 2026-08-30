import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "YouCanDoIt — AI 면접 코치",
  description:
    "자기소개서로 예상 질문을 만들고, 답변하는 동안 시선·자세·표정을 분석해 코칭 리포트를 만듭니다.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
