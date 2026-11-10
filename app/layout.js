import "./globals.css";
import Link from "next/link";

export const metadata = {
  title: "Mini-LLM TH v1.0 — โปรไฟล์บริการ",
  description: "โปรไฟล์บริการ Mini-LLM TH v1.0 และหน้า Chatbot สำหรับลูกค้า",
};

export default function RootLayout({ children }) {
  return (
    <html lang="th">
      <head>
        <meta charSet="utf-8" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Kanit:wght@400;600;700&family=Noto+Sans+Thai:wght@400;600;700&family=Sarabun:wght@400;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <nav className="topnav">
          <Link href="/" className="brand">Mini-LLM TH <span className="nav-ver">v1.0</span></Link>
          <div className="topnav-links">
            <Link href="/">โปรไฟล์บริการ</Link>
            <Link href="/chat">Chatbot</Link>
            <Link href="/playground">Playground</Link>
            <Link href="/ocr">OCR ตรวจสอบภาพ</Link>
            <Link href="/plant-disease">🌿 ตรวจโรคพืช</Link>
            <Link href="/docs">Docs</Link>
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}