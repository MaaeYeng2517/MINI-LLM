import ChatBot from "../components/ChatBot";
import Link from "next/link";
import QRCodeLink from "../components/QRCodeLink";

export const metadata = {
  title: "Mini-LLM V1.0 — Chatbot",
  description: "บริการแชทบอท Mini-LLM V1.0 เลือกโมเดลและใช้งานได้ทันที",
};

export default function ChatPage() {
  return (
    <main className="page chat-page">
      <header className="hero chat-hero">
        <p className="eyebrow"><Link href="/" className="classic-link">← โปรไฟล์บริการ</Link></p>
        <h1>Mini-LLM V1.0</h1>
        <p className="lede">แชทบอทอัจฉริยะ — เลือกโมเดลและเริ่มสนทนาได้ทันที</p>
        <div className="hero-qr">
          <QRCodeLink size={180} label="สแกนเพื่อเปิด chatbot นี้" sublabel="ใช้กล้องือก phone" />
        </div>
      </header>
      <ChatBot />
    </main>
  );
}