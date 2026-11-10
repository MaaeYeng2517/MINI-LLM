import Link from "next/link";

export const metadata = {
  title: "Playground — Mini-LLM TH v1.0",
  description: " Playground Mini-LLM TH v1.0 สำหรับสร้าง/ฝึก/resume โมเดล",
};

export default function PlaygroundPage() {
  return (
    <main className="page">
      <section className="hero home-hero">
        <p className="eyebrow">Playground</p>
        <h1>Mini-LLM TH <span className="version">v1.0</span></h1>
        <p className="lede">
          สถานที่สำหรับดูข้อมูล Tokenizer สร้าง/ฝึก/Resume โมเดล Loss/Metrics Checkpoint Generation Evaluation
        </p>
      </section>

      <section className="card" style={{ marginTop: 14 }}>
        <h2>ฟี feature Playground</h2>
        <p className="meta">
          ฟี feature Playground ในรุ่น v1.0 ปัจจุบันแสดงข้อมูลโมเดลและเชื่อมต่อกับหน้า Chatbot
          หากต้องการฝึกโมเดลใหม่ ให้ใช้ <code>thai_mini/train.py</code> (PyTorch)
          หรือฝึกผ่านเบราเซอร์จากหน้า Chatbot (Quick train)
        </p>
        <table className="loss" style={{ marginTop: 12 }}><tbody>
          <tr><th>Dataset</th><td><code>public/data/train.txt</code> / <code>test.txt</code></td></tr>
          <tr><th>Tokenizer</th><td>Char + BPE/Subword → Token ID + preview</td></tr>
          <tr><th>Training</th><td>Next Token Prediction (Embedding + LSTM + Dense softmax, Adam)</td></tr>
          <tr><th>Loss/Metrics</th><td>loss, val_loss, accuracy, perplexity, history</td></tr>
          <tr><th>Checkpoint</th><td>Save/Load <code>mini-llm-v1.0</code> ผ่าน TensorFlow.js LocalStorage</td></tr>
          <tr><th>Generation</th><td>สร้างข้อความจาก prompt (temperature sampling)</td></tr>
          <tr><th>Evaluation</th><td>ทดสอบ loss/perplexity/accuracy กับข้อมูลที่ไม่เคยฝึก</td></tr>
        </tbody></table>
      </section>

      <section className="card" style={{ marginTop: 14 }}>
        <h2>ข้อมูลต่อโมเดล</h2>
        <p className="meta">
          ข้อมูลฝึก ตรวจสอบ ทดสอบ แยกต่างหากในรูปแบบ JSON (UTF-8) รองรับทั้งไฟล์
          <code>.txt</code> และ <code>.json</code>
        </p>
        <table className="loss" style={{ marginTop: 12 }}><tbody>
          <tr><th><code>public/data/train.txt</code></th><td>ข้อมูลฝึก JSON (Q&amp;A) — 99 pairs</td></tr>
          <tr><th><code>public/data/models/default/train.json</code></th><td>ข้อมูลฝึก Q&amp;A ของ default model</td></tr>
          <tr><th><code>public/data/models/default/val.json</code></th><td>ข้อมูลตรวจสอบของ default model</td></tr>
          <tr><th><code>public/data/models/default/test.json</code></th><td>ข้อมูลทดสอบของ default model</td></tr>
        </tbody></table>
      </section>

      <div style={{ marginTop: 20 }}>
        <Link href="/" className="cta">กลับสู่หน้าแรก</Link>
        <Link href="/chat" className="cta primary" style={{ marginLeft: 10 }}>ไปที่ Chatbot</Link>
        <Link href="/docs" className="cta" style={{ marginLeft: 10 }}>ไปที่ Docs</Link>
      </div>
    </main>
  );
}