import Link from "next/link";

export const metadata = {
  title: "Docs documentation — Mini-LLM TH v1.0",
  description: "Docs Mini-LLM TH v1.0",
};

export default function DocsPage() {
  return (
    <main className="page">
      <section className="hero home-hero">
        <p className="eyebrow">Docs • Documentation</p>
        <h1>Mini-LLM TH <span className="version">v1.0</span></h1>
        <p className="lede">Docs Mini-LLM TH v1.0 แบบ next-token prediction บนเบราเซอร์</p>
      </section>

      <section className="card" style={{ marginTop: 14 }}>
        <h2>โครงสร้างโปรเจกต์</h2>
        <table className="loss"><tbody>
          <tr><th>เส้นทาง</th><td><code>/</code> — หน้าแรก โปรไฟล์บริการ</td></tr>
          <tr><th>เส้นทาง</th><td><code>/chat</code> — หน้า Chatbot</td></tr>
          <tr><th>เส้นทาง</th><td><code>/docs</code> — หน้าDocsนี้</td></tr>
          <tr><th>เส้นทาง</th><td><code>/graphs/[modelId]</code> — กราฟการฝึกต่อโมเดล</td></tr>
        </tbody></table>

        <h2 style={{ marginTop: 16 }}>ไฟล์สำคัญ</h2>
        <table className="loss"><tbody>
          <tr><th><code>public/data/train.txt</code></th><td>ข้อมูลฝึก JSON (Q&amp;A)</td></tr>
          <tr><th><code>public/data/test.txt</code></th><td>ข้อมูลทดสอบ JSON (Q&amp;A)</td></tr>
          <tr><th><code>lib/tokenizer.js</code></th><td>Char / BPE tokenizer</td></tr>
          <tr><th><code>lib/dataset.js</code></th><td>load / clean / inspect / split dataset</td></tr>
          <tr><th><code>lib/mini-llm.js</code></th><td>ML core: build / train / eval / generate / checkpoint</td></tr>
          <tr><th><code>lib/model-runtime.js</code></th><td>runtime กลาง Playground ↔ Chatbot</td></tr>
          <tr><th><code>app/components/ChatBot.js</code></th><td>Chatbot UI</td></tr>
        </tbody></table>
      </section>

      <section className="card" style={{ marginTop: 14 }}>
        <h2>รันโปรเจกต์</h2>
        <pre>{`npm install\nnpm run dev\n# เปิด http://localhost:3000`}</pre>
      </section>

      <div style={{ marginTop: 20 }}>
        <Link href="/" className="cta">กลับสู่หน้าแรก</Link>
        <Link href="/chat" className="cta primary" style={{ marginLeft: 10 }}>/chat</Link>
      </div>
    </main>
  );
}