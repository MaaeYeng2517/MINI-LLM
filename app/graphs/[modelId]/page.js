import Link from "next/link";
import fs from "fs";
import path from "path";

export const generateStaticParams = async () => {
  try {
    const p = path.join(process.cwd(), "public", "summary.json");
    const summary = JSON.parse(fs.readFileSync(p, "utf8"));
    return (summary.models || []).map((m) => ({ modelId: m.model_id }));
  } catch {
    return [];
  }
};

export const metadata = async ({ modelId }) => {
  return {
    title: `กราฟการฝึก — ${modelId} — Mini-LLM TH v1.0`,
    description: `กราฟการฝึกโมเดล ${modelId} แสดงความสูญเสียและค่าความยากต่อการทำนาย`,
  };
};

function getSummary() {
  try {
    const p = path.join(process.cwd(), "public", "summary.json");
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return { models: [] };
  }
}

function getModels() {
  try {
    const p = path.join(process.cwd(), "public", "data", "models.json");
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return { models: [] };
  }
}

export default function GraphPage({ params }) {
  const { modelId } = params;
  const summary = getSummary();
  const manifest = getModels();
  const stats = (summary.models || []).find((m) => m.model_id === modelId) || {};
  const modelInfo = (manifest.models || []).find((m) => m.id === modelId) || {};
  const imagePath = `/${modelId}_history.png`;

  return (
    <main className="page chat-page">
      <header className="hero chat-hero">
        <p className="eyebrow"><Link href="/" className="classic-link">← กลับสู่หน้าแรก</Link></p>
        <h1>กราฟการฝึก: {modelInfo.name || modelId}</h1>
        <p className="lede">
          แผนภูมิแสดงความสูญเสีย (Loss) และค่าความยากต่อการทำนาย (Perplexity) ของโมเดล
          {modelInfo.name || modelId} ตลอดกระบวนการฝึก
        </p>
      </header>

      <section className="card" style={{ marginTop: 14 }}>
        <h2 style={{ margin: "0 0 10px", fontSize: "1.3rem" }}>สรุปผลการฝึก</h2>
        <table className="loss"><tbody>
          <tr><th>รหัสโมเดล</th><td>{modelId}</td></tr>
          <tr><th>ชื่อโมเดล</th><td>{modelInfo.name || "-"}</td></tr>
          <tr><th>Backend</th><td>{modelInfo.backend || "-"}</td></tr>
          <tr><th>จำนวนรอบฝึก (Epochs)</th><td>{stats.epochs || "-"}</td></tr>
          <tr><th>Train Loss (ท้ายสุด)</th><td>{stats.final_train_loss != null ? stats.final_train_loss.toFixed(4) : "-"}</td></tr>
          <tr><th>Val Loss (ท้ายสุด)</th><td>{stats.final_val_loss != null ? stats.final_val_loss.toFixed(4) : "-"}</td></tr>
          <tr><th>Perplexity (ท้ายสุด)</th><td>{stats.final_perplexity != null ? stats.final_perplexity.toFixed(2) : "-"}</td></tr>
          <tr><th>ข้อมูลฝึก</th><td><code>{modelInfo.train || "-"}</code></td></tr>
          <tr><th>ข้อมูลตรวจสอบ</th><td><code>{modelInfo.val || "-"}</code></td></tr>
          <tr><th>ข้อมูลทดสอบ</th><td><code>{modelInfo.test || "-"}</code></td></tr>
          <tr><th>Checkpoint</th><td><code>{modelInfo.checkpoint || "-"}</code></td></tr>
        </tbody></table>
      </section>

      <section className="card" style={{ marginTop: 14 }}>
        <h2 style={{ margin: "0 0 10px", fontSize: "1.3rem" }}>กราฟความสูญเสียและ Perplexity</h2>
        <img
          src={imagePath}
          alt={`${modelId} training history`}
          style={{ width: "100%", maxWidth: 800, height: "auto", border: "1px solid var(--line)", borderRadius: 4, display: "block", margin: "0 auto" }}
        />
        <p className="meta" style={{ textAlign: "center", marginTop: 8, fontSize: 12 }}>
          เส้นสีดำ = ความสูญเสียชุดฝึก (Train Loss) &nbsp;•&nbsp; เส้นสีแดง = ความสูญเสียชุดตรวจสอบ (Val Loss)
        </p>
      </section>

      <section className="card" style={{ marginTop: 14 }}>
        <h2 style={{ margin: "0 0 10px", fontSize: "1.3rem" }}>ข้อมูลต่อโมเดล</h2>
        <p className="meta">
          โมเดลนี้ใช้ข้อมูลฝึกแยกต่างหากในรูปแบบ JSON (UTF-8) รองรับทั้งไฟล์
          <code>.txt</code> และ <code>.json</code> โดยระบบจะแปลงคำถาม–คำตอบ
          ให้อยู่ในรูปแบบพิเศษก่อนส่งให้โมเดลฝึกและสร้างข้อความ
        </p>
      </section>

      <div style={{ marginTop: 20, textAlign: "center" }}>
        <Link href="/" className="cta">กลับสู่หน้าแรก</Link>
        <Link href="/chat" className="cta primary" style={{ marginLeft: 10 }}>ทดสอบแชทบอท</Link>
      </div>
    </main>
  );
}
