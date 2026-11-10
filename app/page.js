import Link from "next/link";
import fs from "fs";
import path from "path";
import QRCodeLink from "./components/QRCodeLink";

export const metadata = {
  title: "โปรไฟล์บริการ — Mini-LLM TH v1.0",
  description: "ข้อมูลโปรไฟล์บริการ Mini-LLM TH v1.0 สำหรับเปิดให้บริการแชทบอทด้วยโมเดลภาษาขนาดเล็ก",
};

function getProfile() {
  try {
    const p = path.join(process.cwd(), "public", "data", "models.json");
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return { version: "v1.0", models: [] };
  }
}

function getSummary() {
  try {
    const p = path.join(process.cwd(), "public", "summary.json");
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return { models: [] };
  }
}

function checkpointExists(modelId) {
  try {
    const ckpt = path.join(process.cwd(), "thai_mini", "checkpoints", `${modelId}.pt`);
    return fs.existsSync(ckpt);
  } catch {
    return false;
  }
}

export default function Home() {
  const manifest = getProfile();
  const summary = getSummary();
  const models = manifest.models || [];
  const modelStats = {};
  (summary.models || []).forEach((m) => { modelStats[m.model_id] = m; });

  return (
    <main className="page">
      <section className="hero home-hero">
        <p className="eyebrow">ข้อมูลโปรไฟล์ • เปิดให้บริการ</p>
        <h1>Mini-LLM TH <span className="version">v1.0</span></h1>
        <p className="lede">
          บริการแชทบอทที่ใช้โมเดลภาษาขนาดเล็ก (Mini Language Model) แบบ next-token prediction.
          ลูกค้าสามารถเลือกโมเดลและเริ่มใช้งานได้ทันที โดยระบบเชื่อมต่อโมเดลและข้อมูลแยกตามโมเดลอัตโนมัติ.
        </p>
        <div className="row hero-actions">
          <Link href="/chat" className="cta primary">เริ่มใช้งาน Chatbot</Link>
          <Link href="/ocr" className="cta">OCR ตรวจสอบภาพ</Link>
          <a href="#models" className="cta">ดูโมเดลที่ให้บริการ</a>
        </div>
        <div className="hero-qr" aria-label="สแกน QR เพื่อเข้าถึง chatbot">
          <QRCodeLink size={200} />
        </div>
        <div className="status-pill" aria-label="service status">
          <span className="dot" aria-hidden="true"></span> สถานะบริการ: พร้อมใช้งาน
        </div>
      </section>

      <section className="shell home-grid" style={{ marginTop: 14 }}>
        <article className="card">
          <h2>โปรไฟล์บริการ</h2>
          <table className="loss"><tbody>
            <tr><th>ชื่อบริการ</th><td>Mini-LLM TH Chatbot</td></tr>
            <tr><th>เวอร์ชัน</th><td>v1.0</td></tr>
            <tr><th>รูปแบบ</th><td>ถาม–ตอบภาษาไทย/ข้อความ (next-token generation)</td></tr>
            <tr><th>การเชื่อมต่อ</th><td>เลือกโมเดล → เชื่อมต่ออัตโนมัติ</td></tr>
            <tr><th>ข้อมูลต่อโมเดล</th>
              <td><code>train.json</code>, <code>val.json</code>, <code>test.json</code> + checkpoint แยกกัน</td>
            </tr>
          </tbody></table>
        </article>

        <article className="card">
          <h2>วิธีใช้งานลูกค้า</h2>
          <ol className="steps">
            <li>กด <Link href="/chat">เริ่มใช้งาน Chatbot</Link></li>
            <li>เลือกโมเดลที่ต้องการจากรายการ</li>
            <li>รอระบบเชื่อมต่อโมเดลอัตโนมัติ แล้วพิมพ์คำถาม</li>
          </ol>
        </article>
      </section>

      <section className="card" id="models" style={{ marginTop: 14 }}>
        <h2>โมเดลที่ให้บริการ (ข้อมูลแยกตามโมเดล)</h2>
        <table className="loss"><thead><tr><th>โมเดล</th><th>Backend</th><th>สถานะ</th><th>Train Loss</th><th>Val Loss</th><th>Perplexity</th><th>Epochs</th><th>ดูกราฟ</th></tr></thead><tbody>
          {models.map((m) => {
            const stats = modelStats[m.id] || {};
            const ready = checkpointExists(m.id);
            return (
              <tr key={m.id}>
                <td>
                  <strong>{m.name || m.id}</strong>
                  <div className="meta">{m.id} • {m.backend || "browser"}</div>
                </td>
                <td>{m.backend || "-"}</td>
                <td>
                  {ready ? (
                    <span style={{ color: "#2e7d32", fontWeight: 600 }}>ฝึกแล้ว • พร้อมใช้งาน</span>
                  ) : (
                    <span style={{ color: "#d32f2f" }}>ยังไม่ได้ฝึก</span>
                  )}
                </td>
                <td>{stats.final_train_loss != null ? stats.final_train_loss.toFixed(4) : "-"}</td>
                <td>{stats.final_val_loss != null ? stats.final_val_loss.toFixed(4) : "-"}</td>
                <td>{stats.final_perplexity != null ? stats.final_perplexity.toFixed(2) : "-"}</td>
                <td>{stats.epochs != null ? stats.epochs : "-"}</td>
                <td>
                  {ready ? (
                    <Link href={`/graphs/${m.id}`} className="link-btn" style={{ fontSize: 12, padding: "4px 8px" }}>
                      ดูกราฟ
                    </Link>
                  ) : (
                    <span style={{ fontSize: 12, color: "#999" }}>—</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody></table>
      </section>

      <section className="card" id="graphs" style={{ marginTop: 14 }}>
        <h2>กราฟการฝึกโมเดล (Training Graphs)</h2>
        <p className="meta">แสดงเส้นความสูญเสีย (Loss) และความยากต่อการทำนาย (Perplexity) ตามจำนวนรอบการฝึก (Epoch)</p>
        <div className="shell" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 14 }}>
          {models.map((m) => {
            const stats = modelStats[m.id] || {};
            const ready = checkpointExists(m.id);
            return (
              <article className="card" key={m.id} style={{ padding: 12 }}>
                <h3 style={{ margin: "0 0 8px", fontSize: "1.1rem" }}>{m.name || m.id}</h3>
                {ready ? (
                  <>
                    <img
                      src={`/${m.id}_history.png`}
                      alt={`${m.id} training history`}
                      style={{ width: "100%", height: "auto", border: "1px solid var(--line)", borderRadius: 4, marginBottom: 8 }}
                    />
                    <div className="meta" style={{ fontSize: 12 }}>
                      Epochs: {stats.epochs || "-"} &nbsp;|&nbsp;
                      Train Loss: {stats.final_train_loss != null ? stats.final_train_loss.toFixed(4) : "-"} &nbsp;|&nbsp;
                      Val Loss: {stats.final_val_loss != null ? stats.final_val_loss.toFixed(4) : "-"} &nbsp;|&nbsp;
                      Perplexity: {stats.final_perplexity != null ? stats.final_perplexity.toFixed(2) : "-"}
                    </div>
                  </>
                ) : (
                  <p style={{ color: "#999", fontSize: 13, textAlign: "center", padding: "20px 0" }}>
                    ยังไม่มีข้อมูลการฝึกสำหรับโมเดลนี้
                  </p>
                )}
              </article>
            );
          })}
        </div>
        <div style={{ marginTop: 10 }}>
          <img
            src="/model_comparison.png"
            alt="Model comparison"
            style={{ width: "100%", maxWidth: 700, height: "auto", border: "1px solid var(--line)", borderRadius: 4, display: "block", margin: "0 auto" }}
          />
          <p className="meta" style={{ textAlign: "center", marginTop: 6, fontSize: 12 }}>
            เปรียบเทียบประสิทธิภาพโมเดลทุกรุ่น
          </p>
        </div>
      </section>
    </main>
  );
}
