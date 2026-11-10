"use client";

import { useEffect, useRef, useState } from "react";
import { ensureTf, loadRuntimeModel, quickTrainRuntime } from "../../lib/model-runtime";
import { checkpointForModel, getModel, loadModelManifest } from "../../lib/model-registry";
import ModelSelector from "./ModelSelector";

function cleanReply(text) {
  let s = String(text || "");
  s = s.replace(/\uFFFD/g, "").replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, "");
  s = s.replace(/^[\s:：\-–—]+/, "");
  const cut = s.search(/\n\s*(คำถาม|ผู้ใช้|User|คำตอบ|ผู้ช่วย|Bot)\s*[:：]/);
  if (cut > 0) s = s.slice(0, cut);
  // If model emits mixed-script garbage (Latin/digit noise inside Thai), keep readable Thai+common chars.
  const thaiCount = (s.match(/[\u0E00-\u0E7F]/g) || []).length;
  if (thaiCount >= 2) {
    const cleaned = s
      .replace(/[^ \n\t\u0E00-\u0E7F0-9.,!?%:;()'"“”‘’\-–—…ฯๆ฿\/]/g, " ")
      .replace(/([A-Za-z])(?=[\u0E00-\u0E7F])/g, " ")
      .replace(/([\u0E00-\u0E7F])(?=[A-Za-z])/g, " ")
      .replace(/[ \t]{2,}/g, " ")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
    if (cleaned.replace(/\s/g, "").length >= 2) s = cleaned;
  }
  const lines = s.split(/\r?\n/).map((x) => x.replace(/[ \t]{2,}/g, " ").trim()).filter(Boolean);
  s = lines.length ? lines.slice(0, 4).join("\n") : s.trim();
  if (s.length > 260) s = s.slice(0, 260).replace(/[^\n.!?…ฯๆก-๙0-9 )\]}'"”’]+$/u, "").trim();
  return s.trim();
}

function buildQaPrompt(messages, text) {
  const history = messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .slice(-2)
    .map((m) => (m.role === "user" ? `คำถาม: ${m.text}` : `คำตอบ: ${m.text}`))
    .join("\n");
  return (history ? history + "\n\n" : "") + `คำถาม: ${text}\nคำตอบ:`;
}

export default function ChatBot() {
  const modelRef = useRef(null);
  const listRef = useRef(null);
  const reqRef = useRef(0);
  const [models, setModels] = useState([]);
  const [modelId, setModelId] = useState("");
  const [modelReady, setModelReady] = useState(false);
  const [status, setStatus] = useState("กำลังค้นหาโมเดลที่พร้อมใช้งาน...");
  const [messages, setMessages] = useState([{ role: "assistant", text: "สวัสดีครับ ยินดีต้อนรับสู่ Mini-LLM V1.0 ระบบกำลังค้นหาและเชื่อมต่อโมเดลให้อัตโนมัติ กรุณารอสักครู่..." }]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState("search"); // search | load | train | generate
  const [temperature] = useState(0.6);
  const [maxTokens] = useState(90);
  const [trainEpoch, setTrainEpoch] = useState(0);
  const [trainLoss, setTrainLoss] = useState(null);

  function push(role, text) { setMessages((m) => [...m, { role, text }]); }
  const activeModel = getModel(models, modelId);
  const isPy = activeModel?.backend === "pytorch";

  useEffect(() => {
    (async () => {
      setBusy("search");
      setStatus("กำลังค้นหาโมเดลที่พร้อมใช้งาน...");
      try {
        const manifest = await loadModelManifest();
        const ms = manifest.models || [];
        setModels(ms);
        if (!ms.length) {
          setStatus("ไม่พบโมเดลที่พร้อมใช้งาน");
          setMessages([{ role: "assistant", text: "ไม่พบรายการโมเดล กรุณาตรวจสอบข้อมูลระบบ" }]);
          setBusy("");
          return;
        }
        const preferred = (manifest.active && ms.find((m) => m.id === manifest.active)) ? manifest.active : (ms.find((m) => m.id === "thai-v1-opto")?.id || ms.find((m) => m.backend === "pytorch")?.id || ms[0]?.id || "");
        setStatus(`พบ ${ms.length} โมเดล — กำลังเชื่อมต่อโมเดล "${(ms.find(m=>m.id===preferred)||{}).name || preferred}"...`);
        setModelId(preferred);
      } catch (err) {
        setStatus(`ค้นหาโมเดลไม่สำเร็จ: ${err.message}`);
        push("system", `เกิดข้อผิดพลาด: ${err.message}`);
        setBusy("");
      }
    })();
  }, []);

  useEffect(() => {
    if (!modelId || !models.length) return;
    const req = ++reqRef.current;
    let mounted = true;
    const m = getModel(models, modelId);
    (async () => {
      setModelReady(false); setTrainEpoch(0); setTrainLoss(null);
      const mname = m?.name || modelId;
      setBusy("load");
      setStatus(`กำลังค้นหาโมเดล "${mname}"...`);
      try {
        if (m?.backend === "pytorch") {
          setStatus(`กำลังเชื่อมต่อโมเดล "${mname}" (PyTorch)...`);
          const res = await fetch("/api/py-generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: "คำถาม: ping\nคำตอบ:", max_new_tokens: 1, temperature: 0.7, modelId }) });
          const data = await res.json().catch(() => ({}));
          if (!mounted || req !== reqRef.current) return;
          if (data.ok) {
            setModelReady(true);
            setStatus(`พร้อมใช้งาน: ${mname}`);
            setMessages([{ role: "assistant", text: `สวัสดีครับ ผม Mini-LLM V1.0 (โมเดล: ${mname}) พร้อมช่วยตอบคำถามแล้วครับ` }]);
          } else {
            setStatus(`ไม่สามารถโหลดโมเดล "${mname}" ได้`);
            push("system", `เชื่อมต่อ ${modelId} ไม่สำเร็จ: ${data.error || "unknown"}`);
          }
        } else {
          setStatus(`กำลังเตรียมระบบสำหรับโมเดล "${mname}"...`);
          await ensureTf();
          if (!mounted || req !== reqRef.current) return;
          const ckpt = checkpointForModel(m || modelId);
          const has = !!window.localStorage.getItem(`tensorflowjs_models/${ckpt}/info`);
          if (has) {
            setBusy("load");
            setStatus(`กำลังโหลดโมเดล "${mname}"...`);
            const { llm } = await loadRuntimeModel(ckpt, modelId);
            if (!mounted || req !== reqRef.current) return;
            modelRef.current = llm; setModelReady(true);
            setStatus(`พร้อมใช้งาน: ${mname}`);
            setMessages([{ role: "assistant", text: `สวัสดีครับ ผม Mini-LLM V1.0 (โมเดล: ${mname}) พร้อมช่วยตอบคำถามแล้วครับ` }]);
          } else {
            setBusy("train");
            const ds = m?.train || "/data/models/default/train.json";
            setStatus(`กำลังค้นหาและเตรียมโมเดล "${mname}" จากข้อมูลฝึก (ครั้งแรก)...`);
            setMessages([{ role: "assistant", text: `กำลังเตรียมโมเดล "${mname}" ให้พร้อมใช้งาน กรุณารอสักครู่...` }]);
            const { llm } = await quickTrainRuntime({ modelId, checkpoint: ckpt, datasetPath: ds, epochs: 45, earlyStopPatience: 6, config: { seqLen: 64, embedDim: 128, lstmUnits: 160, learningRate: 0.006, tokenizerType: "char", dropout: 0.12, batchSize: 24, valSplit: 0.1 } }, (p) => {
              if (mounted && req === reqRef.current) {
                setTrainEpoch(p.epoch);
                setTrainLoss(p.loss);
                setStatus(`กำลังโหลด/เตรียมโมเดล "${mname}"... (epoch ${p.epoch}, loss ${typeof p.loss === "number" ? p.loss.toFixed(4) : "-"})`);
              }
            });
            if (!mounted || req !== reqRef.current) return;
            modelRef.current = llm; setModelReady(true);
            setStatus(`พร้อมใช้งาน: ${mname}`);
            setMessages([{ role: "assistant", text: `สวัสดีครับ ผม Mini-LLM V1.0 (โมเดล: ${mname}) พร้อมช่วยตอบคำถามแล้วครับ` }]);
          }
        }
      } catch (err) {
        if (mounted) { setStatus(`เชื่อมต่อไม่สำเร็จ: ${err.message}`); push("system", `Error: ${err.message}`); }
      } finally { if (mounted && req === reqRef.current) setBusy(""); }
    })();
    return () => { mounted = false; };
  }, [modelId, models]);

  useEffect(() => { if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight; }, [messages, busy, trainEpoch, status]);

  async function handleSend(e) {
    e?.preventDefault();
    const text = input.trim();
    if (!text || !modelReady || busy) return;
    setInput(""); push("user", text); setBusy("generate"); setStatus("กำลังตอบ...");
    try {
      const prompt = buildQaPrompt(messages, text);
      let reply = "";
      if (isPy) {
        const res = await fetch("/api/py-generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt, max_new_tokens: Number(maxTokens), temperature: Number(temperature), modelId }) });
        const data = await res.json().catch(() => ({}));
        if (!data.ok) throw new Error(data.error || "Inference failed");
        reply = cleanReply(data.output || "");
      } else {
        if (!modelRef.current?.ready) throw new Error("Model not loaded");
        const full = await modelRef.current.generate(prompt, { maxTokens: Number(maxTokens), temperature: Number(temperature), repetitionPenalty: 1.2, stopMarkers: ["\n\nคำถาม:", "\nคำถาม:", "\nผู้ใช้:"] });
        reply = cleanReply(full.startsWith(prompt) ? full.slice(prompt.length) : full);
      }
      // keep Thai display readable; fallback if model output is too short/noisy
      const thaiVisible = (reply.match(/[\u0E00-\u0E7F]/g) || []).length;
      const compact = reply.replace(/\s+/g, "");
      const chars = Array.from(compact);
      const uniq = new Set(chars).size;
      const tooShort = compact.length < 2;
      const repetitive = compact.length >= 6 && uniq <= 3;
      const lowDiversity = compact.length >= 12 && (uniq / Math.max(chars.length, 1)) < 0.22;
      const mostlySame = compact.length >= 8 && chars.length > 0 && chars.filter(c => c === chars[0]).length / chars.length > 0.55;
      const maxRun = (() => { let m=0,r=1; for(let i=1;i<chars.length;i++){ if(chars[i]===chars[i-1]) r++; else { if(r>m)m=r; r=1; } } return Math.max(m,r); })();
      const longRun = maxRun >= 4 && compact.length >= 8;
      const digitNoise = compact.length >= 8 && (compact.match(/[0-9]/g) || []).length / compact.length > 0.18;
      const fragmented = compact.length >= 10 && uniq <= 6 && (compact.match(/[\u0E00-\u0E7F]/g) || []).length / compact.length < 0.75;
      if (!reply || tooShort || (thaiVisible < 2 && compact.length < 4) || repetitive || lowDiversity || mostlySame || longRun || (digitNoise && fragmented)) {
        reply = "ขออภัยครับ ตอนนี้โมเดลยังตอบได้ไม่ชัดเจน ลองถามใหม่อีกครั้ง หรือเลือกโมเดลที่ฝึกด้วยข้อมูลภาษาไทยให้เหมาะสมกับคำถามครับ";
      }
      push("assistant", reply);
      setStatus(`พร้อมใช้งาน: ${activeModel?.name || modelId}`);
    } catch (err) { push("system", `เกิดข้อผิดพลาด: ${err.message}`); setStatus("เกิดข้อผิดพลาด"); }
    finally { setBusy(""); }
  }

  const canSend = modelReady && !busy && input.trim();
  const loadingText = busy === "search"
    ? "กำลังค้นหาโมเดล..."
    : busy === "train"
      ? `กำลังโหลดโมเดล...${trainEpoch ? ` (กำลังเตรียม epoch ${trainEpoch})` : ""}`
      : "กำลังโหลดโมเดล...";

  return (
    <div className="shell">
      <section className="card chat-controls">
        <ModelSelector models={models} value={modelId} onChange={setModelId} disabled={busy === "search" || busy === "load" || busy === "train"} />
        <p className="meta"><strong>Mini-LLM V1.0</strong> — {status}</p>
      </section>

      <section className="card chat-card">
        <div className="chat-list" ref={listRef} aria-live="polite">
          {messages.map((m, i) => (<div key={i} className={`chat-msg chat-${m.role}`}><div className="chat-role">{m.role === "assistant" ? "Mini-LLM" : m.role}</div><div className="chat-text">{m.text}</div></div>))}
          {(busy === "generate" || busy === "search" || busy === "load" || busy === "train") ? (
            <div className="chat-msg chat-assistant">
              <div className="chat-role">Mini-LLM</div>
              <div className="chat-text">{busy === "generate" ? "กำลังคิดคำตอบ..." : loadingText}</div>
            </div>
          ) : null}
        </div>
        <form className="chat-form" onSubmit={handleSend}>
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder={modelReady ? "พิมพ์คำถามภาษาไทย..." : "กำลังค้นหา/โหลดโมเดล..."} disabled={!!busy} aria-label="ข้อความแชท" />
          <button type="submit" disabled={!canSend}>{busy === "generate" ? "..." : "ส่ง"}</button>
        </form>
      </section>
    </div>
  );
}
