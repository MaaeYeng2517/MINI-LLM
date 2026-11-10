"use client";

import { useEffect, useRef, useState } from "react";
import { createWorker } from "tesseract.js";

/**
 * ThaiImageOCR provides a browser-side OCR page for inspecting images.
 * Users upload an image and the engine runs Tesseract.js (Thai + English)
 * to extract text. The extracted text is then analyzed against a
 * user-defined purpose.
 */
export default function ThaiImageOCR() {
  const [imageSrc, setImageSrc] = useState("");
  const [rawText, setRawText] = useState("");
  const [purpose, setPurpose] = useState("");
  const [analysis, setAnalysis] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");
  const [lang, setLang] = useState("tha+eng");
  const fileRef = useRef(null);
  const imgRef = useRef(null);
  const workerRef = useRef(null);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const worker = await createWorker(["tha", "eng"]);
        if (mounted) workerRef.current = worker;
      } catch (e) {
        if (mounted) setError(`ไม่สามารถเริ่มต้นเครื่องมือ OCR ได้: ${e.message}`);
      }
    })();
    return () => {
      mounted = false;
      if (workerRef.current) workerRef.current.terminate().catch(() => {});
    };
  }, []);

  async function handleFile(file) {
    if (!file) return;
    setError("");
    setRawText("");
    setAnalysis("");
    const url = URL.createObjectURL(file);
    setImageSrc(url);
    runOcr(url);
  }

  async function runOcr(url) {
    setBusy(true);
    setProgress("กำลังเริ่มต้น OCR...");
    try {
      const worker = workerRef.current;
      if (!worker) throw new Error("OCR engine not ready");
      const { data } = await worker.recognize(url, { lang });
      setRawText(data.text || "");
      setProgress("อ่านข้อความแล้ว");
    } catch (e) {
      setError(`OCR ล้มเหลว: ${e.message}`);
      setProgress("");
    } finally {
      setBusy(false);
    }
  }

  function analyze() {
    const text = (rawText || "").trim();
    if (!text) {
      setAnalysis("ไม่มีข้อความที่จะวิเคราะห์");
      return;
    }
    const p = purpose.trim();
    const lines = [];
    lines.push(`ข้อความที่อ่านได้: ${text}`);
    if (p) {
      lines.push(`วัตถุประสงค์: ${p}`);
      const contains = text.toLowerCase().includes(p.toLowerCase());
      lines.push(
        `ความสัมพันธ์กับวัตถุประสงค์: ${
          contains ? "พบข้อความที่ตรงกับวัตถุประสงค์" : "ไม่พบข้อความที่ตรงกับวัตถุประสงค์"
        }`
      );
    }
    const thaiCount = (text.match(/[\u0E00-\u0E7F]/g) || []).length;
    const asciiCount = (text.match(/[A-Za-z0-9]/g) || []).length;
    lines.push(`จำนวนตัวอักษรไทย: ${thaiCount}`);
    lines.push(`จำนวนตัวอักษรภาษาอังกฤษและตัวเลข: ${asciiCount}`);
    lines.push(`ความยาวทั้งหมด: ${text.length}`);
    setAnalysis(lines.join("\n"));
  }

  return (
    <div className="ocr-page">
      <section className="card">
        <h2>OCR ตรวจสอบภาพ</h2>
        <p className="meta">
          อัป loaded ภาพ หรือลาก dropped ลงในช่อง — ระบบจะอ่านข้อความภาษาไทย/
          อังกฤษด้วย Tesseract.js
        </p>

        <div className="ocr-controls">
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
          <button type="button" className="btn" onClick={() => fileRef.current?.click()} disabled={busy}>
            {busy ? "กำลังอ่านข้อความ..." : "Choose ภาพ"}
          </button>
          <label className="meta">
            ภาษา:
            <select value={lang} onChange={(e) => setLang(e.target.value)} disabled={busy}>
              <option value="tha+eng">ไทย + อังกฤษ</option>
              <option value="tha">ไทย</option>
              <option value="eng">อังกฤษ</option>
            </select>
          </label>
        </div>

        {busy && <p className="meta">{progress}</p>}
        {error && <p className="meta" style={{ color: "#d32f2f" }}>{error}</p>}
      </section>

      {imageSrc && (
        <section className="card">
          <h3>ภาพที่อัป loaded</h3>
          <div className="ocr-image-wrap">
            <img ref={imgRef} src={imageSrc} alt="OCR input" />
          </div>
        </section>
      )}

      {rawText && (
        <section className="card">
          <h3>ข้อความที่อ่านได้</h3>
          <textarea className="ocr-text" value={rawText} readOnly rows={6} />
          <div className="ocr-controls" style={{ marginTop: 10 }}>
            <input
              className="ocr-purpose"
              placeholder="ระบุวัตถุประสงค์การตรวจสอบ (เช่น ตรวจสอบชื่อ, วิเคราะห์ข้อความ, หาคำ Specific)"
              value={purpose}
              onChange={(e) => setPurpose(e.target.value)}
            />
            <button type="button" className="btn" onClick={analyze}>
              วิเคราะห์ตามวัตถุประสงค์
            </button>
          </div>
          {analysis && <pre className="ocr-analysis">{analysis}</pre>}
        </section>
      )}
    </div>
  );
}