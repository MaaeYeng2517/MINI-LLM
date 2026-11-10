"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { QRCodeCanvas } from "qrcode.react";

/**
 * QRCodeLink renders a scannable QR code pointing at the chatbot URL.
 * The URL is derived from the current location so it always reflects the
 * live deployment (localhost during dev, production host otherwise).
 */
export default function QRCodeLink({
  value: valueProp,
  label = "สแกนเพื่อเปิด chatbot",
  sublabel = "ใช้กล้องือก phone เพื่อเข้าถึงลิงก์นี้",
  size = 220,
  fgColor = "#1f2937",
  bgColor = "#ffffff",
  includeMargin = true,
  className = "",
  showCopy = true,
  showDownload = true,
}) {
  const [mounted, setMounted] = useState(false);
  const [copied, setCopied] = useState(false);
  const canvasRef = useRef(null);

  const defaultUrl = useMemo(() => {
    if (typeof window === "undefined") return "";
    const base = window.location.origin;
    return valueProp || `${base}/chat`;
  }, [valueProp]);

  // Avoid hydration mismatch: render placeholder until client mounts.
  useEffect(() => setMounted(true), []);

  async function handleCopy() {
    try {
      if (navigator.clipboard && defaultUrl) {
        await navigator.clipboard.writeText(defaultUrl);
        setCopied(true);
        setTimeout(() => setCopied(false), 1800);
      }
    } catch {
      // ignore
    }
  }

  function handleDownload() {
    const canvas = canvasRef.current?._canvas;
    if (!canvas || !defaultUrl) return;
    const url = canvas.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url;
    a.download = "mini-llm-chatbot-qr.png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  if (!mounted) {
    return (
      <div className={`qr-link qr-link--skeleton ${className}`} style={{ width: size, height: size }} aria-hidden="true" />
    );
  }

  return (
    <div className={`qr-link ${className}`}>
      <div className="qr-link__card">
        <QRCodeCanvas
          ref={canvasRef}
          value={defaultUrl || "https://example.com"}
          size={size}
          fgColor={fgColor}
          bgColor={bgColor}
          includeMargin={includeMargin}
          level="M"
        />
      </div>
      <div className="qr-link__meta">
        <p className="qr-link__label">{label}</p>
        <p className="qr-link__sub">{sublabel}</p>
        <code className="qr-link__url" title={defaultUrl}>
          {defaultUrl}
        </code>
        <div className="qr-link__actions">
          {showCopy && (
            <button type="button" className="btn btn--ghost" onClick={handleCopy}>
              {copied ? "คัดลอกแล้ว" : "คัดลอกลิงก์"}
            </button>
          )}
          {showDownload && (
            <button type="button" className="btn btn--ghost" onClick={handleDownload}>
              ดาวน์ QR
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function QRCodeInline({ value, size = 160, fgColor = "#1f2937", bgColor = "#ffffff" }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div style={{ width: size, height: size }} aria-hidden="true" />;
  return (
    <QRCodeCanvas
      value={value || "https://example.com"}
      size={size}
      fgColor={fgColor}
      bgColor={bgColor}
      includeMargin
      level="M"
    />
  );
}