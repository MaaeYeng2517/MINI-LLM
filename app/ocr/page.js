import ThaiImageOCR from "../components/ThaiImageOCR";

export const metadata = {
  title: "Mini-LLM V1.0 — OCR ตรวจสอบภาพ",
  description: "หน้า OCR ตรวจสอบภาพ วิเคราะห์ข้อความภาษาไทย/อังกฤษจากภาพตามวัตถุประสงค์",
};

export default function OCRPage() {
  return (
    <main className="page ocr-page">
      <header className="hero chat-hero">
        <p className="eyebrow">เครื่องมือ • ตรวจสอบภาพ</p>
        <h1>OCR ตรวจสอบภาพ</h1>
        <p className="lede">
          อัป loaded ภาพแล้วให้ระบบอ่านข้อความด้วย Tesseract.js (ไทย +
          อังกฤษ) แล้ววิเคราะห์ตามวัตถุประสงค์ที่คุณกำหนด
        </p>
      </header>
      <ThaiImageOCR />
    </main>
  );
}