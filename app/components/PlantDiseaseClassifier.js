"use client";

import { useState, useRef, useCallback } from "react";

const DISEASE_INFO = {
  "Apple___Apple_scab": { th: "โรคราสแอปเปิ้ล", en: "Apple Scab", desc: "เชื้อรา Venturia inaequalis ทำให้ใบและผลมีจุดสีดำ" },
  "Apple___Black_rot": { th: "โรคเน่าสีดำแอปเปิ้ล", en: "Black Rot", desc: "เชื้อรา Botryosphaeria obtusa ทำให้ผลเน่าและใบมีจุด" },
  "Apple___Cedar_apple_rust": { th: "โรคราสซีดาร์-แอปเปิ้ล", en: "Cedar Apple Rust", desc: "เชื้อรา Gymnosporangium juniperi-virginianae ต้องมีพืชหอผ่าน" },
  "Apple___healthy": { th: "แอปเปิ้ลแข็งแรง", en: "Healthy Apple", desc: "พืชแข็งแรง ไม่พบอาการโรค" },
  "Corn___Cercospora_leaf_spot": { th: "โรคจุดใบเซอร์โคสปอรา", en: "Cercospora Leaf Spot", desc: "เชื้อรา Cercospora zeae-maydis ทำให้ใบมีจุดสีน้ำตาล" },
  "Corn___Common_rust": { th: "โรคราสทั่วไปข้าวโพด", en: "Common Rust", desc: "เชื้อรา Puccinia sorghi ทำให้ใบมีพิลล์สีส้ม" },
  "Corn___Northern_Leaf_Blight": { th: "โรคไหม้ใบทางเหนือ", en: "Northern Leaf Blight", desc: "เชื้อรา Exserohilum turcicum ทำให้ใบมีแผลยาว" },
  "Corn___healthy": { th: "ข้าวโพดแข็งแรง", en: "Healthy Corn", desc: "พืชแข็งแรง ไม่พบอาการโรค" },
  "Grape___Black_rot": { th: "โรคเน่าสีดำองุ่น", en: "Black Rot", desc: "เชื้อรา Guignardia bidwellii ทำให้ผลเน่าและใบมีจุด" },
  "Grape___Esca_(Black_Measles)": { th: "โรคเอสก้า (Black Measles)", en: "Esca", desc: "กลุ่มเชื้อราทำให้กิ่งเหี่ยวและใบมีจุดสีดำ" },
  "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": { th: "โรคไหม้ใบอิซาเรีย", en: "Leaf Blight", desc: "เชื้อรา Isariopsis clavispora ทำให้ใบไหม้" },
  "Grape___healthy": { th: "องุ่นแข็งแรง", en: "Healthy Grape", desc: "พืชแข็งแรง ไม่พบอาการโรค" },
  "Tomato___Bacterial_spot": { th: "โรคจุดใบเชื้อแบคทีเรีย", en: "Bacterial Spot", desc: "เชื้อแบคทีเรีย Xanthomonas ทำให้ใบและผลมีจุดน้ำ" },
  "Tomato___Early_blight": { th: "โรคไหม้ใบเร็ว", en: "Early Blight", desc: "เชื้อรา Alternaria solani ทำให้ใบมีแผ่นวงแหวน" },
  "Tomato___Late_blight": { th: "โรคไหม้ใบช้า", en: "Late Blight", desc: "เชื้อรา Phytophthora infestans ทำให้ใบและผลเน่าร่วดเร็ว" },
  "Tomato___Leaf_Mold": { th: "โรคราบน้ำมันใบมะเขือเทศ", en: "Leaf Mold", desc: "เชื้อรา Passalora fulva ทำให้ใบมีรูปน้ำมันสีเหลือง" },
  "Tomato___Septoria_leaf_spot": { th: "โรคจุดใบเซปทอเรีย", en: "Septoria Leaf Spot", desc: "เชื้อรา Septoria lycopersici ทำให้ใบมีจุดวงกลมเล็ก" },
  "Tomato___Spider_mites": { th: "โรคแมลงกัดมดลูก", en: "Spider Mites", desc: "แมลงกัดมดลูก ทำให้ใบมีจุดสีขาวและใยแมงมุม" },
  "Tomato___Target_Spot": { th: "โรคจุดเป้ากลาง", en: "Target Spot", desc: "เชื้อรา Corynespora cassiicola ทำให้ใบมีแผ่นคล้ายเป้ากลาง" },
  "Tomato___Yellow_Leaf_Curl_Virus": { th: "โรคไวรัสใบม้วนเหลือง", en: "Yellow Leaf Curl Virus", desc: "ไวรัสทำให้ใบม้วน เหลือง และพืชแคระ" },
  "Tomato___healthy": { th: "มะเขือเทศแข็งแรง", en: "Healthy Tomato", desc: "พืชแข็งแรง ไม่พบอาการโรค" },
};

export default function PlantDiseaseClassifier() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [useTTA, setUseTTA] = useState(true);
  const fileInputRef = useRef(null);

  const handleFileChange = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    if (!file.type.startsWith("image/")) {
      setError("กรุณาเลือกไฟล์รูปภาพเท่านั้น");
      return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
      setError("ไฟล์ใหญ่เกิน 10MB");
      return;
    }

    setImage(file);
    setError(null);
    setResult(null);
    
    const reader = new FileReader();
    reader.onload = (ev) => setPreview(ev.target.result);
    reader.readAsDataURL(file);
  }, []);

  const handlePredict = async () => {
    if (!image) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Convert image to base64
      const base64 = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(",")[1]);
        reader.onerror = reject;
        reader.readAsDataURL(image);
      });

      const res = await fetch("/api/plant-disease", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64, tta: useTTA, top_k: 5 }),
      });

      const data = await res.json();
      
      if (!data.ok) {
        throw new Error(data.error || "Inference failed");
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add("drag-over");
  };

  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove("drag-over");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove("drag-over");
    const file = e.dataTransfer.files?.[0];
    if (file) {
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInputRef.current.files = dt.files;
      handleFileChange({ target: fileInputRef.current });
    }
  };

  const getDiseaseInfo = (className) => {
    return DISEASE_INFO[className] || { th: className, en: className, desc: "ไม่มีข้อมูลเพิ่มเติม" };
  };

  return (
    <div className="shell plant-disease-shell">
      <section className="card">
        <h2>🌿 ตรวจจับโรคพืชด้วย AI</h2>
        <p className="meta">อัปโหลดรูปใบพืชเพื่อตรวจสอบโรค (รองรับ PlantVillage dataset 38 คลาส)</p>
      </section>

      <section className="card">
        <div className="upload-area" onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            id="plant-image-input"
            className="file-input"
            disabled={loading}
          />
          <label htmlFor="plant-image-input" className="upload-label">
            {preview ? (
              <>
                <img src={preview} alt="Preview" className="preview-img" />
                <p>คลิกเพื่อเปลี่ยนรูปภาพ</p>
              </>
            ) : (
              <>
                <span className="upload-icon">📷</span>
                <p>ลากและวางรูปภาพที่นี่</p>
                <p className="meta">หรือคลิกเพื่อเลือกไฟล์ (PNG, JPG, JPEG)</p>
              </>
            )}
          </label>
        </div>

        {error && (
      <div className="error-msg">
        ❌ {error}
        {error.includes("checkpoint") && (
          <div className="setup-hint" style={{ marginTop: 8, fontSize: 13 }}>
            <strong>วิธีการ preparing:</strong>
            <ol style={{ margin: "4px 0 0 16px", padding: 0 }}>
              <li>Source code: <code>cd plant_disease_classifier && bash setup.sh</code></li>
              <li>Prepare data: <code>python main.py prepare-data --source /path/to/plantvillage --output data</code></li>
              <li>Train: <code>python main.py train --epochs 50 --model resnet50</code></li>
            </ol>
          </div>
        )}
      </div>
    )}

        <div className="options">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={useTTA}
              onChange={(e) => setUseTTA(e.target.checked)}
              disabled={loading}
            />
            <span>ใช้ Test-Time Augmentation (TTA) เพิ่มความแม่นยำ</span>
          </label>
        </div>

        <button
          className="predict-btn"
          onClick={handlePredict}
          disabled={!image || loading}
        >
          {loading ? "🔄 กำลังวิเคราะห์..." : "🔍 ตรวจสอบโรค"}
        </button>
      </section>

      {result && (
        <section className="card result-card">
          <h3>📊 ผลการวิเคราะห์</h3>
          
          <div className="main-result">
            <div className="result-info">
              <div className="result-class">
                {result.prediction}
                <span className="confidence-badge">{result.confidence.toFixed(1)}%</span>
              </div>
              <div className="result-thai">{getDiseaseInfo(result.prediction).th}</div>
              <div className="result-desc">{getDiseaseInfo(result.prediction).desc}</div>
            </div>
          </div>

          <div className="top-k-section">
            <h4>Top {result.top_k?.length || 5} ผลลัพธ์:</h4>
            <div className="top-k-list">
              {result.top_k?.map((item, idx) => (
                <div key={idx} className="top-k-item">
                  <div className="top-k-rank">#{idx + 1}</div>
                  <div className="top-k-class">
                    <span className="class-name">{item.class}</span>
                    <span className="class-thai">{getDiseaseInfo(item.class).th}</span>
                  </div>
                  <div className="top-k-bar">
                    <div 
                      className="bar-fill" 
                      style={{ width: `${item.confidence}%` }}
                    ></div>
                  </div>
                  <div className="top-k-conf">{item.confidence.toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>

          <details className="all-probs">
            <summary>ดูความน่าจะเป็นทั้งหมด ({Object.keys(result.all_probabilities || {}).length} คลาส)</summary>
            <div className="probs-grid">
              {Object.entries(result.all_probabilities || {})
                .sort(([,a], [,b]) => b - a)
                .map(([cls, prob]) => (
                  <div key={cls} className="prob-item">
                    <span className="prob-class">{cls}</span>
                    <span className="prob-value">{prob.toFixed(2)}%</span>
                  </div>
                ))}
            </div>
          </details>

          <div className="meta">
            รันบน: {result.device} | โมเดล: {result.checkpoint?.split("/").pop() || "best_model.pt"}
          </div>
        </section>
      )}

      <section className="card info-card">
        <h3>ℹ️ ข้อมูลเพิ่มเติม</h3>
        <ul>
          <li><strong>Dataset:</strong> PlantVillage (38 คลาส: Apple, Corn, Grape, Tomato)</li>
          <li><strong>Model:</strong> Transfer Learning (ResNet50 / EfficientNet / ViT)</li>
          <li><strong>TTA:</strong> Test-Time Augmentation 5-crop เพิ่มความแม่นยำ</li>
          <li><strong>การใช้งาน:</strong> ถ่ายรูปใบพืชในสภาพแสงดี พื้นหลังสะอาด จะได้ผลแม่นยำที่สุด</li>
        </ul>
      </section>
    </div>
  );
}