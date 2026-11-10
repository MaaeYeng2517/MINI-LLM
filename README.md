# Mini-LLM TH v1.0 (Next.js + JavaScript + TensorFlow.js)

บริการ/เดโม Mini Language Model แบบ character & BPE next-token prediction ทำงานบนเบราว์เซอร์ด้วย TensorFlow.js

## เว็บไซต์/เส้นทาง

- `/` — หน้าแรก Mini-LLM TH v1.0 (มี เมนูนำทาง)
- `/playground` — Dataset, Tokenizer, สร้าง/ฝึก/Resume โมเดล, Loss/Metrics, Checkpoint, Generation, Evaluation
- `/chat` — Chatbot เชื่อมต่อโมเดลจาก checkpoint (หรือ Quick train & connect)
- `/docs` — เอกสารการใช้งาน
- `/graphs/[modelId]` — กราฟการฝึกต่อโมเดล

## เกณฑ์ v1.0

| ความสามารถ | สถานะ |
|---|---|
| โหลด Dataset | อ่าน `public/data/train.txt` / `test.txt` ได้ (clean/split/inspect) |
| Tokenizer | Character + BPE/Subword → Token ID + preview |
| Training | Next Token Prediction (Embedding + LSTM + Dense softmax, Adam) |
| Loss/Metrics | loss, val_loss, accuracy, perplexity, history (browser storage) |
| Checkpoint | Save/Load `mini-llm-v1.0` ผ่าน TensorFlow.js LocalStorage |
| Generation | สร้างข้อความจาก prompt (temperature sampling) |
| Evaluation | ทดสอบ loss/perplexity/accuracy กับข้อมูลที่ไม่เคยฝึก |
| Chatbot | ใช้ checkpoint เดียวกับ Playground / Quick train |

## รันโปรเจกต์

```bash
npm install
npm run dev
```

เปิด `http://localhost:3000`

## โครงสร้างสำคัญ

- `public/data/train.txt`, `public/data/test.txt`
- `lib/tokenizer.js` — Char/BPE tokenizer
- `lib/dataset.js` — dataset load/clean/inspect/split
- `lib/mini-llm.js` — ML core (build/train/resume/eval/generate/checkpoint)
- `lib/model-runtime.js` — runtime กลาง Playground ↔ Chatbot
- `app/components/MiniLLMDemo.js` — Playground UI
- `app/components/ChatBot.js` — Chatbot UI
- `app/docs/page.js` — เอกสาร

## หมายเหตุ

- Mini-LLM ขนาดเล็กเพื่อเดโม/เรียนรู้ ไม่ใช่ production LLM
- คุณภาพขึ้นกับ dataset, tokenizer, hyperparameters และ epochs
- Checkpoint เก็บใน LocalStorage ของเบราว์เซอร์ (origin เดียวกัน)
