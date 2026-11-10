import * as tf from "@tensorflow/tfjs";
import { createTokenizer, tokenizerFromJSON, SPECIAL_TOKENS, formatQA } from "./tokenizer";
import { cleanText } from "./dataset";

export const DEFAULT_DATASET_PATH = "/data/train.json";
export const DEFAULT_CONFIG = {
  seqLen: 256, embedDim: 384, lstmUnits: 384, learningRate: 0.005,
  tokenizerType: "char", vocabSize: 2000, dropout: 0.1, batchSize: 8, valSplit: 0.15,
};
const LOSS_KEY = "mini-llm-v1:loss-history";
const META_KEY = "mini-llm-v1:meta";

function sampleIndex(probs, temperature = 1) {
  const t = Math.max(temperature, 1e-6);
  const logits = probs.map((p) => Math.log(Math.max(p, 1e-8)) / t);
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  let r = Math.random() * sum;
  for (let i = 0; i < exps.length; i++) { r -= exps[i]; if (r <= 0) return i; }
  return exps.length - 1;
}

function applyRepetitionPenalty(probs, recent, penalty = 1.15) {
  if (!recent || !recent.length || !penalty || penalty === 1) return probs;
  const out = probs.slice();
  for (const id of new Set(recent.slice(-64))) {
    if (id >= 0 && id < out.length) out[id] = Math.max(out[id] / penalty, 1e-10);
  }
  const s = out.reduce((a, b) => a + b, 0) || 1;
  return out.map((v) => v / s);
}

function makeSequences(ids, seqLen) {
  const xs = []; const ys = [];
  for (let i = 0; i + seqLen < ids.length; i++) { xs.push(ids.slice(i, i + seqLen)); ys.push(ids[i + seqLen]); }
  return { xs, ys };
}

function tensorsFromSequences(xs, ys, seqLen, vocabSize) {
  const x = tf.tensor2d(xs, [xs.length, seqLen], "int32");
  const y = tf.oneHot(tf.tensor1d(ys, "int32"), vocabSize);
  return { x, y, dispose() { x.dispose(); y.dispose(); } };
}

function movingAverage(values, window = 3) {
  return values.map((_, i) => {
    const start = Math.max(0, i - window + 1);
    const slice = values.slice(start, i + 1).filter((v) => typeof v === "number");
    return slice.length ? slice.reduce((a, b) => a + b, 0) / slice.length : null;
  });
}

export class MiniLLM {
  constructor(config = {}) {
    this.setConfig(config);
    this.tokenizer = null;
    this.model = null;
    this.lossHistory = [];
    this.meta = null;
  }

  setConfig(config = {}) {
    this.seqLen = Number(config.seqLen ?? this.seqLen ?? DEFAULT_CONFIG.seqLen);
    this.embedDim = Number(config.embedDim ?? this.embedDim ?? DEFAULT_CONFIG.embedDim);
    this.lstmUnits = Number(config.lstmUnits ?? this.lstmUnits ?? DEFAULT_CONFIG.lstmUnits);
    this.learningRate = Number(config.learningRate ?? this.learningRate ?? DEFAULT_CONFIG.learningRate);
    this.tokenizerType = config.tokenizerType ?? this.tokenizerType ?? DEFAULT_CONFIG.tokenizerType;
    this.vocabSize = Number(config.vocabSize ?? this.vocabSize ?? DEFAULT_CONFIG.vocabSize);
    this.dropout = Number(config.dropout ?? this.dropout ?? DEFAULT_CONFIG.dropout);
    this.batchSize = Number(config.batchSize ?? this.batchSize ?? DEFAULT_CONFIG.batchSize);
    this.valSplit = Number(config.valSplit ?? this.valSplit ?? DEFAULT_CONFIG.valSplit);
  }

  get config() {
    return {
      seqLen: this.seqLen, embedDim: this.embedDim, lstmUnits: this.lstmUnits,
      learningRate: this.learningRate, tokenizerType: this.tokenizerType, vocabSize: this.vocabSize,
      dropout: this.dropout, batchSize: this.batchSize, valSplit: this.valSplit,
    };
  }

  fitTokenizer(text) {
    this.tokenizer = createTokenizer(this.tokenizerType, text, { vocabSize: this.vocabSize });
    return { vocabSize: this.tokenizer.vocabSize, type: this.tokenizer.type };
  }

  compile() {
    if (!this.model) throw new Error("Model is not built.");
    this.model.compile({ optimizer: tf.train.adam(this.learningRate), loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  }

  buildModel(config) {
    if (config) this.setConfig(config);
    if (!this.tokenizer) throw new Error("Tokenizer is not initialized. Load dataset and fit tokenizer first.");
    if (this.model) this.model.dispose();
    const model = tf.sequential();
    model.add(tf.layers.embedding({ inputDim: this.tokenizer.vocabSize, outputDim: this.embedDim, inputLength: this.seqLen, name: "token_embedding" }));
    model.add(tf.layers.lstm({ units: this.lstmUnits, returnSequences: false, dropout: this.dropout, name: "lstm" }));
    if (this.dropout > 0) model.add(tf.layers.dropout({ rate: Math.min(this.dropout, 0.5), name: "dropout" }));
    model.add(tf.layers.dense({ units: this.tokenizer.vocabSize, activation: "softmax", name: "next_token" }));
    this.model = model;
    this.compile();
    return this.modelSummary();
  }

  modelSummary() {
    if (!this.model) return null;
    const layers = this.model.layers.map((l) => {
      const out = l.outputShape;
      const shape = Array.isArray(out) ? out.map((s) => (Array.isArray(s) ? `[${s.join(",")}]` : s)).join(" | ") : String(out);
      return { name: l.name, className: l.getClassName(), outputShape: shape, params: l.countParams() };
    });
    return { config: this.config, vocabSize: this.tokenizer?.vocabSize ?? null, tokenizerType: this.tokenizer?.type ?? null, layers, totalParams: layers.reduce((a, l) => a + l.params, 0), built: true };
  }

  async train(text, options = {}, onProgress) {
    const epochs = options.epochs ?? 20;
    const batchSize = options.batchSize ?? this.batchSize;
    const valSplit = Math.min(Math.max(options.valSplit ?? this.valSplit, 0), 0.4);
    const resume = !!options.resume && this.model && this.tokenizer;
    const earlyStopPatience = Math.max(0, Number(options.earlyStopPatience ?? 0));

    const prepared = cleanText(text);
    if (!resume) this.fitTokenizer(prepared);
    const ids = this.tokenizer.encode(prepared, false);
    if (ids.length <= this.seqLen + 2) throw new Error("Training text is too short for the selected sequence length.");
    if (!this.model) this.buildModel();

    const { xs, ys } = makeSequences(ids, this.seqLen);
    const splitAt = Math.max(1, Math.floor(xs.length * (1 - valSplit)));
    const train = { xs: xs.slice(0, splitAt), ys: ys.slice(0, splitAt) };
    const val = { xs: xs.slice(splitAt), ys: ys.slice(splitAt) };
    const trainT = tensorsFromSequences(train.xs, train.ys, this.seqLen, this.tokenizer.vocabSize);
    const valT = val.xs.length ? tensorsFromSequences(val.xs, val.ys, this.seqLen, this.tokenizer.vocabSize) : null;

    if (!resume) this.lossHistory = [];
    const startEpoch = resume ? this.lossHistory.length : 0;
    let best = Infinity; let wait = 0; let stoppedEarly = false;
    try {
      await this.model.fit(trainT.x, trainT.y, {
        epochs: startEpoch + epochs,
        initialEpoch: startEpoch,
        batchSize,
        shuffle: true,
        validationData: valT ? [valT.x, valT.y] : undefined,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            const entry = { epoch: epoch + 1, loss: logs?.loss ?? null, val_loss: logs?.val_loss ?? null, acc: logs?.acc ?? logs?.accuracy ?? null };
            this.lossHistory.push(entry);
            this.persistLossHistory();
            const monitor = typeof entry.val_loss === "number" ? entry.val_loss : entry.loss;
            if (typeof monitor === "number" && monitor < best - 1e-5) { best = monitor; wait = 0; } else { wait++; }
            if (onProgress) onProgress({ ...entry, epochs: startEpoch + epochs, addedEpochs: epochs, best: Number.isFinite(best) ? best : null });
            if (earlyStopPatience > 0 && wait >= earlyStopPatience) { stoppedEarly = true; this.model.stopTraining = true; }
            await tf.nextFrame();
          },
        },
      });
      this.meta = { vocabSize: this.tokenizer.vocabSize, tokenizerType: this.tokenizer.type, trainSamples: train.xs.length, valSamples: val.xs.length, ...this.config, trainedAt: new Date().toISOString(), stoppedEarly };
      return { ...this.meta, lossHistory: this.lossHistory, model: this.modelSummary(), resume, metrics: this.getMetrics() };
    } finally {
      trainT.dispose();
      if (valT) valT.dispose();
    }
  }

  getMetrics() {
    const h = this.lossHistory || [];
    if (!h.length) return null;
    const losses = h.map((r) => r.loss).filter((v) => typeof v === "number");
    const vals = h.map((r) => r.val_loss).filter((v) => typeof v === "number");
    const lossMA = movingAverage(h.map((r) => r.loss));
    return {
      epochs: h.length,
      finalLoss: losses.at(-1) ?? null,
      finalValLoss: vals.at(-1) ?? null,
      bestLoss: losses.length ? Math.min(...losses) : null,
      bestValLoss: vals.length ? Math.min(...vals) : null,
      perplexity: typeof losses.at(-1) === "number" ? Math.exp(Math.min(losses.at(-1), 20)) : null,
      lossMA,
      trend: losses.length >= 2 ? (losses.at(-1) < losses[0] ? "improving" : "not_improving") : "n/a",
    };
  }

  async evaluate(text) {
    if (!this.model || !this.tokenizer) throw new Error("Model is not trained or loaded.");
    const ids = this.tokenizer.encode(cleanText(text), false);
    if (ids.length <= this.seqLen + 1) throw new Error("Evaluation text is too short for the selected sequence length.");
    const { xs, ys } = makeSequences(ids, this.seqLen);
    const data = tensorsFromSequences(xs, ys, this.seqLen, this.tokenizer.vocabSize);
    try {
      const result = this.model.evaluate(data.x, data.y);
      const tensors = Array.isArray(result) ? result : [result];
      const loss = (await tensors[0].data())[0];
      const acc = tensors[1] ? (await tensors[1].data())[0] : null;
      tensors.forEach((t) => t.dispose());
      return { loss, acc, perplexity: Math.exp(Math.min(loss, 20)), samples: xs.length };
    } finally { data.dispose(); }
  }

  async generate(prompt, options = {}) {
    if (!this.model || !this.tokenizer) throw new Error("Model is not trained or loaded.");
    const maxTokens = options.maxTokens ?? 80;
    const temperature = options.temperature ?? 0.8;
    const repetitionPenalty = options.repetitionPenalty ?? 1.15;
    const stopMarkers = options.stopMarkers || ["\nคำถาม:", "\nผู้ใช้:", "\nUser:", "\nคำตอบ:", "\nผู้ช่วย:", "\nBot:"];
    let ids = this.tokenizer.encode(prompt, false);
    if (ids.length === 0) ids = [this.tokenizer.tokenToId?.[SPECIAL_TOKENS.BOS] ?? this.tokenizer.charToId?.[SPECIAL_TOKENS.BOS] ?? 2];
    const generated = [];
    const map = this.tokenizer.idToToken || this.tokenizer.idToChar || {};
    let tail = "";
    for (let i = 0; i < maxTokens; i++) {
      const input = ids.slice(-this.seqLen);
      const padId = this.tokenizer.tokenToId?.[SPECIAL_TOKENS.PAD] ?? this.tokenizer.charToId?.[SPECIAL_TOKENS.PAD] ?? 0;
      while (input.length < this.seqLen) input.unshift(padId);
      const probs = tf.tidy(() => {
        const x = tf.tensor2d([input], [1, this.seqLen], "int32");
        return Array.from(this.model.predict(x).dataSync());
      });
      const adjusted = applyRepetitionPenalty(probs, [...ids.slice(-96), ...generated], repetitionPenalty);
      const nextId = sampleIndex(adjusted, temperature);
      generated.push(nextId); ids.push(nextId);
      tail = (tail + (map[nextId] ?? "")).slice(-120);
      if (stopMarkers.some((m) => tail.includes(m))) break;
      await tf.nextFrame();
    }
    return prompt + this.tokenizer.decode(generated);
  }

  tokenizePreview(text, limit = 120) {
    const src = (text || "").slice(0, limit * 2);
    if (!this.tokenizer) this.tokenizer = createTokenizer(this.tokenizerType, src || " ", { vocabSize: this.vocabSize });
    const ids = this.tokenizer.encode(src, false).slice(0, limit);
    const map = this.tokenizer.idToToken || this.tokenizer.idToChar || {};
    return ids.map((id) => ({ id, token: map[id] ?? "?" }));
  }

  persistLossHistory() {
    if (typeof window === "undefined") return;
    try { window.localStorage.setItem(LOSS_KEY, JSON.stringify(this.lossHistory)); } catch {}
  }

  async saveCheckpoint(name = "mini-llm-v1") {
    if (!this.model || !this.tokenizer) throw new Error("No trained model to save.");
    await this.model.save(`localstorage://${name}`);
    const meta = { tokenizer: this.tokenizer.toJSON(), config: this.config, lossHistory: this.lossHistory, meta: this.meta, metrics: this.getMetrics(), savedAt: new Date().toISOString(), name };
    if (typeof window !== "undefined") window.localStorage.setItem(META_KEY, JSON.stringify(meta));
    return meta;
  }

  async loadCheckpoint(name = "mini-llm-v1") {
    const raw = typeof window !== "undefined" ? window.localStorage.getItem(META_KEY) : null;
    if (!raw) throw new Error("No saved checkpoint metadata found.");
    const meta = JSON.parse(raw);
    this.setConfig(meta.config || {});
    this.tokenizer = tokenizerFromJSON(meta.tokenizer);
    this.lossHistory = meta.lossHistory || [];
    this.meta = meta.meta || null;
    if (this.model) this.model.dispose();
    this.model = await tf.loadLayersModel(`localstorage://${name}`);
    this.compile();
    return { ...meta, model: this.modelSummary(), metrics: this.getMetrics() };
  }

  hasSavedCheckpoint(name = "mini-llm-v1") {
    if (typeof window === "undefined") return false;
    try {
      const meta = JSON.parse(window.localStorage.getItem(META_KEY) || "null");
      return !!meta && meta.name === name && !!window.localStorage.getItem(`tensorflowjs_models/${name}/info`);
    } catch { return false; }
  }

  get ready() { return !!this.model && !!this.tokenizer; }
  get built() { return !!this.model; }
}

export { cleanText };
function formatThaiQA(pairs) {
  return pairs
    .map((p) => `คำถาม: ${(p.question || "").trim()}\nคำตอบ: ${(p.answer || "").trim()}`)
    .filter((s) => s.includes("คำถาม:") && s.includes("คำตอบ:"))
    .join("\n\n");
}

export async function loadTrainDataset(path = DEFAULT_DATASET_PATH) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load dataset: ${res.status} ${res.statusText}`);
  const text = await res.text();
  try {
    const data = JSON.parse(text);
    if (Array.isArray(data) && data.length > 0 && typeof data[0] === "object") {
      return formatThaiQA(data);
    }
    if (Array.isArray(data)) {
      return data.filter((x) => typeof x === "string" && x.trim()).join("\n");
    }
    return text;
  } catch {
    return text;
  }
}
