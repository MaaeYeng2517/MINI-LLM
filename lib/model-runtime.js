import { DEFAULT_CONFIG, DEFAULT_DATASET_PATH, MiniLLM, loadTrainDataset } from "./mini-llm";
import { checkpointForModel, runtimeKeyForModel } from "./model-registry";

export const APP_VERSION = "v1.0";
export const CHECKPOINT_NAME = "mini-llm-v1.0-th3/default";
export const RUNTIME_KEY = "mini-llm-v1.0-th3:runtime:default";
const LEGACY_CHECKPOINTS = ["mini-llm-v1.0", "mini-llm-v1"];
const tfState = { promise: null, backend: "" };

export async function ensureTf() {
  if (!tfState.promise) {
    tfState.promise = import("@tensorflow/tfjs").then(async (tf) => {
      await tf.ready();
      tfState.backend = tf.getBackend?.() || "";
      return tf;
    });
  }
  await tfState.promise;
  return tfState;
}

export function createMiniLLM(config = {}) {
  return new MiniLLM({ ...DEFAULT_CONFIG, ...config });
}

export function readRuntimeMeta(modelId = "default") {
  if (typeof window === "undefined") return null;
  try { return JSON.parse(window.localStorage.getItem(runtimeKeyForModel(modelId)) || "null"); } catch { return null; }
}

export function writeRuntimeMeta(patch, modelId = "default") {
  if (typeof window === "undefined") return null;
  const prev = readRuntimeMeta(modelId) || {};
  const next = { ...prev, ...patch, checkpoint: checkpointForModel(patch?.checkpoint || modelId), modelId, updatedAt: new Date().toISOString() };
  try { window.localStorage.setItem(runtimeKeyForModel(modelId), JSON.stringify(next)); } catch {}
  return next;
}

export function checkpointAvailable(name = CHECKPOINT_NAME) {
  if (typeof window === "undefined") return false;
  const names = [name, ...(name === CHECKPOINT_NAME ? LEGACY_CHECKPOINTS : [])];
  return names.some((n) => !!window.localStorage.getItem(`tensorflowjs_models/${n}/info`));
}

export async function loadRuntimeModel(name = CHECKPOINT_NAME, modelId = "default") {
  await ensureTf();
  const llm = createMiniLLM();
  const names = [name, ...LEGACY_CHECKPOINTS];
  let meta = null;
  for (const n of names) { try { meta = await llm.loadCheckpoint(n); break; } catch {} }
  if (!meta) throw new Error("No saved checkpoint found.");
  if (meta?.name && meta.name !== name) { try { await llm.saveCheckpoint(name); } catch {} }
  const runtime = writeRuntimeMeta({ ready: true, loadedAt: new Date().toISOString(), config: llm.config, meta: meta.meta || null, metrics: meta.metrics || llm.getMetrics?.() || null }, modelId);
  return { llm, meta, runtime, backend: tfState.backend };
}

export async function quickTrainRuntime(options = {}, onProgress) {
  await ensureTf();
  const modelId = options.modelId || "default";
  const checkpoint = options.checkpoint || checkpointForModel(modelId);
  const llm = createMiniLLM(options.config || {});
  const datasetPath = options.datasetPath || DEFAULT_DATASET_PATH;
  const text = options.text || await loadTrainDataset(datasetPath);
  const result = await llm.train(text, { epochs: options.epochs ?? 12, batchSize: options.batchSize ?? 32, valSplit: options.valSplit ?? 0.15, earlyStopPatience: options.earlyStopPatience ?? 6 }, onProgress);
  await llm.saveCheckpoint(checkpoint);
  const runtime = writeRuntimeMeta({ ready: true, trainedAt: new Date().toISOString(), datasetPath, config: llm.config, meta: result, metrics: result.metrics || llm.getMetrics?.() || null, checkpoint }, modelId);
  return { llm, result, runtime, backend: tfState.backend };
}
