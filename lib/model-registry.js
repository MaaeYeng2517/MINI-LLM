export const MANIFEST_URL = "/data/models.json";
export const DEFAULT_MODEL_ID = "default";

export async function loadModelManifest() {
  const res = await fetch(MANIFEST_URL, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load model manifest: ${res.status}`);
  const data = await res.json();
  const models = Array.isArray(data.models) ? data.models : [];
  return { ...data, models };
}

export function getModel(models, id) {
  return models.find((m) => m.id === id) || models.find((m) => m.id === DEFAULT_MODEL_ID) || models[0] || null;
}

export function checkpointForModel(modelOrId) {
  if (!modelOrId) return "mini-llm-v1.0-th3/default";
  if (typeof modelOrId === "string") return modelOrId.includes("/") || modelOrId.includes("mini_transformer") ? modelOrId : `mini-llm-v1.0/${modelOrId}`;
  return modelOrId.checkpoint || `mini-llm-v1.0-th3/${modelOrId.id}`;
}

export function runtimeKeyForModel(modelId) {
  return `mini-llm-v1.0-th3:runtime:${modelId || DEFAULT_MODEL_ID}`;
}
