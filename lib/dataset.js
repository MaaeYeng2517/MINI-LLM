export const DEFAULT_TRAIN_PATH = "/data/train.json";

export function normalizeText(text) {
  return (text || "")
    .replace(/\r\n?/g, "\n")
    .replace(/[\t\f\v]+/g, " ")
    .replace(/ +/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function cleanText(text, opts = {}) {
  let t = normalizeText(text);
  if (opts.lowercase) t = t.toLowerCase();
  if (opts.stripPunct) t = t.replace(/[^\p{L}\p{N}\s.,!?'"()\-:;]/gu, "");
  if (opts.maxChars) t = t.slice(0, opts.maxChars);
  return normalizeText(t);
}

export function inspectDataset(text) {
  const raw = text || "";
  const lines = raw.split(/\r?\n/);
  const nonEmpty = lines.filter((l) => l.trim().length > 0);
  const chars = Array.from(raw);
  const words = raw.trim() ? raw.trim().split(/\s+/) : [];
  const freq = new Map();
  for (const ch of chars) freq.set(ch, (freq.get(ch) || 0) + 1);
  const topChars = [...freq.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12).map(([ch, n]) => ({ ch, n }));
  return {
    chars: chars.length,
    lines: lines.length,
    nonEmptyLines: nonEmpty.length,
    words: words.length,
    uniqueChars: freq.size,
    topChars,
    empty: !raw.trim(),
    warnings: [
      ...(chars.length < 200 ? ["Dataset is very small; training quality will be limited."] : []),
      ...(nonEmpty.length < 2 ? ["Dataset has very few lines; add more examples for better generalization."] : []),
    ],
  };
}

export function splitDataset(text, opts = {}) {
  const valRatio = Math.min(Math.max(opts.valRatio ?? 0.15, 0.05), 0.4);
  const testRatio = Math.min(Math.max(opts.testRatio ?? 0.1, 0), 0.3);
  const clean = cleanText(text, opts);
  const lines = clean.split("\n").filter((l) => l.trim());
  if (lines.length >= 6) {
    const n = lines.length;
    const testN = Math.max(testRatio > 0 ? 1 : 0, Math.floor(n * testRatio));
    const valN = Math.max(1, Math.floor(n * valRatio));
    const trainN = Math.max(1, n - valN - testN);
    const train = lines.slice(0, trainN).join("\n");
    const val = lines.slice(trainN, trainN + valN).join("\n");
    const test = lines.slice(trainN + valN).join("\n");
    return { train, val, test: test || val, mode: "line", stats: { trainLines: trainN, valLines: valN, testLines: Math.max(0, n - trainN - valN) } };
  }
  // char fallback
  const n = clean.length;
  const testN = Math.max(0, Math.floor(n * testRatio));
  const valN = Math.max(1, Math.floor(n * valRatio));
  const trainN = Math.max(1, n - valN - testN);
  return {
    train: clean.slice(0, trainN),
    val: clean.slice(trainN, trainN + valN),
    test: clean.slice(trainN + valN) || clean.slice(trainN, trainN + valN),
    mode: "char",
    stats: { trainChars: trainN, valChars: valN, testChars: Math.max(0, n - trainN - valN) },
  };
}

export async function loadDatasetText(path = DEFAULT_TRAIN_PATH) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load dataset: ${res.status} ${res.statusText}`);
  return res.text();
}
