const PAD = "<PAD>";
const UNK = "<UNK>";
const BOS = "<BOS>";
const EOS = "<EOS>";
const QUESTION = "<|question|>";
const ANSWER = "<|answer|>";
const SEP = "<|sep|>";
const SPECIAL = [PAD, UNK, BOS, EOS, QUESTION, ANSWER, SEP];

function isThai(ch) {
  const cp = ch.codePointAt(0);
  return (cp >= 0x0E00 && cp <= 0x0E7F);
}

function isLatin(ch) {
  const cp = ch.codePointAt(0);
  return (cp >= 0x0041 && cp <= 0x005A) || (cp >= 0x0061 && cp <= 0x007A);
}

function scriptTag(ch) {
  if (isThai(ch)) return "TH";
  if (isLatin(ch)) return "LA";
  return "OT";
}

function bytesToSymbols(text) {
  return Array.from(text).map((ch) => Array.from(ch).join("") + "</w>");
}

function getPairs(symbols) {
  const pairs = new Map();
  for (const word of symbols) {
    for (let i = 0; i < word.length - 1; i++) {
      const key = word[i] + "\u0000" + word[i + 1];
      pairs.set(key, (pairs.get(key) || 0) + 1);
    }
  }
  return pairs;
}

function mergePair(symbols, a, b) {
  const merged = a + b;
  return symbols.map((word) => {
    const out = [];
    for (let i = 0; i < word.length; i++) {
      if (i < word.length - 1 && word[i] === a && word[i + 1] === b) {
        out.push(merged);
        i++;
      } else out.push(word[i]);
    }
    return out;
  });
}

function scriptAwareSplit(text) {
  const parts = [];
  let cur = "";
  let curScript = null;
  for (const ch of text) {
    const s = scriptTag(ch);
    if (curScript === null) { curScript = s; cur += ch; }
    else if (s === curScript || s === "OT" || curScript === "OT") { cur += ch; if (s !== "OT") curScript = s; }
    else { parts.push(cur); cur = ch; curScript = s; }
  }
  if (cur) parts.push(cur);
  return parts;
}

export class CharTokenizer {
  constructor(text = "") {
    this.type = "char";
    this.special = SPECIAL;
    this.chars = [];
    this.charToId = {};
    this.idToChar = {};
    if (text) this.fit(text);
  }
  fit(text) {
    const chars = new Set();
    for (const ch of text) chars.add(ch);
    this.chars = Array.from(chars).sort();
    const vocab = [...this.special, ...this.chars];
    this.charToId = {}; this.idToChar = {};
    vocab.forEach((ch, i) => { this.charToId[ch] = i; this.idToChar[i] = ch; });
    return this;
  }
  get vocabSize() { return this.special.length + this.chars.length; }
  encode(text, addSpecial = false) {
    const ids = Array.from(text).map((ch) => this.charToId[ch] ?? this.charToId[UNK]);
    if (!addSpecial) return ids;
    return [this.charToId[BOS], ...ids, this.charToId[EOS]];
  }
  decode(ids) {
    return ids.map((id) => this.idToChar[id] ?? "").filter((ch) => !this.special.includes(ch)).join("");
  }
  toJSON() { return { type: "char", chars: this.chars }; }
  static fromJSON(data) { return new CharTokenizer().fit((data.chars || []).join(""))._restore(data); }
  _restore(data) {
    this.chars = data.chars || [];
    const vocab = [...this.special, ...this.chars];
    this.charToId = {}; this.idToChar = {};
    vocab.forEach((ch, i) => { this.charToId[ch] = i; this.idToChar[i] = ch; });
    return this;
  }
}

export class BPETokenizer {
  constructor(text = "", opts = {}) {
    this.type = "bpe";
    this.special = SPECIAL;
    this.merges = [];
    this.vocab = [];
    this.tokenToId = {};
    this.idToToken = {};
    this.targetVocabSize = opts.vocabSize ?? 300;
    this.scriptAware = opts.scriptAware !== false;
    if (text) this.fit(text, opts);
  }
  fit(text, opts = {}) {
    this.targetVocabSize = opts.vocabSize ?? this.targetVocabSize;
    this.scriptAware = opts.scriptAware !== false;
    let parts = this.scriptAware ? scriptAwareSplit(text) : [text];
    let symbols = parts.map((p) => Array.from(p).join("") + "</w>");
    const base = Array.from(new Set(symbols.flat())).sort();
    this.merges = [];
    let vocab = [...base];
    const maxMerges = Math.max(0, this.targetVocabSize - this.special.length - base.length);
    for (let i = 0; i < maxMerges; i++) {
      const pairs = getPairs(symbols);
      let best = null; let bestCount = 0;
      for (const [k, c] of pairs.entries()) if (c > bestCount) { bestCount = c; best = k; }
      if (!best || bestCount < 2) break;
      const [a, b] = best.split("\u0000");
      symbols = mergePair(symbols, a, b);
      this.merges.push([a, b]);
      const tok = a + b;
      if (!vocab.includes(tok)) vocab.push(tok);
    }
    this.vocab = [...this.special, ...vocab];
    this.tokenToId = {}; this.idToToken = {};
    this.vocab.forEach((t, i) => { this.tokenToId[t] = i; this.idToToken[i] = t; });
    return this;
  }
  _encodeWord(word) {
    let symbols = Array.from(word + "</w>");
    for (const [a, b] of this.merges) {
      const out = [];
      for (let i = 0; i < symbols.length; i++) {
        if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) { out.push(a + b); i++; }
        else out.push(symbols[i]);
      }
      symbols = out;
      if (symbols.length === 1) break;
    }
    return symbols.map((s) => this.tokenToId[s] ?? this.tokenToId[UNK]);
  }
  encode(text, addSpecial = false) {
    const parts = this.scriptAware ? scriptAwareSplit(text) : [text];
    const ids = parts.flatMap((p) => this._encodeWord(p));
    if (!addSpecial) return ids;
    return [this.tokenToId[BOS], ...ids, this.tokenToId[EOS]];
  }
  decode(ids) {
    const text = ids
      .map((id) => this.idToToken[id] ?? "")
      .filter((t) => !this.special.includes(t))
      .join("")
      .replaceAll("</w>", " ");
    return text;
  }
  get vocabSize() { return this.vocab.length; }
  toJSON() { return { type: "bpe", merges: this.merges, vocab: this.vocab, scriptAware: this.scriptAware, targetVocabSize: this.targetVocabSize }; }
  static fromJSON(data) {
    const t = new BPETokenizer("", { vocabSize: data.targetVocabSize ?? 300, scriptAware: data.scriptAware !== false });
    t.merges = data.merges || [];
    t.vocab = data.vocab || [...SPECIAL];
    t.tokenToId = {}; t.idToToken = {};
    t.vocab.forEach((tok, i) => { t.tokenToId[tok] = i; t.idToToken[i] = tok; });
    return t;
  }
}

export function createTokenizer(type, text, opts) {
  return type === "bpe" ? new BPETokenizer(text, opts) : new CharTokenizer(text);
}

export function tokenizerFromJSON(data) {
  if (!data) return null;
  return data.type === "bpe" ? BPETokenizer.fromJSON(data) : CharTokenizer.fromJSON(data);
}

export const SPECIAL_TOKENS = { PAD, UNK, BOS, EOS, QUESTION, ANSWER, SEP };

export function formatQA(pairs) {
  return pairs.map((p) => `${QUESTION} ${p.question} ${SEP} ${ANSWER} ${p.answer}`).join(" ");
}

export function parseQA(text) {
  const pairs = [];
  const qRe = new RegExp(`\\${QUESTION}\\s+([^${SEP.replace(/[.*+?^${}()|[\]\\\\]/g, '\\$&')}]+?)\\s*\\${SEP.replace(/[.*+?^${}()|[\]\\\\]/g, '\\$&')}\\s*\\${ANSWER}\\s+`, "g");
  let m;
  while ((m = qRe.exec(text)) !== null) {
    const rest = text.slice(m.index + m[0].length);
    const nextQ = rest.indexOf(QUESTION);
    const answer = nextQ === -1 ? rest.trim() : rest.slice(0, nextQ).trim();
    pairs.push({ question: m[1].trim(), answer });
  }
  return pairs;
}
