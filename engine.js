export const PROMPT = '你叫什么名字';
export const MAX_NEW = 14;
export const WIDTH = 768;
export const LAYERS = 12;
export const HEADS = 12;
export const HEAD_WIDTH = WIDTH / HEADS;
export const VOCAB_SIZE = 50257;

export function dot(a, b) {
  let result = 0;
  for (let i = 0; i < a.length; i++) result += a[i] * b[i];
  return result;
}

export function softmax(row) {
  let max = -Infinity;
  for (const value of row) max = Math.max(max, value);
  const exps = Array.from(row, value => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map(value => value / sum);
}

export function splitQKV(qkv, head) {
  return [0, WIDTH, WIDTH * 2].map(part => {
    const offset = part + head * HEAD_WIDTH;
    return qkv.subarray(offset, offset + HEAD_WIDTH);
  });
}

export function projectAttention(query, keys, values, index) {
  const dotProducts = keys.map(key => dot(query, key));
  const scaled = dotProducts.map(value => value / Math.sqrt(HEAD_WIDTH));
  const mask = scaled.map((_, i) => i > index ? -Infinity : 0);
  const masked = scaled.map((value, i) => value + mask[i]);
  const probabilities = softmax(masked);
  const mixed = new Float32Array(HEAD_WIDTH);
  for (let j = 0; j < HEAD_WIDTH; j++) {
    let sum = 0;
    for (let i = 0; i <= index; i++) sum += values[i][j] * probabilities[i];
    mixed[j] = sum;
  }
  return { dotProducts, scaled, mask, masked, probabilities, mixed };
}

export function viewAttention(qkv, head) {
  const split = qkv.map(row => splitQKV(row, head));
  const [q, k, v] = [0, 1, 2].map(part => split.map(row => row[part]));
  const rows = q.map((query, i) => projectAttention(query, k, v, i));
  return { q, k, v, ...Object.fromEntries(Object.keys(rows[0]).map(key => [key, rows.map(row => row[key])])) };
}

export function distribution(logits, { temperature, method, k, p }) {
  const scaled = logits.map(x => x / temperature);
  const probabilities = softmax(scaled);
  // Sort by logit so low-probability ties retain their true ordering.
  const sorted = probabilities.map((prob, id) => ({ id, prob, logit: logits[id], scaled: scaled[id] }))
    .sort((a, b) => b.logit - a.logit || a.id - b.id);
  let cumulative = 0;
  const rows = sorted.map((row, i) => {
    const keep = method === 'greedy' ? i === 0 : method === 'top-k' ? i < k : cumulative < p || p === 1;
    cumulative += row.prob;
    return { ...row, keep, cumulative };
  });
  const mass = rows.reduce((sum, row) => sum + (row.keep ? row.prob : 0), 0);
  for (const row of rows) row.filtered = row.keep ? row.prob / mass : 0;
  return rows;
}

export function randomAt(seed, index) {
  let h = (seed + Math.imul(index + 1, 0x9e3779b9)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x21f0aaad);
  h = Math.imul(h ^ (h >>> 15), 0x735a2d97);
  return ((h ^ (h >>> 15)) >>> 0) / 4294967296;
}

export function choose(rows, sample) {
  let boundary = 0;
  let fallback = rows[0].id;
  for (const row of rows) {
    if (!row.keep) break;
    boundary += row.filtered;
    fallback = row.id;
    if (sample < boundary) return row.id;
  }
  return fallback;
}
