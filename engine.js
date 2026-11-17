const PROMPT = '你叫什么名字';
const EOS = 1;
const MAX_NEW = 14;
const encode = text => Array.from(text, char => Math.max(2, MODEL.vocab.indexOf(char)));
const add = (a, b) => a.map((v, i) => v + b[i]);
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const mean = a => a.reduce((sum, x) => sum + x, 0) / a.length;
const affine = (x, [w, b]) => x.map(row => {
  const result = b.slice();
  for (let i = 0; i < row.length; i++) for (let j = 0; j < b.length; j++) result[j] += row[i] * w[i][j];
  return result;
});
const normalize = (x, [scale, shift]) => {
  const stats = x.map(row => {
    const average = mean(row);
    return [average, mean(row.map(v => (v - average) ** 2))];
  });
  const normalized = x.map((row, i) => row.map(v => (v - stats[i][0]) / Math.sqrt(stats[i][1] + 1e-5)));
  const out = normalized.map(row => row.map((v, j) => v * scale[j] + shift[j]));
  return { stats, normalized, out };
};
function softmax(row) {
  const max = Math.max(...row);
  const exponentials = row.map(v => Math.exp(v - max));
  const sum = exponentials.reduce((a, b) => a + b, 0);
  return exponentials.map(v => v / sum);
}

function forward(ids) {
  const { width, heads, embedding, position, blocks, finalNorm } = MODEL;
  const span = width / heads;
  const token = ids.map(id => embedding[id]);
  const pos = ids.map((_, i) => position[i]);
  const initial = token.map((row, i) => add(row, pos[i]));
  let x = initial;
  const layers = blocks.map(weights => {
    const input = x;
    const ln1 = normalize(input, weights.ln1);
    const qkv = affine(ln1.out, weights.qkv);
    const attention = Array.from({ length: heads }, (_, head) => {
      const [q, k, v] = [0, 1, 2].map(part => qkv.map(row => {
        const offset = part * width + head * span;
        return row.slice(offset, offset + span);
      }));
      const dotProducts = q.map(query => k.map(key => dot(query, key)));
      const scaled = dotProducts.map(row => row.map(n => n / Math.sqrt(span)));
      const mask = scaled.map((row, i) => row.map((_, j) => j > i ? -Infinity : 0));
      const masked = scaled.map((row, i) => add(row, mask[i]));
      const probabilities = masked.map(softmax);
      const mixed = probabilities.map(row => v[0].map((_, j) => dot(row, v.map(value => value[j]))));
      return { q, k, v, dotProducts, scaled, mask, masked, probabilities, mixed };
    });
    const joined = input.map((_, i) => attention.flatMap(head => head.mixed[i]));
    const projected = affine(joined, weights.att);
    const residual = input.map((row, i) => add(row, projected[i]));
    const ln2 = normalize(residual, weights.ln2);
    const expanded = affine(ln2.out, weights.up);
    const gelu = expanded.map(row => row.map(v => 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3)))));
    const compressed = affine(gelu, weights.down);
    x = residual.map((row, i) => add(row, compressed[i]));
    return { input, ln1, qkv, attention, joined, projected, residual, ln2, expanded, gelu, compressed, output: x };
  });
  const final = normalize(x, finalNorm);
  const last = final.out.at(-1);
  const logits = embedding.map(row => dot(last, row));
  return { ids, token, pos, initial, layers, final, last, logits };
}

function distribution(logits, { temperature, method, k, p }) {
  const scaled = logits.map(x => x / temperature);
  const probabilities = softmax(scaled);
  const sorted = probabilities.map((prob, id) => ({ id, prob, logit: logits[id], scaled: scaled[id] }))
    .sort((a, b) => b.prob - a.prob || a.id - b.id);
  let cumulative = 0;
  const rows = sorted.map((row, i) => {
    const keep = method === 'greedy' ? i === 0 : method === 'top-k' ? i < k : cumulative < p;
    cumulative += row.prob;
    return { ...row, keep, cumulative };
  });
  const mass = rows.reduce((sum, row) => sum + (row.keep ? row.prob : 0), 0);
  return rows.map(row => ({ ...row, filtered: row.keep ? row.prob / mass : 0 }));
}

function randomAt(seed, index) {
  let h = (seed + Math.imul(index + 1, 0x9e3779b9)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x21f0aaad);
  h = Math.imul(h ^ (h >>> 15), 0x735a2d97);
  return ((h ^ (h >>> 15)) >>> 0) / 4294967296;
}

function choose(rows, sample) {
  const allowed = rows.filter(row => row.keep);
  let boundary = 0;
  for (const row of allowed) {
    boundary += row.filtered;
    if (sample < boundary) return row.id;
  }
  return allowed.at(-1).id;
}
