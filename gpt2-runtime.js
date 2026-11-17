import { WIDTH, LAYERS, HEADS, HEAD_WIDTH, VOCAB_SIZE, projectAttention, splitQKV } from './engine.js';

const MODEL_URL = new URL('./gpt2/q8-v1/', import.meta.url);
const FLOATS = Float32Array.BYTES_PER_ELEMENT;
const RANGE_BYTES = 4 * 1024 * 1024;

async function fetchChecked(fetcher, url, options) {
  const response = await fetcher(url, { cache: 'force-cache', ...options });
  if (!response.ok) throw new Error(`资源响应 HTTP ${response.status}（${url.pathname.split('/').at(-1)}）`);
  return response;
}

async function readPartition(response, bytes, start, end, advance, file) {
  let offset = start;
  if (response.body) {
    const reader = response.body.getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (offset + value.length > end) {
        await reader.cancel();
        throw new Error(`权重长度有误（${file}）`);
      }
      bytes.set(value, offset);
      offset += value.length;
      advance(value.length);
    }
  } else {
    const part = new Uint8Array(await response.arrayBuffer());
    if (part.length > end - start) throw new Error(`权重长度有误（${file}）`);
    bytes.set(part, offset);
    offset += part.length;
    advance(part.length);
  }
  if (offset !== end) throw new Error(`权重下载不完整（${file}）`);
}

async function readShard(shard, fetcher, advance) {
  const url = new URL(shard.file, MODEL_URL);
  const bytes = new Uint8Array(shard.bytes);
  const range = (start, end) => fetchChecked(fetcher, url, {
    headers: { Range: `bytes=${start}-${end - 1}` },
  });
  const readRange = (response, start, end) => {
    if (response.status !== 206 || response.headers.get('content-range') !== `bytes ${start}-${end - 1}/${bytes.length}`) {
      throw new Error(`权重分段有误（${shard.file}）`);
    }
    return readPartition(response, bytes, start, end, advance, shard.file);
  };

  const firstEnd = Math.min(RANGE_BYTES, bytes.length);
  const first = await range(0, firstEnd);
  if (first.status === 200) {
    // Static development servers may respond with the entire file.
    await readPartition(first, bytes, 0, bytes.length, advance, shard.file);
  } else {
    let next = firstEnd;
    await Promise.all([
      readRange(first, 0, firstEnd),
      ...Array.from({ length: 3 }, async () => {
        while (next < bytes.length) {
          const start = next;
          const end = Math.min(start + RANGE_BYTES, bytes.length);
          next = end;
          await readRange(await range(start, end), start, end);
        }
      }),
    ]);
  }
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const hex = Array.from(hash, value => value.toString(16).padStart(2, '0')).join('');
  if (hex !== shard.sha256) throw new Error(`权重校验失败（${shard.file}）`);

  return Object.fromEntries(Object.entries(shard.tensors).map(([key, { kind, offset, shape, scaleOffset }]) => {
    const size = shape.reduce((product, dimension) => product * dimension, 1);
    const tensor = kind === 'float' ? new Float32Array(bytes.buffer, offset, size) : {
      weight: new Int8Array(bytes.buffer, offset, size),
      scale: new Float32Array(bytes.buffer, scaleOffset, shape[0]),
      rows: shape[0],
      columns: shape[1],
    };
    return [key, tensor];
  }));
}

export async function loadGPT2(progress = () => {}, fetcher = fetch) {
  const manifest = await (await fetchChecked(fetcher, new URL('weights.json', MODEL_URL))).json();
  if (manifest.format !== 'gpt2-row-int8-v1' || manifest.width !== WIDTH || manifest.depth !== LAYERS
      || manifest.heads !== HEADS || manifest.vocab !== VOCAB_SIZE || manifest.shards.length !== LAYERS + 1) {
    throw new Error('GPT‑2 权重配置不匹配');
  }
  let loaded = 0;
  let reported = 0;
  let nextShard = 0;
  const weights = {};
  const advance = length => {
    loaded += length;
    if (loaded - reported >= 512 * 1024 || loaded === manifest.totalBytes) {
      reported = loaded;
      progress({ phase: 'download', loaded, total: manifest.totalBytes });
    }
  };
  progress({ phase: 'download', loaded: 0, total: manifest.totalBytes });
  await Promise.all(Array.from({ length: 3 }, async () => {
    while (nextShard < manifest.shards.length) {
      const shard = manifest.shards[nextShard++];
      Object.assign(weights, await readShard(shard, fetcher, advance));
    }
  }));
  return new GPT2(weights);
}

export function matvec(x, matrix, bias) {
  const { weight, scale, rows, columns } = matrix;
  const output = new Float32Array(rows);
  for (let row = 0, offset = 0; row < rows; row++, offset += columns) {
    let a = 0;
    let b = 0;
    let c = 0;
    let d = 0;
    let i = 0;
    for (; i + 3 < columns; i += 4) {
      a += x[i] * weight[offset + i];
      b += x[i + 1] * weight[offset + i + 1];
      c += x[i + 2] * weight[offset + i + 2];
      d += x[i + 3] * weight[offset + i + 3];
    }
    for (; i < columns; i++) a += x[i] * weight[offset + i];
    output[row] = (a + b + c + d) * scale[row] + (bias ? bias[row] : 0);
  }
  return output;
}

const add = (a, b) => a.map((value, i) => value + b[i]);

function layerNorm(x, [gain, bias]) {
  const mean = x.reduce((sum, value) => sum + value, 0) / x.length;
  const variance = x.reduce((sum, value) => sum + (value - mean) ** 2, 0) / x.length;
  const divisor = Math.sqrt(variance + 1e-5);
  const normalized = x.map(value => (value - mean) / divisor);
  return { stats: [mean, variance], normalized, out: normalized.map((value, i) => value * gain[i] + bias[i]) };
}

const gelu = value => 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));

class GPT2 {
  constructor(weights) {
    const norm = key => [weights[key + '.weight'], weights[key + '.bias']];
    const affine = key => [weights[key + '.weight'], weights[key + '.bias']];
    this.embedding = weights.wte;
    this.positions = weights.wpe;
    this.finalNorm = norm('ln_f');
    this.blocks = Array.from({ length: LAYERS }, (_, i) => {
      const prefix = `h.${i}.`;
      return {
        ln1: norm(prefix + 'ln_1'), ln2: norm(prefix + 'ln_2'),
        qkv: affine(prefix + 'attn.c_attn'), att: affine(prefix + 'attn.c_proj'),
        up: affine(prefix + 'mlp.c_fc'), down: affine(prefix + 'mlp.c_proj'),
      };
    });
    this.norms = { blocks: this.blocks.map(({ ln1, ln2 }) => ({ ln1, ln2 })), final: this.finalNorm };
    this.reset();
  }

  reset() {
    this.cache = this.blocks.map(() => []);
    this.length = 0;
  }

  forward(id) {
    const index = this.length++;
    const { weight, scale } = this.embedding;
    const token = new Float32Array(WIDTH);
    for (let i = 0; i < WIDTH; i++) token[i] = weight[id * WIDTH + i] * scale[id];
    const pos = this.positions.subarray(index * WIDTH, (index + 1) * WIDTH);
    const initial = add(token, pos);
    let x = initial;
    const layers = this.blocks.map((params, i) => {
      const input = x;
      const ln1 = layerNorm(input, params.ln1);
      const qkv = matvec(ln1.out, ...params.qkv);
      this.cache[i].push(Array.from({ length: HEADS }, (_, head) => splitQKV(qkv, head)));
      const joined = new Float32Array(WIDTH);
      for (let head = 0; head < HEADS; head++) {
        const history = this.cache[i].map(item => item[head]);
        const [query] = history[index];
        const { mixed } = projectAttention(query, history.map(row => row[1]), history.map(row => row[2]), index);
        joined.set(mixed, head * HEAD_WIDTH);
      }
      const projected = matvec(joined, ...params.att);
      const residual = add(input, projected);
      const ln2 = layerNorm(residual, params.ln2);
      const expanded = matvec(ln2.out, ...params.up);
      const activated = expanded.map(gelu);
      const compressed = matvec(activated, ...params.down);
      x = add(residual, compressed);
      return { input, ln1, qkv, joined, projected, residual, ln2, expanded, gelu: activated, compressed, output: x };
    });
    return { token, pos: pos.slice(), initial, layers, final: layerNorm(x, this.finalNorm) };
  }

  logits(row) {
    return matvec(row.final.out, this.embedding);
  }
}
