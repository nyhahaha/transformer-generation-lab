import { PROMPT, MAX_NEW, WIDTH, LAYERS, HEADS, HEAD_WIDTH, VOCAB_SIZE, viewAttention, distribution, randomAt, choose } from './engine.js';
import { GPT2Tokenizer, EOS } from './gpt2-tokenizer.js';

const byId = id => document.getElementById(id);
const pad = n => String(n).padStart(2, '0');
const num = (n, digits = 3) => Number.isFinite(n) ? (Math.abs(n) < 0.5 * 10 ** -digits ? 0 : n).toFixed(digits) : '−∞';
const pct = n => `${(n * 100).toFixed(n > 0 && n < 0.001 ? 3 : 1)}%`;
const label = id => tokenizer?.labels[id] ?? (id === EOS ? 'EOS' : String(id));
const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const showToken = id => escape(label(id));
let tokenizer;
let promptIds = [];
let worker;
let workerRequest = 0;
let workerPending;
let norms;
let busy = false;
const initialConfig = () => ({ temperature: 1, method: 'greedy', k: 5, p: 0.9, seed: 42 });
const startState = () => ({ stage: 0, layer: 0, head: 0, row: Math.max(0, promptIds.length - 1), round: 0, tabs: {}, pages: {}, history: [], ids: [...promptIds], config: initialConfig(), done: false, activations: [], logits: null });
let state = startState();

const stages = [
  ['切分文本', 'UTF‑8 → 字节级 BPE → Token ID', 'GPT‑2 先把文本转换为 UTF‑8 字节，再按预训练的合并规则和词表切成 token。一个汉字可以跨多个 token。'],
  ['词向量与位置', '词嵌入 + 位置嵌入', '查两张训练得到的表，再逐维相加。每个位置得到一个 768 维向量。'],
  ['注意力前归一化', 'LayerNorm 1', '把每个 token 自己的 768 个特征标准化，再用训练得到的参数缩放和平移。'],
  ['Q / K / V 投影', '线性映射与分头', '同一个输入经过线性投影，得到查询 Q、键 K、值 V。切换 12 个注意力头可查看各自的 64 维切片。'],
  ['缩放与因果遮罩', '点积 → 缩放 → 遮罩', '每个查询与各个键做点积。除以 √64 后，把未来位置的分数设为 −∞。'],
  ['注意力权重与聚合', 'Softmax → 加权求和', '沿每一行做 Softmax，得到各位置的权重，然后用这些权重对 V 向量求和。'],
  ['合并头与残差', '拼接 → 投影 → 残差相加', '把 12 个头在同一位置的输出拼接，通过输出投影，再加回该层的输入。'],
  ['前馈前归一化', 'LayerNorm 2', '在送入前馈网络之前，对刚更新的每个 token 向量再次做层归一化。'],
  ['前馈网络与残差', '升维 → GELU → 降维 → 残差', '前馈网络独立处理每个位置。先从 768 维展开到 3072 维，经 GELU 激活后降回 768 维，再加回残差。'],
  ['最终归一化与打分', '最终 LayerNorm → 词表 logits', '对最后一个解码块的输出做归一化，取序列最后一个位置的向量，与每个词嵌入做点积。'],
  ['候选概率与采样', '温度 → Softmax → 筛选 → 选择', '用温度缩放 logits，再计算全词表概率。采样策略决定保留哪些候选，以及怎样从中选出一个 token。'],
  ['拼接与循环生成', '追加 → 再推理 / 停止', '把生成的 token 追加到序列，拼回字节并解码为文本。遇到 EOS 或达到步数上限时停止。'],
];

function drawHero() {
  const canvas = byId('flow-art');
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, 570, 260);
  const points = [[22, 134], [193, 110], [269, 136], [380, 137], [491, 135], [562, 127]];
  for (let lane = 0; lane < 8; lane++) {
    const trace = points.map(([x, y], i) => [x, y + (lane - 3.5) * [22, 14, 8, 8, 13, 18][i]]);
    ctx.beginPath();
    ctx.strokeStyle = `rgba(${lane % 2 ? '142,223,194' : '109,162,175'},${lane === 4 ? '.76' : '.28'})`;
    ctx.lineWidth = lane === 4 ? 2 : 1;
    ctx.moveTo(...trace[0]);
    trace.slice(1).forEach(([x, y], i) => {
      const [previousX, previousY] = trace[i];
      const mid = (previousX + x) / 2;
      ctx.bezierCurveTo(mid, previousY, mid, y, x, y);
    });
    ctx.stroke();
  }
  [[68, 55], [222, 114], [340, 143], [449, 132], [545, 204], [109, 215]].forEach(([x, y]) => {
    ctx.beginPath(); ctx.fillStyle = '#83d3b7'; ctx.arc(x, y, 2, 0, Math.PI * 2); ctx.fill();
  });
}

function choiceButtons(items, key, current) {
  return `<div class="segmented">${items.map(([name, value]) => `<button type="button" data-${key}="${value}" aria-pressed="${value === current}">${name}</button>`).join('')}</div>`;
}

function selectedRound() {
  return state.history[state.round] ?? { length: state.ids.length, logits: state.logits, config: state.config };
}

function composeTrace(round) {
  const ids = state.ids.slice(0, round.length);
  if (!round.logits) return { ids };
  const rows = state.activations.slice(0, round.length);
  const gather = (records, keys) => Object.fromEntries(keys.map(key => [key, records.map(item => item[key])]));
  const norm = items => gather(items, ['stats', 'normalized', 'out']);
  const layers = Array.from({ length: LAYERS }, (_, index) => {
    const items = rows.map(row => row.layers[index]);
    return {
      ...gather(items, ['input', 'qkv', 'joined', 'projected', 'residual', 'expanded', 'gelu', 'compressed', 'output']),
      ln1: norm(items.map(item => item.ln1)), ln2: norm(items.map(item => item.ln2)),
    };
  });
  const final = norm(rows.map(row => row.final));
  return { ids, ...gather(rows, ['token', 'pos', 'initial']), layers, final, last: final.out.at(-1), logits: round.logits };
}

function renderNav() {
  if (!byId('stage-nav').children.length) {
    byId('stage-nav').innerHTML = stages.map(([title], index) => `${({ 0: '<div class="stage-group">输入</div>', 2: '<div class="stage-group">解码层 × 12</div>', 9: '<div class="stage-group">输出与生成</div>' })[index] ?? ''}<button type="button" class="stage-nav-button" data-stage="${index}"><span>${pad(index + 1)}</span>${title}<span class="stage-chevron" aria-hidden="true">›</span></button>`).join('');
  }
  byId('stage-nav').querySelectorAll('button').forEach(button => {
    if (+button.dataset.stage === state.stage) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
  });
  byId('step-count').textContent = `${pad(state.stage + 1)} / ${stages.length}`;
  byId('previous').disabled = state.stage === 0;
  byId('next').disabled = state.stage === stages.length - 1;
}

function renderConsole() {
  const emitted = state.history.filter(round => round.chosen !== EOS).length;
  byId('generation-count').textContent = `${emitted} 个 token`;
  const reply = (tokenizer && decode(state.history.map(round => round.chosen), !state.done)) ?? '';
  byId('decoded-reply').textContent = reply || (emitted ? '等待拼成完整字符' : '');
  byId('decoded-reply').hidden = emitted === 0;
  const html = state.history.length ? state.history.map((round, i) => `<button type="button" data-round="${i}" class="generated-token ${round.chosen === EOS ? 'eos' : ''} ${i === state.round ? 'is-selected' : ''}" aria-pressed="${i === state.round}" aria-label="第 ${i + 1} 轮，${showToken(round.chosen)}">${showToken(round.chosen)}</button>`).join('') : '<span class="placeholder">等待生成</span>';
  if (byId('generated').innerHTML !== html) byId('generated').innerHTML = html;
  byId('generate').disabled = byId('run').disabled = state.done || busy || !state.logits;
  byId('reset').disabled = busy || !tokenizer;
  byId('round').innerHTML = state.history.map((round, i) => `<option value="${i}">第 ${i + 1} 轮 · ${showToken(round.chosen)}</option>`).join('') + (state.done ? '' : `<option value="${state.history.length}">第 ${state.history.length + 1} 轮 · 待生成</option>`);
  byId('round').value = state.round;
  const { temperature, method, k, p, seed } = state.config;
  byId('method').value = method;
  byId('temperature').value = temperature;
  byId('temperature-value').textContent = temperature.toFixed(1);
  byId('method-settings').hidden = method === 'greedy';
  const threshold = byId('threshold');
  const config = method === 'top-k' ? [1, VOCAB_SIZE, 1, k] : [0.05, 1, 0.01, p];
  [threshold.min, threshold.max, threshold.step, threshold.value] = config;
  threshold.setAttribute('aria-label', method === 'top-k' ? '保留候选数 k' : '累计概率 p');
  byId('threshold-label').textContent = method === 'top-k' ? '保留候选数 k' : '累计概率 p';
  byId('threshold-value').textContent = method === 'top-k' ? k : p.toFixed(2);
  if (document.activeElement !== byId('seed')) byId('seed').value = seed;
}

function metric(title, value) {
  return `<div class="metric"><small>${title}</small><strong>${value}</strong></div>`;
}

function heat(value, mode, max) {
  if (!Number.isFinite(value)) return '';
  const amount = Math.min(0.94, Math.abs(value) / (max || 1));
  const color = mode === 'probability' || value >= 0 ? '79,166,136' : '112,150,178';
  return ` style="--cell:rgba(${color},${(0.07 + 0.47 * amount).toFixed(3)})"`;
}

function matrix(title, values, trace, { seqColumns = false, rowLabels, mode, rowOffset = 0 } = {}) {
  const columns = values[0].length;
  const size = seqColumns ? columns : 12;
  const page = Math.max(0, Math.min(state.pages[title] ?? 0, Math.ceil(columns / size) - 1));
  const start = page * size;
  const end = Math.min(columns, start + size);
  const visible = Array.from({ length: end - start }, (_, index) => start + index);
  let max = 1;
  if (mode !== 'probability') {
    max = 0;
    for (const row of values) for (const value of row) if (Number.isFinite(value)) max = Math.max(max, Math.abs(value));
  }
  const colNames = visible.map(i => seqColumns ? `${showToken(trace.ids[i])}·${i}` : `${i}`);
  function rowName(i) {
    return rowLabels ? rowLabels[i] : `${showToken(trace.ids[i + rowOffset])}·${i + rowOffset}`;
  }
  const body = values.map((row, i) => `<tr${!rowLabels && state.row === i + rowOffset ? ' class="active-row"' : ''}><th scope="row" class="row-name">${rowName(i)}</th>${visible.map((column, j) => {
    const value = row[column];
    const detail = `${title} [${rowName(i)}, ${colNames[j]}] = ${num(value, 6)}`;
    return `<td><button type="button" class="matrix-cell${Number.isFinite(value) ? '' : ' masked'}"${heat(value, mode, max)} data-cell="${escape(detail)}"${rowLabels ? '' : ` data-row="${i + rowOffset}"`} aria-label="${escape(detail)}" title="${escape(detail)}">${num(value, 2)}</button></td>`;
  }).join('')}</tr>`).join('');
  const pageChoice = columns > size ? `<label class="matrix-pages"><span>查看列</span><select class="tiny-button" data-matrix-page="${escape(title)}" aria-label="${escape(title)} 列范围">${Array.from({ length: Math.ceil(columns / size) }, (_, i) => `<option value="${i}"${i === page ? ' selected' : ''}>${i * size}–${Math.min(columns, (i + 1) * size) - 1}</option>`).join('')}</select></label>` : '';
  return `<section class="matrix-card"><h4 class="matrix-title"><span>${title}</span><small>${values.length} × ${columns}</small></h4>${pageChoice}<div class="matrix-scroll"><table class="matrix" aria-label="${title}"><thead><tr><th scope="col">${seqColumns ? 'Q ＼ K' : '位置'}</th>${colNames.map(name => `<th scope="col">${name}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div><p class="cell-detail" aria-live="polite">点击单元格查看数值</p></section>`;
}

const legend = '<div class="legend"><span>负值</span><i aria-hidden="true"></i><span>正值</span></div>';
const formula = text => `<p class="formula">${escape(text)}</p>`;
const chain = steps => `<div class="transform-chain">${steps.map(name => `<b>${name}</b>`).join('<span aria-hidden="true">→</span>')}</div>`;
const explain = html => `<p class="explain">${html}</p>`;

function views(names) {
  const chosen = state.tabs[state.stage] ?? 0;
  return [`<div class="tab-line">${choiceButtons(names.map((name, i) => [name, i]), 'view', chosen)}</div>`, chosen];
}

function tokens(trace) {
  if (!tokenizer) return '<div class="token-panel"><p class="section-label">正在读取 GPT‑2 词表…</p></div>';
  const selected = trace.ids[state.row];
  const bytes = Array.from(new TextEncoder().encode(PROMPT), byte => byte.toString(16).padStart(2, '0').toUpperCase());
  return chain(['原始文字', 'UTF‑8 字节', 'BPE 合并', '词表 ID']) + `<div class="token-panel"><p class="section-label">当前序列 · ${trace.ids.length} 个 token</p><div class="token-strip">${trace.ids.map((id, i) => `<button type="button" class="token-box ${i >= promptIds.length ? 'is-generated' : ''}" data-token="${i}" aria-pressed="${state.row === i}" aria-label="位置 ${i}，${showToken(id)}，ID ${id}"><strong>${showToken(id)}</strong><small>ID ${id}</small></button>`).join('')}</div><div class="token-details">${metric('原始 UTF‑8 字节', tokenizer.hex(selected))}${metric('词表 ID', selected)}${metric('位置索引', state.row)}</div></div>${formula('输入 UTF‑8 = ' + bytes.join(' ') + '\n输入 Token ID = [' + promptIds.join(', ') + ']')}${explain('尖括号里的十六进制表示还不能单独解码的字节。按顺序拼回这些字节才得到原文。虚线框为前序轮次追加的 token。')}`;
}

function embeddings(trace) {
  const [tabs, active] = views(['选中位置', '词嵌入表查找', '位置嵌入表查找', '相加后的全序列']);
  const row = state.row;
  const example = `${num(trace.token[row][0])} + ${num(trace.pos[row][0])} = ${num(trace.initial[row][0])}`;
  return tabs + (active === 0 ? `<div class="matrix-group">${matrix('词嵌入 E[token_id]', [trace.token[row]], trace, { rowOffset: row })}${matrix('位置嵌入 P[position]', [trace.pos[row]], trace, { rowOffset: row })}${matrix('相加后的表示 X₀', [trace.initial[row]], trace, { rowOffset: row })}</div>` : matrix(['', '词嵌入 E', '位置嵌入 P', '输入表示 X₀'][active], [null, trace.token, trace.pos, trace.initial][active], trace)) + legend + formula(`X₀[i, j] = E[idᵢ, j] + P[i, j]\n位置 ${row}，维度 0：${example}`);
}

function normContent(trace, input, norm, params, name) {
  const [tabs, active] = views(['输出', '归一化前', '标准化后', '缩放与偏移']);
  const [avg, variance] = norm.stats[state.row];
  const matrices = [norm.out, input, norm.normalized, params];
  const names = [`${name} 输出`, `${name} 输入`, '标准化结果 z', '训练参数'];
  return tabs + matrix(names[active], matrices[active], trace, active === 3 ? { rowLabels: ['γ', 'β'] } : {}) + legend + `<div class="metric-row">${metric('位置', `${state.row} · ${showToken(trace.ids[state.row])}`)}${metric('均值 μ', num(avg))}${metric('方差 σ²', num(variance))}</div>` + formula('z = (x − μ) / √(σ² + 0.00001)\n输出 = γ ⊙ z + β');
}

function qkv(trace, layer, head) {
  const [tabs, active] = views(['查询 Q', '键 K', '值 V', '投影后全部特征']);
  const part = ['Q', 'K', 'V'][active];
  return tabs + matrix(part ? `头 ${state.head + 1} · ${part}` : 'QKV 拼接矩阵', [head.q, head.k, head.v, layer.qkv][active], trace) + legend + formula(`Q = LayerNorm₁(X) · W_Q + b_Q\nK 与 V 使用各自的权重和偏差。头 ${state.head + 1} 取每种投影的第 ${state.head * HEAD_WIDTH}–${(state.head + 1) * HEAD_WIDTH - 1} 维。`);
}

function masks(trace, head) {
  const [tabs, active] = views(['遮罩后', '原始点积', '缩放后', '因果遮罩']);
  const values = [head.masked, head.dotProducts, head.scaled, head.mask][active];
  const names = ['S + M', 'QKᵀ', 'S = QKᵀ / √64', '因果遮罩 M'];
  return tabs + matrix(names[active], values, trace, { seqColumns: true }) + legend + formula('S[i, j] = (qᵢ · kⱼ) / √64\n当 j > i，M[i, j] = −∞；否则 M[i, j] = 0。') + explain('行表示查询位置，列表示键位置。带斜纹的单元格表示该查询无法读取的未来位置。');
}

function weighted(trace, head) {
  const [tabs, active] = views(['注意力权重 A', '值向量 V', '头输出 AV']);
  const sum = head.probabilities[state.row].reduce((a, b) => a + b, 0);
  return tabs + matrix(['权重 A', '值向量 V', '加权结果 H'][active], [head.probabilities, head.v, head.mixed][active], trace, active === 0 ? { seqColumns: true, mode: 'probability' } : {}) + formula(`Aᵢ = softmax(Sᵢ + Mᵢ)\nH[i, d] = Σⱼ A[i, j] × V[j, d]`) + explain(`查询位置 ${state.row}（${showToken(trace.ids[state.row])}）的权重之和为 <strong>${num(sum, 6)}</strong>。推理时关闭 Dropout。`);
}

function combine(trace, layer) {
  const [tabs, active] = views(['残差相加后', '头输出拼接', '输出投影后', '残差来源']);
  return chain(['拼接 12 个头', '线性投影', '加回输入']) + tabs + matrix(['残差 R', '各头拼接 H', '投影 O', '该层输入 X'][active], [layer.residual, layer.joined, layer.projected, layer.input][active], trace) + legend + formula('H = concat(H₁, …, H₁₂)\nO = H · W_O + b_O\nR = X + O');
}

function mlp(trace, layer) {
  const [tabs, active] = views(['层输出', '升维 3072', 'GELU', '降维 768', '残差来源']);
  return chain(['输入 768 维', '线性 3072 维', 'GELU', '线性 768 维', '残差相加']) + tabs + matrix(['层输出 Y', '升维 U', '激活 G', '降维 F', '残差 R'][active], [layer.output, layer.expanded, layer.gelu, layer.compressed, layer.residual][active], trace) + legend + formula('U = LayerNorm₂(R) · W₁ + b₁\nG = GELU(U), F = G · W₂ + b₂\nY = R + F') + explain(state.layer < LAYERS - 1 ? `第 ${state.layer + 1} 层输出接入第 ${state.layer + 2} 层。` : '第 12 层输出传给最终归一化。');
}

const LIST_SIZE = 24;

function pageList(length, key) {
  const pages = Math.max(1, Math.ceil(length / LIST_SIZE));
  const page = Math.max(0, Math.min(state.pages[key] ?? 0, pages - 1));
  const start = page * LIST_SIZE;
  const end = Math.min(start + LIST_SIZE, length);
  const controls = pages > 1 ? `<div class="list-pages"><button type="button" class="tiny-button" data-list-prev="${key}" ${page === 0 ? 'disabled' : ''}>← 前页</button><span>${start + 1}–${end} / ${length.toLocaleString()}</span><label>页 <input type="number" data-list-page="${key}" aria-label="页码" min="1" max="${pages}" value="${page + 1}"></label><button type="button" class="tiny-button" data-list-next="${key}" ${page + 1 === pages ? 'disabled' : ''}>后页 →</button></div>` : '';
  return { start, end, controls };
}

function logitContent(trace) {
  const [tabs, active] = views(['最后位置', '全部位置', '词表打分', '归一化前', '缩放与偏移']);
  const last = trace.ids.length - 1;
  const best = trace.logits.reduce((id, value, index, all) => value > all[id] ? index : id, 0);
  const final = trace.layers.at(-1).output;
  const logitList = () => {
    const sorted = Array.from(trace.logits, (value, id) => [id, value]).sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    const { start, end, controls } = pageList(sorted.length, 'logits-list');
    const page = sorted.slice(start, end);
    return controls + matrix('词表 logits（按分数降序）', page.map(([, value]) => [value]), trace,
      { rowLabels: page.map(([id], index) => `${index + start + 1} · ${showToken(id)} · ID ${id}`) });
  };
  const plots = [
    () => matrix('最后位置的归一化向量 h', [trace.last], trace, { rowOffset: last }),
    () => matrix('最终层归一化输出', trace.final.out, trace),
    logitList,
    () => matrix('第 12 层输出', final, trace),
    () => matrix('最终 LayerNorm 参数', norms.final, trace, { rowLabels: ['γ', 'β'] }),
  ];
  return tabs + plots[active]() + legend + formula(`h = LayerNorm_final(Y)[${last}]\nlogit[t] = h · E[t]`) + explain(`输出投影与输入词嵌入共用权重。当前最大 logit 对应 <strong>${showToken(best)}</strong>（ID ${best}），分数为 ${num(trace.logits[best])}。`);
}

function sampleContent(round) {
  const { trace, config, chosen, random } = round;
  const rows = distribution(trace.logits, config);
  const sample = random ?? randomAt(config.seed, state.round);
  const preview = chosen ?? choose(rows, sample);
  const [tabs, active] = views(['排名前 8', '全部 50,257 项', '筛选保留项']);
  const group = active === 0 ? rows.slice(0, 8) : active === 1 ? rows : rows.filter(row => row.keep);
  const page = pageList(group.length, active === 2 ? 'kept-list' : 'probability-list');
  const shown = group.slice(page.start, page.end);
  const strategy = { greedy: '贪心', 'top-k': `Top-k · k=${config.k}`, 'top-p': `Top-p · p=${config.p.toFixed(2)}` }[config.method];
  const chart = `<div class="prob-scroll" role="table" aria-label="本轮候选概率"><div class="prob-header" role="row"><span role="columnheader">Token</span><span role="columnheader">Logit</span><span role="columnheader">Softmax 概率</span><span role="columnheader">筛选后</span></div>${shown.map(row => `<div class="prob-row${row.id === preview ? ' selected' : ''}" role="row"><strong role="cell" title="词表 ID ${row.id}${row.id === preview ? '，当前选中' : ''}">${showToken(row.id)}${row.id === preview ? ' ✓' : ''}</strong><em role="cell" title="缩放后 logit ${num(row.scaled)}">${num(row.logit, 2)}</em><span class="bar-field" role="cell" title="累计概率 ${pct(row.cumulative)}"><span class="bar-track" aria-hidden="true"><i style="width:${row.prob * 100}%"></i></span><em>${pct(row.prob)}</em></span><em role="cell">${pct(row.filtered)}</em></div>`).join('')}</div>`;
  const picked = rows.find(row => row.id === preview);
  const high = rows.slice(0, rows.indexOf(picked) + 1).reduce((sum, row) => sum + row.filtered, 0);
  const reason = config.method === 'greedy' ? '贪心策略取最大值，正温度缩放保持原来的排序。' : `随机值 u = ${num(sample, 6)}，落在该项的累计概率区间 [${num(high - picked.filtered, 6)}, ${num(high, 6)})。`;
  return `<div class="metric-row">${metric('本轮策略', strategy)}${metric('温度 T', config.temperature.toFixed(1))}${metric(chosen === undefined ? '待生成' : '已选 token', showToken(preview))}</div><p class="explain">${reason}</p><div class="tabs-label" style="margin-top:10px">候选分布</div>${tabs}${page.controls}${chart}${formula('p[t] = softmax(logit[t] / T)\n筛选后概率 = p[t] / 保留项概率之和')}${explain('Top-k 保留排名前 k 项。Top-p 保留累计概率第一次达到 p 所需的最短前缀。右侧设置作用于待生成轮次，已完成轮次保留原设置。')}`;
}

function decode(ids, streaming = false) {
  return tokenizer.decode(ids, streaming);
}

function sequenceContent(round) {
  const { trace, chosen } = round;
  const old = escape(decode(trace.ids, true));
  const expanded = chosen === undefined || chosen === EOS ? old : escape(decode([...trace.ids, chosen], true));
  const newText = chosen === undefined ? '等待生成本轮 token' : `${expanded} <mark>${showToken(chosen)}</mark>`;
  const reply = escape(decode(state.history.map(item => item.chosen), !state.done));
  const stop = state.done ? state.history.at(-1).chosen === EOS ? '已停止 · 生成了 EOS' : `已停止 · 达到 ${MAX_NEW} 个新 token` : '可继续生成';
  return chain(['追加 token', '经过 12 层解码', '预测下一个 token']) + `<div class="sequence-box"><small>第 ${state.round + 1} 轮输入</small><p>${old}</p></div><div class="sequence-box"><small>追加后的可解码文本与本轮 token</small><p>${newText}</p></div><div class="sequence-box"><small>当前续写 · ${stop}</small><p>${reply || (state.history.length ? '等待拼成完整字符' : '等待生成')}</p></div>` + (state.history.length ? `<div class="history">${state.history.map((entry, i) => `<button type="button" class="tiny-button" data-round="${i}" aria-pressed="${state.round === i}">${i + 1} · ${showToken(entry.chosen)}</button>`).join('')}</div>` : '') + explain('点击任意已生成 token 可回放那一轮。历史位置的表示由因果遮罩保持不变，接下来的计算会复用已有 K、V。');
}

function renderStage() {
  const round = selectedRound();
  const trace = composeTrace(round);
  state.row = Math.max(0, Math.min(state.row, trace.ids.length - 1));
  const inLayer = state.stage >= 2 && state.stage <= 8;
  const withHead = state.stage >= 3 && state.stage <= 5;
  const [title, name, description] = stages[state.stage];
  byId('stage-intro').innerHTML = `<div class="stage-overline"><span>${pad(state.stage + 1)} / ${name}</span><span class="tag">第 ${state.round + 1} 轮</span></div><h3 id="stage-title">${title}</h3><p>${description}</p>`;
  byId('next').innerHTML = state.stage === 8 && state.layer < LAYERS - 1 ? `进入第 ${state.layer + 2} 层 <span aria-hidden="true">→</span>` : '下一步 <span aria-hidden="true">→</span>';
  const selector = (name, id, options, value) => `<div class="tool-part"><label for="${id}">${name}</label><select id="${id}" class="tiny-button">${options.map((option, i) => `<option value="${i}"${i === value ? ' selected' : ''}>${option}</option>`).join('')}</select></div>`;
  byId('stage-tools').innerHTML = trace.logits && (inLayer || state.stage === 1) ? `<div class="stage-tools">${inLayer ? selector('解码层', 'layer', Array.from({ length: LAYERS }, (_, i) => `第 ${i + 1} 层`), state.layer) : ''}${withHead ? selector('注意力头', 'head', Array.from({ length: HEADS }, (_, i) => `头 ${i + 1}`), state.head) : ''}${selector('位置', 'position', trace.ids.map((id, i) => `${i} · ${showToken(id)}`), state.row)}</div>` : '';
  if (state.stage === 0) {
    byId('stage-content').innerHTML = tokens(trace);
    return;
  }
  if (!trace.logits) {
    byId('stage-content').innerHTML = `<div class="load-placeholder"><p>${busy ? '正在准备模型和中间值…' : '加载 GPT‑2 后即可查看本阶段的真实计算。'}</p><button type="button" class="button primary" data-load ${busy ? 'disabled' : ''}>加载 GPT‑2 ↓</button></div>`;
    return;
  }
  const weights = norms.blocks[state.layer];
  const layer = trace.layers[state.layer];
  const head = withHead ? viewAttention(layer.qkv, state.head) : null;
  const panels = [
    () => tokens(trace), () => embeddings(trace), () => normContent(trace, layer.input, layer.ln1, weights.ln1, 'LayerNorm 1'),
    () => qkv(trace, layer, head), () => masks(trace, head), () => weighted(trace, head), () => combine(trace, layer),
    () => normContent(trace, layer.residual, layer.ln2, weights.ln2, 'LayerNorm 2'), () => mlp(trace, layer),
    () => logitContent(trace), () => sampleContent({ ...round, trace }), () => sequenceContent({ ...round, trace }),
  ];
  byId('stage-content').innerHTML = panels[state.stage]();
}

function render() {
  renderNav();
  renderConsole();
  renderStage();
}

function go(stage, scroll = false) {
  state.stage = stage;
  render();
  if (scroll && matchMedia('(max-width: 570px)').matches) byId('stage-intro').scrollIntoView({ block: 'start' });
}

function navigate(direction) {
  if (direction === 1 && state.stage === 8 && state.layer < LAYERS - 1) {
    state.layer++;
    go(2);
  } else if (direction === -1 && state.stage === 2 && state.layer > 0) {
    state.layer--;
    go(8);
  } else {
    if (direction === 1 && state.stage === 1) state.layer = 0;
    if (direction === -1 && state.stage === 9) state.layer = LAYERS - 1;
    go(state.stage + direction);
  }
}

function selectRound(index) {
  state.round = index;
  state.row = selectedRound().length - 1;
  render();
}

async function emitToken() {
  if (state.done) return;
  const config = { ...state.config };
  const random = randomAt(config.seed, state.history.length);
  const rows = distribution(state.logits, config);
  const chosen = choose(rows, random);
  const done = chosen === EOS || state.history.length + 1 >= MAX_NEW;
  const result = done ? null : await askWorker('append', [chosen]);
  state.history.push({ length: state.ids.length, logits: state.logits, config, chosen, random });
  state.round = state.history.length - 1;
  state.row = state.ids.length - 1;
  state.done = done;
  if (result) {
    state.ids.push(chosen);
    state.activations.push(...result.rows);
    state.logits = result.logits;
  }
}

function applySetting(key, value) {
  state.config[key] = value;
  if (!state.done && state.round !== state.history.length) {
    state.round = state.history.length;
    state.row = state.ids.length - 1;
  }
  render();
}

function restoreFocus(button) {
  const key = Object.keys(button.dataset)[0];
  [...document.querySelectorAll('button')].find(item => item.dataset[key] === button.dataset[key])?.focus({ preventScroll: true });
}

document.addEventListener('click', event => {
  const button = event.target.closest('button');
  if (!button) return;
  const data = button.dataset;
  if (data.stage !== undefined) go(+data.stage, true);
  if (data.round !== undefined) { selectRound(+data.round); restoreFocus(button); }
  if (data.view !== undefined) { state.tabs[state.stage] = +data.view; renderStage(); restoreFocus(button); }
  if (data.token !== undefined) { state.row = +data.token; renderStage(); restoreFocus(button); }
  if (data.load !== undefined) loadModel();
  if (data.listPrev || data.listNext) {
    const key = data.listPrev || data.listNext;
    state.pages[key] = (state.pages[key] ?? 0) + (data.listNext ? 1 : -1);
    renderStage();
    restoreFocus(button);
  }
  if (data.cell !== undefined) {
    const card = button.closest('.matrix-card');
    const detail = card.querySelector('.cell-detail');
    if (detail) detail.textContent = data.cell;
    if (data.row !== undefined) {
      state.row = +data.row;
      const position = byId('position');
      if (position) position.value = state.row;
      card.querySelector('.active-row')?.classList.remove('active-row');
      button.closest('tr').classList.add('active-row');
      if ([2, 5, 7].includes(state.stage)) updateSelectedMetrics();
    }
  }
});

function updateSelectedMetrics() {
  const trace = composeTrace(selectedRound());
  const layer = trace.layers[state.layer];
  if (state.stage === 5) {
    const sum = viewAttention(layer.qkv, state.head).probabilities[state.row].reduce((a, b) => a + b, 0);
    byId('stage-content').querySelector('.explain').innerHTML = `查询位置 ${state.row}（${showToken(trace.ids[state.row])}）的权重之和为 <strong>${num(sum, 6)}</strong>。推理时关闭 Dropout。`;
  } else {
    const n = state.stage === 2 ? layer.ln1 : layer.ln2;
    const [avg, variance] = n.stats[state.row];
    byId('stage-content').querySelector('.metric-row').innerHTML = metric('位置', `${state.row} · ${showToken(trace.ids[state.row])}`) + metric('均值 μ', num(avg)) + metric('方差 σ²', num(variance));
  }
}

byId('stage-tools').addEventListener('change', ({ target }) => {
  const key = { position: 'row', layer: 'layer', head: 'head' }[target.id];
  if (!key) return;
  state[key] = Number(target.value);
  renderStage();
  byId(target.id).focus({ preventScroll: true });
});
byId('stage-content').addEventListener('change', ({ target }) => {
  const field = ['matrixPage', 'listPage'].find(key => target.dataset[key]);
  if (!field) return;
  const key = target.dataset[field];
  state.pages[key] = Math.max(0, Math.trunc(Number(target.value)) - (field === 'listPage' ? 1 : 0));
  renderStage();
  [...byId('stage-content').querySelectorAll('input,select')].find(input => input.dataset[field] === key)?.focus({ preventScroll: true });
});

let tokenizerPromise;
function ensureTokenizer() {
  if (tokenizer) return Promise.resolve();
  if (!tokenizerPromise) tokenizerPromise = (async () => {
    const data = await fetch(new URL('./gpt2/tokenizer.json', import.meta.url), { cache: 'force-cache' });
    if (!data.ok) throw new Error(`分词器读取失败（HTTP ${data.status}）`);
    tokenizer = new GPT2Tokenizer(await data.json());
    promptIds = tokenizer.encode(PROMPT);
    state.ids = promptIds.slice();
    state.row = Math.max(promptIds.length - 1, 0);
    byId('prompt-count').textContent = `${promptIds.length} 个 BPE token`;
    render();
  })().catch(error => {
    tokenizerPromise = undefined;
    throw error;
  });
  return tokenizerPromise;
}

function status(message, fraction = null) {
  byId('model-message').textContent = message;
  const progress = byId('model-progress');
  progress.hidden = fraction === null;
  if (fraction !== null) progress.value = fraction;
}

function showProgress(progress) {
  if (progress.phase === 'download') {
    const megabytes = amount => (amount / 1e6).toFixed(1);
    status(`正在下载权重 · ${megabytes(progress.loaded)} / ${megabytes(progress.total)} MB`, progress.loaded / progress.total);
  } else {
    status(`正在处理输入 · ${progress.done} / ${progress.total} 个 token`, progress.done / progress.total);
  }
}

function startWorker() {
  worker = new Worker(new URL('./gpt2-worker.js', import.meta.url), { type: 'module' });
  worker.addEventListener('message', ({ data }) => {
    if (data.progress) showProgress(data.progress);
    if (!workerPending || data.request !== workerPending.request) return;
    const { resolve, reject } = workerPending;
    workerPending = undefined;
    if (data.error) reject(new Error(data.error));
    else resolve(data);
  });
  worker.addEventListener('error', error => {
    workerPending?.reject(new Error(error.message || '浏览器无法启动推理进程'));
    workerPending = undefined;
  });
}

function askWorker(action, ids) {
  const request = ++workerRequest;
  return new Promise((resolve, reject) => {
    workerPending = { request, resolve, reject };
    worker.postMessage({ request, action, ids });
  });
}

function modelError(error) {
  worker?.terminate();
  worker = undefined;
  norms = undefined;
  const { stage, config } = state;
  state = { ...startState(), stage, config };
  byId('load-model').hidden = false;
  status(`${error.message}。可重新加载 GPT‑2。`);
}

async function prefill() {
  const result = await askWorker('prefill', promptIds);
  state.activations = result.rows;
  state.logits = result.logits;
  const first = distribution(result.logits, initialConfig());
  const artLabel = label(first[0].id);
  byId('art-next').textContent = artLabel.length <= 9 ? artLabel : artLabel.slice(0, 7) + '…';
  document.querySelectorAll('.art-bars i').forEach((bar, i) => { bar.style.height = `${Math.max(5, first[i].prob * 100)}%`; });
  status('权重已加载。选择解码层和注意力头，或从控制台逐个生成 token。');
}

async function loadModel() {
  if (busy || norms) return;
  busy = true;
  byId('load-model').disabled = true;
  status('正在读取 GPT‑2 模型配置…', 0);
  render();
  try {
    await ensureTokenizer();
    startWorker();
    norms = (await askWorker('load')).norms;
    await prefill();
    byId('load-model').hidden = true;
  } catch (error) {
    modelError(error);
  } finally {
    busy = false;
    byId('load-model').disabled = false;
    render();
  }
}

async function resetModel() {
  if (busy) return;
  state = startState();
  if (!norms) {
    render();
    return;
  }
  busy = true;
  status('正在恢复初始输入…', 0);
  render();
  try {
    await prefill();
  } catch (error) {
    modelError(error);
  } finally {
    busy = false;
    render();
  }
}

async function generate(all) {
  if (busy || state.done || !state.logits) return;
  busy = true;
  renderConsole();
  status('GPT‑2 正在生成续写…');
  try {
    do {
      await emitToken();
      renderConsole();
    } while (all && !state.done);
    state.stage = all ? 11 : 10;
    status(state.done ? '本次续写已结束，可以逐轮回放。' : '权重已加载，下一轮可以继续生成。');
  } catch (error) {
    modelError(error);
  } finally {
    busy = false;
    render();
  }
}

byId('previous').addEventListener('click', () => navigate(-1));
byId('next').addEventListener('click', () => navigate(1));
byId('reset').addEventListener('click', resetModel);
byId('load-model').addEventListener('click', loadModel);
byId('generate').addEventListener('click', () => generate(false));
byId('run').addEventListener('click', () => generate(true));
byId('round').addEventListener('change', event => selectRound(+event.target.value));
byId('method').addEventListener('change', event => applySetting('method', event.target.value));
byId('temperature').addEventListener('input', event => applySetting('temperature', +event.target.value));
byId('threshold').addEventListener('input', event => applySetting(state.config.method === 'top-k' ? 'k' : 'p', +event.target.value));
byId('seed').addEventListener('change', event => {
  const seed = Math.max(0, Math.min(4294967295, Math.trunc(+event.target.value || 0)));
  event.target.value = seed;
  applySetting('seed', seed);
});
byId('sample-link').addEventListener('click', event => { event.preventDefault(); go(10, true); });

drawHero();
render();
ensureTokenizer().catch(error => status(`${error.message}。点击「加载 GPT‑2」重试。`));
