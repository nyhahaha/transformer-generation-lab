const byId = id => document.getElementById(id);
const pad = n => String(n).padStart(2, '0');
const num = (n, digits = 3) => Number.isFinite(n) ? (Math.abs(n) < 0.5 * 10 ** -digits ? 0 : n).toFixed(digits) : '−∞';
const pct = n => `${(n * 100).toFixed(n > 0 && n < 0.001 ? 3 : 1)}%`;
const label = id => id === EOS ? 'EOS' : MODEL.vocab[id];
const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const showToken = id => escape(label(id));
const promptIds = encode(PROMPT);
const initialConfig = () => ({ temperature: 1, method: 'greedy', k: 5, p: 0.9, seed: 42 });
const startState = () => ({ stage: 0, layer: 0, head: 0, row: promptIds.length - 1, round: 0, tabs: {}, history: [], ids: [...promptIds], config: initialConfig(), done: false, trace: forward(promptIds) });
let state = startState();

const stages = [
  ['切分文本', '字符 → Token ID', '按字符切分文本，再查词表，把每个字符映射成一个整数 ID。点击 token 可查看它的编号和位置。'],
  ['词向量与位置', '词嵌入 + 位置嵌入', '查两张训练得到的表，再逐维相加。每个位置得到一个 16 维向量。'],
  ['注意力前归一化', 'LayerNorm 1', '把每个 token 自己的 16 个特征标准化，再用训练得到的参数缩放和平移。'],
  ['Q / K / V 投影', '线性映射与分头', '同一个输入经过三个线性变换，得到查询 Q、键 K、值 V。切换注意力头可查看各自的 8 维切片。'],
  ['缩放与因果遮罩', '点积 → 缩放 → 遮罩', '每个查询与各个键做点积。除以 √8 后，把未来位置的分数设为 −∞。'],
  ['注意力权重与聚合', 'Softmax → 加权求和', '沿每一行做 Softmax，得到各位置的权重，然后用这些权重对 V 向量求和。'],
  ['合并头与残差', '拼接 → 投影 → 残差相加', '把两个头在同一位置的输出拼接，通过输出投影，再加回该层的输入。'],
  ['前馈前归一化', 'LayerNorm 2', '在送入前馈网络之前，对刚更新的每个 token 向量再次做层归一化。'],
  ['前馈网络与残差', '升维 → GELU → 降维 → 残差', '前馈网络独立处理每个位置。先从 16 维展开到 64 维，经 GELU 激活后降回 16 维，再加回残差。'],
  ['最终归一化与打分', '最终 LayerNorm → 词表 logits', '对最后一个解码块的输出做归一化，取序列最后一个位置的向量，与每个词嵌入做点积。'],
  ['候选概率与采样', '温度 → Softmax → 筛选 → 选择', '用温度缩放 logits，再计算全词表概率。采样策略决定保留哪些候选，以及怎样从中选出一个 token。'],
  ['拼接与循环生成', '追加 → 再推理 / 停止', '将结果解码为文本并追加到序列。只要还未生成 EOS 或达到步数上限，下一轮就会使用更新后的序列。'],
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
  return state.history[state.round] ?? { trace: state.trace, config: state.config };
}

function renderNav() {
  if (!byId('stage-nav').children.length) {
    byId('stage-nav').innerHTML = stages.map(([title], index) => `${({ 0: '<div class="stage-group">输入</div>', 2: '<div class="stage-group">解码层 × 2</div>', 9: '<div class="stage-group">输出与生成</div>' })[index] ?? ''}<button type="button" class="stage-nav-button" data-stage="${index}"><span>${pad(index + 1)}</span>${title}<span class="stage-chevron" aria-hidden="true">›</span></button>`).join('');
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
  const html = state.history.length ? state.history.map((round, i) => `<button type="button" data-round="${i}" class="generated-token ${round.chosen === EOS ? 'eos' : ''} ${i === state.round ? 'is-selected' : ''}" aria-pressed="${i === state.round}" aria-label="第 ${i + 1} 轮，${showToken(round.chosen)}">${showToken(round.chosen)}</button>`).join('') : '<span class="placeholder">等待生成</span>';
  if (byId('generated').innerHTML !== html) byId('generated').innerHTML = html;
  byId('generate').disabled = byId('run').disabled = state.done;
  byId('round').innerHTML = state.history.map((round, i) => `<option value="${i}">第 ${i + 1} 轮 · ${showToken(round.chosen)}</option>`).join('') + (state.done ? '' : `<option value="${state.history.length}">第 ${state.history.length + 1} 轮 · 待生成</option>`);
  byId('round').value = state.round;
  const { temperature, method, k, p, seed } = state.config;
  byId('method').value = method;
  byId('temperature').value = temperature;
  byId('temperature-value').textContent = temperature.toFixed(1);
  byId('method-settings').hidden = method === 'greedy';
  const threshold = byId('threshold');
  const config = method === 'top-k' ? [1, MODEL.vocab.length, 1, k] : [0.05, 1, 0.01, p];
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

function matrix(title, values, trace, { seqColumns = false, rowLabels, mode, rowOffset = 0, details = true } = {}) {
  const max = mode === 'probability' ? 1 : Math.max(0, ...values.flat().filter(Number.isFinite).map(Math.abs));
  const colNames = values[0].map((_, i) => seqColumns ? `${showToken(trace.ids[i])}·${i}` : `${i}`);
  function rowName(i) {
    return rowLabels ? rowLabels[i] : `${showToken(trace.ids[i + rowOffset])}·${i + rowOffset}`;
  }
  const body = values.map((row, i) => `<tr${!rowLabels && state.row === i + rowOffset ? ' class="active-row"' : ''}><th scope="row" class="row-name">${rowName(i)}</th>${row.map((value, j) => {
    const formatted = num(value, mode === 'probability' ? 2 : 2);
    const detail = `${title} [${rowName(i)}, ${colNames[j]}] = ${num(value, 6)}`;
    return `<td><button type="button" class="matrix-cell${Number.isFinite(value) ? '' : ' masked'}"${heat(value, mode, max)} data-cell="${escape(detail)}"${rowLabels ? '' : ` data-row="${i + rowOffset}"`} aria-label="${escape(detail)}" title="${escape(detail)}">${formatted}</button></td>`;
  }).join('')}</tr>`).join('');
  return `<section class="matrix-card"><h4 class="matrix-title"><span>${title}</span><small>${values.length} × ${values[0].length}</small></h4><div class="matrix-scroll"><table class="matrix" aria-label="${title}"><thead><tr><th scope="col">${seqColumns ? 'Q ＼ K' : '位置'}</th>${colNames.map(name => `<th scope="col">${name}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>${details ? '<p class="cell-detail" aria-live="polite">点击单元格查看精确值</p>' : ''}</section>`;
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
  const selected = trace.ids[state.row];
  return `<div class="token-panel"><p class="section-label">当前序列</p><div class="token-strip">${trace.ids.map((id, i) => `<button type="button" class="token-box ${i >= promptIds.length ? 'is-generated' : ''}" data-token="${i}" aria-pressed="${state.row === i}" aria-label="位置 ${i}，${showToken(id)}，ID ${id}"><strong>${showToken(id)}</strong><small>ID ${id}</small></button>`).join('')}</div><div class="token-details">${metric('已选 token', showToken(selected))}${metric('词表 ID', selected)}${metric('位置索引', state.row)}</div></div>${formula('“' + PROMPT + '” → [' + promptIds.join(', ') + ']')}${explain('位置从 0 开始。虚线表示前面轮次生成后追加的 token。')}`;
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
  return tabs + matrix(part ? `头 ${state.head + 1} · ${part}` : 'QKV 拼接矩阵', [head.q, head.k, head.v, layer.qkv][active], trace) + legend + formula(`Q = LayerNorm₁(X) · W_Q + b_Q\nK 与 V 使用各自的权重和偏差。头 ${state.head + 1} 取每种投影的第 ${state.head * 8}–${state.head * 8 + 7} 维。`);
}

function masks(trace, head) {
  const [tabs, active] = views(['遮罩后', '原始点积', '缩放后', '因果遮罩']);
  const values = [head.masked, head.dotProducts, head.scaled, head.mask][active];
  const names = ['S + M', 'QKᵀ', 'S = QKᵀ / √8', '因果遮罩 M'];
  return tabs + matrix(names[active], values, trace, { seqColumns: true }) + legend + formula('S[i, j] = (qᵢ · kⱼ) / √8\n当 j > i，M[i, j] = −∞；否则 M[i, j] = 0。') + explain('行表示查询位置，列表示键位置。带斜纹的单元格表示该查询无法读取的未来位置。');
}

function weighted(trace, head) {
  const [tabs, active] = views(['注意力权重 A', '值向量 V', '头输出 AV']);
  const sum = head.probabilities[state.row].reduce((a, b) => a + b, 0);
  return tabs + matrix(['权重 A', '值向量 V', '加权结果 H'][active], [head.probabilities, head.v, head.mixed][active], trace, active === 0 ? { seqColumns: true, mode: 'probability' } : {}) + formula(`Aᵢ = softmax(Sᵢ + Mᵢ)\nH[i, d] = Σⱼ A[i, j] × V[j, d]`) + explain(`查询位置 ${state.row}（${showToken(trace.ids[state.row])}）的权重之和为 <strong>${num(sum, 6)}</strong>。推理时关闭 Dropout。`);
}

function combine(trace, layer) {
  const [tabs, active] = views(['残差相加后', '头输出拼接', '输出投影后', '残差来源']);
  return chain(['头 1 + 头 2 拼接', '线性投影', '加回输入']) + tabs + matrix(['残差 R', '各头拼接 H', '投影 O', '该层输入 X'][active], [layer.residual, layer.joined, layer.projected, layer.input][active], trace) + legend + formula('H = concat(H₁, H₂)\nO = H · W_O + b_O\nR = X + O');
}

function mlp(trace, layer) {
  const [tabs, active] = views(['层输出', '升维 64', 'GELU', '降维 16', '残差来源']);
  return chain(['输入 16 维', '线性 64 维', 'GELU', '线性 16 维', '残差相加']) + tabs + matrix(['层输出 Y', '升维 U', '激活 G', '降维 F', '残差 R'][active], [layer.output, layer.expanded, layer.gelu, layer.compressed, layer.residual][active], trace) + legend + formula('U = LayerNorm₂(R) · W₁ + b₁\nG = GELU(U), F = G · W₂ + b₂\nY = R + F') + explain(state.layer === 0 ? '第 1 层输出接入第 2 层。切换到第 2 层可追踪后续计算。' : '第 2 层输出传给最终归一化。');
}

function logitContent(trace) {
  const [tabs, active] = views(['最后位置', '全部位置', '词表打分', '归一化前', '缩放与偏移']);
  const last = trace.ids.length - 1;
  const sorted = trace.logits.map((logit, id) => [id, logit]).sort((a, b) => b[1] - a[1]);
  const final = trace.layers.at(-1).output;
  const plots = [
    () => matrix('最后位置的归一化向量 h', [trace.last], trace, { rowOffset: last }),
    () => matrix('最终层归一化输出', trace.final.out, trace),
    () => matrix('所有词表 logits（按分数降序）', sorted.map(([, value]) => [value]), trace, { rowLabels: sorted.map(([id]) => `${showToken(id)} · ${id}`) }),
    () => matrix('第 2 层输出', final, trace),
    () => matrix('最终 LayerNorm 参数', MODEL.finalNorm, trace, { rowLabels: ['γ', 'β'] }),
  ];
  return tabs + plots[active]() + legend + formula(`h = LayerNorm_final(Y)[${last}]\nlogit[t] = h · E[t]`) + explain(`输出投影与输入词嵌入共用权重。当前最大 logit 对应 <strong>${showToken(sorted[0][0])}</strong>，分数为 ${num(sorted[0][1])}。`);
}

function sampleContent(round) {
  const { trace, config, chosen, random } = round;
  const rows = distribution(trace.logits, config);
  const sample = random ?? randomAt(config.seed, state.round);
  const preview = chosen ?? choose(rows, sample);
  const [tabs, active] = views(['排名前 8', `全部 ${MODEL.vocab.length} 项`, '筛选保留项']);
  const shown = active === 0 ? rows.slice(0, 8) : active === 1 ? rows : rows.filter(row => row.keep);
  const strategy = { greedy: '贪心', 'top-k': `Top-k · k=${config.k}`, 'top-p': `Top-p · p=${config.p.toFixed(2)}` }[config.method];
  const chart = `<div class="prob-scroll" role="table" aria-label="本轮候选概率"><div class="prob-header" role="row"><span role="columnheader">Token</span><span role="columnheader">Logit</span><span role="columnheader">Softmax 概率</span><span role="columnheader">筛选后</span></div>${shown.map(row => `<div class="prob-row${row.id === preview ? ' selected' : ''}" role="row"><strong role="cell" title="词表 ID ${row.id}${row.id === preview ? '，当前选中' : ''}">${showToken(row.id)}${row.id === preview ? ' ✓' : ''}</strong><em role="cell" title="缩放后 logit ${num(row.scaled)}">${num(row.logit, 2)}</em><span class="bar-field" role="cell" title="累计概率 ${pct(row.cumulative)}"><span class="bar-track" aria-hidden="true"><i style="width:${row.prob * 100}%"></i></span><em>${pct(row.prob)}</em></span><em role="cell">${pct(row.filtered)}</em></div>`).join('')}</div>`;
  const picked = rows.find(row => row.id === preview);
  const high = rows.filter(row => row.keep && (row.prob > picked.prob || (row.prob === picked.prob && row.id <= picked.id))).reduce((sum, row) => sum + row.filtered, 0);
  const reason = config.method === 'greedy' ? '贪心策略取最大值，正温度缩放保持原来的排序。' : `随机值 u = ${num(sample, 6)}，落在该项的累计概率区间 [${num(high - picked.filtered, 6)}, ${num(high, 6)})。`;
  return `<div class="metric-row">${metric('本轮策略', strategy)}${metric('温度 T', config.temperature.toFixed(1))}${metric(chosen === undefined ? '待生成' : '已选 token', showToken(preview))}</div><p class="explain">${reason}</p><div class="tabs-label" style="margin-top:10px">候选分布</div>${tabs}${chart}${formula('p[t] = softmax(logit[t] / T)\n筛选后概率 = p[t] / 保留项概率之和')}${explain('Top-k 保留排名前 k 项。Top-p 保留累计概率第一次达到 p 所需的最短前缀。右侧设置作用于待生成轮次，已完成轮次保留原设置。')}`;
}

function decode(ids) {
  return ids.filter(id => id !== EOS).map(label).join('');
}

function sequenceContent(round) {
  const { trace, chosen } = round;
  const old = escape(decode(trace.ids));
  const newText = chosen === undefined ? '等待生成本轮 token' : chosen === EOS ? `${old} <mark>EOS</mark>` : `${old}<mark>${showToken(chosen)}</mark>`;
  const reply = escape(decode(state.history.map(item => item.chosen)));
  const stop = state.done ? state.history.at(-1).chosen === EOS ? '已停止 · 生成了 EOS' : `已停止 · 达到 ${MAX_NEW} 个新 token` : '可继续生成';
  return chain(['更新序列', '重新前向传播', '预测下一个 token']) + `<div class="sequence-box"><small>第 ${state.round + 1} 轮输入</small><p>${old}</p></div><div class="sequence-box"><small>追加本轮结果</small><p>${newText}</p></div><div class="sequence-box"><small>当前回答 · ${stop}</small><p>${reply || '等待生成'}</p></div>` + (state.history.length ? `<div class="history">${state.history.map((entry, i) => `<button type="button" class="tiny-button" data-round="${i}" aria-pressed="${state.round === i}">${i + 1} · ${showToken(entry.chosen)}</button>`).join('')}</div>` : '') + explain('EOS 是停止信号，显示在生成记录中，解码为回答时会跳过。点击任意已生成 token 可回放它对应的完整前向计算。');
}

function renderStage() {
  const round = selectedRound();
  const { trace } = round;
  state.row = Math.min(state.row, trace.ids.length - 1);
  const layer = trace.layers[state.layer];
  const head = layer.attention[state.head];
  const inLayer = state.stage >= 2 && state.stage <= 8;
  const withHead = state.stage >= 3 && state.stage <= 5;
  const [title, name, description] = stages[state.stage];
  byId('stage-intro').innerHTML = `<div class="stage-overline"><span>${pad(state.stage + 1)} / ${name}</span><span class="tag">第 ${state.round + 1} 轮</span></div><h3 id="stage-title">${title}</h3><p>${description}</p>`;
  byId('stage-tools').innerHTML = inLayer || state.stage === 1 ? `<div class="stage-tools">${inLayer ? `<div class="tool-part"><span>解码层</span>${choiceButtons([['第 1 层', 0], ['第 2 层', 1]], 'layer', state.layer)}</div>` : ''}${withHead ? `<div class="tool-part"><span>注意力头</span>${choiceButtons([['头 1', 0], ['头 2', 1]], 'head', state.head)}</div>` : ''}<div class="tool-part"><label for="position">位置</label><select id="position" class="tiny-button">${trace.ids.map((id, i) => `<option value="${i}"${i === state.row ? ' selected' : ''}>${i} · ${showToken(id)}</option>`).join('')}</select></div></div>` : '';
  const weights = MODEL.blocks[state.layer];
  const panels = [
    () => tokens(trace), () => embeddings(trace), () => normContent(trace, layer.input, layer.ln1, weights.ln1, 'LayerNorm 1'),
    () => qkv(trace, layer, head), () => masks(trace, head), () => weighted(trace, head), () => combine(trace, layer),
    () => normContent(trace, layer.residual, layer.ln2, weights.ln2, 'LayerNorm 2'), () => mlp(trace, layer),
    () => logitContent(trace), () => sampleContent(round), () => sequenceContent(round),
  ];
  byId('stage-content').innerHTML = panels[state.stage]();
  byId('next').innerHTML = state.stage === 8 && state.layer === 0 ? '进入第 2 层 <span aria-hidden="true">→</span>' : '下一步 <span aria-hidden="true">→</span>';
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
  if (direction === 1 && state.stage === 8 && state.layer === 0) {
    state.layer = 1;
    go(2);
  } else if (direction === -1 && state.stage === 2 && state.layer === 1) {
    state.layer = 0;
    go(8);
  } else {
    if (direction === 1 && state.stage === 1) state.layer = 0;
    if (direction === -1 && state.stage === 9) state.layer = 1;
    go(state.stage + direction);
  }
}

function selectRound(index) {
  state.round = index;
  state.row = selectedRound().trace.ids.length - 1;
  render();
}

function emitToken() {
  if (state.done) return;
  const random = randomAt(state.config.seed, state.history.length);
  const rows = distribution(state.trace.logits, state.config);
  const chosen = choose(rows, random);
  state.history.push({ trace: state.trace, config: { ...state.config }, chosen, random });
  state.round = state.history.length - 1;
  state.row = state.trace.ids.length - 1;
  state.done = chosen === EOS || state.history.length >= MAX_NEW;
  if (!state.done) {
    state.ids.push(chosen);
    state.trace = forward(state.ids);
  }
}

function applySetting(key, value) {
  state.config[key] = value;
  if (!state.done && state.round !== state.history.length) {
    state.round = state.history.length;
    state.row = state.trace.ids.length - 1;
  }
  render();
}

function restoreFocus(button) {
  const key = Object.keys(button.dataset)[0];
  document.querySelector(`button[data-${key}="${button.dataset[key]}"]`)?.focus({ preventScroll: true });
}

document.addEventListener('click', event => {
  const button = event.target.closest('button');
  if (!button) return;
  const data = button.dataset;
  if (data.stage !== undefined) go(+data.stage, true);
  if (data.round !== undefined) { selectRound(+data.round); restoreFocus(button); }
  if (data.layer !== undefined) { state.layer = +data.layer; renderStage(); restoreFocus(button); }
  if (data.head !== undefined) { state.head = +data.head; renderStage(); restoreFocus(button); }
  if (data.view !== undefined) { state.tabs[state.stage] = +data.view; renderStage(); restoreFocus(button); }
  if (data.token !== undefined) { state.row = +data.token; renderStage(); restoreFocus(button); }
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
  const { trace } = selectedRound();
  const layer = trace.layers[state.layer];
  if (state.stage === 5) {
    const sum = layer.attention[state.head].probabilities[state.row].reduce((a, b) => a + b, 0);
    byId('stage-content').querySelector('.explain').innerHTML = `查询位置 ${state.row}（${showToken(trace.ids[state.row])}）的权重之和为 <strong>${num(sum, 6)}</strong>。推理时关闭 Dropout。`;
  } else {
    const n = state.stage === 2 ? layer.ln1 : layer.ln2;
    const [avg, variance] = n.stats[state.row];
    byId('stage-content').querySelector('.metric-row').innerHTML = metric('位置', `${state.row} · ${showToken(trace.ids[state.row])}`) + metric('均值 μ', num(avg)) + metric('方差 σ²', num(variance));
  }
}

byId('stage-tools').addEventListener('change', event => {
  if (event.target.id === 'position') { state.row = Number(event.target.value); renderStage(); byId('position').focus({ preventScroll: true }); }
});
byId('previous').addEventListener('click', () => navigate(-1));
byId('next').addEventListener('click', () => navigate(1));
byId('reset').addEventListener('click', () => { state = startState(); render(); });
byId('generate').addEventListener('click', () => { emitToken(); go(10); });
byId('run').addEventListener('click', () => { while (!state.done) emitToken(); go(11); });
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
const firstDistribution = distribution(state.trace.logits, initialConfig());
byId('art-next').textContent = label(firstDistribution[0].id);
document.querySelectorAll('.art-bars i').forEach((bar, i) => { bar.style.height = `${Math.max(7, firstDistribution[i].prob * 100)}%`; });
render();
