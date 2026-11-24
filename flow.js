const get = id => document.getElementById(id);

const colors = {
  query: '#5279ef',
  key: '#df7782',
  value: '#43a886',
  mixed: '#7e62e8',
};

export function createFlow({ pickRow, selectLayer, selectHead, openStage, layers, heads }) {
  const canvas = get('flow-art');
  const board = canvas.parentElement;
  const ctx = canvas.getContext('2d');
  let frame;

  const options = (count, noun) => Array.from({ length: count }, (_, i) => `<option value="${i}">第 ${i + 1} ${noun}</option>`).join('');
  get('flow-layer').innerHTML = options(layers, '层');
  get('flow-head').innerHTML = options(heads, '头');
  get('flow-layer').addEventListener('change', event => selectLayer(Number(event.target.value)));
  get('flow-head').addEventListener('change', event => selectHead(Number(event.target.value)));

  board.addEventListener('click', event => {
    const button = event.target.closest('button');
    if (!button) return;
    if ('flowStage' in button.dataset) {
      openStage(Number(button.dataset.flowStage));
      return;
    }
    if (!('flowRow' in button.dataset)) return;
    const { flowRow, flowKey } = button.dataset;
    pickRow(Number(flowRow));
    const replacement = board.querySelector(`button[data-flow-row="${flowRow}"]${flowKey === undefined ? ':not([data-flow-key])' : `[data-flow-key="${flowKey}"]`}`);
    replacement?.focus({ preventScroll: true });
    if (replacement && flowKey !== undefined) get('flow-insight').textContent = replacement.getAttribute('aria-label');
  });

  function paint() {
    if (!ctx) return;
    const box = board.getBoundingClientRect();
    if (!box.width || !box.height) return;
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.round(box.width * scale);
    canvas.height = Math.round(box.height * scale);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    const position = (node, side) => {
      const rect = node.getBoundingClientRect();
      return [rect[side] - box.left, rect.top + rect.height / 2 - box.top];
    };
    const ribbon = ([x, y], [u, v], color, width, opacity) => {
      const mid = x + (u - x) * .52;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.globalAlpha = opacity;
      ctx.lineWidth = width;
      ctx.moveTo(x, y);
      ctx.bezierCurveTo(mid, y, mid, v, u, v);
      ctx.stroke();
    };
    const tokens = [...get('flow-tokens').children];
    const projections = [...get('flow-projections').children];
    const mlp = [...get('flow-mlp').children];
    const matrixBox = get('flow-grid').getBoundingClientRect();
    const end = get('flow-output').getBoundingClientRect();
    tokens.forEach((token, i) => {
      const current = token.getAttribute('aria-pressed') === 'true';
      const projection = projections[i];
      const source = position(token, 'right');
      const target = position(mlp[i], 'left');
      const fraction = (i + 0.5) / tokens.length;
      const matrixY = matrixBox.top - box.top + matrixBox.height * fraction;
      const matrixX = matrixBox.left - box.left + matrixBox.width * fraction;
      [...projection.children].forEach((chip, channel) => {
        const color = [colors.query, colors.key, colors.value][channel];
        ribbon(source, position(chip, 'left'), color, current ? 3 : 1.7, current ? .55 : .2);
        const entrance = [
          [matrixBox.left - box.left - 12, matrixY],
          [matrixX, matrixBox.top - box.top - 12],
          [matrixX, matrixBox.bottom - box.top + 12],
        ][channel];
        ribbon(position(chip, 'right'), entrance, color, current ? 3 : 1.35, current ? .58 : .17);
      });
      ribbon([matrixBox.right - box.left, matrixY], target, colors.mixed, current ? 5 : 2.3, current ? .5 : .2);
      if (i === tokens.length - 1) {
        ribbon(position(mlp[i], 'right'), [end.left - box.left, end.top - box.top + 12], colors.mixed, 7, .28);
      }
    });
    ctx.globalAlpha = 1;
  }

  function schedulePaint() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(paint);
  }
  new ResizeObserver(schedulePaint).observe(board);

  return function render({ tokens, start, length, selected, layer, head, round, weights, candidates }) {
    get('flow-layer').value = layer;
    get('flow-head').value = head;
    get('flow-mode').textContent = weights ? `第 ${round} 轮 · 位置 ${start}–${start + tokens.length - 1} · 共 ${length} 个 token` : '结构预览';
    get('flow-insight').textContent = weights
      ? '选择左侧 token 或矩阵中的圆点，查看对应查询位置。圆点颜色越深，注意力权重越高。'
      : '加载 GPT‑2 后，圆点和概率条会显示本轮的实际计算结果。';
    get('flow-tokens').innerHTML = tokens.map(({ name, id }, i) => `<button type="button" class="flow-token" data-flow-row="${start + i}" aria-pressed="${start + i === selected}" title="位置 ${start + i} · ID ${id}"${id === undefined ? ' disabled' : ''}><span>${name}</span><small>${start + i}</small></button>`).join('');
    get('flow-projections').innerHTML = tokens.map(() => '<div class="flow-qkv"><span class="query">Q</span><span class="key">K</span><span class="value">V</span></div>').join('');
    get('flow-mlp').innerHTML = tokens.map((_, i) => `<div class="flow-mlp-lane${start + i === selected ? ' selected' : ''}"><span></span><span></span><span></span></div>`).join('');
    const grid = get('flow-grid');
    grid.style.setProperty('--cells', tokens.length);
    grid.innerHTML = tokens.map((_, row) => tokens.map((_, col) => {
      const q = start + row;
      const k = start + col;
      const weight = weights?.[q][k];
      const masked = col > row;
      const summary = `查询 ${q} · 键 ${k} · ${masked ? '因果遮罩' : weight === undefined ? '等待模型加载' : `权重 ${(weight * 100).toFixed(2)}%`}`;
      return `<button type="button" class="flow-cell${masked ? ' masked' : ''}${q === selected ? ' selected' : ''}" data-flow-row="${q}" data-flow-key="${k}" aria-label="${summary}" title="${summary}" style="--strength:${weight === undefined ? 0 : Math.pow(weight, .48)}"${weight === undefined ? ' disabled' : ''}></button>`;
    }).join('')).join('');
    get('flow-output').innerHTML = candidates.length ? candidates.map(({ name, prob, picked, kept }) =>
      `<div class="flow-result${picked ? ' picked' : ''}${kept ? '' : ' filtered'}"><div><strong>${name}</strong><span>${(prob * 100).toFixed(2)}%</span></div><div class="flow-bar"><i style="width:${prob / candidates[0].prob * 100}%"></i></div></div>`).join('')
      : '<div class="flow-empty">等待模型计算</div>';
    schedulePaint();
  };
}
