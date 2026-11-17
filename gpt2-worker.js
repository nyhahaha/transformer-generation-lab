import { loadGPT2 } from './gpt2-runtime.js';

let model;
self.onmessage = async ({ data: { request, action, ids } }) => {
  try {
    if (action === 'load') {
      model = await loadGPT2(progress => self.postMessage({ progress }));
      self.postMessage({ request, norms: model.norms });
      return;
    }
    if (!model) throw new Error('GPT‑2 权重尚未加载');
    if (action === 'prefill') model.reset();
    const rows = ids.map((id, index) => {
      const row = model.forward(id);
      if (action === 'prefill') self.postMessage({ progress: { phase: 'prefill', done: index + 1, total: ids.length } });
      return row;
    });
    const logits = model.logits(rows.at(-1));
    self.postMessage({ request, rows, logits });
  } catch (error) {
    self.postMessage({ request, error: error.message });
  }
};
