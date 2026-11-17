# Transformer 生成实验室

[打开在线实验](https://nyhahaha.github.io/transformer-generation-lab/)

用「你叫什么名字」观察 OpenAI 预训练的 GPT‑2 Small 如何按字节级 BPE 分词，再经 12 层解码器、全词表打分与采样，逐个生成 token。每层的 12 个注意力头和 768 维中间表示均可检查。GPT‑2 以文本续写为目标，同一中文字符也可能跨多个 token。

## 本地运行

按下文生成本地权重，在仓库根目录运行 `python -m http.server 8000`，再打开浏览器中的 `http://localhost:8000/`。点击「加载 GPT‑2」后，浏览器会下载约 128 MB 的权重并在后台线程执行推理。在线版所需权重由 GitHub Actions 构建并随 GitHub Pages 发布。

修改页面结构 `page.html` 或样式 `style.css` 后运行 `node build.mjs`，刷新根目录的 `index.html`。交互逻辑在 `app.js`，BPE 分词在 `gpt2-tokenizer.js`，前向计算在 `gpt2-runtime.js`。

## 权重来源

权重来自 OpenAI GPT‑2 124M 的原始 TensorFlow 检查点。浏览器版线性权重和共享词嵌入做逐行对称 INT8 量化，其余参数保留 FP32。权重清单包含每个分片的 SHA‑256。[GPT‑2 原仓库](https://github.com/openai/gpt-2) · [分词器来源](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e) · [许可证](gpt2/LICENSE)

使用 Python 3.11+ 和 NumPy，运行 `python pack_gpt2.py /path/to/model.ckpt.index gpt2/q8-v1` 可以重打包权重。`model.ckpt.index` 和 `model.ckpt.data-00000-of-00001` 可以从 [OpenAI 官方下载路径](https://github.com/openai/gpt-2/blob/master/download_model.py) 获取。脚本也支持相同模型的 Safetensors 文件。

交互设计参考 [Transformer Explainer](https://arxiv.org/abs/2408.04619)。
