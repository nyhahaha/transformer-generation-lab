# Transformer 生成实验室

一个中文交互网页，用固定输入「你叫什么名字」演示 Transformer 从字符切分到逐 token 生成回答的全过程。

在浏览器里打开 `index.html` 就能开始实验。网页包含模型权重和推理代码，可离线运行。默认贪心策略生成「我叫小元。」并在 EOS 停止。生成记录可以逐轮回放，温度、Top-k 和 Top-p 可以交互调整。

## 编辑与构建

- `page.html`、`style.css`、`app.js` 包含页面结构、样式和交互。
- `engine.js` 执行模型前向计算和采样，`model.json` 提供训练好的参数。
- `train.py` 用 NumPy 在合成中文问答上训练教学模型。

修改后运行 `node build.mjs`，再把生成的 `dist/index.html` 复制到仓库根目录作为新版入口。

模型采用 GPT‑2 风格的两层解码器，每层两个注意力头，隐藏维度为 16。交互设计参考 [Transformer Explainer](https://arxiv.org/abs/2408.04619)。
