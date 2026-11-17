"""Train the small character-level decoder used by the browser demo."""
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np


QUESTIONS = ['你叫什么名字', '你叫啥名字', '请问你叫什么', '你是谁', '你的名字是什么', '怎么称呼你', '可以告诉我你的名字吗', '你有名字吗']
ANSWERS = ['我叫小元。', '我叫小知。', '我叫小智。', '我是小元。', '我的名字是小元。']
OTHER = [('你好', '你好呀。'), ('早上好', '早上好。'), ('谢谢你', '不用谢。'), ('你会做什么', '我会回答问题。'), ('你是机器人吗', '我是一个学习助手。'), ('再见', '下次见。')]
corpus = [(q + p, a) for q in QUESTIONS for p in ('', '？') for a in ANSWERS]
corpus += [(q, a) for q, a in OTHER for _ in range(5)]
vocab = ['<PAD>', '<EOS>', '<UNK>'] + sorted(set(''.join(q + a for q, a in corpus)))
token_id = {c: i for i, c in enumerate(vocab)}
width, heads, depth, context = 16, 2, 2, 32
head_width = width // heads
rng = np.random.default_rng(42)
eps = 1e-5


class Parameter:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)
        self.grad = np.zeros_like(self.value)
        self.m = np.zeros_like(self.value)
        self.v = np.zeros_like(self.value)
        parameters.append(self)


parameters = []


def random(*shape):
    return Parameter(rng.normal(0, 0.13, shape))


def linear(a, b):
    return (random(a, b), Parameter(np.zeros(b)))


def norm():
    return (Parameter(np.ones(width)), Parameter(np.zeros(width)))


embedding = random(len(vocab), width)
position = random(context, width)
blocks = [dict(ln1=norm(), qkv=linear(width, 3 * width), att=linear(width, width),
               ln2=norm(), up=linear(width, 4 * width), down=linear(4 * width, width)) for _ in range(depth)]
final_norm = norm()


def affine(x, params):
    w, b = params
    def backward(dy):
        w.grad += x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        b.grad += dy.sum(axis=(0, 1))
        return dy @ w.value.T
    return x @ w.value + b.value, backward


def layer_norm(x, params):
    scale, shift = params
    centered = x - x.mean(axis=-1, keepdims=True)
    inv_std = 1 / np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + eps)
    z = centered * inv_std
    def backward(dy):
        scale.grad += (dy * z).sum(axis=(0, 1))
        shift.grad += dy.sum(axis=(0, 1))
        dz = dy * scale.value
        return inv_std * (dz - dz.mean(axis=-1, keepdims=True) - z * (dz * z).mean(axis=-1, keepdims=True))
    return z * scale.value + shift.value, backward


def softmax(x):
    exps = np.exp(x - x.max(axis=-1, keepdims=True))
    return exps / exps.sum(axis=-1, keepdims=True)


def block(x, weights):
    batch, seq, _ = x.shape
    z, d_ln1 = layer_norm(x, weights['ln1'])
    qkv, d_qkv = affine(z, weights['qkv'])
    q, k, v = qkv.reshape(batch, seq, 3, heads, head_width).transpose(2, 0, 3, 1, 4)
    scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(head_width)
    masked = np.where(np.triu(np.ones((seq, seq), dtype=bool), k=1), -np.inf, scores)
    att = softmax(masked).astype(np.float32)
    mixed = (att @ v).transpose(0, 2, 1, 3).reshape(batch, seq, width)
    projected, d_att = affine(mixed, weights['att'])
    residual = x + projected
    z, d_ln2 = layer_norm(residual, weights['ln2'])
    expanded, d_up = affine(z, weights['up'])
    tanh = np.tanh(np.sqrt(2 / np.pi) * (expanded + 0.044715 * expanded ** 3))
    gelu = (0.5 * expanded * (1 + tanh)).astype(np.float32)
    compressed, d_down = affine(gelu, weights['down'])

    def backward(dy):
        grad_gelu = 0.5 * (1 + tanh) + 0.5 * expanded * (1 - tanh ** 2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * expanded ** 2)
        dr = (dy + d_ln2(d_up(d_down(dy) * grad_gelu))).astype(np.float32)
        dm = d_att(dr).reshape(batch, seq, heads, head_width).transpose(0, 2, 1, 3)
        dv = att.swapaxes(-1, -2) @ dm
        da = dm @ v.swapaxes(-1, -2)
        ds = (att * (da - (da * att).sum(axis=-1, keepdims=True)) / np.sqrt(head_width)).astype(np.float32)
        dq, dk = ds @ k, ds.swapaxes(-1, -2) @ q
        grad_qkv = np.stack((dq, dk, dv)).transpose(1, 3, 0, 2, 4).reshape(batch, seq, 3 * width)
        return dr + d_ln1(d_qkv(grad_qkv))
    return residual + compressed, backward


def train():
    rows = [[token_id[c] for c in q + a] + [1] for q, a in corpus]
    length = max(map(len, rows))
    data = np.array([row + [0] * (length - len(row)) for row in rows])
    # Several replies share a prefix. Their relative frequencies control the teaching distribution.
    rates = np.array([base * scale for _ in QUESTIONS for scale in (1, 0.16)
                      for base in (9, 4, 2, 1.3, 1)] + [2] * (len(OTHER) * 5))
    rates /= rates.sum()
    for step in range(1, 641):
        batch = data[rng.choice(len(data), size=24, p=rates)]
        ids, targets = batch[:, :-1], batch[:, 1:]
        seq = ids.shape[1]
        x = embedding.value[ids] + position.value[:seq]
        backwards = []
        for weights in blocks:
            x, backward = block(x, weights)
            backwards.append(backward)
        x, backward = layer_norm(x, final_norm)
        probs = softmax(x @ embedding.value.T)
        chosen = np.take_along_axis(probs, targets[..., None], axis=-1)[..., 0]
        valid = targets != 0
        if step in (1, 160, 320, 480, 640):
            print(f'{step}: {(-np.log(chosen[valid])).mean():.4f}', flush=True)
        dy = probs - np.eye(len(vocab), dtype=np.float32)[targets]
        dy *= valid[..., None] / valid.sum()
        embedding.grad += dy.reshape(-1, len(vocab)).T @ x.reshape(-1, width)
        dx = backward(dy @ embedding.value)
        for block_backward in reversed(backwards):
            dx = block_backward(dx)
        np.add.at(embedding.grad, ids, dx)
        position.grad[:seq] += dx.sum(axis=0)
        lr = 0.007 if step <= 480 else 0.003
        for p in parameters:
            p.m = 0.9 * p.m + 0.1 * p.grad
            p.v = 0.99 * p.v + 0.01 * p.grad ** 2
            p.value -= lr * (p.m / (1 - 0.9 ** step)) / (np.sqrt(p.v / (1 - 0.99 ** step)) + 1e-8)
            p.grad.fill(0)


def serialize(value):
    if isinstance(value, Parameter):
        return value.value.round(6).tolist() if value.value.ndim == 0 else np.round(value.value.astype(float), 6).tolist()
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    return [serialize(item) for item in value]


if __name__ == '__main__':
    train()
    model = dict(vocab=vocab, width=width, heads=heads, context=context, embedding=serialize(embedding),
                 position=serialize(position), blocks=serialize(blocks), finalNorm=serialize(final_norm))
    Path('model.json').write_text(json.dumps(model, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
