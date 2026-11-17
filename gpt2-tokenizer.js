// GPT-2 uses a reversible alphabet for the 256 UTF-8 byte values.
const byteValues = [];
for (let b = 0; b < 256; b++) {
  if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) byteValues.push(b);
}
const byteCodePoints = byteValues.slice();
for (let b = 0, extra = 0; b < 256; b++) {
  if (!byteValues.includes(b)) {
    byteValues.push(b);
    byteCodePoints.push(256 + extra++);
  }
}
const byteToSymbol = [];
const symbolToByte = new Map();
for (let i = 0; i < 256; i++) {
  const symbol = String.fromCodePoint(byteCodePoints[i]);
  byteToSymbol[byteValues[i]] = symbol;
  symbolToByte.set(symbol, byteValues[i]);
}

const tokenPattern = /'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { ignoreBOM: true });
const strictDecoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
export const EOS = 50256;

export class GPT2Tokenizer {
  constructor(data) {
    this.vocab = new Map(Object.entries(data.model.vocab));
    this.text = [];
    this.bytes = [];
    this.labels = [];
    this.vocab.forEach((id, text) => {
      this.text[id] = text;
      const bytes = Uint8Array.from(Array.from(text, symbol => symbolToByte.get(symbol)));
      this.bytes[id] = bytes;
      this.labels[id] = this.format(id, bytes);
    });
    this.merges = new Map(data.model.merges.map((entry, index) => [Array.isArray(entry) ? entry.join(' ') : entry, index]));
  }

  encode(text) {
    return [...text.matchAll(tokenPattern)].flatMap(([piece]) => {
      const symbols = Array.from(encoder.encode(piece), byte => byteToSymbol[byte]);
      while (symbols.length > 1) {
        let bestRank = Infinity;
        let bestPair;
        for (let i = 0; i + 1 < symbols.length; i++) {
          const pair = symbols[i] + ' ' + symbols[i + 1];
          const rank = this.merges.get(pair);
          if (rank !== undefined && rank < bestRank) {
            bestPair = pair;
            bestRank = rank;
          }
        }
        if (bestPair === undefined) break;
        for (let i = 0; i + 1 < symbols.length; i++) {
          if (symbols[i] + ' ' + symbols[i + 1] === bestPair) {
            symbols.splice(i, 2, symbols[i] + symbols[i + 1]);
          }
        }
      }
      return symbols.map(symbol => this.vocab.get(symbol));
    });
  }

  hex(id) {
    return Array.from(this.bytes[id], byte => byte.toString(16).padStart(2, '0').toUpperCase()).join(' ');
  }

  format(id, bytes) {
    if (id === EOS) return 'EOS';
    try {
      return strictDecoder.decode(bytes).replace(/ /g, '␣').replace(/\n/g, '↵').replace(/\r/g, '␍').replace(/\t/g, '⇥')
        .replace(/[\u0000-\u001F\u007F]/g, char => `\\x${char.charCodeAt(0).toString(16).padStart(2, '0')}`);
    } catch {
      return '⟨' + Array.from(bytes, byte => byte.toString(16).padStart(2, '0').toUpperCase()).join(' ') + '⟩';
    }
  }

  decode(ids, streaming = false) {
    const bytes = Uint8Array.from(ids.filter(id => id !== EOS).flatMap(id => Array.from(this.bytes[id])));
    // Hide an unfinished trailing character until its remaining UTF-8 bytes arrive.
    return streaming ? new TextDecoder('utf-8', { ignoreBOM: true }).decode(bytes, { stream: true }) : decoder.decode(bytes);
  }
}
