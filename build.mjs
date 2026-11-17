import { readFile, writeFile } from 'node:fs/promises';

const read = name => readFile(new URL(name, import.meta.url), 'utf8');
const [template, css] = await Promise.all(['page.html', 'style.css'].map(read));
const html = template.replace('/* INLINE_STYLE */', () => css);
await writeFile(new URL('index.html', import.meta.url), html);
