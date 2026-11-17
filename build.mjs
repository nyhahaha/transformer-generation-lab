import { readFile, mkdir, writeFile } from 'node:fs/promises';

const read = name => readFile(new URL(name, import.meta.url), 'utf8');
const [template, css, model, engine, app] = await Promise.all(['page.html', 'style.css', 'model.json', 'engine.js', 'app.js'].map(read));
const html = template.replace('/* INLINE_STYLE */', () => css)
  .replace('/* INLINE_MODEL */', () => `const MODEL = ${model};`)
  .replace('/* INLINE_ENGINE */', () => engine)
  .replace('/* INLINE_APP */', () => app);
await mkdir(new URL('dist/', import.meta.url), { recursive: true });
await writeFile(new URL('dist/index.html', import.meta.url), html);
