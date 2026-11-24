const CACHE_NAME = 'transformer-gpt2-model-v1';

export async function openModelCache() {
  try {
    return await globalThis.caches?.open(CACHE_NAME);
  } catch {
    return undefined;
  }
}

export async function cachedResponse(url) {
  try {
    return await (await openModelCache())?.match(url);
  } catch {
    return undefined;
  }
}

export async function saveResponse(url, response) {
  try {
    const cache = await openModelCache();
    if (!cache) return false;
    await cache.put(url, response);
    return true;
  } catch {
    return false;
  }
}

export async function forgetResponse(url) {
  try {
    await (await openModelCache())?.delete(url);
  } catch {
    // Storage access may be disabled while an in-memory model is still usable.
  }
}

export async function modelJSON(url, fetcher = fetch) {
  const saved = await cachedResponse(url);
  if (saved) {
    try {
      return await saved.json();
    } catch {
      await forgetResponse(url);
    }
  }
  const response = await fetcher(url, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`资源响应 HTTP ${response.status}（${url.pathname.split('/').at(-1)}）`);
  const data = await response.clone().json();
  await saveResponse(url, response);
  return data;
}

export function shardCacheURL(base, shard) {
  const url = new URL(shard.file, base);
  url.searchParams.set('sha256', shard.sha256);
  return url;
}

export async function hasCachedGPT2() {
  const base = new URL('./gpt2/q8-v1/', import.meta.url);
  const response = await cachedResponse(new URL('weights.json', base));
  if (!response) return false;
  try {
    const manifest = await response.json();
    if (manifest.format !== 'gpt2-row-int8-v1' || manifest.shards?.length !== 13) return false;
    const shards = await Promise.all(manifest.shards.map(shard => cachedResponse(shardCacheURL(base, shard))));
    return shards.every((saved, i) => saved?.headers.get('content-length') === String(manifest.shards[i].bytes));
  } catch {
    return false;
  }
}
