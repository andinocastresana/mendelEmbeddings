// =========================================
// ID: PHYLOFACE_FIFA_HARVEST_SQUADS
// VERSION: v0.1 (probe — 1 equipo, verboso)
// =========================================
// FILE: client/scripts/fifa_harvest_squads.mjs
//
// Cosecha las fichas oficiales de jugadores del Mundial 2026 desde fifa.com.
// La página /teams/<slug>/team-news es una SPA client-rendered (CRA): el HTML
// crudo no trae jugadores, hay que renderizar JS. Estrategia: Playwright
// renderiza la página, intercepta las respuestas de red de las APIs de FIFA
// (datos estructurados) y además raspa el DOM del carousel (imgs digitalhub).
//
// Esta v0.1 es de SONDEO: corre 1 equipo (env TEAM, default germany), loguea
// los endpoints JSON que matchean hosts FIFA, vuelca los payloads candidatos a
// scratchpad y lista las imgs digitalhub del DOM. Sirve para descubrir el shape
// real antes de escribir la extracción definitiva.
//
// Uso (desde client/):
//   TEAM=germany node scripts/fifa_harvest_squads.mjs
//   OUT=/abs/dump.json DUMP_DIR=/abs/dir TEAM=germany node scripts/...
//
// NO publicar las fotos: son copyright FIFA/Getty (uso local de investigación).

import { chromium } from '@playwright/test';
import { writeFileSync } from 'fs';
import { resolve } from 'path';

const TEAM = process.env.TEAM || 'germany';
const BASE = 'https://www.fifa.com/es/tournaments/mens/worldcup/canadamexicousa2026';
const DUMP_DIR = process.env.DUMP_DIR ||
  resolve(process.env.HOME, 'Proyectos/0_code_(gitHub)/mendelEmbeddings',
    '../../../tmp'); // sobreescribible; en el probe lo pasamos por env
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// Hosts de API de FIFA detectados en el bundle.
const FIFA_API_HOSTS = ['cxm-api.fifa.com', 'fdh-api.fifa.com', 'api.fifa.com'];
const isFifaApi = (url) => FIFA_API_HOSTS.some((h) => url.includes(h));

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1600, height: 1200 },
  locale: 'es-ES',
});
const page = await context.newPage();

// --- Interceptar respuestas JSON de las APIs FIFA ---
const apiHits = [];           // { url, status, ctype, bytes }
const jsonPayloads = [];      // { url, json } cuando parsea y parece relevante
page.on('response', async (resp) => {
  try {
    const url = resp.url();
    if (!isFifaApi(url)) return;
    const ctype = resp.headers()['content-type'] || '';
    const isJson = ctype.includes('json');
    let bytes = 0, text = '';
    if (isJson) {
      text = await resp.text();
      bytes = text.length;
    }
    apiHits.push({ url, status: resp.status(), ctype, bytes });
    if (isJson && (/digitalhub|player|squad|jersey|position/i.test(text))) {
      let json = null;
      try { json = JSON.parse(text); } catch { /* noop */ }
      if (json) jsonPayloads.push({ url, json });
    }
  } catch { /* noop */ }
});

const pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(e.message));

const teamUrl = `${BASE}/teams/${TEAM}/team-news`;
console.log(`[probe] goto ${teamUrl}`);
await page.goto(teamUrl, { waitUntil: 'networkidle', timeout: 60000 }).catch((e) =>
  console.log('  goto warn:', e.message));

// Scroll para disparar lazy-load del carousel de jugadores.
for (let i = 0; i < 6; i++) {
  await page.mouse.wheel(0, 1400);
  await wait(700);
}
await wait(1500);

// --- Raspar imgs digitalhub del DOM ---
const domImgs = await page.evaluate(() => {
  const out = [];
  for (const img of document.querySelectorAll('img')) {
    const src = img.getAttribute('src') || '';
    const srcset = img.getAttribute('srcset') || '';
    if (/digitalhub\.fifa\.com\/transform/.test(src + srcset)) {
      out.push({
        alt: img.getAttribute('alt') || '',
        title: img.getAttribute('title') || '',
        src,
        srcset,
      });
    }
  }
  return out;
});

const shot = `${DUMP_DIR}/fifa-${TEAM}-probe.png`;
await page.screenshot({ path: shot, fullPage: true }).catch(() => {});

// --- Reportar ---
console.log(`\n[probe] API hits (${apiHits.length}):`);
for (const h of apiHits.slice(0, 40)) {
  console.log(`  ${h.status} ${h.bytes}b ${h.ctype.split(';')[0]}  ${h.url.slice(0, 140)}`);
}
console.log(`\n[probe] JSON payloads relevantes: ${jsonPayloads.length}`);
for (const p of jsonPayloads) {
  const keys = Array.isArray(p.json) ? `[array len ${p.json.length}]` : Object.keys(p.json).join(',');
  console.log(`  ${p.url.slice(0, 110)}\n     keys: ${keys.slice(0, 200)}`);
}
console.log(`\n[probe] DOM imgs digitalhub: ${domImgs.length}`);
for (const d of domImgs.slice(0, 8)) {
  const best = (d.srcset.split(',').pop() || d.src).trim().split(' ')[0];
  console.log(`  alt="${d.alt}"  -> ${best.slice(0, 130)}`);
}
if (pageErrors.length) console.log(`\n[probe] pageerrors: ${pageErrors.slice(0, 3).join(' | ')}`);

const dump = { team: TEAM, teamUrl, apiHits, jsonPayloads, domImgs, pageErrors };
const outPath = process.env.OUT || `${DUMP_DIR}/fifa-${TEAM}-probe.json`;
writeFileSync(outPath, JSON.stringify(dump, null, 2));
console.log(`\n[probe] dump -> ${outPath}`);
console.log(`[probe] screenshot -> ${shot}`);

await browser.close();
process.exit(0);
