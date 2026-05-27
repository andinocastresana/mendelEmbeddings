// Smoke headless: PERSISTENCIA + RE-SIEMBRA de occlusion en la App primaria (#12).
//
// El CÁLCULO de occlusion headless (≈24 inferencias ResNet50 en WASM) es
// demasiado lento para un smoke (>5 min); se valida a mano en browser headed
// (WebGPU). Lo que sí valida este smoke —y es exactamente el síntoma reportado
// ("occlusion no se persiste")— es el camino guardar→cargar→sembrar→mostrar:
//   1. analizar (geométrico se computa y persiste).
//   2. INYECTAR en IndexedDB un resultado de occlusion (clon del geométrico, con
//      method='occlusion') simulando que el usuario lo calculó y se guardó.
//   3. RECARGAR → la solapa Occlusion debe quedar HABILITADA (datos cacheados;
//      ver cache no necesita sesión) y mostrar el veredicto de occlusion SIN botón
//      "Calcular" (sembrado, no recomputa).
// Asume vite en http://localhost:5173.

import { chromium } from '@playwright/test';
import { resolve } from 'path';
import { readdirSync } from 'fs';

const APP_URL = 'http://localhost:5173/';
const FIX_DIR = resolve(process.env.HOME, 'Proyectos/0_code_(gitHub)/mendelEmbeddings/client/public/spike_fixtures_detection/images');
const imgs = readdirSync(FIX_DIR).filter((f) => f.endsWith('.png')).sort().slice(0, 3).map((f) => resolve(FIX_DIR, f));
const SHOT = (n) => `/tmp/primaria-occ-${n}.png`;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

if (imgs.length < 3) { console.error(`FAIL: necesito 3 fixtures en ${FIX_DIR}`); process.exit(1); }

const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'] });
const context = await browser.newContext({ viewport: { width: 1400, height: 1100 } });
const page = await context.newPage();
page.on('pageerror', (e) => console.log('  [pageerror]', e.message));

const notComputing = () => page.waitForFunction(
  () => ![...document.querySelectorAll('button')].some((b) => /calculando|analizando/i.test(b.textContent || '')),
  null, { timeout: 240000 },
);

try {
  console.log('1. analizar 3 fotos (geométrico se persiste)');
  await page.goto(APP_URL, { waitUntil: 'load' });
  const fi = page.locator('input[type="file"]');
  await fi.nth(0).setInputFiles(imgs[0]);
  await fi.nth(1).setInputFiles(imgs[1]);
  await fi.nth(2).setInputFiles(imgs[2]);
  await wait(400);
  await page.getByRole('button', { name: /analizar parecido/i }).click();
  await page.getByText(/Heredó de Padre/i).waitFor({ timeout: 240000 });
  await notComputing();
  await wait(800); // dejar que persista geométrico

  console.log('2. inyectar occlusion en IndexedDB (clon del geométrico)');
  const injected = await page.evaluate(async () => {
    const open = () => new Promise((res, rej) => {
      const r = indexedDB.open('phyloface-primaria', 1);
      r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
    });
    const db = await open();
    const rec = await new Promise((res, rej) => {
      const t = db.transaction('state', 'readonly');
      const g = t.objectStore('state').get('current');
      g.onsuccess = () => res(g.result); g.onerror = () => rej(g.error);
    });
    if (!rec || !rec.regional || !rec.regional.geometric) { db.close(); return 'sin geométrico persistido'; }
    const occ = JSON.parse(JSON.stringify(rec.regional.geometric)); // misma shape (sin typed arrays)
    for (const side of Object.keys(occ)) { occ[side].method = 'occlusion'; occ[side].methodLabel = 'Occlusion (contribución)'; occ[side].baseConfidence = 'medium'; }
    rec.regional.occlusion = occ;
    await new Promise((res, rej) => {
      const t = db.transaction('state', 'readwrite');
      t.objectStore('state').put(rec, 'current');
      t.oncomplete = () => res(); t.onerror = () => rej(t.error);
    });
    db.close();
    return 'ok';
  });
  if (injected !== 'ok') throw new Error(`inyección falló: ${injected}`);
  console.log('   occlusion inyectada en IDB');

  console.log('3. RECARGAR → occlusion sembrada debe verse sin recomputar');
  await page.reload({ waitUntil: 'load' });
  await page.getByText(/Heredó de Padre/i).waitFor({ timeout: 30000 });
  await notComputing();
  const diag = await page.evaluate(async () => {
    const open = () => new Promise((res, rej) => { const r = indexedDB.open('phyloface-primaria', 1); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
    const db = await open();
    const rec = await new Promise((res, rej) => { const t = db.transaction('state', 'readonly'); const g = t.objectStore('state').get('current'); g.onsuccess = () => res(g.result); g.onerror = () => rej(g.error); });
    db.close();
    return { regionalKeys: rec && rec.regional ? Object.keys(rec.regional) : null, modelVersion: rec ? rec.modelVersion : null };
  });
  console.log('   [diag] IDB regional tras recarga:', JSON.stringify(diag));
  // La solapa Occlusion debe estar habilitada DE ENTRADA (datos cacheados, sin sesión).
  const occDisabledAtStart = await page.evaluate(() => {
    const occ = [...document.querySelectorAll('button')].find((b) => /occlusion/i.test(b.textContent || ''));
    return occ ? occ.disabled : 'no-tab';
  });
  if (occDisabledAtStart !== false) throw new Error(`solapa Occlusion no habilitada al cargar (disabled=${occDisabledAtStart}) — no se puede ver lo persistido`);
  await page.getByRole('button', { name: /occlusion/i }).click();
  await wait(400);
  await page.getByText(/Según.*Occlusion/i).first().waitFor({ timeout: 15000 });
  const calcBtns = await page.getByRole('button', { name: /^calcular/i }).count();
  await page.screenshot({ path: SHOT('after-reload'), fullPage: true });
  if (calcBtns > 0) throw new Error('apareció "Calcular" en occlusion → NO se sembró (recomputaría)');

  console.log('\nPASS — occlusion persiste y se RE-SIEMBRA: tras recargar se ve sin recomputar y la solapa queda habilitada con datos cacheados.');
  console.log('SMOKE_RESULT occlusion_persist=ok');
} catch (e) {
  console.error('FAIL:', e.message);
  await page.screenshot({ path: SHOT('FAIL'), fullPage: true }).catch(() => {});
  await browser.close();
  process.exit(1);
}
await browser.close();
process.exit(0);
