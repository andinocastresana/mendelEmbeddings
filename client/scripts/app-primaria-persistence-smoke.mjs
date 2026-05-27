// Smoke headless de la PERSISTENCIA de la App primaria (Tarea #12, lib/primariaStore).
//
// Valida que una recarga NO borre el trabajo:
//   1. analizar 3 fotos → veredicto global + regional.
//   2. RECARGAR la página → el veredicto vuelve SIN re-analizar (global rearmado
//      de embeddings, regional geométrico de landmarks; sin inferencia ni modelos).
//   3. "Limpiar" → borra todo; tras recargar no queda nada restaurado.
// IndexedDB persiste a través de page.reload() (mismo contexto/origen).
// Asume vite en http://localhost:5173.

import { chromium } from '@playwright/test';
import { resolve } from 'path';
import { readdirSync } from 'fs';

const APP_URL = 'http://localhost:5173/';
const FIX_DIR = resolve(process.env.HOME, 'Proyectos/0_code_(gitHub)/mendelEmbeddings/client/public/spike_fixtures_detection/images');
const imgs = readdirSync(FIX_DIR).filter((f) => f.endsWith('.png')).sort().slice(0, 3).map((f) => resolve(FIX_DIR, f));
const SHOT = (name) => `/tmp/primaria-persist-${name}.png`;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

if (imgs.length < 3) { console.error(`FAIL: necesito 3 fixtures en ${FIX_DIR}`); process.exit(1); }

const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'] });
const context = await browser.newContext({ viewport: { width: 1400, height: 1100 } });
const page = await context.newPage();
page.on('pageerror', (e) => console.log('  [pageerror]', e.message));

const headline = () => page.getByText(/^Se parece (más a (Padre|Madre)|de forma pareja a ambos)$|^Comparación con/i).first();
const notComputing = () => page.waitForFunction(
  () => ![...document.querySelectorAll('button')].some((b) => /calculando|analizando/i.test(b.textContent || '')),
  null, { timeout: 240000 },
);

try {
  console.log('1. analizar 3 fotos');
  await page.goto(APP_URL, { waitUntil: 'load' });
  const fileInputs = page.locator('input[type="file"]');
  await fileInputs.nth(0).setInputFiles(imgs[0]);
  await fileInputs.nth(1).setInputFiles(imgs[1]);
  await fileInputs.nth(2).setInputFiles(imgs[2]);
  await wait(400);
  await page.getByRole('button', { name: /analizar parecido/i }).click();
  await headline().waitFor({ timeout: 240000 });
  await page.getByText(/Heredó de Padre/i).waitFor({ timeout: 60000 });
  const before = (await headline().textContent())?.trim();
  console.log(`   veredicto pre-recarga: "${before}"`);
  await page.screenshot({ path: SHOT('1-before-reload'), fullPage: true });

  console.log('2. RECARGAR — el veredicto debe volver SIN re-analizar');
  const t0 = Date.now();
  await page.reload({ waitUntil: 'load' });
  // Sin tocar nada: el veredicto global debe reaparecer (rearmado de embeddings).
  await headline().waitFor({ timeout: 30000 });
  await notComputing();
  await page.getByText(/Heredó de Padre/i).waitFor({ timeout: 30000 });
  const after = (await headline().textContent())?.trim();
  const previews = await page.locator('img[alt="Padre"], img[alt="Hijo/a"], img[alt="Madre"]').count();
  console.log(`   veredicto post-recarga: "${after}" en ${Date.now() - t0} ms · ${previews}/3 fotos restauradas`);
  await page.screenshot({ path: SHOT('2-after-reload'), fullPage: true });

  if (after !== before) throw new Error(`el veredicto cambió tras recargar: "${before}" → "${after}"`);
  if (previews < 3) throw new Error(`se restauraron ${previews}/3 fotos`);

  console.log('2b. la solapa Occlusion debe habilitarse tras recargar (init ONNX en background)');
  await page.getByRole('button', { name: /occlusion/i }).waitFor({ timeout: 30000 });
  await page.waitForFunction(() => {
    const occ = [...document.querySelectorAll('button')].find((b) => /occlusion/i.test(b.textContent || ''));
    return occ && !occ.disabled;
  }, null, { timeout: 90000 });
  console.log('   solapa Occlusion habilitada ✓');

  console.log('3. Limpiar → recargar → no debe quedar nada');
  await page.getByRole('button', { name: /limpiar informe completo/i }).click();
  await wait(400);
  if (await headline().count() > 0) throw new Error('el veredicto sigue tras Limpiar');
  await page.reload({ waitUntil: 'load' });
  await wait(2500); // dar tiempo a un eventual restore (no debería haberlo)
  const restoredAfterClear = await headline().count();
  await page.screenshot({ path: SHOT('3-after-clear-reload'), fullPage: true });
  if (restoredAfterClear > 0) throw new Error('quedó estado tras Limpiar + recargar');

  console.log('\nPASS — persistencia OK: el veredicto sobrevive la recarga sin re-analizar; Limpiar borra todo.');
  console.log('SMOKE_RESULT persistence=ok');
} catch (e) {
  console.error('FAIL:', e.message);
  await page.screenshot({ path: SHOT('FAIL'), fullPage: true }).catch(() => {});
  await browser.close();
  process.exit(1);
}
await browser.close();
process.exit(0);
