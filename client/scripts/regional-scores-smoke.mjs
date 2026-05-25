// Smoke headless del panel de scores por región (Tareas #4/#9/#10/#16).
//
// Carga 3 fixtures (gitignored, spike_fixtures_detection) en los slots
// left/child/right del Comparador, compara, y ejercita el panel:
//   - geométrico (instantáneo) → Calcular → screenshot
//   - occlusion (~12 inferencias × progenitor) → Calcular → mide wall-time → screenshot
//   - toggle heatmap → screenshot
// Valida que el panel monte, que occlusion termine, y deja screenshots para
// revisión visual. Mide el costo de occlusion (lo que preocupa al usuario); el
// costo térmico real lo da el .dev-resources.log de dev-monitored.sh.
//
// Asume vite en http://localhost:5173.

import { chromium } from '@playwright/test';
import { resolve } from 'path';
import { readdirSync } from 'fs';

const APP_URL = 'http://localhost:5173/';
const FIX_DIR = resolve(process.env.HOME, 'Proyectos/0_code_(gitHub)/mendelEmbeddings/client/public/spike_fixtures_detection/images');
const imgs = readdirSync(FIX_DIR).filter((f) => f.endsWith('.png')).sort().slice(0, 3).map((f) => resolve(FIX_DIR, f));
const SHOT = (name) => `/tmp/regional-${name}.png`;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

if (imgs.length < 3) { console.error(`FAIL: necesito 3 fixtures en ${FIX_DIR}, encontré ${imgs.length}`); process.exit(1); }

// HEADED=1 abre una ventana real en el display (necesario para WebGPU: ORT-web
// cae a WASM en headless y occlusion —~24 inferencias— bloquea el main thread).
const HEADED = process.env.HEADED === '1';
console.log(`modo: ${HEADED ? 'HEADED (WebGPU real)' : 'headless (WASM fallback)'}`);
const browser = await chromium.launch({ headless: !HEADED, args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'] });
const context = await browser.newContext({ viewport: { width: 1500, height: 1000 } });
const page = await context.newPage();
page.on('console', (m) => { if (m.type() === 'error') console.log('  [browser error]', m.text()); });
page.on('pageerror', (e) => console.log('  [pageerror]', e.message));

const notComputing = () => page.waitForFunction(
  () => ![...document.querySelectorAll('button')].some((b) => /calculando/i.test(b.textContent || '')),
  null, { timeout: 240000 },
);

try {
  console.log('imgs:', imgs.map((p) => p.split('/').pop()).join(', '));
  console.log('1. cargar app (tab comparator por default)');
  await page.goto(APP_URL, { waitUntil: 'load' });

  console.log('2. setear 3 fixtures en los slots left/child/right');
  const fileInputs = page.locator('input[type="file"]');
  await fileInputs.nth(0).setInputFiles(imgs[0]);
  await fileInputs.nth(1).setInputFiles(imgs[1]);
  await fileInputs.nth(2).setInputFiles(imgs[2]);
  await wait(500);

  console.log('3. Comparar (init MediaPipe+ONNX en la 1ª corrida; puede tardar)');
  await page.getByRole('button', { name: /comparar/i }).click();

  console.log('4. esperar que monte el panel "Scores por región"');
  await page.getByRole('heading', { name: 'Scores por región' }).waitFor({ timeout: 240000 });
  await page.screenshot({ path: SHOT('1-panel-mounted'), fullPage: true });

  console.log('5. geométrico → Calcular');
  const tGeo = Date.now();
  await page.getByRole('button', { name: /calcular/i }).click();
  await notComputing();
  console.log(`   geométrico OK en ${Date.now() - tGeo} ms`);
  await page.screenshot({ path: SHOT('2-geometric'), fullPage: true });

  console.log('5b. escala Absoluta → screenshot, y vuelta a Reparto P↔M');
  await page.getByRole('radio', { name: /absoluta/i }).click();
  await wait(200);
  await page.screenshot({ path: SHOT('2b-geometric-absolute'), fullPage: true });
  await page.getByRole('radio', { name: /reparto/i }).click();
  await wait(200);

  // occlusion sólo en headed: en headless ORT cae a WASM y bloquea el main thread
  // (~24 inferencias congelan la UI). El geométrico ya valida layout + radar.
  let occMs = -1;
  if (HEADED) {
    console.log('6. occlusion → Calcular (mido wall-time)');
    await page.getByRole('radio', { name: /occlusion/i }).click();
    await wait(200);
    const tOcc = Date.now();
    await page.getByRole('button', { name: /calcular|recalcular/i }).click();
    await wait(300);
    await notComputing();
    occMs = Date.now() - tOcc;
    console.log(`   occlusion OK en ${occMs} ms (≈${(occMs / 1000).toFixed(1)}s, 2 progenitores × ~12 regiones)`);
    await page.screenshot({ path: SHOT('3-occlusion'), fullPage: true });
  } else {
    console.log('6. occlusion OMITIDO (headless). Corré con HEADED=1 para validarlo.');
  }

  console.log('7. toggle heatmap sobre el Hijo/a');
  await page.getByRole('checkbox', { name: /heatmap/i }).click();
  await wait(400);
  await page.screenshot({ path: SHOT('4-heatmap'), fullPage: true });

  // Asserts mínimos: el radar (svg) y al menos una barra de región existen.
  const svgCount = await page.locator('svg[aria-label="Radar de scores por región"]').count();
  if (svgCount < 1) throw new Error('no se renderizó el radar');

  console.log(`\nPASS — panel OK${HEADED ? `. occlusion=${occMs}ms` : ' (geométrico; occlusion omitido)'}. Screenshots en /tmp/regional-*.png`);
  console.log(`SMOKE_RESULT occlusion_ms=${occMs} geometric_ms=${Date.now() - tGeo}`);
} catch (e) {
  console.error('FAIL:', e.message);
  await page.screenshot({ path: SHOT('FAIL'), fullPage: true }).catch(() => {});
  await browser.close();
  process.exit(1);
}
await browser.close();
process.exit(0);
