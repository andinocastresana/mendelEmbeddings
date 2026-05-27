// Smoke headless de la descarga de informe PDF de la App primaria (#31).
//
// Analiza 3 fotos, espera el veredicto, dispara "📄 Descargar PDF" y verifica que
// se descargue un PDF válido (cabecera %PDF-) y no trivial (caras embebidas).
// Todo client-side (jsPDF); las imágenes no salen del browser.
// Asume vite en http://localhost:5173.

import { chromium } from '@playwright/test';
import { resolve } from 'path';
import { readdirSync, readFileSync } from 'fs';

const APP_URL = 'http://localhost:5173/';
const FIX_DIR = resolve(process.env.HOME, 'Proyectos/0_code_(gitHub)/mendelEmbeddings/client/public/spike_fixtures_detection/images');
const imgs = readdirSync(FIX_DIR).filter((f) => f.endsWith('.png')).sort().slice(0, 3).map((f) => resolve(FIX_DIR, f));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

if (imgs.length < 3) { console.error(`FAIL: necesito 3 fixtures en ${FIX_DIR}`); process.exit(1); }

const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'] });
const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1400, height: 1100 } });
const page = await context.newPage();
page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
page.on('console', (m) => { if (m.type() === 'error') console.log('  [browser error]', m.text()); });

try {
  console.log('1. analizar 3 fotos');
  await page.goto(APP_URL, { waitUntil: 'load' });
  const fi = page.locator('input[type="file"]');
  await fi.nth(0).setInputFiles(imgs[0]);
  await fi.nth(1).setInputFiles(imgs[1]);
  await fi.nth(2).setInputFiles(imgs[2]);
  await wait(400);
  await page.getByRole('button', { name: /analizar parecido/i }).click();
  await page.getByText(/Heredó de Padre/i).waitFor({ timeout: 240000 }); // veredicto + regional listos

  await page.screenshot({ path: '/tmp/primaria-page.png', fullPage: true });
  console.log('   screenshot de la página → /tmp/primaria-page.png (verifica botón arriba-izquierda)');

  console.log('2. click "📄 Descargar PDF" y capturar la descarga');
  const [download] = await Promise.all([
    page.waitForEvent('download', { timeout: 30000 }),
    page.getByRole('button', { name: /descargar pdf/i }).click(),
  ]);
  await download.saveAs('/tmp/primaria-informe.pdf');
  const path = await download.path();
  const buf = readFileSync(path);
  const header = buf.slice(0, 5).toString('latin1');
  console.log(`   descargado: ${download.suggestedFilename()} · ${buf.length} bytes · header="${header}"`);

  if (header !== '%PDF-') throw new Error(`no es un PDF (header="${header}")`);
  if (!/^informe-parecido-.*\.pdf$/.test(download.suggestedFilename())) throw new Error(`nombre inesperado: ${download.suggestedFilename()}`);
  if (buf.length < 8000) throw new Error(`PDF sospechosamente chico (${buf.length} bytes) — ¿faltan las caras?`);

  console.log('\nPASS — informe PDF OK: descarga un %PDF- válido con las caras embebidas.');
  console.log(`SMOKE_RESULT pdf_bytes=${buf.length}`);
} catch (e) {
  console.error('FAIL:', e.message);
  await page.screenshot({ path: '/tmp/primaria-pdf-FAIL.png', fullPage: true }).catch(() => {});
  await browser.close();
  process.exit(1);
}
await browser.close();
process.exit(0);
