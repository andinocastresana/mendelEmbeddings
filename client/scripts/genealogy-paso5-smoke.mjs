// Smoke headless del paso 5 (Tarea #26 — comparación on-demand).
// Crea árbol limpio, dos personas con foto, activa modo comparación, dispara
// cosine, verifica persistencia tras reload. Screenshots en /tmp/genealogy-p5-*.
//
// Asume vite corriendo en http://localhost:5173.

import { chromium } from '@playwright/test';
import { resolve } from 'path';

const APP_URL = 'http://localhost:5173/';
const IMG_DIR = resolve(process.env.HOME, 'Proyectos/0_code_(gitHub)/mendelEmbeddings/data/input/img/spike_e2e_set');
const IMG_BRUNO = resolve(IMG_DIR, 'BrunoFondoBlanco.jpeg');
const IMG_MATEO = resolve(IMG_DIR, 'mateoFotoTarjetaTransporte.jpeg');

const SHOT = (name) => `/tmp/genealogy-p5-${name}.png`;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function fail(msg, e) {
  console.error(`FAIL: ${msg}`);
  if (e) console.error(e);
  process.exit(1);
}

const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
});
const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await context.newPage();

page.on('console', (msg) => {
  if (msg.type() === 'error') console.log('  [browser error]', msg.text());
});
page.on('pageerror', (err) => console.log('  [pageerror]', err.message));

try {
  console.log('1. Cargar app + reset IndexedDB');
  await page.goto(APP_URL, { waitUntil: 'load' });
  await page.evaluate(() => new Promise((resolve, reject) => {
    const req = indexedDB.deleteDatabase('phyloface-genealogy');
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error('blocked'));
  }));
  await page.reload({ waitUntil: 'load' });

  // En App.tsx los tabs son <div onClick>, no <button>: usar selector por
  // texto y esperar a que el h2 del componente árbol haya montado (no solo
  // por el texto del tab, que matchea aunque sigamos en otra pestaña).
  const goToTreeTab = async () => {
    await page.locator('div').filter({ hasText: /^Árbol genealógico$/ }).first().click();
    await page.waitForSelector('h2:has-text("Árbol genealógico (Track 2b")', { timeout: 5000 });
  };

  console.log('2. Navegar a tab Árbol genealógico');
  await goToTreeTab();

  console.log('3. Crear árbol "Familia Test"');
  await page.fill('input[placeholder="Nombre del árbol nuevo"]', 'Familia Test');
  await page.click('button:has-text("+ Árbol")');
  await wait(500);

  console.log('4. Crear personas Bruno y Mateo');
  for (const name of ['Bruno', 'Mateo']) {
    await page.fill('input[placeholder="Nombre de la persona nueva"]', name);
    await page.click('button:has-text("+ Persona")');
    await wait(400);
  }

  console.log('5. Subir fotos a ambos nodos vía input file oculto');
  // Los inputs file están dentro del foreignObject por nodo SVG. Hay uno por
  // persona; el orden en el DOM coincide con el orden de persons (createdAt
  // ASC). El primero en el DOM es Bruno (creado primero), el segundo Mateo.
  const fileInputs = page.locator('svg input[type="file"]');
  await fileInputs.nth(0).setInputFiles(IMG_BRUNO);
  await wait(600);
  await fileInputs.nth(1).setInputFiles(IMG_MATEO);
  await wait(600);

  await page.screenshot({ path: SHOT('01-tree-with-photos'), fullPage: true });

  console.log('6. Activar modo comparación');
  const toggleLabel = page.locator('label:has-text("Modo comparación")');
  await toggleLabel.locator('input[type="checkbox"]').check();
  await wait(300);
  await page.screenshot({ path: SHOT('02-comparison-mode-on'), fullPage: true });

  // El paso 4 capturó el patrón: `.click()` de Playwright centra sobre el
  // `<g>` interno de la foto que hace stopPropagation (intencional del producto,
  // no se toca). Workaround: dispatchEvent dirigido al `<rect>` background, que
  // es el que tiene el handler onClick → onSelect. Episodio
  // [[2026-05-22-playwright-headless-plus-multimodal-llm-closes-ui-validation-loop]].
  const clickNodeBackground = async (name) => {
    await page.evaluate((n) => {
      const g = document.querySelector(`g[aria-label^="${n}"]`);
      if (!g) throw new Error(`No SVG g for ${n}`);
      const rect = g.querySelector('rect');
      if (!rect) throw new Error(`No background rect inside ${n}`);
      rect.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    }, name);
  };

  console.log('7. Click sobre Bruno (P1)');
  await clickNodeBackground('Bruno');
  await wait(300);
  await page.screenshot({ path: SHOT('03-p1-selected'), fullPage: true });

  console.log('8. Click sobre Mateo (P2) → dispara cómputo');
  await clickNodeBackground('Mateo');

  // Esperar a que el cosine aparezca (no isComputing).
  await page.waitForFunction(() => {
    const el = document.querySelector('[data-testid="cosine-value"]');
    if (!el) return false;
    const t = el.textContent || '';
    return t !== '…' && t !== '—';
  }, null, { timeout: 60000 });
  await wait(500);

  const cosineText = await page.locator('[data-testid="cosine-value"]').textContent();
  console.log(`   cosine = ${cosineText}`);
  await page.screenshot({ path: SHOT('04-comparison-result'), fullPage: true });

  console.log('9. Recompute (debe re-correr y dar mismo resultado)');
  await page.click('button:has-text("↻ recompute")');
  await page.waitForFunction(() => {
    const el = document.querySelector('[data-testid="cosine-value"]');
    return el && el.textContent === '…';
  }, null, { timeout: 5000 }).catch(() => { /* puede ser muy rápido */ });
  await page.waitForFunction(() => {
    const el = document.querySelector('[data-testid="cosine-value"]');
    return el && el.textContent !== '…' && el.textContent !== '—';
  }, null, { timeout: 60000 });
  await wait(500);
  const cosineText2 = await page.locator('[data-testid="cosine-value"]').textContent();
  console.log(`   cosine post-recompute = ${cosineText2}`);
  if (cosineText !== cosineText2) {
    console.log(`   ⚠ cosine cambió tras recompute (era ${cosineText}, ahora ${cosineText2})`);
  }
  await page.screenshot({ path: SHOT('05-after-recompute'), fullPage: true });

  console.log('10. Refresh y verificar que historial persiste');
  await page.reload({ waitUntil: 'load' });
  await goToTreeTab();
  // Activar modo comparación para ver el panel.
  await page.locator('label:has-text("Modo comparación")').locator('input[type="checkbox"]').check();
  await wait(500);
  await page.screenshot({ path: SHOT('06-after-reload'), fullPage: true });

  // Verificar que hay >= 1 entrada en el historial.
  const histText = await page.locator('text=Historial').first().textContent();
  console.log(`   historial label = "${histText}"`);
  const histMatch = histText?.match(/Historial \((\d+)\)/);
  const histCount = histMatch ? parseInt(histMatch[1], 10) : 0;
  if (histCount < 2) {
    await fail(`Historial debería tener ≥2 entradas (cómputo inicial + recompute), tiene ${histCount}`);
  }
  console.log(`   ✓ historial tiene ${histCount} entradas`);

  console.log('11. Borrar una entrada del historial');
  const firstDeleteBtn = page.locator('ul li button:has-text("✕")').first();
  await firstDeleteBtn.click();
  await wait(400);
  const histText2 = await page.locator('text=Historial').first().textContent();
  const histMatch2 = histText2?.match(/Historial \((\d+)\)/);
  const histCount2 = histMatch2 ? parseInt(histMatch2[1], 10) : 0;
  if (histCount2 !== histCount - 1) {
    await fail(`Historial debería bajar a ${histCount - 1} tras borrar, está en ${histCount2}`);
  }
  console.log(`   ✓ historial bajó a ${histCount2}`);
  await page.screenshot({ path: SHOT('07-after-delete-from-history'), fullPage: true });

  console.log('OK paso 5 smoke');
} catch (e) {
  await page.screenshot({ path: SHOT('99-error'), fullPage: true });
  await fail('exception', e);
} finally {
  await browser.close();
}
