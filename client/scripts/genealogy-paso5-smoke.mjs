// Smoke headless del paso 5 (Tarea #26) — iteración multi-selección.
//
// Crea 3 personas (Bruno, Mateo, Hijo) con Bruno+Mateo padres de Hijo para
// ejercitar el pedigree canónico (línea de unión + bus). Luego selecciona los
// 3 con ctrl+click y verifica que aparezcan 3 líneas naranjas (3 pares =
// C(3,2)). Deselecciona uno y verifica que queden 1 línea (pares restantes).
//
// Asume vite corriendo en http://localhost:5173.

import { chromium } from '@playwright/test';
import { resolve } from 'path';

const APP_URL = 'http://localhost:5173/';
const IMG_DIR = resolve(process.env.HOME, 'Proyectos/0_code_(gitHub)/mendelEmbeddings/data/input/img/spike_e2e_set');
const IMG_BRUNO = resolve(IMG_DIR, 'BrunoFondoBlanco.jpeg');
const IMG_MATEO = resolve(IMG_DIR, 'mateoFotoTarjetaTransporte.jpeg');
const IMG_HIJO = resolve(IMG_DIR, 'IMG-20191018-WA0000.jpg');

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

const clickNodeBackground = async (name, withCtrl = false) => {
  await page.evaluate(({ n, ctrl }) => {
    const g = document.querySelector(`g[aria-label^="${n}"]`);
    if (!g) throw new Error(`No SVG g for ${n}`);
    const rect = g.querySelector('rect');
    if (!rect) throw new Error(`No background rect inside ${n}`);
    rect.dispatchEvent(new MouseEvent('click', {
      bubbles: true, cancelable: true, ctrlKey: ctrl,
    }));
  }, { n: name, ctrl: withCtrl });
};

// Espera a que todas las cosines pendientes se hayan resuelto: contamos
// labels que muestran "…" (computing) y esperamos que sean 0.
async function waitAllCosinesReady() {
  await page.waitForFunction(() => {
    const labels = Array.from(document.querySelectorAll('[data-testid="cosine-value"]'));
    if (labels.length === 0) return false;
    return labels.every((el) => {
      const t = el.textContent || '';
      return t !== '…' && t !== '—';
    });
  }, null, { timeout: 180000 });
}

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

  const goToTreeTab = async () => {
    await page.locator('div').filter({ hasText: /^Árbol genealógico$/ }).first().click();
    await page.waitForSelector('h2:has-text("Árbol genealógico (Track 2b")', { timeout: 5000 });
  };

  console.log('2. Navegar a tab Árbol genealógico');
  await goToTreeTab();

  console.log('3. Crear árbol "Familia Test"');
  await page.fill('input[placeholder="Nombre del árbol nuevo"]', 'Familia Test');
  await page.click('button:has-text("+ Árbol")');
  await wait(400);

  console.log('4. Crear personas Bruno, Mateo, Hijo');
  for (const name of ['Bruno', 'Mateo', 'Hijo']) {
    await page.fill('input[placeholder="Nombre de la persona nueva"]', name);
    await page.click('button:has-text("+ Persona")');
    await wait(300);
  }

  console.log('5. Subir fotos a los 3 nodos');
  const fileInputs = page.locator('svg input[type="file"]');
  await fileInputs.nth(0).setInputFiles(IMG_BRUNO);
  await wait(500);
  await fileInputs.nth(1).setInputFiles(IMG_MATEO);
  await wait(500);
  await fileInputs.nth(2).setInputFiles(IMG_HIJO);
  await wait(500);

  console.log('6. Asignar Bruno padre y Mateo madre de Hijo');
  await clickNodeBackground('Hijo');
  await wait(300);
  await page.locator('label:has-text("Padre:") select').selectOption({ label: 'Bruno' });
  await wait(300);
  await page.locator('label:has-text("Madre:") select').selectOption({ label: 'Mateo' });
  await wait(300);
  await page.click('button:has-text("cerrar")');
  await wait(200);
  await page.screenshot({ path: SHOT('01-pedigree-canonical'), fullPage: true });

  console.log('7. Ctrl+click Bruno → sel 1');
  await clickNodeBackground('Bruno', true);
  await wait(300);
  const after1Pairs = await page.locator('g[data-pair-key]').count();
  if (after1Pairs !== 0) {
    await fail(`Con 1 seleccionado no debería haber pares; encontré ${after1Pairs}`);
  }
  await page.screenshot({ path: SHOT('02-sel-1'), fullPage: true });

  console.log('8. Ctrl+click Mateo → sel 2 (dispara cómputo de 1 par)');
  await clickNodeBackground('Mateo', true);
  await waitAllCosinesReady();
  const after2Pairs = await page.locator('g[data-pair-key]').count();
  if (after2Pairs !== 1) {
    await fail(`Con 2 seleccionados esperaba 1 par, encontré ${after2Pairs}`);
  }
  const cosines2 = await page.locator('[data-testid="cosine-value"]').allTextContents();
  console.log(`   1 cosine: ${cosines2.join(' ')}`);
  await page.screenshot({ path: SHOT('03-sel-2-one-line'), fullPage: true });

  console.log('9. Ctrl+click Hijo → sel 3 (dispara cómputo de 2 pares nuevos; total 3)');
  await clickNodeBackground('Hijo', true);
  await waitAllCosinesReady();
  const after3Pairs = await page.locator('g[data-pair-key]').count();
  if (after3Pairs !== 3) {
    await fail(`Con 3 seleccionados esperaba 3 pares, encontré ${after3Pairs}`);
  }
  const cosines3 = await page.locator('[data-testid="cosine-value"]').allTextContents();
  console.log(`   3 cosines: ${cosines3.join(' · ')}`);
  await page.screenshot({ path: SHOT('04-sel-3-three-lines'), fullPage: true });

  console.log('10. Click sobre cosine label Bruno↔Mateo → abre modal de tripleta');
  // El label tiene data-testid="cosine-svg-label" envuelto en un <g>. Tomamos
  // el primero (el del par 1, que es Bruno↔Mateo según el orden de selección).
  // Para hacer click sobre el <g> SVG con cosine cacheado, dispatchEvent al
  // <rect> hijo del label.
  await page.evaluate(() => {
    const labels = document.querySelectorAll('[data-testid="cosine-svg-label"]');
    if (labels.length === 0) throw new Error('No cosine labels en SVG');
    const first = labels[0];
    const rect = first.querySelector('rect');
    if (!rect) throw new Error('Label sin rect');
    rect.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  });
  await wait(400);
  const modalVisible = await page.locator('[data-testid="triplet-modal"]').count();
  if (modalVisible !== 1) {
    await fail(`Esperaba modal de tripleta abierto, encontré ${modalVisible}`);
  }
  // En el modal, el cosine A↔B debería estar visible (mismo valor: 0.2302).
  const modalCosines = await page.locator('[data-testid="cosine-modal-value"]').allTextContents();
  console.log(`   modal cosines: ${modalCosines.join(' · ')}`);
  if (!modalCosines.some((c) => c === '0.2302')) {
    await fail(`Esperaba cosine 0.2302 en el modal, encontré: ${modalCosines.join(', ')}`);
  }
  await page.screenshot({ path: SHOT('05-modal-pair'), fullPage: true });

  console.log('11. Agregar tercero (Hijo) → dispara 2 cómputos extras');
  await page.locator('[data-testid="triplet-add-third"]').selectOption({ label: 'Hijo' });
  // Esperar a que las 3 cosines estén listas en el modal.
  await page.waitForFunction(() => {
    const labels = Array.from(document.querySelectorAll('[data-testid="cosine-modal-value"]'));
    return labels.length === 3 && labels.every((el) => (el.textContent || '') !== '…');
  }, null, { timeout: 60000 });
  const modal3Cosines = await page.locator('[data-testid="cosine-modal-value"]').allTextContents();
  console.log(`   3 cosines en modal: ${modal3Cosines.join(' · ')}`);
  await page.screenshot({ path: SHOT('06-modal-triplet'), fullPage: true });

  console.log('12. Handoff → click "abrir en Comparador MVP"');
  await page.locator('[data-testid="triplet-handoff-mvp"]').click();
  await wait(400);
  // Verificar que cambió al tab Comparador (h1 del componente).
  await page.waitForSelector('h1:has-text("Comparador anónimo")', { timeout: 5000 });
  // El Comparator carga los slots desde IDB; las fotos pueden tardar un
  // par de ticks en aparecer (read from IDB + createObjectURL). Esperamos
  // que los 3 slots tengan <img>.
  await page.waitForFunction(() => {
    // Comparator pinta cada slot cargado con un <img>. Si hay >=3 imgs en la
    // página del Comparator, el prefill funcionó.
    const imgs = document.querySelectorAll('img');
    return imgs.length >= 3;
  }, null, { timeout: 10000 });
  await page.screenshot({ path: SHOT('07-comparator-prefilled'), fullPage: true });
  console.log('   ✓ Comparador prellenado');

  console.log('OK smoke multi-selección + tripleta + handoff');
} catch (e) {
  await page.screenshot({ path: SHOT('99-error'), fullPage: true });
  await fail('exception', e);
} finally {
  await browser.close();
}
