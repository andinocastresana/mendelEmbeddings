// Smoke headless del paso 6 (Tarea #26) — export / import del árbol.
//
// Flujo:
//   1. Construye un árbol "Familia Export" con Bruno+Mateo padres de Hijo, las
//      3 fotos cargadas, y dispara 1 comparación (Bruno↔Mateo) para CACHEAR
//      embeddings — así el export lleva embeddings reales que validar.
//   2. ⬇ Exportar → intercepta la descarga, parsea el JSON y valida el schema
//      (format/v/modelVersion, 3 personas, ≥2 fotos con embedding, 1 comparación).
//   3. ⬆ Importar el mismo JSON vía el file input → crea un árbol NUEVO.
//   4. Lee IndexedDB cruda y verifica la rehidratación:
//        - 2 árboles tras el primer import,
//        - el árbol importado tiene 3 personas con los mismos nombres,
//        - Hijo conserva la topología (padre+madre apuntando dentro del árbol),
//        - ids REMAPEADOS (Bruno importado ≠ Bruno original),
//        - 1 comparación con el mismo cosine,
//        - fotos con embedding preservado (versión de modelo coincide).
//   5. Doble import: reimporta y verifica que el árbol original sigue intacto
//      (3 personas) — el remapeo de ids evita el clobber del store `persons`.
//
// Asume vite corriendo en http://localhost:5173.

import { chromium } from '@playwright/test';
import { resolve } from 'path';
import { readFileSync } from 'fs';

const APP_URL = 'http://localhost:5173/';
const IMG_DIR = resolve(process.env.HOME, 'Proyectos/0_code_(gitHub)/mendelEmbeddings/data/input/img/spike_e2e_set');
const IMG_BRUNO = resolve(IMG_DIR, 'BrunoFondoBlanco.jpeg');
const IMG_MATEO = resolve(IMG_DIR, 'mateoFotoTarjetaTransporte.jpeg');
const IMG_HIJO = resolve(IMG_DIR, 'IMG-20191018-WA0000.jpg');
const EXPORT_PATH = '/tmp/genealogy-p6-export.json';

const SHOT = (name) => `/tmp/genealogy-p6-${name}.png`;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function fail(msg, e) {
  console.error(`FAIL: ${msg}`);
  if (e) console.error(e);
  process.exit(1);
}
function assert(cond, msg) {
  if (!cond) throw new Error(`assert: ${msg}`);
}

const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
});
const context = await browser.newContext({ viewport: { width: 1400, height: 900 }, acceptDownloads: true });
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
    rect.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, ctrlKey: ctrl }));
  }, { n: name, ctrl: withCtrl });
};

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

// Lee todos los records de un store de la DB de genealogía (IDB cruda; no
// depende de importar módulos de la app en el contexto de página).
async function readAll(storeName) {
  return page.evaluate((store) => new Promise((res, rej) => {
    const open = indexedDB.open('phyloface-genealogy');
    open.onsuccess = () => {
      const db = open.result;
      const tx = db.transaction(store, 'readonly');
      const rq = tx.objectStore(store).getAll();
      rq.onsuccess = () => res(rq.result);
      rq.onerror = () => rej(rq.error);
    };
    open.onerror = () => rej(open.error);
  }), storeName);
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

  console.log('2. Navegar a tab Árbol genealógico');
  await page.locator('div').filter({ hasText: /^Árbol genealógico$/ }).first().click();
  await page.waitForSelector('h2:has-text("Árbol genealógico (Track 2b")', { timeout: 5000 });

  console.log('3. Crear árbol "Familia Export"');
  await page.fill('input[placeholder="Nombre del árbol nuevo"]', 'Familia Export');
  await page.click('button:has-text("+ Árbol")');
  await wait(400);

  console.log('4. Crear personas Bruno, Mateo, Hijo');
  for (const name of ['Bruno', 'Mateo', 'Hijo']) {
    await page.fill('input[placeholder="Nombre de la persona nueva"]', name);
    await page.click('button:has-text("+ Persona")');
    await wait(300);
  }

  console.log('5. Subir las 3 fotos');
  const fileInputs = page.locator('svg input[type="file"]');
  await fileInputs.nth(0).setInputFiles(IMG_BRUNO);
  await wait(400);
  await fileInputs.nth(1).setInputFiles(IMG_MATEO);
  await wait(400);
  await fileInputs.nth(2).setInputFiles(IMG_HIJO);
  await wait(400);

  console.log('6. Bruno padre + Mateo madre de Hijo');
  await clickNodeBackground('Hijo');
  await wait(300);
  await page.locator('label:has-text("Padre:") select').selectOption({ label: 'Bruno' });
  await wait(200);
  await page.locator('label:has-text("Madre:") select').selectOption({ label: 'Mateo' });
  await wait(200);
  await page.click('button:has-text("cerrar")');
  await wait(200);

  console.log('7. Comparar Bruno↔Mateo (cachea embeddings + crea 1 comparación)');
  await clickNodeBackground('Bruno', true);
  await wait(200);
  await clickNodeBackground('Mateo', true);
  await waitAllCosinesReady();
  const uiCosine = (await page.locator('[data-testid="cosine-value"]').first().textContent())?.trim();
  console.log(`   cosine UI Bruno↔Mateo = ${uiCosine}`);
  await page.screenshot({ path: SHOT('01-tree-built'), fullPage: true });

  // Capturar estado original (árbol + personas) para comparar tras importar.
  const treesBefore = await readAll('trees');
  assert(treesBefore.length === 1, `esperaba 1 árbol antes de exportar, hay ${treesBefore.length}`);
  const originalTreeId = treesBefore[0].id;
  const personsBefore = (await readAll('persons')).filter((p) => p.treeId === originalTreeId);
  const origByName = Object.fromEntries(personsBefore.map((p) => [p.name, p]));
  assert(personsBefore.length === 3, `esperaba 3 personas originales, hay ${personsBefore.length}`);

  console.log('8. ⬇ Exportar → interceptar descarga + validar JSON');
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.click('button:has-text("Exportar")'),
  ]);
  await download.saveAs(EXPORT_PATH);
  const exp = JSON.parse(readFileSync(EXPORT_PATH, 'utf8'));
  assert(exp.format === 'phyloface-genealogy', `format inesperado: ${exp.format}`);
  assert(exp.v === 1, `v inesperado: ${exp.v}`);
  assert(typeof exp.modelVersion === 'string' && exp.modelVersion.length > 0, 'falta modelVersion');
  assert(exp.persons.length === 3, `export: esperaba 3 personas, hay ${exp.persons.length}`);
  assert(exp.photos.length >= 2, `export: esperaba ≥2 fotos, hay ${exp.photos.length}`);
  const photosWithEmb = exp.photos.filter((p) => Array.isArray(p.embedding) && p.embedding.length === 512);
  assert(photosWithEmb.length >= 2, `export: esperaba ≥2 fotos con embedding 512-d, hay ${photosWithEmb.length}`);
  assert(exp.comparisons.length === 1, `export: esperaba 1 comparación, hay ${exp.comparisons.length}`);
  const expHijo = exp.persons.find((p) => p.name === 'Hijo');
  assert(expHijo.fatherId && expHijo.motherId, 'export: Hijo debería conservar padre+madre');
  console.log(`   ✓ JSON válido: ${exp.persons.length} personas, ${exp.photos.length} fotos (${photosWithEmb.length} c/embedding), ${exp.comparisons.length} comparación, modelVersion=${exp.modelVersion}`);

  console.log('9. ⬆ Importar el mismo JSON → árbol nuevo');
  await page.locator('input[accept="application/json,.json"]').setInputFiles(EXPORT_PATH);
  await page.waitForSelector('text=/Importado «/', { timeout: 15000 });
  await wait(500);
  await page.screenshot({ path: SHOT('02-after-import'), fullPage: true });

  const treesAfter = await readAll('trees');
  assert(treesAfter.length === 2, `tras import esperaba 2 árboles, hay ${treesAfter.length}`);
  const importedTree = treesAfter.find((t) => t.id !== originalTreeId);
  assert(importedTree, 'no encontré el árbol importado');
  const importedPersons = (await readAll('persons')).filter((p) => p.treeId === importedTree.id);
  assert(importedPersons.length === 3, `árbol importado: esperaba 3 personas, hay ${importedPersons.length}`);
  const impByName = Object.fromEntries(importedPersons.map((p) => [p.name, p]));
  for (const n of ['Bruno', 'Mateo', 'Hijo']) {
    assert(impByName[n], `falta "${n}" en el árbol importado`);
  }
  // Topología: Hijo importado apunta a Bruno+Mateo importados (no a los originales).
  assert(impByName['Hijo'].fatherId === impByName['Bruno'].id, 'Hijo.fatherId no apunta al Bruno importado');
  assert(impByName['Hijo'].motherId === impByName['Mateo'].id, 'Hijo.motherId no apunta al Mateo importado');
  // Remapeo de ids: el Bruno importado tiene id distinto al original.
  assert(impByName['Bruno'].id !== origByName['Bruno'].id, 'ids NO remapeados (Bruno importado == original)');
  // Foto compartida por sha256 (dedup): mismo photoSha256.
  assert(impByName['Bruno'].photoSha256 === origByName['Bruno'].photoSha256, 'sha256 de foto no preservado');
  // Comparación importada: 1, mismo cosine que el export.
  const importedComps = (await readAll('comparisons')).filter((c) => c.treeId === importedTree.id);
  assert(importedComps.length === 1, `árbol importado: esperaba 1 comparación, hay ${importedComps.length}`);
  assert(Math.abs(importedComps[0].cosine - exp.comparisons[0].cosine) < 1e-9, 'cosine de la comparación no coincide');
  // Embeddings preservados (versión coincide): la foto sigue teniendo embedding.
  const allPhotos = await readAll('photos');
  const brunoPhoto = allPhotos.find((p) => p.sha256 === impByName['Bruno'].photoSha256);
  assert(brunoPhoto && brunoPhoto.embedding && brunoPhoto.embedding.length === 512, 'embedding de foto no preservado tras import');
  console.log(`   ✓ Import OK: árbol nuevo con 3 personas, topología preservada, ids remapeados, cosine ${importedComps[0].cosine.toFixed(4)}, embeddings reusados`);

  console.log('10. Doble import → original intacto (sin clobber)');
  await page.locator('input[accept="application/json,.json"]').setInputFiles(EXPORT_PATH);
  await page.waitForSelector('text=/Importado «/', { timeout: 15000 });
  await wait(500);
  const treesFinal = await readAll('trees');
  assert(treesFinal.length === 3, `tras doble import esperaba 3 árboles, hay ${treesFinal.length}`);
  const originalPersonsStill = (await readAll('persons')).filter((p) => p.treeId === originalTreeId);
  assert(originalPersonsStill.length === 3, `árbol original clobbereado: quedan ${originalPersonsStill.length} personas (esperaba 3)`);
  await page.screenshot({ path: SHOT('03-double-import'), fullPage: true });
  console.log('   ✓ Árbol original conserva sus 3 personas tras el doble import');

  console.log('OK smoke export/import roundtrip');
} catch (e) {
  await page.screenshot({ path: SHOT('99-error'), fullPage: true });
  await fail('exception', e);
} finally {
  await browser.close();
}
