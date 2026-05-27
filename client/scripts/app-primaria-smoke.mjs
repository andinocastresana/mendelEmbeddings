// Smoke headless de la App primaria (Tarea #12).
//
// Carga 3 fixtures (gitignored, spike_fixtures_detection) en los slots
// Padre / Hijo/a / Madre, dispara "Analizar parecido" y valida:
//   - la pestaña "App primaria" es la default (heading presente al cargar)
//   - aparece el VEREDICTO global (headline "Se parece más a…" / "pareja")
//   - el panel de scores por región auto-computa geométrico y el veredicto
//     regional ("Heredó de Padre/Madre: …") se puebla sincronizado
//   - el radar (svg) se renderiza
// Deja screenshots en /tmp/primaria-*.png para revisión visual multimodal.
// El costo térmico real lo da el .dev-resources.log de dev-monitored.sh.
//
// Headless usa el fallback WASM de ORT: los 3 embeddings van serializados por
// runSessionExclusive (3 runs secuenciales, no concurrentes) → OK headless.
// Asume vite en http://localhost:5173.

import { chromium } from '@playwright/test';
import { resolve } from 'path';
import { readdirSync } from 'fs';

const APP_URL = 'http://localhost:5173/';
const FIX_DIR = resolve(process.env.HOME, 'Proyectos/0_code_(gitHub)/mendelEmbeddings/client/public/spike_fixtures_detection/images');
const imgs = readdirSync(FIX_DIR).filter((f) => f.endsWith('.png')).sort().slice(0, 3).map((f) => resolve(FIX_DIR, f));
const SHOT = (name) => `/tmp/primaria-${name}.png`;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

if (imgs.length < 3) { console.error(`FAIL: necesito 3 fixtures en ${FIX_DIR}, encontré ${imgs.length}`); process.exit(1); }

const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'] });
const context = await browser.newContext({ viewport: { width: 1400, height: 1100 } });
const page = await context.newPage();
page.on('console', (m) => { if (m.type() === 'error') console.log('  [browser error]', m.text()); });
page.on('pageerror', (e) => console.log('  [pageerror]', e.message));

const notComputing = () => page.waitForFunction(
  () => ![...document.querySelectorAll('button')].some((b) => /calculando|analizando/i.test(b.textContent || '')),
  null, { timeout: 240000 },
);

try {
  console.log('imgs:', imgs.map((p) => p.split('/').pop().slice(0, 12)).join(', '));

  console.log('1. cargar app — la pestaña App primaria debe ser la default');
  await page.goto(APP_URL, { waitUntil: 'load' });
  await page.getByRole('heading', { name: /¿A quién se parece\?/i }).waitFor({ timeout: 30000 });
  await page.screenshot({ path: SHOT('0-loaded'), fullPage: true });

  console.log('2. setear 3 fixtures en Padre / Hijo-a / Madre');
  const fileInputs = page.locator('input[type="file"]');
  await fileInputs.nth(0).setInputFiles(imgs[0]); // Padre
  await fileInputs.nth(1).setInputFiles(imgs[1]); // Hijo/a
  await fileInputs.nth(2).setInputFiles(imgs[2]); // Madre
  await wait(500);

  console.log('3. Analizar parecido (init MediaPipe+ONNX en la 1ª corrida; puede tardar)');
  const t0 = Date.now();
  await page.getByRole('button', { name: /analizar parecido/i }).click();

  console.log('4. esperar el VEREDICTO global');
  // Ancla ^ al inicio del texto del elemento: el headline del veredicto empieza
  // con "Se parece…"/"Comparación con…"; así NO matchea la descripción del panel
  // ("Reparto P↔M: … ¿se parece más a uno o a otro?").
  const headline = page.getByText(/^Se parece (más a (Padre|Madre)|de forma pareja a ambos)$|^Comparación con/i).first();
  await headline.waitFor({ timeout: 240000 });
  const headlineText = (await headline.textContent())?.trim();
  console.log(`   veredicto global en ${Date.now() - t0} ms → "${headlineText}"`);
  await page.screenshot({ path: SHOT('1-verdict-global'), fullPage: true });

  console.log('5. esperar que monte el panel y termine el auto-cómputo geométrico');
  await page.getByRole('heading', { name: 'Scores por región' }).waitFor({ timeout: 60000 });
  await notComputing();

  console.log('6. esperar el veredicto REGIONAL (herencia por región)');
  await page.getByText(/Heredó de Padre/i).waitFor({ timeout: 60000 });
  await wait(400);
  await page.screenshot({ path: SHOT('2-verdict-regional'), fullPage: true });

  // Extraer las líneas de herencia para el log.
  const herenciaPadre = (await page.getByText(/Heredó de Padre/i).first().evaluate((el) => el.parentElement?.textContent || '')).trim();
  const herenciaMadre = (await page.getByText(/Heredó de Madre/i).first().evaluate((el) => el.parentElement?.textContent || '')).trim();
  console.log(`   ${herenciaPadre}`);
  console.log(`   ${herenciaMadre}`);

  console.log('7. toggle heatmap sobre el Hijo/a');
  await page.getByRole('checkbox', { name: /heatmap/i }).click();
  await wait(400);
  await page.screenshot({ path: SHOT('3-heatmap'), fullPage: true });

  // Asserts mínimos.
  const svgCount = await page.locator('svg[aria-label="Radar de scores por región"]').count();
  if (svgCount < 1) throw new Error('no se renderizó el radar');
  if (!headlineText || !/se parece|pareja|comparación/i.test(headlineText)) throw new Error('no se renderizó el veredicto global');

  console.log(`\nPASS — App primaria OK. Veredicto global + regional renderizados. Screenshots en /tmp/primaria-*.png`);
  console.log(`SMOKE_RESULT total_ms=${Date.now() - t0}`);
} catch (e) {
  console.error('FAIL:', e.message);
  await page.screenshot({ path: SHOT('FAIL'), fullPage: true }).catch(() => {});
  await browser.close();
  process.exit(1);
}
await browser.close();
process.exit(0);
