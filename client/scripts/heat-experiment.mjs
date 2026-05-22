// =========================================
// ID: PHYLO_HEAT_EXPERIMENT_PLAYWRIGHT
// VERSION: v1.0
// =========================================
// Script Playwright que orquesta las Fases 1-6 del experimento de temperatura
// del Track 2b. Lo invoca `scripts/heat-experiment.sh`; éste se encarga de
// arrancar/parar vite, samplear el sistema, y de Fase 0 (baseline sin browser).
//
// Estrategia para asociar muestras a fases: escribimos el nombre de la fase a
// un archivo (`.heat-experiment-current-phase.txt`) que el sampler bash lee
// en cada muestra. Sincronización barata y suficiente para granularidad 5s.
//
// Modo headed (no headless) por dos razones:
//   1) WebGPU puede no estar disponible en headless según Chromium build.
//   2) Validación visual del experimento — vos ves lo que pasa.
//
// Imágenes usadas: `data/input/img/spike_e2e_set/` (ya conocidas del Track 2a).

import { chromium } from '@playwright/test';
import { writeFileSync } from 'fs';
import { resolve } from 'path';

const PHASE_FILE = '.heat-experiment-current-phase.txt';
const PHASE_DURATION_MS = 30000; // 30s por fase

// Asumimos cwd = client/ (el bash hace `cd client && node ../scripts/...`).
const TEST_IMG_DIR = resolve('..', 'data/input/img/spike_e2e_set');
const IMG_1 = resolve(TEST_IMG_DIR, 'BrunoFondoBlanco.jpeg');
const IMG_2 = resolve(TEST_IMG_DIR, 'IMG-20191018-WA0000.jpg');

function setPhase(name) {
  writeFileSync(PHASE_FILE, name);
  const ts = new Date().toISOString().substring(11, 19);
  console.log(`\n[${ts}] PHASE → ${name}`);
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({
  headless: false,
  // Permitir WebGPU. Por default Chromium puede no exponerlo en algunos contextos.
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
});
const context = await browser.newContext();
const page = await context.newPage();

try {
  // Cargar app. Esperar a que el tab default (Comparador) esté listo.
  await page.goto('http://localhost:5173/', { waitUntil: 'load' });
  await page.waitForSelector('text="Comparador (MVP)"', { timeout: 15000 });

  // -------------------------------------
  // Phase 1: Comparador idle (default)
  // -------------------------------------
  setPhase('1-comparator-default-idle');
  await wait(PHASE_DURATION_MS);

  // -------------------------------------
  // Phase 2: Árbol genealógico vacío
  // -------------------------------------
  setPhase('2-genealogy-empty');
  await page.click('text="Árbol genealógico"');
  await page.waitForSelector('text=/Árbol genealógico \\(Track 2b/i', { timeout: 5000 });
  await wait(PHASE_DURATION_MS);

  // -------------------------------------
  // Phase 3: árbol con 1 persona + 1 foto
  // -------------------------------------
  // Crear árbol
  await page.fill('input[placeholder*="árbol nuevo"]', 'heat-test');
  await page.getByRole('button', { name: '+ Árbol' }).click();
  await page.waitForTimeout(500);

  // Crear persona
  await page.fill('input[placeholder*="persona nueva"]', 'Persona1');
  await page.getByRole('button', { name: '+ Persona' }).click();
  await page.waitForTimeout(500);

  // Asignar foto — el file input está oculto pero Playwright lo encuentra igual
  const fileInput = page.locator('input[type="file"][accept="image/*"]').first();
  await fileInput.setInputFiles(IMG_1);
  await page.waitForTimeout(500);

  setPhase('3-genealogy-with-photo');
  await wait(PHASE_DURATION_MS);

  // -------------------------------------
  // Phase 4: Comparador + click Comparar (inicializa MediaPipe + ONNX)
  // -------------------------------------
  await page.click('text="Comparador (MVP)"');
  await page.waitForSelector('text=/Padre/i', { timeout: 5000 }).catch(() => {});

  // El Comparador tiene 3 slots; cada uno con su file input oculto.
  // Cargamos imágenes en slot left (Padre) y slot child (Hijo/a).
  const allFileInputs = await page.locator('input[type="file"]').all();
  if (allFileInputs.length < 2) {
    throw new Error(`Comparator file inputs: esperaba 3, encontré ${allFileInputs.length}`);
  }
  // Orden esperado: left (padre), child (hijo/a), right (madre)
  await allFileInputs[0].setInputFiles(IMG_1);
  await page.waitForTimeout(500);
  await allFileInputs[1].setInputFiles(IMG_2);
  await page.waitForTimeout(500);

  // Click Comparar. Texto exacto "Comparar"; durante init cambia a "Inicializando modelos…".
  await page.getByRole('button', { name: 'Comparar' }).click();

  // Esperar al cosine en el resultado. Init de MediaPipe + ONNX puede tardar
  // varios segundos la primera vez (descarga modelos).
  setPhase('4-comparator-after-compare');
  await page.waitForSelector('text=/cosine|similitud/i', { timeout: 120000 }).catch((e) => {
    console.warn(`[playwright] No apareció cosine: ${e.message}`);
  });
  await wait(PHASE_DURATION_MS);

  // -------------------------------------
  // Phase 5: Árbol genealógico (con GPU ya cargada)
  // -------------------------------------
  setPhase('5-genealogy-after-gpu-init');
  await page.click('text="Árbol genealógico"');
  await wait(PHASE_DURATION_MS);

  // -------------------------------------
  // Phase 6: tab cerrada (page closed; browser sigue vivo)
  // -------------------------------------
  setPhase('6-tab-closed');
  await page.close();
  await wait(PHASE_DURATION_MS);
} catch (e) {
  console.error(`[playwright] ERROR: ${e.message}`);
  console.error(e.stack);
} finally {
  await context.close().catch(() => {});
  await browser.close().catch(() => {});
}
