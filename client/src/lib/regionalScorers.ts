// =========================================
// ID: PHYLOFACE_LIB_REGIONAL_SCORERS
// VERSION: v1.0
// =========================================
// Implementaciones concretas del contrato `RegionalScorer` (lib/regionalScores.ts).
// Dos métodos browser-native, ambos sobre la cara ALINEADA 112×112 y los 478
// landmarks mapeados a ese espacio (los expone lib/pipeline.ts):
//
//   - geometricScorer (#4): compara la GEOMETRÍA de cada región (centroide, ancho,
//     alto, proporción) entre las dos caras alineadas. Como ambas se alinean al
//     mismo template ArcFace (ojos/nariz/boca forzados a posiciones canónicas), la
//     variación que queda en cada región es señal de forma real (ancho de nariz,
//     arco de cejas, contorno de mentón/mejillas). Barato: sin inferencia.
//     Semántica del score: similitud de forma 0..1 (exp(-dist)).
//
//   - occlusionScorer (#10): por región, tapa su bbox en la cara A, re-corre ONNX y
//     mide la CAÍDA del coseno A↔B. Drop grande = esa región sostiene el parecido
//     global. ~12 inferencias por comparación (una por región, NO ventana densa).
//     Semántica del score: contribución relativa al parecido global 0..1 (NO es
//     "similitud de la región"); el drop crudo va en `note`/meta.
//
// Las dos semánticas son distintas (forma vs contribución) — por eso el método
// viaja siempre con el resultado y la UI lo muestra. No son comparables entre sí.

import * as ort from 'onnxruntime-web';
import {
  CANONICAL_REGIONS, FOREHEAD_REF_BROW_IDX, FOREHEAD_REF_EYE_IDX,
  type RegionName, type RegionSpec,
} from './regions';
import { cosineSimilarity, imageDataToTensorRGB, runSessionExclusive } from './pipeline';
import {
  registerScorer, type RegionalScorer,
  type RegionalScorerContext, type RegionalScoresResult, type RegionalScore,
} from './regionalScores';

const ALIGNED_SIZE = 112;

export interface Box { x1: number; y1: number; x2: number; y2: number; }

// Puntos (en espacio alineado) que definen una región. Forehead se deriva de
// cejas+ojos (Face Mesh no da contorno cerrado), igual que geometry.get_forehead_bbox.
function regionPoints(region: RegionSpec, lm: number[][]): number[][] {
  if (region.landmarkIdx) {
    return region.landmarkIdx.map((i) => lm[i]).filter(Boolean);
  }
  return [];
}

// bbox min/max + padding relativo, clamp a [0, ALIGNED_SIZE].
function bboxOf(pts: number[][], pad: number): Box | null {
  if (pts.length === 0) return null;
  let xMin = Infinity, yMin = Infinity, xMax = -Infinity, yMax = -Infinity;
  for (const [x, y] of pts) {
    if (x < xMin) xMin = x; if (y < yMin) yMin = y;
    if (x > xMax) xMax = x; if (y > yMax) yMax = y;
  }
  const bw = Math.max(1, xMax - xMin);
  const bh = Math.max(1, yMax - yMin);
  const clamp = (v: number) => Math.max(0, Math.min(ALIGNED_SIZE, v));
  return {
    x1: clamp(xMin - bw * pad), y1: clamp(yMin - bh * pad),
    x2: clamp(xMax + bw * pad), y2: clamp(yMax + bh * pad),
  };
}

// bbox de la frente: ancho de cejas, alto = 2× distancia centro-cejas↔centro-ojos
// hacia arriba (espacio alineado: y crece hacia abajo). Espejo de get_forehead_bbox.
function foreheadBox(lm: number[][]): Box | null {
  const brow = FOREHEAD_REF_BROW_IDX.map((i) => lm[i]).filter(Boolean);
  const eye = FOREHEAD_REF_EYE_IDX.map((i) => lm[i]).filter(Boolean);
  if (brow.length === 0 || eye.length === 0) return null;
  const browXs = brow.map((p) => p[0]);
  const browYs = brow.map((p) => p[1]);
  const xMin = Math.min(...browXs), xMax = Math.max(...browXs);
  const browCenterY = browYs.reduce((a, b) => a + b, 0) / browYs.length;
  const eyeCenterY = eye.map((p) => p[1]).reduce((a, b) => a + b, 0) / eye.length;
  const delta = Math.max(8, Math.abs(eyeCenterY - browCenterY) * 2.0);
  const yBottom = Math.min(...browYs) + 2;
  const clamp = (v: number) => Math.max(0, Math.min(ALIGNED_SIZE, v));
  return { x1: clamp(xMin - 4), y1: clamp(yBottom - delta), x2: clamp(xMax + 4), y2: clamp(yBottom) };
}

function regionBox(region: RegionSpec, lm: number[][]): Box | null {
  if (region.derived && region.name === 'forehead') return foreheadBox(lm);
  return bboxOf(regionPoints(region, lm), region.name.includes('eye') ? 0.25 : 0.2);
}

// Bboxes (espacio alineado 112×112) de TODAS las regiones canónicas. Lo usa el
// overlay heatmap del panel para teñir las regiones sobre la cara del Hijo/a.
export function regionBoxesAligned(landmarksAligned: number[][]): Record<RegionName, Box | null> {
  const out = {} as Record<RegionName, Box | null>;
  for (const region of CANONICAL_REGIONS) out[region.name] = regionBox(region, landmarksAligned);
  return out;
}

function activeRegions(ctx: RegionalScorerContext): RegionSpec[] {
  if (!ctx.regions) return [...CANONICAL_REGIONS];
  const set = new Set(ctx.regions);
  return CANONICAL_REGIONS.filter((r) => set.has(r.name));
}

// =========================================================
// geometricScorer (#4)
// =========================================================
// Feature por región en espacio alineado: [centroidX, centroidY, ancho, alto].
// dist = ||featA - featB|| (px del canvas 112); score = exp(-dist / SCALE).
const GEOM_SCALE_PX = 14; // px: una diferencia de ~14px baja el score a ~1/e.

function regionGeomFeature(box: Box): [number, number, number, number] {
  return [(box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2, box.x2 - box.x1, box.y2 - box.y1];
}

export const geometricScorer: RegionalScorer = {
  method: 'geometric',
  label: 'Geométrico (forma)',
  baseConfidence: 'medium',
  description:
    'Compara la forma de cada región (centroide, ancho, alto) sobre las caras ' +
    'alineadas. Mide parecido de forma/proporción, no de textura ni identidad. ' +
    'Determinístico y barato (sin inferencia).',
  async score(a, b, ctx): Promise<RegionalScoresResult> {
    const scores: RegionalScore[] = activeRegions(ctx).map((region) => {
      const ba = regionBox(region, a.landmarksAligned);
      const bb = regionBox(region, b.landmarksAligned);
      if (!ba || !bb) {
        return { region: region.name, score: NaN, confidence: 'low', valid: false, note: 'región sin landmarks' };
      }
      const fa = regionGeomFeature(ba);
      const fb = regionGeomFeature(bb);
      let d2 = 0;
      for (let i = 0; i < fa.length; i++) d2 += (fa[i] - fb[i]) ** 2;
      const dist = Math.sqrt(d2);
      const score = Math.exp(-dist / GEOM_SCALE_PX);
      return { region: region.name, score, raw: dist, confidence: 'medium', valid: true };
    });
    return { method: 'geometric', methodLabel: this.label, baseConfidence: this.baseConfidence, scores };
  },
};

// =========================================================
// occlusionScorer (#10)
// =========================================================
// Por región: copia la cara alineada de A, rellena la bbox de la región con gris
// neutro (127 → ~0 tras la normalización del modelo), re-corre ONNX y mide la
// caída del coseno A↔B. score = contribución RELATIVA (min-max sobre las regiones).
const OCCLUSION_FILL = 127;
// Δcos que mapea a score absoluto = 1 (parecido global totalmente sostenido por una
// sola región). El panel hace la normalización relativa (min-max) como opción de display.
const OCCLUSION_ABS_SCALE = 0.2;

function maskRegion(aligned: ImageData, box: Box): ImageData {
  const out = new ImageData(new Uint8ClampedArray(aligned.data), aligned.width, aligned.height);
  const x1 = Math.round(box.x1), y1 = Math.round(box.y1);
  const x2 = Math.round(box.x2), y2 = Math.round(box.y2);
  for (let y = y1; y < y2; y++) {
    for (let x = x1; x < x2; x++) {
      const o = (y * out.width + x) * 4;
      out.data[o] = OCCLUSION_FILL; out.data[o + 1] = OCCLUSION_FILL; out.data[o + 2] = OCCLUSION_FILL;
    }
  }
  return out;
}

async function embedAligned(aligned: ImageData, session: ort.InferenceSession): Promise<Float32Array> {
  const tensorData = imageDataToTensorRGB(aligned);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const tensor = new ort.Tensor('float32', tensorData, [1, 3, ALIGNED_SIZE, ALIGNED_SIZE]);
  // Serializado con el resto de runs de la sesión (ver runSessionExclusive):
  // evita solaparse con el pipeline del Comparador (ORT-web no admite runs
  // concurrentes en una misma sesión).
  const outputs = await runSessionExclusive(() => session.run({ [inputName]: tensor }));
  return new Float32Array(outputs[outputName].data as Float32Array);
}

export const occlusionScorer: RegionalScorer = {
  method: 'occlusion',
  label: 'Occlusion (contribución)',
  baseConfidence: 'medium',
  description:
    'Tapa cada región de la cara A, recalcula el embedding y mide cuánto cae el ' +
    'coseno con B. Score = contribución relativa de la región al parecido global ' +
    '(no es "similitud de la región"). ~1 inferencia ONNX por región.',
  async score(a, b, ctx): Promise<RegionalScoresResult> {
    if (!ctx.session) throw new Error('occlusionScorer requiere ctx.session (ONNX)');
    const regions = activeRegions(ctx);
    const baseCos = cosineSimilarity(a.embedding, b.embedding);

    const drops: { region: RegionName; drop: number; valid: boolean }[] = [];
    for (const region of regions) {
      const box = regionBox(region, a.landmarksAligned);
      if (!box || box.x2 - box.x1 < 1 || box.y2 - box.y1 < 1) {
        drops.push({ region: region.name, drop: NaN, valid: false });
        continue;
      }
      try {
        const occEmb = await embedAligned(maskRegion(a.aligned, box), ctx.session);
        const occCos = cosineSimilarity(occEmb, b.embedding);
        drops.push({ region: region.name, drop: baseCos - occCos, valid: true });
      } catch (err) {
        // Una inferencia que falla (p.ej. flakiness de ORT-web) no debe abortar
        // toda la occlusion: marca la región inválida y sigue.
        drops.push({ region: region.name, drop: NaN, valid: false });
        console.warn(`[occlusion] región ${region.name} falló:`, err);
      }
      // Yield entre runs: deja respirar a ORT/GPU (evita errores por muchas
      // inferencias encadenadas sin ceder el event loop).
      await new Promise((r) => setTimeout(r, 0));
    }

    // Score ABSOLUTO 0..1 = Δcos / escala fija. El panel hace la normalización
    // relativa (min-max) como opción de display; el Δcos crudo va en `raw`/`note`.
    const scores: RegionalScore[] = drops.map((d) => ({
      region: d.region,
      score: d.valid ? Math.max(0, Math.min(1, d.drop / OCCLUSION_ABS_SCALE)) : NaN,
      raw: d.valid ? d.drop : undefined,
      confidence: 'medium',
      valid: d.valid,
      note: d.valid ? `Δcos = ${d.drop.toFixed(4)}` : 'región sin bbox válida',
    }));

    return {
      method: 'occlusion',
      methodLabel: this.label,
      baseConfidence: this.baseConfidence,
      scores,
      meta: { baseCosine: baseCos, inferencesRun: drops.filter((d) => d.valid).length },
    };
  },
};

// Auto-registro en el registry (la UI los lista para el selector).
registerScorer(geometricScorer);
registerScorer(occlusionScorer);
