// =========================================
// ID: PHYLOFACE_LIB_REGIONAL_SCORES
// VERSION: v1.0
// =========================================
// Contrato DESACOPLADO de scores por región: la UI consume `RegionalScoresResult`
// sin saber qué algoritmo lo produjo. Distintos `RegionalScorer` (geométrico,
// occlusion, arcface-crop, futuro server/MLP) son intercambiables detrás de esta
// interfaz y se eligen con un selector en la UI.
//
// Decisión de diseño (2026-05-25): aunque la validación de embeddings regionales
// ArcFace (#5) dio NEGATIVA, eso no bloquea la interfaz. Cada método declara su
// `baseConfidence`; la UI muestra scores con su etiqueta de método + confiabilidad,
// así un método débil se presenta honestamente como "experimental" y se reemplaza
// sin tocar la UI. Mismo patrón "adapter intercambiable" que el detector de caras.
//
// Semántica del score (0..1): "cuánto se parece la región de A a la de B".
// Cada método documenta CÓMO lo calcula (similitud de forma, contribución al
// parecido global, etc.) — no son directamente comparables entre métodos, por eso
// el método siempre viaja con el resultado.

import type * as ort from 'onnxruntime-web';
import type { RegionName } from './regions';

export type RegionalMethod = 'geometric' | 'occlusion' | 'arcface-crop';

export type ScoreConfidence = 'high' | 'medium' | 'low' | 'experimental';

export interface RegionalScore {
  region: RegionName;
  /** Score ABSOLUTO 0..1; mayor = más parecido/contribución. NaN si `valid:false`.
   *  El panel puede re-normalizar (relativo min-max) como opción de display. */
  score: number;
  /** Valor crudo subyacente del método (geométrico: dist px; occlusion: Δcos). */
  raw?: number;
  confidence: ScoreConfidence;
  valid: boolean;
  note?: string;
}

export interface RegionalScoresResult {
  method: RegionalMethod;
  methodLabel: string;
  /** Confiabilidad global del método (la UI la muestra como disclaimer). */
  baseConfidence: ScoreConfidence;
  scores: RegionalScore[];
  /** Métrica auxiliar opcional (p.ej. cosine global, timings). */
  meta?: Record<string, number | string>;
}

// -----------------------------------------
// Datos por cara que consumen los scorers. Los produce el pipeline
// (lib/pipeline.ts) — ver el paso de "exponer landmarks + M" pendiente.
// -----------------------------------------
export interface FaceRegionData {
  /** 478 landmarks Face Mesh mapeados al espacio ALINEADO 112×112 (x,y en px). */
  landmarksAligned: number[][];
  /** Cara alineada 112×112 RGBA (para enmascarar regiones y re-embeddear). */
  aligned: ImageData;
  /** Embedding global 512-d de la cara alineada sin ocluir. */
  embedding: Float32Array;
}

export interface RegionalScorerContext {
  /** Sesión ONNX, para métodos que re-corren inferencia (occlusion). */
  session?: ort.InferenceSession;
  /** Subconjunto de regiones a evaluar; por defecto todas las canónicas. */
  regions?: RegionName[];
  /** Progreso opcional (done/total) para métodos largos (occlusion): la UI lo
   *  usa para una barra real región-a-región. Los métodos instantáneos lo ignoran. */
  onProgress?: (done: number, total: number) => void;
}

export interface RegionalScorer {
  method: RegionalMethod;
  /** Etiqueta para el selector de la UI. */
  label: string;
  baseConfidence: ScoreConfidence;
  description: string;
  /** Mide, región por región, cuánto se parece `a` a `b`. */
  score(
    a: FaceRegionData,
    b: FaceRegionData,
    ctx: RegionalScorerContext,
  ): Promise<RegionalScoresResult>;
}

// -----------------------------------------
// Registry: los scorers se registran a sí mismos al importarse; la UI los
// lista para el selector y resuelve por método.
// -----------------------------------------
const REGISTRY = new Map<RegionalMethod, RegionalScorer>();

export function registerScorer(scorer: RegionalScorer): void {
  REGISTRY.set(scorer.method, scorer);
}

export function getScorer(method: RegionalMethod): RegionalScorer | undefined {
  return REGISTRY.get(method);
}

export function listScorers(): RegionalScorer[] {
  return [...REGISTRY.values()];
}

const CONFIDENCE_LABEL_ES: Record<ScoreConfidence, string> = {
  high: 'alta',
  medium: 'media',
  low: 'baja',
  experimental: 'experimental',
};

export function confidenceLabelEs(c: ScoreConfidence): string {
  return CONFIDENCE_LABEL_ES[c];
}
