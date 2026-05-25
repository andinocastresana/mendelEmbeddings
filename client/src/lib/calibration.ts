// =========================================
// ID: PHYLOFACE_LIB_CALIBRATION
// VERSION: v1.0
// =========================================
// Consumo client-side del artefacto de calibración de la Tarea #6 (Fase B).
//
// La Fase A (Python, `scripts/run_calibration_kinfacew.py`) emite un JSON
// versionado con, por métrica (cosine/euclidean) y por relación
// (FS/MD/FD/MS/ALL): umbral de Youden, accuracy 5-CV, AUC y un histograma
// pre-binneado (kin vs non-kin). El JSON NO lleva embeddings ni imágenes —
// sólo conteos — así que es seguro y liviano servirlo al browser.
//
// Este módulo:
//   - tipa el contrato (CalibrationArtifact);
//   - lo carga vía fetch desde `/calibration/<dataset>_calibration.json`
//     (cacheado por URL: una sola request por dataset y sesión). Servirlo
//     desde public/ —en vez de bundlearlo— permite refrescar la calibración
//     copiando un JSON nuevo sin rebuild (workflow de "ajustes sucesivos");
//   - convierte un cosine crudo en métricas interpretables (scoreValue):
//       MÉTRICAS PREVIAS (Fase A, se mantienen visibles para comparar entre
//       ajustes): el propio cosine, el veredicto vs umbral de Youden, y la
//       calidad agregada del clasificador (accuracy ± std, AUC).
//       MÉTRICA NUEVA (basada en las distribuciones): probabilidad calibrada
//       de parentesco P(kin|cos) por densidad-ratio (ventana de Parzen sobre
//       los histogramas, priors iguales), más percentiles dentro de cada
//       distribución y el likelihood ratio.
//
// Por qué densidad-ratio y no la regla del umbral: el umbral da un sí/no; la
// pregunta del usuario es "cuánto", y las distribuciones kin/non-kin ya están
// medidas. Con priors iguales, posterior = ppos/(ppos+pneg). El suavizado
// (kernel gaussiano sobre bins + piso de conteo) evita el ruido de bins
// dispersos (~150 muestras / 40 bins).

// -----------------------------------------
// Contrato (espejo del JSON de la Fase A)
// -----------------------------------------
export type Metric = 'cosine' | 'euclidean';
export type Relation = 'FS' | 'MD' | 'FD' | 'MS' | 'ALL';

export const RELATIONS: Relation[] = ['FS', 'MD', 'FD', 'MS', 'ALL'];

// Etiquetas legibles. FS=Father-Son, etc. ALL = agregada (sin discriminar
// relación), que es el fallback cuando no se conoce la relación exacta.
export const RELATION_LABEL: Record<Relation, string> = {
  FS: 'Padre–Hijo',
  MD: 'Madre–Hija',
  FD: 'Padre–Hija',
  MS: 'Madre–Hijo',
  ALL: 'Todas (agregada)',
};

export interface Histogram {
  bin_edges: number[];   // longitud N+1 (bordes de los bins)
  pos_counts: number[];  // longitud N — pares con parentesco (kin)
  neg_counts: number[];  // longitud N — pares sin parentesco (non-kin)
}

export interface RelationCalibration {
  n_pos: number;
  n_neg: number;
  n_folds: number;
  accuracy_mean: number;
  accuracy_std: number;
  threshold_mean: number;       // umbral de Youden promediado sobre folds
  fold_accuracies: number[];
  auc: number;
  histogram: Histogram;
}

export interface CalibrationArtifact {
  v: number;
  computedAt: number;           // epoch ms — distingue corridas sucesivas
  modelVersion: string;         // debe coincidir con pipeline.MODEL_VERSION
  dataset: string;
  protocol: string;
  primaryDataset?: string;
  evaluationRole?: 'primary' | 'secondary-biased' | string;
  warning?: string | null;
  note?: string;
  limit?: number | null;
  metrics: Record<Metric, Record<Relation, RelationCalibration>>;
}

export function calibrationWarning(cal: CalibrationArtifact): string | null {
  if (cal.warning) return cal.warning;
  if (cal.dataset === 'KinFaceW-II') {
    return 'KinFaceW-II se reporta solo como referencia secundaria: sus pares positivos pueden compartir foto familiar original, introduciendo senales de captura/contexto que no son parentesco facial. Usar KinFaceW-I como evaluacion primaria.';
  }
  return null;
}

// Para cosine, "más alto = más parecido = más kin"; para euclídea sobre
// vectores L2-normalizados es al revés (distancia chica = más parecido).
export function higherIsKin(metric: Metric): boolean {
  return metric === 'cosine';
}

// -----------------------------------------
// Loader cacheado por URL.
// -----------------------------------------
const CACHE = new Map<string, Promise<CalibrationArtifact>>();

export function calibrationUrl(dataset = 'KinFaceW-I'): string {
  return `/calibration/${dataset}_calibration.json`;
}

export function loadCalibration(dataset = 'KinFaceW-I'): Promise<CalibrationArtifact> {
  const url = calibrationUrl(dataset);
  let p = CACHE.get(url);
  if (!p) {
    p = fetch(url).then((r) => {
      if (!r.ok) throw new Error(`No se pudo cargar la calibración (${r.status}) en ${url}`);
      return r.json() as Promise<CalibrationArtifact>;
    });
    CACHE.set(url, p);
  }
  return p;
}

// -----------------------------------------
// Scoring de un valor concreto contra la distribución calibrada.
// -----------------------------------------
export interface ValueScore {
  metric: Metric;
  relation: Relation;
  value: number;

  // --- Métricas previas (Fase A) — se conservan para comparar entre ajustes ---
  threshold: number;            // umbral de Youden
  isKin: boolean;               // veredicto duro vs umbral
  auc: number;
  accuracyMean: number;
  accuracyStd: number;

  // --- Métrica nueva (basada en las distribuciones) ---
  posterior: number;            // P(kin | value) ∈ [0,1], priors iguales
  likelihoodRatio: number;      // ppos/pneg (>1 favorece kin)
  percentileKin: number;        // P(cos_kin ≤ value) ∈ [0,1]
  percentileNon: number;        // P(cos_non ≤ value) ∈ [0,1]

  // --- Datos crudos para graficar ---
  histogram: Histogram;
  nPos: number;
  nNeg: number;
}

// Índice de bin que contiene a x (clamp a [0, N-1]). bin i = [edges[i], edges[i+1]).
function binIndex(edges: number[], x: number): number {
  const n = edges.length - 1;
  if (x <= edges[0]) return 0;
  if (x >= edges[n]) return n - 1;
  // búsqueda lineal: N=40, no vale un binary search.
  for (let i = 0; i < n; i++) {
    if (x < edges[i + 1]) return i;
  }
  return n - 1;
}

// CDF empírica P(X ≤ x) a partir de conteos, con interpolación lineal dentro
// del bin que contiene a x. Devuelve fracción de masa en [0,1].
function cdfLeq(counts: number[], edges: number[], x: number): number {
  const total = counts.reduce((a, b) => a + b, 0);
  if (total === 0) return 0;
  const n = counts.length;
  if (x <= edges[0]) return 0;
  if (x >= edges[n]) return 1;
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const lo = edges[i];
    const hi = edges[i + 1];
    if (x >= hi) {
      acc += counts[i];
    } else {
      // x cae dentro del bin i: sumar la fracción proporcional.
      const frac = hi > lo ? (x - lo) / (hi - lo) : 0;
      acc += counts[i] * frac;
      break;
    }
  }
  return acc / total;
}

// Suaviza un vector de conteos con un kernel gaussiano (sigma en bins),
// preservando la escala (~conteos). Reduce el ruido de bins dispersos antes de
// calibrar, sin meter un piso que distorsione las colas vacías.
function gaussianSmoothCounts(counts: number[], sigmaBins: number): number[] {
  const n = counts.length;
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let num = 0;
    let wsum = 0;
    for (let j = 0; j < n; j++) {
      const d = j - i;
      const w = Math.exp(-(d * d) / (2 * sigmaBins * sigmaBins));
      num += w * counts[j];
      wsum += w;
    }
    out[i] = wsum > 0 ? num / wsum : 0;
  }
  return out;
}

// Regresión isotónica no-decreciente por pool-adjacent-violators (PAV),
// ponderada. Devuelve el ajuste monótono de `y` con pesos `w`. Los bins de
// peso ~0 (colas sin datos) se fusionan con el bloque vecino → no introducen
// caídas espurias. Núcleo de la calibración monótona del posterior.
function pavNonDecreasing(y: number[], w: number[]): number[] {
  const n = y.length;
  const gv: number[] = []; // valor del bloque
  const gw: number[] = []; // peso acumulado
  const gs: number[] = []; // tamaño (cantidad de bins) del bloque
  for (let i = 0; i < n; i++) {
    let v = y[i];
    let ww = Math.max(w[i], 1e-12);
    let s = 1;
    // Mientras el bloque previo viole la monotonía (valor ≥ actual), fusionar.
    while (gv.length > 0 && gv[gv.length - 1] >= v) {
      const pv = gv.pop()!;
      const pw = gw.pop()!;
      const ps = gs.pop()!;
      v = (pv * pw + v * ww) / (pw + ww);
      ww = pw + ww;
      s = ps + s;
    }
    gv.push(v); gw.push(ww); gs.push(s);
  }
  const out = new Array<number>(n);
  let k = 0;
  for (let b = 0; b < gv.length; b++) {
    for (let j = 0; j < gs[b]; j++) out[k++] = gv[b];
  }
  return out;
}

// Curva calibrada P(kin | bin) monótona por construcción, evaluada en `idx`.
// Pasos: suavizar conteos → densidades normalizadas por clase (prior igual) →
// posterior por bin pp/(pp+pn) con peso (pp+pn) → isotónica en el sentido
// "más-parecido ⇒ más-kin" (creciente para cosine, decreciente para euclídea).
function calibratedPosteriorAt(
  histogram: Histogram, idx: number, metric: Metric,
): number {
  const { pos_counts, neg_counts } = histogram;
  const n = pos_counts.length;
  const SIGMA = 1.0;
  const sp = gaussianSmoothCounts(pos_counts, SIGMA);
  const sn = gaussianSmoothCounts(neg_counts, SIGMA);
  const tp = sp.reduce((a, b) => a + b, 0) || 1;
  const tn = sn.reduce((a, b) => a + b, 0) || 1;

  const raw = new Array<number>(n);
  const w = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const pp = sp[i] / tp;
    const pn = sn[i] / tn;
    const tot = pp + pn;
    raw[i] = tot > 0 ? pp / tot : 0;
    w[i] = tot;
  }

  // El PAV ajusta no-decreciente. Para cosine eso es lo que queremos (más
  // cosine ⇒ más kin). Para euclídea es al revés: invertimos el orden, ajustamos
  // y desinvertimos.
  if (higherIsKin(metric)) {
    return clamp01(pavNonDecreasing(raw, w)[idx]);
  }
  const fittedRev = pavNonDecreasing(raw.slice().reverse(), w.slice().reverse());
  return clamp01(fittedRev[n - 1 - idx]);
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

export function scoreValue(
  cal: CalibrationArtifact,
  metric: Metric,
  relation: Relation,
  value: number,
): ValueScore {
  const rc = cal.metrics[metric][relation];
  const { bin_edges, pos_counts, neg_counts } = rc.histogram;
  const idx = binIndex(bin_edges, value);

  // Posterior calibrado monótono (isotónico). El LR se deriva de él para ser
  // consistente: con priors iguales, posterior = LR/(1+LR) ⇒ LR = p/(1−p).
  const posterior = calibratedPosteriorAt(rc.histogram, idx, metric);
  const likelihoodRatio = posterior >= 1 ? Infinity : posterior / (1 - posterior);

  const verdictKin = higherIsKin(metric)
    ? value >= rc.threshold_mean
    : value <= rc.threshold_mean;

  return {
    metric,
    relation,
    value,
    threshold: rc.threshold_mean,
    isKin: verdictKin,
    auc: rc.auc,
    accuracyMean: rc.accuracy_mean,
    accuracyStd: rc.accuracy_std,
    posterior,
    likelihoodRatio,
    percentileKin: cdfLeq(pos_counts, bin_edges, value),
    percentileNon: cdfLeq(neg_counts, bin_edges, value),
    histogram: rc.histogram,
    nPos: rc.n_pos,
    nNeg: rc.n_neg,
  };
}

// -----------------------------------------
// Inferencia de relación desde roles (best-effort).
// Las relaciones de KinFaceW cruzan el género del adulto (Father/Mother) con
// el del hijo (Son/Daughter). En el comparador/árbol el adulto suele tener rol
// (Padre/Madre) pero el sexo del hijo NO se conoce → no se puede fijar FS vs FD
// ni MS vs MD. Devolvemos 'ALL' salvo que se conozcan ambos.
//   adultIsFather: true=Padre, false=Madre, null=desconocido
//   childIsSon:    true=hijo,  false=hija,  null=desconocido
// -----------------------------------------
export function inferRelation(
  adultIsFather: boolean | null,
  childIsSon: boolean | null,
): Relation {
  if (adultIsFather === null || childIsSon === null) return 'ALL';
  if (adultIsFather) return childIsSon ? 'FS' : 'FD';
  return childIsSon ? 'MS' : 'MD';
}
