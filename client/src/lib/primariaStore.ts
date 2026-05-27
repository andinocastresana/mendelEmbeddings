// =========================================
// ID: PHYLOFACE_LIB_PRIMARIA_STORE
// VERSION: v1.0
// =========================================
// Persistencia LOCAL de la App primaria (Tarea #12). 100% en el browser
// (IndexedDB en este equipo) — las imágenes NUNCA se suben a ningún lado; esto
// solo evita que una recarga borre el trabajo. El usuario puede limpiar con el
// botón "Limpiar" de la App primaria (clearPrimariaState).
//
// Qué guarda: por slot (Padre/Hijo/Madre) la imagen original (File) y, si ya se
// analizó, el PipelineOutput serializado (embedding + cara alineada + landmarks).
// Con eso, al recargar:
//   - el veredicto GLOBAL se rearma de los embeddings (coseno, sin inferencia);
//   - el regional GEOMÉTRICO se recomputa de los landmarks (sin inferencia);
//   - occlusion NO se restaura (necesita re-correr ONNX) → re-analizar.
//
// DB dedicada (no la de genealogía/treeStore) para no acoplar ni versionar ese
// esquema. Un único registro bajo clave fija KEY. Se descartan los resultados si
// `modelVersion` no coincide con el runtime (embeddings no comparables); las
// imágenes igual se conservan para re-analizar.

import { MODEL_VERSION, type PipelineOutput } from './pipeline';
import type { RegionalScoresResult, RegionalMethod } from './regionalScores';

const DB_NAME = 'phyloface-primaria';
const STORE = 'state';
const KEY = 'current';
const SCHEMA_V = 1;

export type PrimariaSlotKey = 'padre' | 'child' | 'madre';
type Side = 'left' | 'right';

/** Resultados regionales ya calculados, por método (geométrico/occlusion) y lado.
 *  RegionalScoresResult es objeto plano (scores + meta) → structured-cloneable. */
export type RegionalCache = Partial<Record<RegionalMethod, Partial<Record<Side, RegionalScoresResult>>>>;

// PipelineOutput serializado. File/Blob/TypedArray/number[][] son
// structured-cloneable, así que IndexedDB los guarda directo; ImageData se parte
// en data + dimensiones por portabilidad y se reconstruye al leer.
interface SerializedResult {
  embedding: Float32Array;
  alignedData: Uint8ClampedArray;
  alignedW: number;
  alignedH: number;
  landmarksAligned: number[][];
  meshLandmarksImage: number[][];
  kps: number[][];
  M: number[][];
}
interface SerializedSlot { file: File; result?: SerializedResult; }
interface PersistedState {
  v: number;
  modelVersion: string;
  ts: number;
  slots: Partial<Record<PrimariaSlotKey, SerializedSlot>>;
  /** Scores regionales ya calculados (geométrico/occlusion). Se descartan si la
   *  versión de modelo no coincide (occlusion depende del embedding). */
  regional?: RegionalCache;
}

/** Slot tal como lo consume la App primaria (File + PipelineOutput opcional). */
export interface RestoredSlot { file: File; result?: PipelineOutput; }

/** Estado restaurado: slots + cache regional (vacía si no había o versión cambió). */
export interface RestoredState {
  slots: Partial<Record<PrimariaSlotKey, RestoredSlot>>;
  regional: RegionalCache;
}

// -----------------------------------------
// IndexedDB boilerplate (un store, clave fija).
// -----------------------------------------
function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => { req.result.createObjectStore(STORE); };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function idbPut(value: PersistedState): Promise<void> {
  return openDb().then((db) => new Promise<void>((resolve, reject) => {
    const t = db.transaction(STORE, 'readwrite');
    t.objectStore(STORE).put(value, KEY);
    t.oncomplete = () => { db.close(); resolve(); };
    t.onerror = () => { db.close(); reject(t.error); };
  }));
}

function idbGet(): Promise<PersistedState | undefined> {
  return openDb().then((db) => new Promise<PersistedState | undefined>((resolve, reject) => {
    const t = db.transaction(STORE, 'readonly');
    const r = t.objectStore(STORE).get(KEY);
    r.onsuccess = () => { db.close(); resolve(r.result as PersistedState | undefined); };
    r.onerror = () => { db.close(); reject(r.error); };
  }));
}

function idbDel(): Promise<void> {
  return openDb().then((db) => new Promise<void>((resolve, reject) => {
    const t = db.transaction(STORE, 'readwrite');
    t.objectStore(STORE).delete(KEY);
    t.oncomplete = () => { db.close(); resolve(); };
    t.onerror = () => { db.close(); reject(t.error); };
  }));
}

// -----------------------------------------
// (De)serialización de PipelineOutput.
// -----------------------------------------
function serializeResult(r: PipelineOutput): SerializedResult {
  return {
    embedding: r.embedding,
    alignedData: r.aligned.data,
    alignedW: r.aligned.width,
    alignedH: r.aligned.height,
    landmarksAligned: r.landmarksAligned,
    meshLandmarksImage: r.meshLandmarksImage,
    kps: r.kps,
    M: r.M,
  };
}

function deserializeResult(s: SerializedResult): PipelineOutput {
  return {
    embedding: s.embedding instanceof Float32Array ? s.embedding : new Float32Array(s.embedding),
    aligned: new ImageData(new Uint8ClampedArray(s.alignedData), s.alignedW, s.alignedH),
    landmarksAligned: s.landmarksAligned,
    meshLandmarksImage: s.meshLandmarksImage,
    kps: s.kps,
    M: s.M,
    timings: { detectMs: 0, alignMs: 0, preprocessMs: 0, inferMs: 0 },
  };
}

// -----------------------------------------
// API pública.
// -----------------------------------------
/** Guarda el estado actual (slots + cache regional). Si no hay ningún slot con
 *  foto, limpia el registro. */
export async function savePrimariaState(
  slots: Partial<Record<PrimariaSlotKey, RestoredSlot>>,
  regional?: RegionalCache,
): Promise<void> {
  const out: Partial<Record<PrimariaSlotKey, SerializedSlot>> = {};
  for (const k of Object.keys(slots) as PrimariaSlotKey[]) {
    const s = slots[k];
    if (!s?.file) continue;
    out[k] = { file: s.file, result: s.result ? serializeResult(s.result) : undefined };
  }
  if (Object.keys(out).length === 0) { await clearPrimariaState(); return; }
  try {
    await idbPut({
      v: SCHEMA_V, modelVersion: MODEL_VERSION, ts: Date.now(),
      slots: out,
      regional: regional && Object.keys(regional).length ? regional : undefined,
    });
  } catch (e) {
    console.warn('[primariaStore] save falló:', e);
  }
}

/** Lee el estado persistido. Descarta resultados (PipelineOutput + cache regional)
 *  si la versión de modelo cambió; conserva las imágenes para re-analizar. */
export async function loadPrimariaState(): Promise<RestoredState | null> {
  let st: PersistedState | undefined;
  try { st = await idbGet(); } catch (e) { console.warn('[primariaStore] load falló:', e); return null; }
  if (!st || st.v !== SCHEMA_V) return null;
  const versionOk = st.modelVersion === MODEL_VERSION;
  const slots: Partial<Record<PrimariaSlotKey, RestoredSlot>> = {};
  for (const k of Object.keys(st.slots) as PrimariaSlotKey[]) {
    const s = st.slots[k];
    if (!s?.file) continue;
    slots[k] = { file: s.file, result: (versionOk && s.result) ? deserializeResult(s.result) : undefined };
  }
  if (Object.keys(slots).length === 0) return null;
  return { slots, regional: versionOk ? (st.regional ?? {}) : {} };
}

export async function clearPrimariaState(): Promise<void> {
  try { await idbDel(); } catch (e) { console.warn('[primariaStore] clear falló:', e); }
}
