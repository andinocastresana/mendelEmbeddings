// =========================================
// ID: PHYLOFACE_LIB_TREEEXPORT
// VERSION: v1.0
// =========================================
// Export / import del árbol genealógico completo (Tarea #26 paso 6).
//
// Formato: un único JSON autocontenido (schema versionado `v: 1`) que empaqueta
// Tree + Persons + Photos (bytes en base64) + Comparisons. Sin deps nuevas:
// base64 vía btoa/atob, descarga vía Blob + <a download>, lectura vía File.text().
//
// Decisiones de diseño (cerradas con el usuario al arrancar el paso 6):
//   - **Embeddings van en el export, etiquetados con la versión del modelo**
//     (`modelVersion` a nivel de envelope — toda la app usa un solo modelo, así
//     que un campo por foto sería redundante). Al importar: si la versión del
//     archivo coincide con `MODEL_VERSION` del runtime, se reusan los embeddings
//     (import instantáneo); si no, se descartan (null) y se recomputan lazy en
//     la primera comparación. Resuelve el tradeoff caching vs portabilidad.
//   - **El historial de comparaciones va en el export** (metadata útil del árbol).
//   - **Import crea SIEMPRE un árbol nuevo** (sin merge). Por eso se remapean
//     TODOS los ids: treeId, cada PersonId (y sus refs father/mother), y cada
//     ComparisonId (y sus refs p1/p2). Razón no obvia: los PersonId son keyPath
//     del store `persons`; importar el mismo archivo dos veces SIN remapear haría
//     que el segundo import pisara las personas del primero (mismo id, distinto
//     treeId) y el primer árbol perdería sus nodos del índice by-tree. Remapear
//     hace el import idempotente-seguro (cada import es un árbol independiente).
//   - **Fotos NO se remapean**: son content-addressed por sha256. El dedup del
//     store las comparte naturalmente entre árboles sin duplicar bytes.
//   - Refs de padres "dangling" (apuntan a una persona que no está en el export,
//     p.ej. fue borrada) se resuelven a null al importar — el remapeo limpia la
//     ref colgada como efecto colateral deseable.
//
// Sobre el roundtrip: NO es bit-exacto export→import→export porque los ids se
// regeneran (a propósito). El invariante que sí se preserva: misma cantidad de
// personas/fotos/comparaciones, mismos nombres, misma topología de parentesco,
// mismos bytes de foto (sha256 idéntico) y mismos cosines.

import {
  newId,
  type Comparison,
  type ComparisonId,
  type Person,
  type PersonId,
  type PhotoRecord,
  type Sha256Hex,
  type Tree,
  type TreeId,
} from './genealogy';
import { MODEL_VERSION } from './pipeline';
import {
  getPhoto,
  importPhotoRecord,
  listComparisons,
  listPersons,
  getTree,
  putPerson,
  putTree,
  saveComparison,
} from './treeStore';

// -----------------------------------------
// Schema del archivo exportado (v1).
// -----------------------------------------

export const EXPORT_FORMAT = 'phyloface-genealogy';
export const EXPORT_VERSION = 1 as const;

export interface ExportPerson {
  id: PersonId; // id original; se remapea al importar (sirve para rearmar refs)
  name: string;
  birthYear: number | null;
  fatherId: PersonId | null;
  motherId: PersonId | null;
  photoSha256: Sha256Hex | null;
  createdAt: number;
}

export interface ExportPhoto {
  sha256: Sha256Hex;
  mime: string;
  base64: string; // bytes del blob, sin prefijo data:
  width: number;
  height: number;
  createdAt: number;
  /** Embedding 512-d como array plano; null si no estaba cacheado. */
  embedding: number[] | null;
}

export interface ExportComparison {
  id: ComparisonId;
  p1Id: PersonId;
  p2Id: PersonId;
  p1Sha256: Sha256Hex;
  p2Sha256: Sha256Hex;
  cosine: number;
  computedAt: number;
}

export interface TreeExportV1 {
  format: typeof EXPORT_FORMAT;
  v: typeof EXPORT_VERSION;
  exportedAt: number;
  /** Versión del modelo que produjo los embeddings (MODEL_VERSION al exportar). */
  modelVersion: string;
  tree: { name: string; createdAt: number };
  persons: ExportPerson[];
  photos: ExportPhoto[];
  comparisons: ExportComparison[];
}

// -----------------------------------------
// base64 <-> bytes. btoa/atob trabajan sobre "binary strings" (1 char = 1 byte),
// así que convertimos por chunks para no reventar el call stack con spread de
// arrays grandes en String.fromCharCode.
// -----------------------------------------

async function blobToBase64(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  const CHUNK = 0x8000; // 32k chars por pasada
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

function base64ToBlob(b64: string, mime: string): Blob {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime || 'application/octet-stream' });
}

// -----------------------------------------
// Export: arma el objeto serializable. El caller decide cómo bajarlo a disco
// (helper `downloadTreeExport` más abajo lo hace via <a download>).
// -----------------------------------------

export async function buildTreeExport(treeId: TreeId): Promise<TreeExportV1> {
  const tree = await getTree(treeId);
  if (!tree) throw new Error(`Árbol ${treeId} no existe`);

  const persons = await listPersons(treeId);
  const comparisons = await listComparisons(treeId);

  // Conjunto de sha256 referenciados: por la foto actual de cada persona +
  // por los snapshots de cada comparación (pueden apuntar a fotos viejas que
  // ya no son la foto actual de nadie, pero igual queremos exportarlas para
  // que el historial sea autoexplicable).
  const shaSet = new Set<Sha256Hex>();
  for (const p of persons) if (p.photoSha256) shaSet.add(p.photoSha256);
  for (const c of comparisons) {
    if (c.p1Sha256) shaSet.add(c.p1Sha256);
    if (c.p2Sha256) shaSet.add(c.p2Sha256);
  }

  const photos: ExportPhoto[] = [];
  for (const sha of shaSet) {
    const rec = await getPhoto(sha);
    if (!rec) continue; // defensivo: ref a foto inexistente, se saltea
    photos.push({
      sha256: rec.sha256,
      mime: rec.blob.type || 'image/png',
      base64: await blobToBase64(rec.blob),
      width: rec.width,
      height: rec.height,
      createdAt: rec.createdAt,
      embedding: rec.embedding ? Array.from(rec.embedding) : null,
    });
  }

  return {
    format: EXPORT_FORMAT,
    v: EXPORT_VERSION,
    exportedAt: Date.now(),
    modelVersion: MODEL_VERSION,
    tree: { name: tree.name, createdAt: tree.createdAt },
    persons: persons.map((p) => ({
      id: p.id,
      name: p.name,
      birthYear: p.birthYear,
      fatherId: p.fatherId,
      motherId: p.motherId,
      photoSha256: p.photoSha256,
      createdAt: p.createdAt,
    })),
    photos,
    comparisons: comparisons.map((c) => ({
      id: c.id,
      p1Id: c.p1Id,
      p2Id: c.p2Id,
      p1Sha256: c.p1Sha256,
      p2Sha256: c.p2Sha256,
      cosine: c.cosine,
      computedAt: c.computedAt,
    })),
  };
}

/** Dispara la descarga del export como archivo .json. Sólo en browser. */
export async function downloadTreeExport(treeId: TreeId, treeName: string): Promise<void> {
  const data = await buildTreeExport(treeId);
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement('a');
    const safe = (treeName || 'arbol').replace(/[^\w-]+/g, '_').slice(0, 40);
    const stamp = new Date().toISOString().slice(0, 10);
    a.href = url;
    a.download = `phyloface-${safe}-${stamp}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    URL.revokeObjectURL(url);
  }
}

// -----------------------------------------
// Import. Valida schema, crea árbol nuevo, remapea ids, rehidrata IDB.
// -----------------------------------------

export interface ImportResult {
  treeId: TreeId;
  treeName: string;
  personCount: number;
  /** Fotos efectivamente insertadas (las ya presentes por sha256 no cuentan). */
  photosInserted: number;
  photosDeduped: number;
  comparisonCount: number;
  /** True si los embeddings se descartaron por mismatch de versión de modelo. */
  embeddingsDropped: boolean;
}

/** Valida que el objeto parseado sea un TreeExportV1 coherente; lanza si no. */
function validateExport(data: unknown): asserts data is TreeExportV1 {
  if (typeof data !== 'object' || data === null) {
    throw new Error('El archivo no es un objeto JSON');
  }
  const d = data as Record<string, unknown>;
  if (d.format !== EXPORT_FORMAT) {
    throw new Error(`Formato desconocido (esperaba "${EXPORT_FORMAT}", vino "${String(d.format)}")`);
  }
  if (d.v !== EXPORT_VERSION) {
    throw new Error(`Versión de schema no soportada: ${String(d.v)} (esta build entiende v${EXPORT_VERSION})`);
  }
  if (typeof d.tree !== 'object' || d.tree === null || typeof (d.tree as Record<string, unknown>).name !== 'string') {
    throw new Error('Falta el objeto `tree` con `name`');
  }
  if (!Array.isArray(d.persons) || !Array.isArray(d.photos) || !Array.isArray(d.comparisons)) {
    throw new Error('Faltan los arrays `persons`, `photos` o `comparisons`');
  }
}

/**
 * Importa un export JSON (string) creando un árbol NUEVO. Devuelve el resumen.
 * Lanza con mensaje legible si el schema es inválido.
 */
export async function importTreeFromJson(jsonText: string): Promise<ImportResult> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonText);
  } catch (e) {
    throw new Error(`JSON inválido: ${(e as Error).message}`, { cause: e });
  }
  validateExport(parsed);
  const data = parsed;

  // ¿Reusamos los embeddings del archivo? Sólo si la versión del modelo coincide.
  const reuseEmbeddings = data.modelVersion === MODEL_VERSION;

  // 1) Árbol nuevo (id nuevo, name + createdAt preservados).
  const newTreeId: TreeId = newId();
  const tree: Tree = {
    id: newTreeId,
    name: data.tree.name,
    createdAt: data.tree.createdAt ?? Date.now(),
    updatedAt: Date.now(),
  };
  await putTree(tree);

  // 2) Remapeo de PersonId viejo → nuevo.
  const idMap = new Map<PersonId, PersonId>();
  for (const ep of data.persons) idMap.set(ep.id, newId());
  const remap = (id: PersonId | null): PersonId | null =>
    id === null ? null : idMap.get(id) ?? null; // ref dangling → null

  // 3) Personas con ids remapeados.
  for (const ep of data.persons) {
    const person: Person = {
      id: idMap.get(ep.id)!,
      treeId: newTreeId,
      name: ep.name,
      birthYear: ep.birthYear ?? null,
      fatherId: remap(ep.fatherId ?? null),
      motherId: remap(ep.motherId ?? null),
      photoSha256: ep.photoSha256 ?? null,
      createdAt: ep.createdAt ?? Date.now(),
    };
    await putPerson(person);
  }

  // 4) Fotos (dedup por sha256). Embedding sólo si la versión coincide.
  let photosInserted = 0;
  let photosDeduped = 0;
  for (const ph of data.photos) {
    const record: PhotoRecord = {
      sha256: ph.sha256,
      blob: base64ToBlob(ph.base64, ph.mime),
      embedding: reuseEmbeddings && ph.embedding ? new Float32Array(ph.embedding) : null,
      width: ph.width,
      height: ph.height,
      createdAt: ph.createdAt ?? Date.now(),
    };
    const inserted = await importPhotoRecord(record);
    if (inserted) photosInserted++;
    else photosDeduped++;
  }

  // 5) Comparaciones: remapear id + treeId + p1Id/p2Id. Saltear si algún
  //    extremo no existe en el remapeo (no debería pasar; defensivo).
  let comparisonCount = 0;
  for (const ec of data.comparisons) {
    const p1 = idMap.get(ec.p1Id);
    const p2 = idMap.get(ec.p2Id);
    if (!p1 || !p2) continue;
    const comp: Comparison = {
      id: newId(),
      treeId: newTreeId,
      p1Id: p1,
      p2Id: p2,
      p1Sha256: ec.p1Sha256,
      p2Sha256: ec.p2Sha256,
      cosine: ec.cosine,
      computedAt: ec.computedAt ?? Date.now(),
    };
    await saveComparison(comp);
    comparisonCount++;
  }

  return {
    treeId: newTreeId,
    treeName: tree.name,
    personCount: data.persons.length,
    photosInserted,
    photosDeduped,
    comparisonCount,
    embeddingsDropped: !reuseEmbeddings && data.photos.some((p) => p.embedding !== null),
  };
}
