// =========================================
// ID: PHYLOFACE_LIB_GENEALOGY
// VERSION: v1.2
// =========================================
// Cambio v1.1 → v1.2 (Tareas #9/#10/#16 — scores por región persistidos):
// - `Comparison` gana campo opcional `regional`: los scores por región (por
//   método) se guardan en el MISMO registro que el cosine del par, así viven
//   "como las comparaciones generales" del árbol. Opcional → las comparaciones
//   viejas (solo cosine) siguen válidas. Import type-only de los tipos de
//   `regionalScores` (se borra en runtime; el modelo sigue sin deps de runtime).
//
// Cambio v1.0 → v1.1 (Tarea #26 paso 5 — comparación on-demand):
// - Agregado `ComparisonId` + `Comparison` interface + `newComparison(...)`.
//   Una Comparison persiste el resultado de comparar dos Person del mismo
//   Tree: guarda los `*Sha256` snapshot al momento del cómputo (porque la
//   foto de una persona puede cambiar después y queremos que el historial
//   refleje qué se comparó, no el estado actual). El `cosine` se calcula
//   con los embeddings cacheados en PhotoRecord.embedding.
//
// Modelo de datos puro para el Track 2b (Tarea #26 — comparador con árbol
// genealógico). Sin dependencias del DOM ni de IndexedDB: tipos + helpers
// determinísticos. La capa de persistencia vive en `lib/treeStore.ts`.
//
// Modelo: pedigree formal.
//   - Cada Person admite a lo sumo 1 fatherId + 1 motherId (no más, no grafo
//     libre). Decisión del usuario al diseñar Tarea #26 (2026-05-21).
//   - Una Person puede tener photoSha256 o no (alta gradual; primero personas,
//     después fotos). Las fotos viven en un store separado dedupado por hash.
//   - "Generación" no se guarda: se deriva del árbol en `lib/treeLayout.ts`
//     (paso 3 del plan) para no caer en estado redundante que requiera
//     mantenerse en sync con los parentIds.
//
// Decisiones de diseño no obvias:
//   - El embedding NO vive en Person ni en Tree: vive en PhotoRecord, indexado
//     por SHA-256. Si dos personas comparten foto (mismo bebé reusado de un
//     álbum, sibling con foto similar reciclada), el embedding se computa una
//     sola vez. Dedup natural sin ref-counting.
//   - PersonId, TreeId = strings opacos (UUID v4 vía crypto.randomUUID). No
//     enteros autoincrement: facilita el merge tras import (chocar IDs es muy
//     improbable y se puede regenerar si hace falta).
//   - validateAcyclic recorre ancestros hacia arriba — pedigree es DAG por
//     definición ("ser ancestro de uno mismo" es lo único a prevenir). No hay
//     que chequear ciclos generales porque cada nodo tiene a lo sumo 2 padres
//     y los hijos no se almacenan explícitamente.

import type { RegionalMethod, RegionalScoresResult } from './regionalScores';

// -----------------------------------------
// Tipos
// -----------------------------------------

export type PersonId = string;
export type TreeId = string;
export type ComparisonId = string;

/** Hash hex lowercase, 64 chars (SHA-256). */
export type Sha256Hex = string;

export interface Person {
  id: PersonId;
  treeId: TreeId;
  name: string;
  /** Año opcional; sólo informativo, no afecta el layout. */
  birthYear: number | null;
  fatherId: PersonId | null;
  motherId: PersonId | null;
  /** FK a PhotoRecord.sha256 en el store `photos`. */
  photoSha256: Sha256Hex | null;
  /** Timestamp ms epoch. */
  createdAt: number;
}

export interface Tree {
  id: TreeId;
  name: string;
  createdAt: number;
  updatedAt: number;
}

/**
 * Resultado persistido de comparar dos personas. Snapshotea los sha256 al
 * momento del cómputo: si después le cambian la foto a P1 o P2, esta
 * Comparison sigue representando el cálculo original. El historial UI puede
 * marcarlas como "stale" comparando p{1,2}Sha256 contra la photoSha256
 * actual de la persona.
 */
export interface Comparison {
  id: ComparisonId;
  treeId: TreeId;
  p1Id: PersonId;
  p2Id: PersonId;
  p1Sha256: Sha256Hex;
  p2Sha256: Sha256Hex;
  /** cosineSimilarity de los embeddings ArcFace 512-d (no L2-normalizados). */
  cosine: number;
  /** Timestamp ms epoch. */
  computedAt: number;
  /** Scores por región persistidos (Tareas #9/#10/#16), uno por método. Opcional:
   *  ausente en comparaciones viejas (solo cosine). El RegionalScoresResult es
   *  autodescriptivo (child↔este par), así que no depende del orden p1/p2. */
  regional?: Partial<Record<RegionalMethod, RegionalScoresResult>>;
}

/**
 * Foto deduplicada por hash. Vive en su propio store de IDB; varias Person
 * pueden referenciar el mismo sha256 sin duplicar bytes.
 */
export interface PhotoRecord {
  sha256: Sha256Hex;
  blob: Blob;
  /**
   * Embedding ArcFace 512-d L2-no-normalizado (compatible con cosineSimilarity
   * de lib/pipeline.ts). null = aún no computado. Se cachea al primer cálculo.
   */
  embedding: Float32Array | null;
  /** Ancho/alto en px de la imagen original (no del aligned). Útil para UI. */
  width: number;
  height: number;
  createdAt: number;
}

// -----------------------------------------
// Constructores puros
// -----------------------------------------

/** UUID v4 vía Web Crypto. */
export function newId(): string {
  return crypto.randomUUID();
}

export function newTree(name: string): Tree {
  const now = Date.now();
  return {
    id: newId(),
    name: name.trim() || 'Sin nombre',
    createdAt: now,
    updatedAt: now,
  };
}

export function newComparison(
  treeId: TreeId,
  p1Id: PersonId,
  p2Id: PersonId,
  p1Sha256: Sha256Hex,
  p2Sha256: Sha256Hex,
  cosine: number,
): Comparison {
  return {
    id: newId(),
    treeId,
    p1Id,
    p2Id,
    p1Sha256,
    p2Sha256,
    cosine,
    computedAt: Date.now(),
  };
}

export function newPerson(treeId: TreeId, name: string): Person {
  return {
    id: newId(),
    treeId,
    name: name.trim() || 'Sin nombre',
    birthYear: null,
    fatherId: null,
    motherId: null,
    photoSha256: null,
    createdAt: Date.now(),
  };
}

// -----------------------------------------
// SHA-256 de un Blob (hex lowercase).
// Usa SubtleCrypto; disponible en todo browser moderno con contexto seguro
// (localhost y https). Sin contexto seguro: throws desde subtle.digest.
// -----------------------------------------
export async function sha256OfBlob(blob: Blob): Promise<Sha256Hex> {
  const buf = await blob.arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-256', buf);
  const bytes = new Uint8Array(digest);
  let hex = '';
  for (let i = 0; i < bytes.length; i++) {
    hex += bytes[i].toString(16).padStart(2, '0');
  }
  return hex;
}

// -----------------------------------------
// Validación de aciclicidad
// -----------------------------------------

export type AcyclicResult =
  | { ok: true }
  | { ok: false; cycle: PersonId[] };

/**
 * Antes de asignar `candidateParentId` como padre/madre de `personId`,
 * verificar que no se introduzca un ciclo (que el candidato no sea descendiente
 * del propio personId). Si el resultado.ok es false, devuelve el camino del
 * ciclo encontrado para diagnóstico.
 *
 * Importante: chequea sobre el grafo *resultante* de aplicar el cambio, no
 * sobre el grafo actual. Por eso recibe el set completo + la asignación
 * propuesta y simula localmente.
 */
export function wouldCreateCycle(
  persons: Person[],
  personId: PersonId,
  candidateParentId: PersonId,
): AcyclicResult {
  if (personId === candidateParentId) {
    return { ok: false, cycle: [personId, candidateParentId] };
  }
  const byId = new Map<PersonId, Person>();
  for (const p of persons) byId.set(p.id, p);

  // Subir desde candidateParent por sus ancestros. Si llegamos a personId,
  // entonces personId ya es ancestro del candidato → asignarlo como padre
  // crearía un ciclo.
  const visited = new Set<PersonId>();
  const stack: PersonId[][] = [[candidateParentId]];
  while (stack.length > 0) {
    const path = stack.pop()!;
    const head = path[path.length - 1];
    if (visited.has(head)) continue;
    visited.add(head);
    const p = byId.get(head);
    if (!p) continue;
    for (const parentId of [p.fatherId, p.motherId]) {
      if (parentId === null) continue;
      if (parentId === personId) {
        return { ok: false, cycle: [...path, parentId] };
      }
      stack.push([...path, parentId]);
    }
  }
  return { ok: true };
}

// -----------------------------------------
// Queries simples sobre el set de personas
// -----------------------------------------

/** Hijos directos de personId (busca quién lo tiene como father o mother). */
export function childrenOf(persons: Person[], personId: PersonId): Person[] {
  return persons.filter((p) => p.fatherId === personId || p.motherId === personId);
}

/** Personas sin padres registrados — raíces del árbol (generación 0). */
export function roots(persons: Person[]): Person[] {
  return persons.filter((p) => p.fatherId === null && p.motherId === null);
}
