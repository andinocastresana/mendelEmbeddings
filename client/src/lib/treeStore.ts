// =========================================
// ID: PHYLOFACE_LIB_TREESTORE
// VERSION: v1.0
// =========================================
// Capa de persistencia IndexedDB para el Track 2b (Tarea #26). Wrappea la API
// nativa de IDB con promesas; no agrega deps al cliente.
//
// Por qué IDB nativo y no `idb` (Jake Archibald) u otra librería:
//   - 0 KB de bundle extra.
//   - El surface que usamos es chico (3 stores, ~10 operaciones), el wrapper
//     queda en < 200 LOC.
//   - Si en el futuro el surface crece (transacciones cross-store anidadas,
//     cursors complejos, migraciones), reevaluar `idb`.
//
// Estructura de la DB (`phyloface-genealogy`, version 1):
//   - trees   (keyPath: id)
//   - persons (keyPath: id, index: by-tree → treeId)
//   - photos  (keyPath: sha256)
//
// Decisiones de diseño:
//   - **Dedup natural**: putPhoto(blob) calcula sha256 internamente y devuelve
//     el record existente si ya está. El caller no necesita chequear primero.
//     Esto significa también que persons.photoSha256 puede compartirse entre
//     personas sin ref-counting: borrar una persona NO borra la foto (decisión
//     conservadora; vendrá GC manual o nunca).
//   - **embedding null por default**: el embedding se computa lazily la primera
//     vez que se compara con esa foto. setEmbedding lo cachea para próximas
//     comparaciones. Si el modelo cambia, hay que invalidar manualmente (no
//     hay versionado de embeddings todavía — agregable si surge).
//   - **Una única DB para todas las versiones**: si en el futuro tenemos múltiples
//     árboles, viven todos en la misma DB. El scope "tree" se filtra por treeId.
//   - **No transacciones cross-store automáticas**: cada operación abre su
//     propia transacción. Es más simple y suficiente para el patrón de uso del
//     MVP (1 operación de usuario = 1 escritura). Si surge una operación
//     compuesta (ej. importar árbol completo), se hará una helper específica.

import type {
  Person,
  PersonId,
  PhotoRecord,
  Sha256Hex,
  Tree,
  TreeId,
} from './genealogy';
import { sha256OfBlob } from './genealogy';

const DB_NAME = 'phyloface-genealogy';
const DB_VERSION = 1;

const STORE_TREES = 'trees';
const STORE_PERSONS = 'persons';
const STORE_PHOTOS = 'photos';
const INDEX_PERSONS_BY_TREE = 'by-tree';

// -----------------------------------------
// Apertura de DB (cacheada).
// La promesa de la primera apertura se reusa; no abrimos múltiples conexiones
// concurrentes desde un mismo tab.
// -----------------------------------------

let dbPromise: Promise<IDBDatabase> | null = null;

export function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_TREES)) {
        db.createObjectStore(STORE_TREES, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(STORE_PERSONS)) {
        const store = db.createObjectStore(STORE_PERSONS, { keyPath: 'id' });
        store.createIndex(INDEX_PERSONS_BY_TREE, 'treeId', { unique: false });
      }
      if (!db.objectStoreNames.contains(STORE_PHOTOS)) {
        db.createObjectStore(STORE_PHOTOS, { keyPath: 'sha256' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('Failed to open IndexedDB'));
    req.onblocked = () => reject(new Error('IndexedDB open blocked by another connection'));
  });
  return dbPromise;
}

/**
 * Borra la DB entera. Sólo para testing/reset; no exponer en UI sin
 * confirmación explícita. Resetea también el cache de la promesa.
 */
export async function deleteDb(): Promise<void> {
  dbPromise = null;
  await new Promise<void>((resolve, reject) => {
    const req = indexedDB.deleteDatabase(DB_NAME);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error ?? new Error('Failed to delete IndexedDB'));
    req.onblocked = () => reject(new Error('IndexedDB delete blocked'));
  });
}

// -----------------------------------------
// Helper: promisificar una IDBRequest.
// -----------------------------------------

function req<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error ?? new Error('IDBRequest failed'));
  });
}

// -----------------------------------------
// Trees CRUD
// -----------------------------------------

export async function listTrees(): Promise<Tree[]> {
  const db = await openDb();
  const tx = db.transaction(STORE_TREES, 'readonly');
  return req<Tree[]>(tx.objectStore(STORE_TREES).getAll());
}

export async function getTree(id: TreeId): Promise<Tree | null> {
  const db = await openDb();
  const tx = db.transaction(STORE_TREES, 'readonly');
  const result = await req<Tree | undefined>(tx.objectStore(STORE_TREES).get(id));
  return result ?? null;
}

export async function putTree(tree: Tree): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(STORE_TREES, 'readwrite');
  await req(tx.objectStore(STORE_TREES).put({ ...tree, updatedAt: Date.now() }));
}

/**
 * Borra el árbol y todas sus personas. Las fotos NO se borran (pueden estar
 * compartidas con otros árboles; GC manual queda fuera del MVP).
 */
export async function deleteTree(id: TreeId): Promise<void> {
  const db = await openDb();
  const tx = db.transaction([STORE_TREES, STORE_PERSONS], 'readwrite');
  const personsStore = tx.objectStore(STORE_PERSONS);
  const idx = personsStore.index(INDEX_PERSONS_BY_TREE);
  const personIds = await req<PersonId[]>(idx.getAllKeys(id) as IDBRequest<PersonId[]>);
  for (const pid of personIds) {
    personsStore.delete(pid);
  }
  tx.objectStore(STORE_TREES).delete(id);
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('deleteTree tx failed'));
    tx.onabort = () => reject(tx.error ?? new Error('deleteTree tx aborted'));
  });
}

// -----------------------------------------
// Persons CRUD
// -----------------------------------------

export async function listPersons(treeId: TreeId): Promise<Person[]> {
  const db = await openDb();
  const tx = db.transaction(STORE_PERSONS, 'readonly');
  const idx = tx.objectStore(STORE_PERSONS).index(INDEX_PERSONS_BY_TREE);
  return req<Person[]>(idx.getAll(treeId));
}

export async function getPerson(id: PersonId): Promise<Person | null> {
  const db = await openDb();
  const tx = db.transaction(STORE_PERSONS, 'readonly');
  const result = await req<Person | undefined>(tx.objectStore(STORE_PERSONS).get(id));
  return result ?? null;
}

export async function putPerson(person: Person): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(STORE_PERSONS, 'readwrite');
  await req(tx.objectStore(STORE_PERSONS).put(person));
}

/**
 * Borra una persona. NO toca otras personas que la tengan como padre/madre —
 * sus fatherId/motherId quedan apuntando a un id inexistente y la UI debe
 * tratarlos como "padre desconocido". Decisión conservadora: borrar en cascada
 * podría destruir genealogía valiosa.
 *
 * El caller puede querer correr una pasada de "limpieza de refs colgadas"
 * después; lo dejamos como hook futuro.
 */
export async function deletePerson(id: PersonId): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(STORE_PERSONS, 'readwrite');
  await req(tx.objectStore(STORE_PERSONS).delete(id));
}

// -----------------------------------------
// Photos: dedup por sha256
// -----------------------------------------

/**
 * Inserta una foto dedupando por SHA-256. Si el sha256 ya existe, devuelve
 * el record existente sin sobrescribir (preserva el embedding cacheado).
 *
 * Necesita además width/height — los calcula creando un ImageBitmap del blob.
 * Si esto falla (blob no es imagen), throws.
 */
export async function putPhoto(blob: Blob): Promise<PhotoRecord> {
  const sha256 = await sha256OfBlob(blob);
  const existing = await getPhoto(sha256);
  if (existing) return existing;

  const bitmap = await createImageBitmap(blob);
  const record: PhotoRecord = {
    sha256,
    blob,
    embedding: null,
    width: bitmap.width,
    height: bitmap.height,
    createdAt: Date.now(),
  };
  bitmap.close();

  const db = await openDb();
  const tx = db.transaction(STORE_PHOTOS, 'readwrite');
  await req(tx.objectStore(STORE_PHOTOS).put(record));
  return record;
}

export async function getPhoto(sha256: Sha256Hex): Promise<PhotoRecord | null> {
  const db = await openDb();
  const tx = db.transaction(STORE_PHOTOS, 'readonly');
  const result = await req<PhotoRecord | undefined>(tx.objectStore(STORE_PHOTOS).get(sha256));
  return result ?? null;
}

/**
 * Cachea el embedding computado para una foto. Es idempotente: re-llamar con
 * otro embedding lo reemplaza (útil si cambiamos de modelo y reseteamos).
 */
export async function setPhotoEmbedding(
  sha256: Sha256Hex,
  embedding: Float32Array,
): Promise<void> {
  const existing = await getPhoto(sha256);
  if (!existing) throw new Error(`Photo ${sha256} no existe en el store`);
  const updated: PhotoRecord = { ...existing, embedding };
  const db = await openDb();
  const tx = db.transaction(STORE_PHOTOS, 'readwrite');
  await req(tx.objectStore(STORE_PHOTOS).put(updated));
}
