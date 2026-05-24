// =========================================
// ID: PHYLOFACE_LIB_ACTIVE_TRIPLET
// VERSION: v1.0
// =========================================
// Estado COMPARTIDO y PERSISTENTE de la "tripleta activa" — el puente vivo
// entre el Árbol genealógico y el Comparador MVP (3 slots). Reemplaza al
// handoff de un solo uso (prefill con TTL 60s) por un vínculo persistente que
// ambas vistas leen y escriben.
//
// Vive en localStorage (no se consume al leerse, sin TTL): así sobrevive a los
// cambios de pestaña (que desmontan el componente) y a refreshes. Cada vista lo
// lee al montar y lo refleja; cada cambio relevante lo reescribe.
//
// Flujo:
//   Árbol → Comparador: al seleccionar 2-3 nodos con foto (ctrl+click), el
//     árbol escribe la tripleta (mapeo nodos→slots con `assignTripletSlots`).
//     El Comparador la lee al montar y precarga los slots.
//   Comparador → Árbol: al cambiar la foto de un slot vinculado, el Comparador
//     resuelve/crea el nodo en el árbol (aditivo, nunca pisa fotos existentes)
//     y reescribe la tripleta. El Árbol, al volver a montarse, re-hidrata su
//     selección desde la tripleta y recarga personas (los nodos nuevos
//     aparecen solos).
//
// `personId` es opcional sólo por robustez de parsing; en la práctica siempre
// está (todo slot mapea a un nodo del árbol activo).

import type { Person, PersonId } from './genealogy';

export type SlotKey = 'left' | 'child' | 'right';

export interface TripletSlot {
  slot: SlotKey;
  personId?: PersonId;   // nodo del árbol al que mapea
  sha256: string;        // FK a PhotoRecord.sha256 (la foto)
  role?: string;         // sólo left/right (Padre/Madre/…)
}

export interface ActiveTriplet {
  v: 1;
  treeId: string;        // árbol al que pertenecen estos nodos
  slots: TripletSlot[];
  updatedAt: number;
}

const KEY = 'phyloface-active-triplet';

// Evento que se dispara al escribir/limpiar. Permite reaccionar en vivo si en
// el futuro ambas vistas coexisten; hoy cada una lee al montar, pero el evento
// no molesta y deja la puerta abierta.
export const TRIPLET_EVENT = 'phyloface-triplet-changed';

export function readActiveTriplet(): ActiveTriplet | null {
  const raw = localStorage.getItem(KEY);
  if (!raw) return null;
  try {
    const t = JSON.parse(raw) as ActiveTriplet;
    if (t.v !== 1 || typeof t.treeId !== 'string' || !Array.isArray(t.slots)) return null;
    return t;
  } catch {
    return null;
  }
}

export function writeActiveTriplet(treeId: string, slots: TripletSlot[]): void {
  const t: ActiveTriplet = { v: 1, treeId, slots, updatedAt: Date.now() };
  localStorage.setItem(KEY, JSON.stringify(t));
  window.dispatchEvent(new CustomEvent(TRIPLET_EVENT));
}

export function clearActiveTriplet(): void {
  localStorage.removeItem(KEY);
  window.dispatchEvent(new CustomEvent(TRIPLET_EVENT));
}

// Mapeo puro: lista ordenada de personas seleccionadas → slots de la tripleta.
// Reglas:
//   - Se consideran sólo personas con foto (sin foto no se puede comparar), tope 3.
//   - Si alguna es hija de las otras (ambos padres en el set, o al menos uno),
//     esa va al slot central (Hijo/a) y las demás a left/right con su rol real
//     (Padre/Madre) inferido del pedigree.
//   - Si no hay relación derivable, la primera seleccionada va al centro y el
//     resto a left/right (roles los pone el Comparador por default).
export function assignTripletSlots(sel: Person[]): TripletSlot[] {
  const photod = sel.filter((p) => p.photoSha256).slice(0, 3);
  if (photod.length === 0) return [];

  const idSet = new Set(photod.map((p) => p.id));
  const hasBothParents = (p: Person) =>
    p.fatherId != null && p.motherId != null && idSet.has(p.fatherId) && idSet.has(p.motherId);
  const hasAnyParent = (p: Person) =>
    (p.fatherId != null && idSet.has(p.fatherId)) || (p.motherId != null && idSet.has(p.motherId));

  let childIdx = photod.findIndex(hasBothParents);
  if (childIdx === -1) childIdx = photod.findIndex(hasAnyParent);

  const child = childIdx !== -1 ? photod[childIdx] : photod[0];
  const others = photod.filter((p) => p.id !== child.id);

  const slots: TripletSlot[] = [
    { slot: 'child', personId: child.id, sha256: child.photoSha256! },
  ];
  const sideKeys: SlotKey[] = ['left', 'right'];
  others.forEach((p, i) => {
    let role: string | undefined;
    if (child.fatherId === p.id) role = 'Padre';
    else if (child.motherId === p.id) role = 'Madre';
    slots.push({ slot: sideKeys[i] ?? 'right', personId: p.id, sha256: p.photoSha256!, role });
  });
  return slots;
}
