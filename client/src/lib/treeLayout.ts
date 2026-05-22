// =========================================
// ID: PHYLOFACE_LIB_TREELAYOUT
// VERSION: v1.0
// =========================================
// Paso 3 del plan Tarea #26 (Track 2b — comparador con árbol genealógico).
// Función pura que asigna a cada Person su `generation` y su `indexInGen`,
// derivado del grafo de parentesco. Base para el render SVG del paso 4.
//
// Por qué la generación no vive en `Person` (memoria `lib/genealogy.ts`):
// state derivado del grafo, mantenerlo en sync con los parentIds requiere
// invalidar en cada edit. Mucho más simple recomputarlo en el render.
//
// Definición operativa:
//   - `generation` = profundidad desde las raíces, top-down (0 = raíces, +1
//     por cada ancestro). Para Person p:
//         gen(p) = max(gen(father), gen(mother)) + 1
//       donde gen de un parent ausente o dangling cuenta como -1, lo cual
//       hace que personas sin padres válidos caigan en gen 0.
//   - `indexInGen` = posición ordinal dentro de la generación, determinístico:
//     `createdAt` ASC, tiebreak por `id` ASC. NO minimiza cruces de líneas
//     (eso quedaría para iteraciones posteriores si el grafo crece y se
//     vuelve ilegible — para MVP, orden estable es suficiente).
//
// Tolerancia a datos sucios:
//   - **Refs colgadas**: si `fatherId`/`motherId` apunta a un id que no está
//     en el set (la persona padre fue borrada — ver `deletePerson` en
//     `lib/genealogy.ts` que NO toca refs colgadas), ese parent se trata como
//     null. La persona promueve generación por el otro padre si existe; si
//     ambos son danglings, cae en gen 0.
//   - **Ciclos**: `wouldCreateCycle` en `lib/genealogy.ts` los previene en la
//     UI, pero datos importados desde JSON externo podrían colarse. Defensa:
//     guard de stack de visitados durante la recursión. Si se detecta ciclo,
//     se tratan los nodos del ciclo como gen 0 (mejor degradar a un layout
//     incorrecto que loopear infinitamente). No es la mejor UX pero es
//     improbable y el comportamiento defensivo es mínimo.
//
// Sin acoplamiento al DOM ni a IDB — testeable directo con un fixture de
// Person[]. El render SVG (paso 4) consume `TreeLayout` y decide coordenadas.
//
// API esperada por el render del paso 4 (sugerido, a confirmar al implementar):
//   - `computeTreeLayout(persons)` → Map<PersonId, LayoutNode>: lookup por id.
//   - Si el render quiere iterar generación a generación en orden, puede
//     hacer el reagrupamiento ahí mismo. Si emerge como repetido, se agrega
//     un helper `bucketByGeneration(persons, layout)` en una v1.1.

import type { Person, PersonId } from './genealogy';

// -----------------------------------------
// Tipos
// -----------------------------------------

export interface LayoutNode {
  /** 0 = raíz; +1 por cada ancestro. */
  generation: number;
  /** Posición ordinal dentro de la generación (0-based). */
  indexInGen: number;
}

export type TreeLayout = Map<PersonId, LayoutNode>;

// -----------------------------------------
// Internal: cómputo de generaciones por DFS con memoización
// -----------------------------------------

/**
 * Devuelve un Map<id, generation> con la generación asignada a cada Person.
 * Implementación: DFS recursiva con memo + stack de visitando para detectar
 * ciclos. O(n) en el caso típico (memo evita re-visitar).
 *
 * Decisión: padres ausentes o danglings cuentan como gen = -1, así
 * `max(parents) + 1` cae naturalmente en 0 cuando ningún padre es válido.
 */
function computeGenerations(persons: Person[]): Map<PersonId, number> {
  const byId = new Map<PersonId, Person>();
  for (const p of persons) byId.set(p.id, p);

  const memo = new Map<PersonId, number>();
  // Si un nodo aparece en `visiting` durante la recursión, hay ciclo —
  // ver nota de tolerancia a datos sucios en la cabecera del archivo.
  const visiting = new Set<PersonId>();

  function genOf(id: PersonId): number {
    const cached = memo.get(id);
    if (cached !== undefined) return cached;

    if (visiting.has(id)) {
      // Ciclo detectado. Degradar a gen 0 para el nodo de cierre; los nodos
      // ancestros del ciclo van a heredar este valor por el max(), también 0.
      memo.set(id, 0);
      return 0;
    }

    const p = byId.get(id);
    if (!p) {
      // Dangling ref: el caller (genOf de un padre) interpreta -1 como
      // "no contribuye" y queda con el otro padre o cae a 0.
      return -1;
    }

    visiting.add(id);

    let maxParentGen = -1;
    for (const parentId of [p.fatherId, p.motherId]) {
      if (parentId === null) continue;
      const g = genOf(parentId);
      if (g > maxParentGen) maxParentGen = g;
    }

    visiting.delete(id);

    const gen = maxParentGen + 1; // -1 + 1 = 0 si no hay padres válidos
    memo.set(id, gen);
    return gen;
  }

  for (const p of persons) genOf(p.id);
  return memo;
}

// -----------------------------------------
// Internal: orden dentro de cada generación
// -----------------------------------------

/**
 * Asigna `indexInGen` (0-based) a cada Person, agrupando por generación y
 * ordenando por (`createdAt` ASC, tiebreak `id` ASC).
 *
 * Determinístico: dos invocaciones sobre el mismo input devuelven el mismo
 * orden. Necesario para que el render SVG no salte de posición entre re-renders.
 */
function indicesByGeneration(
  persons: Person[],
  gens: Map<PersonId, number>,
): Map<PersonId, number> {
  const byGen = new Map<number, Person[]>();
  for (const p of persons) {
    const g = gens.get(p.id) ?? 0;
    const bucket = byGen.get(g);
    if (bucket) bucket.push(p);
    else byGen.set(g, [p]);
  }

  const indices = new Map<PersonId, number>();
  for (const group of byGen.values()) {
    group.sort((a, b) => {
      if (a.createdAt !== b.createdAt) return a.createdAt - b.createdAt;
      // tiebreak por id (lexicográfico ASC). UUID v4 son aleatorios pero
      // estables — la comparación es consistente entre invocaciones.
      if (a.id < b.id) return -1;
      if (a.id > b.id) return 1;
      return 0;
    });
    group.forEach((p, i) => indices.set(p.id, i));
  }
  return indices;
}

// -----------------------------------------
// Public API
// -----------------------------------------

/**
 * Asigna a cada Person su LayoutNode (generation + indexInGen). El caller
 * provee el set de Persons del árbol activo (ya filtrado por treeId).
 *
 * Devuelve un Map vacío si `persons` está vacío. No lanza excepciones por
 * datos sucios — degrada (ver nota de tolerancia en la cabecera).
 */
export function computeTreeLayout(persons: Person[]): TreeLayout {
  const gens = computeGenerations(persons);
  const indices = indicesByGeneration(persons, gens);

  const layout: TreeLayout = new Map();
  for (const p of persons) {
    layout.set(p.id, {
      generation: gens.get(p.id) ?? 0,
      indexInGen: indices.get(p.id) ?? 0,
    });
  }
  return layout;
}
