// =========================================
// ID: PHYLOFACE_GENEALOGY_TREE
// VERSION: v3.1
// =========================================
// Cambio v3.0 → v3.1 (Tarea #26 paso 6 del plan — export/import del árbol):
// - Dos botones en la toolbar de árbol: **⬇ Exportar** (baja el árbol activo a
//   un JSON autocontenido — Tree+Persons+Photos en base64+Comparisons) y
//   **⬆ Importar** (file picker → crea un árbol NUEVO con ids remapeados).
//   Toda la lógica de serialización/rehidratación vive en `lib/treeExport.ts`;
//   acá sólo va el cableado de UI (handlers + input file oculto + feedback).
// - Estado `info` nuevo (caja verde) para feedback de éxito del import; el
//   `error` existente cubre los fallos. Tras importar se recarga la lista de
//   árboles y se selecciona el recién creado.
//
// Cambio v2.0 → v3.0 (Tarea #26 paso 5 del plan — comparación on-demand):
// - **Toggle "Modo comparación"** en una toolbar nueva arriba del SVG.
//   Mientras está ON, los clicks sobre nodos NO seleccionan para detalle:
//   primer click setea P1 (azul), segundo click setea P2 (verde) y dispara
//   el cosine; un tercer click resetea a "ese nodo como P1 nuevo".
// - **Panel "Comparación"** al lado del panel detalle (flex row): muestra
//   foto P1, foto P2, cosine grande, botón ↻ recompute (fuerza re-cálculo
//   sin usar el embedding cacheado), historial filtrado por treeId.
// - **Cómputo de embeddings lazy + cacheado** vía `ensureEmbedding(sha256)`:
//   si `PhotoRecord.embedding` no es null, se reusa; sino se carga la foto
//   en un HTMLImage + ImageData, se corre el pipeline browser-only de
//   `lib/pipeline.ts` (Face Mesh → align → ArcFace) y se cachea con
//   `setPhotoEmbedding`. Init de FaceLandmarker + ONNX session es lazy y
//   compartido entre todas las comparaciones del mismo mount.
// - **Cleanup obligatorio** (lección Tarea #27, episodio
//   [[react-cleanup-gpu-wasm-resources-or-leak]]): `useEffect([])` con
//   return que llama `landmarkerRef.current?.close()` +
//   `void sessionRef.current?.release().catch(...)` al desmontar. Las refs
//   se setean adentro del init lazy para que el cleanup vea la instancia
//   incluso si init terminó después de un unmount in-flight (StrictMode dev
//   double-mount).
// - **Persistencia**: cada cómputo nuevo crea una `Comparison` que se guarda
//   en IDB (store `comparisons`, índice by-tree). El historial se recarga
//   al cambiar de árbol. Las comparisons cuyos `p{1,2}Sha256` ya no
//   coinciden con la `photoSha256` actual de la persona se marcan como
//   "stale" en el historial (el cómputo es válido para esas fotos, pero
//   las fotos asociadas a esas personas cambiaron desde entonces).
//
// Cambio v1.0 → v2.0 (Tarea #26 paso 4 del plan):
// - Reemplazada la tabla de personas por un **render SVG pedigree** top-down,
//   con cajas por persona dispuestas por generación y líneas de parentesco.
// - Layout calculado por `computeTreeLayout` (lib/treeLayout.ts, paso 3): la
//   generación de cada persona es `max(gen_padre, gen_madre) + 1`; el orden
//   intra-generación es estable por `createdAt` ASC.
// - **Selección por click sobre nodo** → panel inferior con detalle: foto
//   grande, dropdowns padre/madre, botón borrar. Esto reemplaza la edición
//   inline por fila que tenía la tabla v1.0.
// - **Drag-and-drop foto sobre nodo** (filesystem → nodo SVG): el nodo
//   resalta durante drag-over y al soltar carga la imagen a esa persona.
//   Click sobre el área de foto del nodo también abre el file picker como
//   alternativa.
//
// Lo que se conservó v1.0 → v2.0:
//   - Toolbars de árbol (selector + crear/borrar) y de persona (+ Persona).
//   - Multi-tree en el store con una sola activa en UI; LAST_TREE_KEY en
//     localStorage para persistir la selección entre refreshes.
//   - Object URLs de fotos cacheados por sha256 y revocados en cleanup del
//     effect (cumple [[react-cleanup-gpu-wasm-resources-or-leak]] para
//     recursos del DOM que no son GPU pero igualmente externos).
//   - Validación de aciclicidad antes de asignar padres.
//   - Manejo de refs colgadas (parentId apunta a persona borrada): el dropdown
//     del panel muestra "(borrado: XXXXXX...)" en rojo; el render SVG las
//     ignora silenciosamente (la persona huérfana sube a gen 0 si pierde
//     todos los padres válidos).
//
// Diseño del SVG (geometría simple, no minimiza cruces — MVP):
//   - Cada nodo es un `<g>` con un rectángulo + foto + nombre debajo.
//   - Coordenadas: y = generation * (BOX_H + GEN_GAP); x = centrado dentro de
//     la generación, posición ordinal del nodo en su gen.
//   - viewBox dinámico según contenido + padding fijo.
//   - Líneas de parentesco: del centro-bottom del padre/madre al centro-top
//     del hijo, una por relación (puede haber 2 por persona si tiene ambos
//     padres). Sin curvas; rectos simples.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { FaceLandmarker } from '@mediapipe/tasks-vision';
import type { InferenceSession } from 'onnxruntime-web';
import {
  newComparison,
  newPerson,
  newTree,
  wouldCreateCycle,
  type Person,
  type PersonId,
  type Sha256Hex,
  type Tree,
} from './lib/genealogy';
import {
  deletePerson,
  deleteTree,
  getPhoto,
  listPersons,
  listTrees,
  putPerson,
  putPhoto,
  putTree,
  saveComparison,
  setPhotoEmbedding,
} from './lib/treeStore';
import { computeTreeLayout } from './lib/treeLayout';
import { downloadTreeExport, importTreeFromJson } from './lib/treeExport';
import {
  computeEmbedding,
  cosineSimilarity,
  initFaceLandmarker,
  initOnnxSession,
  loadImage,
} from './lib/pipeline';
import TripletModal from './TripletModal';

const LAST_TREE_KEY = 'phyloface-genealogy-last-tree';

// -----------------------------------------
// Constantes geométricas del SVG (px del viewBox; el browser escala con CSS).
// -----------------------------------------
const BOX_W = 120;
const BOX_H = 140;
const PHOTO_SIZE = 80;
const IDX_GAP = 40;   // separación horizontal entre nodos en la misma gen
const GEN_GAP = 60;   // separación vertical entre generaciones
const PADDING = 30;

export default function GenealogyTree() {
  const [trees, setTrees] = useState<Tree[]>([]);
  const [selectedTreeId, setSelectedTreeId] = useState<string | null>(null);
  const [persons, setPersons] = useState<Person[]>([]);
  const [photoUrls, setPhotoUrls] = useState<Map<string, string>>(new Map());
  const [newPersonName, setNewPersonName] = useState('');
  const [newTreeName, setNewTreeName] = useState('');
  const [error, setError] = useState<string | null>(null);
  // Feedback de éxito (caja verde) — hoy lo usa sólo el import. Efímero: se
  // limpia al próximo error o al cerrarlo.
  const [info, setInfo] = useState<string | null>(null);
  // Input file oculto para ⬆ Importar (vive fuera del SVG, en el árbol de la
  // toolbar). Se dispara con .click() desde el botón.
  const importInputRef = useRef<HTMLInputElement>(null);
  const [selectedPersonId, setSelectedPersonId] = useState<PersonId | null>(null);
  // Persona-target del drag-over actual (para highlight visual).
  const [dragOverPersonId, setDragOverPersonId] = useState<PersonId | null>(null);

  // ----- Selección múltiple para comparar (paso 5, iter multi-select) -----
  // Lista ordenada de PersonIds seleccionados via ctrl/cmd+click. El orden
  // refleja el orden de selección y se usa para numerar los badges (1, 2, …).
  // Toggle por click: si el id ya está en la lista, se saca (y todas las
  // comparaciones que lo involucran); si no, se agrega y se disparan los
  // cómputos de los pares nuevos. Para N seleccionados hay N*(N-1)/2 pares.
  // Click sin modifier sigue abriendo el panel detalle.
  const [selectedForCompare, setSelectedForCompare] = useState<PersonId[]>([]);
  // Cosine por par (key canónica `pairKey(a, b)`). Sólo entries para pares
  // *activos* (ambos extremos están en selectedForCompare); al deseleccionar
  // un nodo, sus entries se podan. Si se vuelve a seleccionar, se recomputa
  // (rápido si los embeddings ya están cacheados en PhotoRecord).
  const [cosineByPair, setCosineByPair] = useState<Map<string, number>>(() => new Map());
  const [isComputing, setIsComputing] = useState(false);
  // Modal de tripleta abierto. Contiene los ids del par origen + el cosine
  // cacheado del par (que ya conocemos al click). null = cerrado.
  const [tripletModalState, setTripletModalState] = useState<{
    aId: PersonId; bId: PersonId; cosine: number;
  } | null>(null);

  // Recursos pesados (GPU/WASM) compartidos entre todas las comparaciones del
  // mismo mount. Init lazy en `ensureEmbedding`. Cleanup obligatorio al
  // desmontar — lección [[react-cleanup-gpu-wasm-resources-or-leak]].
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const sessionRef = useRef<InferenceSession | null>(null);

  // -------------------------------------
  // Carga inicial: trees + selección persistida.
  // -------------------------------------
  useEffect(() => {
    void (async () => {
      try {
        const all = await listTrees();
        setTrees(all);
        const last = localStorage.getItem(LAST_TREE_KEY);
        const initial =
          last && all.some((t) => t.id === last) ? last : all[0]?.id ?? null;
        setSelectedTreeId(initial);
      } catch (e) {
        setError(`Cargando árboles: ${(e as Error).message}`);
      }
    })();
  }, []);

  // -------------------------------------
  // Persistir selección activa.
  // -------------------------------------
  useEffect(() => {
    if (selectedTreeId) localStorage.setItem(LAST_TREE_KEY, selectedTreeId);
  }, [selectedTreeId]);

  // -------------------------------------
  // Cargar personas del árbol activo.
  // -------------------------------------
  const reloadPersons = useCallback(async () => {
    if (!selectedTreeId) {
      setPersons([]);
      return;
    }
    try {
      const list = await listPersons(selectedTreeId);
      setPersons(list);
    } catch (e) {
      setError(`Cargando personas: ${(e as Error).message}`);
    }
  }, [selectedTreeId]);

  useEffect(() => {
    void reloadPersons();
  }, [reloadPersons]);

  // Nota: no hace falta deseleccionar persona al cambiar de árbol. El
  // `selectedPersonId` puede quedar con un valor stale pero `selectedPerson`
  // se computa con `find` y retorna null si el id no existe en el árbol
  // actual, así que el panel no se muestra. Colisión de IDs entre árboles
  // es ~0 por UUID v4.

  // -------------------------------------
  // Object URLs de fotos: rehacer cuando cambia la lista de personas;
  // revocar las viejas en el cleanup. Indexamos por sha256 para no recrear
  // URLs si la misma foto la usan dos personas.
  // -------------------------------------
  useEffect(() => {
    let cancelled = false;
    const created: string[] = [];
    void (async () => {
      const next = new Map<string, string>();
      const seen = new Set<string>();
      for (const p of persons) {
        if (!p.photoSha256 || seen.has(p.photoSha256)) continue;
        seen.add(p.photoSha256);
        const rec = await getPhoto(p.photoSha256);
        if (!rec) continue;
        const url = URL.createObjectURL(rec.blob);
        created.push(url);
        next.set(p.photoSha256, url);
      }
      if (!cancelled) setPhotoUrls(next);
    })();
    return () => {
      cancelled = true;
      for (const url of created) URL.revokeObjectURL(url);
    };
  }, [persons]);

  // -------------------------------------
  // Cleanup recursos GPU/WASM al desmontar (Tarea #27 / paso 5).
  // Las refs se setean dentro de `ensureEmbedding` cuando el init lazy
  // termina; este cleanup las libera al unmount aunque el init haya corrido
  // mucho después del mount inicial.
  // -------------------------------------
  useEffect(() => {
    return () => {
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
      void sessionRef.current?.release().catch(() => {});
      sessionRef.current = null;
    };
  }, []);

  // Helper para resetear la selección viva sin tocar el historial persistido
  // en IDB. Lo invocan: cambio de árbol, botón "↺ limpiar selección" del
  // hint bar, errores fatales de cómputo.
  const resetComparisonSelection = useCallback(() => {
    setSelectedForCompare([]);
    setCosineByPair(new Map());
  }, []);

  // Key canónica para un par (independiente del orden). Usamos lexicográfico
  // sobre los UUIDs porque comparar strings es barato y `selectedForCompare`
  // siempre tiene ≤ pocas decenas de entradas.
  const pairKey = (a: PersonId, b: PersonId): string =>
    a < b ? `${a}|${b}` : `${b}|${a}`;

  // -------------------------------------
  // ensureEmbedding(sha256): devuelve el embedding ArcFace 512-d para una
  // foto, computándolo si no estaba cacheado en PhotoRecord.embedding.
  //
  // Init lazy de FaceLandmarker + ONNX session: el costo es alto (~segundos
  // por descarga de modelos), por eso se cachea en refs y se reusa entre
  // comparaciones del mismo mount. La asignación a la ref ocurre AL TERMINAR
  // el init, así que si el componente desmonta in-flight, el cleanup del
  // effect [] todavía puede liberar correctamente cuando termine — el
  // problema clásico de StrictMode dev se evita porque el effect cleanup
  // mira `.current` al unmount.
  //
  // Throws si no encuentra la foto en el store, si el blob no decodea, o si
  // el pipeline no detecta cara.
  // -------------------------------------
  const ensureEmbedding = useCallback(async (sha256: Sha256Hex, force = false): Promise<Float32Array> => {
    const rec = await getPhoto(sha256);
    if (!rec) throw new Error(`Foto ${sha256.slice(0, 8)}… no está en el store`);
    if (!force && rec.embedding) return rec.embedding;

    if (!landmarkerRef.current) {
      landmarkerRef.current = await initFaceLandmarker();
    }
    if (!sessionRef.current) {
      sessionRef.current = await initOnnxSession();
    }

    const url = URL.createObjectURL(rec.blob);
    try {
      const { img, imageData } = await loadImage(url);
      const out = await computeEmbedding(img, imageData, landmarkerRef.current, sessionRef.current);
      await setPhotoEmbedding(sha256, out.embedding);
      return out.embedding;
    } finally {
      URL.revokeObjectURL(url);
    }
  }, []);

  // -------------------------------------
  // computeMissingPairsFor(sel): para una selección actual `sel`, computa
  // todos los pares cuya cosine aún no está en `cosineByPair` y los agrega.
  // Persistente: cada cómputo guarda una `Comparison` en IDB (silencioso —
  // no hay UI de historial todavía pero la data queda).
  //
  // Si falta foto en cualquier nodo del par, ese par se saltea con un setError
  // amistoso (el resto sí se computa).
  // -------------------------------------
  const computeMissingPairsFor = useCallback(async (sel: PersonId[]): Promise<void> => {
    if (!selectedTreeId || sel.length < 2) return;
    // Pares ordenados (i<j en sel) sin entry en cosineByPair.
    const pending: Array<[Person, Person]> = [];
    for (let i = 0; i < sel.length; i++) {
      for (let j = i + 1; j < sel.length; j++) {
        const key = pairKey(sel[i], sel[j]);
        if (cosineByPair.has(key)) continue;
        const a = persons.find((p) => p.id === sel[i]);
        const b = persons.find((p) => p.id === sel[j]);
        if (a && b) pending.push([a, b]);
      }
    }
    if (pending.length === 0) return;
    setIsComputing(true);
    setError(null);
    try {
      for (const [a, b] of pending) {
        if (!a.photoSha256 || !b.photoSha256) {
          const missing = !a.photoSha256 ? a.name : b.name;
          setError(`"${missing}" no tiene foto cargada — par salteado.`);
          continue;
        }
        try {
          const eA = await ensureEmbedding(a.photoSha256);
          const eB = await ensureEmbedding(b.photoSha256);
          const cos = cosineSimilarity(eA, eB);
          const key = pairKey(a.id, b.id);
          // Usar updater funcional: durante un loop largo cosineByPair
          // capturada por closure queda desactualizada.
          setCosineByPair((prev) => {
            const next = new Map(prev);
            next.set(key, cos);
            return next;
          });
          await saveComparison(newComparison(
            selectedTreeId, a.id, b.id, a.photoSha256, b.photoSha256, cos,
          ));
        } catch (e) {
          setError(`Error comparando "${a.name}" ↔ "${b.name}": ${(e as Error).message}`);
        }
      }
    } finally {
      setIsComputing(false);
    }
  }, [selectedTreeId, persons, cosineByPair, ensureEmbedding]);

  // Click sobre nodo. `withModifier=true` (ctrl/cmd+click) → toggle en
  // `selectedForCompare`. `withModifier=false` (click normal) → abre/cambia
  // panel detalle, sin afectar la selección activa.
  //
  // Toggle:
  //   - si el id ya está → quitarlo + podar entries de cosineByPair que lo
  //     involucran.
  //   - si no → agregarlo al final + disparar `computeMissingPairsFor` para
  //     calcular las cosines de los nuevos pares con los previamente
  //     seleccionados.
  const handleNodeClick = useCallback((id: PersonId, withModifier: boolean) => {
    if (!withModifier) {
      setSelectedPersonId(id);
      return;
    }
    if (selectedForCompare.includes(id)) {
      const next = selectedForCompare.filter((x) => x !== id);
      setSelectedForCompare(next);
      // Podar cosines de pares que involucran al id removido.
      setCosineByPair((prev) => {
        const updated = new Map<string, number>();
        for (const [k, v] of prev) {
          const [a, b] = k.split('|');
          if (a !== id && b !== id) updated.set(k, v);
        }
        return updated;
      });
      return;
    }
    const next = [...selectedForCompare, id];
    setSelectedForCompare(next);
    void computeMissingPairsFor(next);
  }, [selectedForCompare, computeMissingPairsFor]);

  const handleSelectTree = useCallback((id: string | null) => {
    setSelectedTreeId(id);
    // Cambio de árbol: limpiar selección de comparación viva y selección de
    // detalle. El historial persistido se recarga vía el effect [selectedTreeId].
    resetComparisonSelection();
    setSelectedPersonId(null);
  }, [resetComparisonSelection]);

  // -------------------------------------
  // Handlers — los mismos que v1.0; sólo el render cambia.
  // -------------------------------------

  const handleCreateTree = async () => {
    const name = newTreeName.trim();
    if (!name) {
      setError('El árbol necesita un nombre');
      return;
    }
    try {
      const tree = newTree(name);
      await putTree(tree);
      setTrees((prev) => [...prev, tree]);
      handleSelectTree(tree.id);
      setNewTreeName('');
      setError(null);
    } catch (e) {
      setError(`Creando árbol: ${(e as Error).message}`);
    }
  };

  const handleDeleteTree = async () => {
    if (!selectedTreeId) return;
    const tree = trees.find((t) => t.id === selectedTreeId);
    if (!tree) return;
    if (!confirm(`¿Borrar el árbol "${tree.name}" y todas sus personas? Las fotos quedan en el store (compartibles).`)) return;
    try {
      await deleteTree(selectedTreeId);
      const remaining = trees.filter((t) => t.id !== selectedTreeId);
      setTrees(remaining);
      handleSelectTree(remaining[0]?.id ?? null);
      setError(null);
    } catch (e) {
      setError(`Borrando árbol: ${(e as Error).message}`);
    }
  };

  // ⬇ Exportar: baja el árbol activo a un JSON autocontenido.
  const handleExport = async () => {
    if (!selectedTreeId) return;
    const tree = trees.find((t) => t.id === selectedTreeId);
    try {
      await downloadTreeExport(selectedTreeId, tree?.name ?? 'arbol');
      setError(null);
      setInfo(`Árbol "${tree?.name ?? ''}" exportado.`);
    } catch (e) {
      setInfo(null);
      setError(`Exportando: ${(e as Error).message}`);
    }
  };

  // ⬆ Importar: lee el JSON elegido y crea un árbol NUEVO (ids remapeados).
  // Tras importar recarga la lista y selecciona el árbol recién creado.
  const handleImportFile = async (file: File) => {
    try {
      const text = await file.text();
      const result = await importTreeFromJson(text);
      const all = await listTrees();
      setTrees(all);
      handleSelectTree(result.treeId);
      setError(null);
      const parts = [
        `${result.personCount} persona${result.personCount === 1 ? '' : 's'}`,
        `${result.photosInserted} foto${result.photosInserted === 1 ? '' : 's'}${result.photosDeduped > 0 ? ` (+${result.photosDeduped} ya existente${result.photosDeduped === 1 ? '' : 's'})` : ''}`,
        `${result.comparisonCount} comparación${result.comparisonCount === 1 ? '' : 'es'}`,
      ];
      setInfo(
        `Importado «${result.treeName}»: ${parts.join(', ')}.` +
          (result.embeddingsDropped
            ? ' Embeddings descartados por versión de modelo distinta — se recomputarán al comparar.'
            : ''),
      );
    } catch (e) {
      setInfo(null);
      setError(`Importando: ${(e as Error).message}`);
    }
  };

  const handleCreatePerson = async () => {
    if (!selectedTreeId) {
      setError('Primero creá o seleccioná un árbol');
      return;
    }
    const name = newPersonName.trim();
    if (!name) {
      setError('La persona necesita un nombre');
      return;
    }
    try {
      const person = newPerson(selectedTreeId, name);
      await putPerson(person);
      setPersons((prev) => [...prev, person]);
      setNewPersonName('');
      setError(null);
    } catch (e) {
      setError(`Creando persona: ${(e as Error).message}`);
    }
  };

  const handleDeletePerson = async (id: PersonId) => {
    const p = persons.find((pp) => pp.id === id);
    if (!p) return;
    if (!confirm(`¿Borrar "${p.name}"? Si era padre/madre de otros, esas referencias quedarán colgadas.`)) return;
    try {
      await deletePerson(id);
      setPersons((prev) => prev.filter((pp) => pp.id !== id));
      if (selectedPersonId === id) setSelectedPersonId(null);
      setError(null);
    } catch (e) {
      setError(`Borrando persona: ${(e as Error).message}`);
    }
  };

  const handleSetParent = async (
    personId: PersonId,
    role: 'father' | 'mother',
    parentId: PersonId | null,
  ) => {
    const person = persons.find((p) => p.id === personId);
    if (!person) return;

    if (parentId !== null) {
      const check = wouldCreateCycle(persons, personId, parentId);
      if (!check.ok) {
        setError(
          `Asignación rechazada: crearía un ciclo (${check.cycle.length} pasos). Una persona no puede ser su propio ancestro.`,
        );
        return;
      }
    }

    const updated: Person = {
      ...person,
      fatherId: role === 'father' ? parentId : person.fatherId,
      motherId: role === 'mother' ? parentId : person.motherId,
    };
    try {
      await putPerson(updated);
      setPersons((prev) => prev.map((p) => (p.id === personId ? updated : p)));
      setError(null);
    } catch (e) {
      setError(`Actualizando padres: ${(e as Error).message}`);
    }
  };

  const handleUploadPhoto = async (personId: PersonId, file: File) => {
    const person = persons.find((p) => p.id === personId);
    if (!person) return;
    try {
      const rec = await putPhoto(file);
      const updated: Person = { ...person, photoSha256: rec.sha256 };
      await putPerson(updated);
      setPersons((prev) => prev.map((p) => (p.id === personId ? updated : p)));
      setError(null);
    } catch (e) {
      setError(`Subiendo foto: ${(e as Error).message}`);
    }
  };

  // -------------------------------------
  // Layout SVG: posiciones absolutas por nodo a partir de computeTreeLayout.
  // -------------------------------------

  const personsById = useMemo(() => {
    const m = new Map<PersonId, Person>();
    for (const p of persons) m.set(p.id, p);
    return m;
  }, [persons]);

  const positions = useMemo(() => {
    const layout = computeTreeLayout(persons);

    // Conteo por generación para poder centrar cada fila.
    const countByGen = new Map<number, number>();
    for (const node of layout.values()) {
      countByGen.set(node.generation, (countByGen.get(node.generation) ?? 0) + 1);
    }
    const maxCount = countByGen.size > 0 ? Math.max(...countByGen.values()) : 0;
    const totalWidth = maxCount * BOX_W + Math.max(0, maxCount - 1) * IDX_GAP;

    const pos = new Map<PersonId, { x: number; y: number }>();
    for (const [id, node] of layout) {
      const nInGen = countByGen.get(node.generation) ?? 1;
      const rowWidth = nInGen * BOX_W + Math.max(0, nInGen - 1) * IDX_GAP;
      const rowOffset = (totalWidth - rowWidth) / 2;
      const x = PADDING + rowOffset + node.indexInGen * (BOX_W + IDX_GAP);
      const y = PADDING + node.generation * (BOX_H + GEN_GAP);
      pos.set(id, { x, y });
    }

    const maxGen = countByGen.size > 0 ? Math.max(...countByGen.keys()) : 0;
    const viewW = totalWidth + PADDING * 2;
    const viewH = (maxGen + 1) * BOX_H + maxGen * GEN_GAP + PADDING * 2;
    return { pos, viewW, viewH };
  }, [persons]);

  const selectedPerson = selectedPersonId
    ? persons.find((p) => p.id === selectedPersonId) ?? null
    : null;

  // -------------------------------------
  // Render
  // -------------------------------------

  return (
    <div style={{ padding: 20, fontFamily: 'sans-serif', fontSize: 14 }}>
      <h2 style={{ marginTop: 0 }}>Árbol genealógico (Track 2b — MVP paso 4)</h2>

      {/* Toolbar de árbol */}
      <div style={toolbarStyle}>
        <label>
          Árbol activo:&nbsp;
          <select
            value={selectedTreeId ?? ''}
            onChange={(e) => handleSelectTree(e.target.value || null)}
            disabled={trees.length === 0}
            style={{ minWidth: 200 }}
          >
            {trees.length === 0 && <option value="">(sin árboles)</option>}
            {trees.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </select>
        </label>
        <button onClick={handleDeleteTree} disabled={!selectedTreeId}>
          ✕ Borrar árbol
        </button>
        <button onClick={handleExport} disabled={!selectedTreeId} title="Bajar el árbol activo a un archivo JSON">
          ⬇ Exportar
        </button>
        <button onClick={() => importInputRef.current?.click()} title="Importar un árbol desde un JSON (crea uno nuevo)">
          ⬆ Importar
        </button>
        <input
          ref={importInputRef}
          type="file"
          accept="application/json,.json"
          style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void handleImportFile(f);
            e.target.value = ''; // permite re-importar el mismo archivo
          }}
        />
        <span style={{ flex: 1 }} />
        <input
          type="text"
          placeholder="Nombre del árbol nuevo"
          value={newTreeName}
          onChange={(e) => setNewTreeName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && void handleCreateTree()}
          style={{ width: 220 }}
        />
        <button onClick={handleCreateTree}>+ Árbol</button>
      </div>

      {/* Mensaje de error */}
      {error && (
        <div style={errorStyle}>
          {error} <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* Mensaje de éxito (import/export) */}
      {info && (
        <div style={infoStyle}>
          {info} <button onClick={() => setInfo(null)}>×</button>
        </div>
      )}

      {/* Toolbar de persona */}
      <div style={toolbarStyle}>
        <input
          type="text"
          placeholder="Nombre de la persona nueva"
          value={newPersonName}
          onChange={(e) => setNewPersonName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && void handleCreatePerson()}
          disabled={!selectedTreeId}
          style={{ flex: 1 }}
        />
        <button onClick={handleCreatePerson} disabled={!selectedTreeId}>
          + Persona
        </button>
      </div>

      {/* Hint de UX: ctrl/cmd+click selecciona/deselecciona nodos para
          comparación múltiple. Cuando hay ≥1 seleccionado muestro estado +
          botón para limpiar. */}
      <div style={{ ...toolbarStyle, background: selectedForCompare.length > 0 ? '#fff7e0' : '#f4f4f4', fontSize: 12 }}>
        <span style={{ color: '#555' }}>
          <strong>Click</strong> sobre un nodo abre el panel de detalle ·{' '}
          <strong>Ctrl/Cmd+click</strong> selecciona (o deselecciona) un nodo
          para comparar. Se calculan todas las cosines par a par.
        </span>
        <span style={{ flex: 1 }} />
        {selectedForCompare.length > 0 && (
          <>
            <span style={{ color: '#666' }}>
              {isComputing && '⏳ computando…'}
              {!isComputing && selectedForCompare.length === 1 && '→ 1 seleccionado. Ctrl+click sobre otro para empezar a comparar.'}
              {!isComputing && selectedForCompare.length >= 2 && `✓ ${selectedForCompare.length} seleccionados · ${cosineByPair.size} comparación${cosineByPair.size === 1 ? '' : 'es'}`}
            </span>
            <button onClick={resetComparisonSelection} disabled={isComputing}>
              ↺ limpiar selección
            </button>
          </>
        )}
      </div>

      {/* Pedigree SVG / estados vacíos */}
      {!selectedTreeId && (
        <p style={{ color: '#888' }}>Creá un árbol para empezar.</p>
      )}
      {selectedTreeId && persons.length === 0 && (
        <p style={{ color: '#888' }}>Sin personas todavía. Agregá la primera con el botón de arriba.</p>
      )}
      {persons.length > 0 && (
        <PedigreeSvg
          persons={persons}
          personsById={personsById}
          positions={positions}
          photoUrls={photoUrls}
          selectedPersonId={selectedPersonId}
          dragOverPersonId={dragOverPersonId}
          selectedForCompare={selectedForCompare}
          cosineByPair={cosineByPair}
          isComputing={isComputing}
          onSelect={handleNodeClick}
          onUploadPhoto={handleUploadPhoto}
          onDragOverPerson={setDragOverPersonId}
          onCosineClick={(aId, bId, cos) => setTripletModalState({ aId, bId, cosine: cos })}
        />
      )}

      {/* Modal de tripleta: detalle del par seleccionado + agregar tercero +
          handoff al Comparador MVP. Sólo visible si hay un par con cosine
          computado. */}
      {tripletModalState && (() => {
        const aPerson = persons.find((p) => p.id === tripletModalState.aId);
        const bPerson = persons.find((p) => p.id === tripletModalState.bId);
        if (!aPerson || !bPerson) return null;
        const candidates = persons.filter(
          (p) => p.id !== aPerson.id && p.id !== bPerson.id && p.photoSha256 !== null,
        );
        return (
          <TripletModal
            a={aPerson}
            b={bPerson}
            initialCosine={tripletModalState.cosine}
            candidates={candidates}
            photoUrls={photoUrls}
            personsById={personsById}
            ensureEmbedding={ensureEmbedding}
            onClose={() => setTripletModalState(null)}
          />
        );
      })()}

      {/* Panel detalle de la persona seleccionada (click normal). */}
      {selectedPerson && (
        <PersonDetailPanel
          person={selectedPerson}
          persons={persons}
          personsById={personsById}
          photoUrl={selectedPerson.photoSha256 ? photoUrls.get(selectedPerson.photoSha256) ?? null : null}
          onSetParent={handleSetParent}
          onUploadPhoto={handleUploadPhoto}
          onDelete={handleDeletePerson}
          onClose={() => setSelectedPersonId(null)}
        />
      )}

      <p style={{ color: '#888', marginTop: 24, fontSize: 12 }}>
        Pedigree canónico (línea de unión padres → bus de hermanos) +
        comparación multi-selección con <code>Ctrl/Cmd+click</code>. Se
        computa el cosine entre cada par de nodos seleccionados y aparece
        sobre la línea que los conecta.
      </p>
    </div>
  );
}

// -----------------------------------------
// SVG pedigree top-down. Stateless: recibe todo lo que necesita por props.
// -----------------------------------------

interface PedigreeSvgProps {
  persons: Person[];
  personsById: Map<PersonId, Person>;
  positions: { pos: Map<PersonId, { x: number; y: number }>; viewW: number; viewH: number };
  photoUrls: Map<string, string>;
  selectedPersonId: PersonId | null;
  dragOverPersonId: PersonId | null;
  /** Lista ordenada de PersonIds seleccionados para comparar (ctrl+click). */
  selectedForCompare: PersonId[];
  /** Cosines de los pares activos (key canónica `min(a,b)|max(a,b)`). */
  cosineByPair: Map<string, number>;
  /** True mientras se computan los cosines (labels pendientes muestran "…"). */
  isComputing: boolean;
  /**
   * `withModifier` = true si el evento original tenía ctrlKey o metaKey, lo
   * que dispara el toggle en `selectedForCompare`. Sin modifier abre el panel
   * de detalle como siempre.
   */
  onSelect: (id: PersonId, withModifier: boolean) => void;
  onUploadPhoto: (id: PersonId, file: File) => void;
  onDragOverPerson: (id: PersonId | null) => void;
  /**
   * Click sobre el label del cosine de un par (rect blanco con borde naranja
   * en el medio de la línea). Abre el modal de tripleta. Sólo dispara si el
   * cosine ya está computado (clickear "…" no abre).
   */
  onCosineClick: (aId: PersonId, bId: PersonId, cosine: number) => void;
}

function PedigreeSvg({
  persons,
  personsById,
  positions,
  photoUrls,
  selectedPersonId,
  dragOverPersonId,
  selectedForCompare,
  cosineByPair,
  isComputing,
  onSelect,
  onUploadPhoto,
  onDragOverPerson,
  onCosineClick,
}: PedigreeSvgProps) {
  const { pos, viewW, viewH } = positions;

  // -------------------------------------
  // Líneas de parentesco — render canónico tipo pedigree clínico:
  //   1) Línea horizontal de UNIÓN entre padre y madre (cuando ambos existen
  //      en el set), trazada justo debajo de las cajas a la altura de su
  //      bottom. Si los padres están en distintas generaciones, dibujamos
  //      dos verticales para igualar altura antes de la línea horizontal.
  //   2) Línea VERTICAL desde el centro de la unión (o desde el único padre)
  //      bajando hasta el "sib-bus" (línea horizontal de hermanos a media
  //      generación entre padres e hijos).
  //   3) Línea horizontal del SIB-BUS desde min(child.cx) hasta max(child.cx)
  //      (incluyendo el punto donde baja la vertical de los padres, así nunca
  //      queda flotante el conector).
  //   4) Línea VERTICAL desde el bus hasta el top de cada hijo.
  // Agrupamos hijos por par (fatherId, motherId) — distintos pares producen
  // buses separados (hermanos completos comparten bus, half-sibs no).
  // Padres dangling (id que no está en `pos`) se ignoran sin romper el bus
  // (el ⚠ del nodo ya marca el problema en el hijo).
  // -------------------------------------

  type Segment = { x1: number; y1: number; x2: number; y2: number; key: string };
  const segments: Segment[] = [];

  // Agrupar por par de padres efectivos (excluyendo danglings).
  type ParentPairKey = string;
  const groups = new Map<ParentPairKey, {
    fatherId: PersonId | null;
    motherId: PersonId | null;
    children: Person[];
  }>();
  for (const p of persons) {
    const fid = p.fatherId !== null && pos.has(p.fatherId) ? p.fatherId : null;
    const mid = p.motherId !== null && pos.has(p.motherId) ? p.motherId : null;
    if (fid === null && mid === null) continue;
    const key: ParentPairKey = `${fid ?? '_'}|${mid ?? '_'}`;
    const existing = groups.get(key) ?? { fatherId: fid, motherId: mid, children: [] };
    existing.children.push(p);
    groups.set(key, existing);
  }

  for (const [key, g] of groups) {
    // Coordenadas de los padres (centro-X y bottom-Y).
    const fp = g.fatherId ? pos.get(g.fatherId) : null;
    const mp = g.motherId ? pos.get(g.motherId) : null;
    // child cxs ordenados.
    const childPositions = g.children
      .map((c) => pos.get(c.id))
      .filter((pp): pp is { x: number; y: number } => pp != null);
    if (childPositions.length === 0) continue;
    const childCxs = childPositions.map((pp) => pp.x + BOX_W / 2);
    const childTopY = childPositions[0].y; // todos los hijos del grupo en misma gen

    // (1) Línea de unión entre padres + punto X desde el que baja la vertical.
    let parentJoinX: number;
    let parentJoinY: number;
    if (fp && mp) {
      const fcx = fp.x + BOX_W / 2;
      const mcx = mp.x + BOX_W / 2;
      // Si están en la misma gen, una sola horizontal. Sino, un par de L
      // para igualar altura. Usamos `parentJoinY` = bottom del más bajo
      // (mayor Y) — el conector se traza a esa altura.
      const fb = fp.y + BOX_H;
      const mb = mp.y + BOX_H;
      parentJoinY = Math.max(fb, mb);
      const leftCx = Math.min(fcx, mcx);
      const rightCx = Math.max(fcx, mcx);
      // Vertical "stub" de cada padre hacia parentJoinY (es 0 px si ya está ahí).
      if (fb < parentJoinY) {
        segments.push({ x1: fcx, y1: fb, x2: fcx, y2: parentJoinY, key: `${key}-fstub` });
      }
      if (mb < parentJoinY) {
        segments.push({ x1: mcx, y1: mb, x2: mcx, y2: parentJoinY, key: `${key}-mstub` });
      }
      // Línea de unión horizontal entre padres.
      segments.push({
        x1: leftCx, y1: parentJoinY, x2: rightCx, y2: parentJoinY,
        key: `${key}-union`,
      });
      parentJoinX = (fcx + mcx) / 2;
    } else {
      // Un solo padre conocido (el otro es null o dangling).
      const only = fp ?? mp!;
      parentJoinX = only.x + BOX_W / 2;
      parentJoinY = only.y + BOX_H;
    }

    // (2) Vertical de los padres al sib-bus.
    //     Sib-bus a mitad de camino vertical entre parentJoinY y childTopY.
    const busY = (parentJoinY + childTopY) / 2;
    segments.push({
      x1: parentJoinX, y1: parentJoinY, x2: parentJoinX, y2: busY,
      key: `${key}-drop`,
    });

    // (3) Horizontal del sib-bus. Incluye parentJoinX para que la "bajada" de
    //     los padres se conecte siempre (caso límite: un único hijo no
    //     alineado con parentJoinX).
    const busAllXs = [parentJoinX, ...childCxs];
    const busMinX = Math.min(...busAllXs);
    const busMaxX = Math.max(...busAllXs);
    if (busMinX < busMaxX) {
      segments.push({
        x1: busMinX, y1: busY, x2: busMaxX, y2: busY,
        key: `${key}-bus`,
      });
    }

    // (4) Vertical de cada hijo al bus.
    for (let i = 0; i < g.children.length; i++) {
      const child = g.children[i];
      const cx = childCxs[i];
      segments.push({
        x1: cx, y1: busY, x2: cx, y2: childTopY,
        key: `${key}-c${child.id}`,
      });
    }
  }

  // Líneas de comparación para cada par (i,j) con i<j en selectedForCompare.
  // Cada segmento va de borde-a-borde (clip contra los rects de ambas cajas).
  // El label sobre la línea muestra el cosine del par si ya está en
  // cosineByPair, sino "…" mientras esté computing, sino "—".
  type ComparisonRender = {
    key: string;
    line: LineSeg;
    cosine: number | null;
    aId: PersonId;
    bId: PersonId;
  };
  const comparisonLines: ComparisonRender[] = [];
  for (let i = 0; i < selectedForCompare.length; i++) {
    for (let j = i + 1; j < selectedForCompare.length; j++) {
      const aId = selectedForCompare[i];
      const bId = selectedForCompare[j];
      const aPos = pos.get(aId);
      const bPos = pos.get(bId);
      if (!aPos || !bPos) continue;
      const line = clipLineBetweenBoxes(aPos, bPos, BOX_W, BOX_H);
      if (!line) continue;
      const k = aId < bId ? `${aId}|${bId}` : `${bId}|${aId}`;
      comparisonLines.push({
        key: k,
        line,
        cosine: cosineByPair.get(k) ?? null,
        aId,
        bId,
      });
    }
  }

  return (
    <div style={{
      border: '1px solid #ddd',
      borderRadius: 4,
      padding: 0,
      background: '#fcfcfc',
      overflow: 'auto',
      maxHeight: '60vh',
    }}>
      <svg
        width={viewW}
        height={viewH}
        viewBox={`0 0 ${viewW} ${viewH}`}
        style={{ display: 'block' }}
      >
        {/* Segmentos de pedigree primero, así quedan debajo de las cajas. */}
        {segments.map((s) => (
          <line
            key={s.key}
            x1={s.x1}
            y1={s.y1}
            x2={s.x2}
            y2={s.y2}
            stroke="#888"
            strokeWidth={1.5}
            strokeLinecap="square"
          />
        ))}
        {/* Líneas de comparación entre cada par seleccionado: por encima del
            pedigree, debajo de los nodos. La línea tiene pointerEvents none
            (no interceptea clicks) pero el label sí — al clickearlo abre el
            modal de tripleta con detalles del par + opción de agregar tercero. */}
        {comparisonLines.map((c) => (
          <g key={c.key} data-pair-key={c.key}>
            <line
              x1={c.line.x1}
              y1={c.line.y1}
              x2={c.line.x2}
              y2={c.line.y2}
              stroke="#d97706"
              strokeWidth={4}
              strokeLinecap="round"
              opacity={0.85}
              pointerEvents="none"
            />
            <ComparisonLabel
              x={(c.line.x1 + c.line.x2) / 2}
              y={(c.line.y1 + c.line.y2) / 2}
              cosine={c.cosine}
              isComputing={isComputing && c.cosine === null}
              onClick={
                c.cosine !== null
                  ? () => onCosineClick(c.aId, c.bId, c.cosine as number)
                  : undefined
              }
            />
          </g>
        ))}
        {/* Nodos. */}
        {persons.map((p) => {
          const pp = pos.get(p.id);
          if (!pp) return null;
          const idx = selectedForCompare.indexOf(p.id);
          return (
            <PersonNode
              key={p.id}
              person={p}
              personsById={personsById}
              x={pp.x}
              y={pp.y}
              photoUrl={p.photoSha256 ? photoUrls.get(p.photoSha256) ?? null : null}
              isSelected={p.id === selectedPersonId}
              isDragOver={p.id === dragOverPersonId}
              selectionIndex={idx >= 0 ? idx : null}
              onSelect={onSelect}
              onUploadPhoto={onUploadPhoto}
              onDragOverPerson={onDragOverPerson}
            />
          );
        })}
      </svg>
    </div>
  );
}

// -----------------------------------------
// Nodo individual del pedigree. Encapsula:
//   - rect de fondo (color según selección)
//   - foto (si tiene) o placeholder
//   - nombre debajo
//   - eventHandlers: click (selecciona), drag-over/leave/drop (foto)
//   - file input oculto activado por click sobre el área de la foto
// -----------------------------------------

interface PersonNodeProps {
  person: Person;
  personsById: Map<PersonId, Person>;
  x: number;
  y: number;
  photoUrl: string | null;
  isSelected: boolean;
  isDragOver: boolean;
  /** 0-based index del nodo en la lista de seleccionados para comparar; null si no está seleccionado. */
  selectionIndex: number | null;
  onSelect: (id: PersonId, withModifier: boolean) => void;
  onUploadPhoto: (id: PersonId, file: File) => void;
  onDragOverPerson: (id: PersonId | null) => void;
}

function PersonNode({
  person,
  personsById,
  x,
  y,
  photoUrl,
  isSelected,
  isDragOver,
  selectionIndex,
  onSelect,
  onUploadPhoto,
  onDragOverPerson,
}: PersonNodeProps) {
  // Para drag desde filesystem necesitamos preventDefault en dragOver/drop,
  // sino el browser navega a la imagen al soltar.
  const handleDragOver = (e: React.DragEvent<SVGGElement>) => {
    e.preventDefault();
    if (!isDragOver) onDragOverPerson(person.id);
  };
  const handleDragLeave = (e: React.DragEvent<SVGGElement>) => {
    e.preventDefault();
    onDragOverPerson(null);
  };
  const handleDrop = (e: React.DragEvent<SVGGElement>) => {
    e.preventDefault();
    onDragOverPerson(null);
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    if (!f.type.startsWith('image/')) return;
    onUploadPhoto(person.id, f);
  };

  // El input file en SVG no se puede embeber directo — vive en un <foreignObject>
  // para que el browser respete el click → file picker.
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Prioridad visual: dragOver > selected-for-compare (naranja) > selected
  // para detalle (azul) > default. Todos los nodos seleccionados para comparar
  // comparten el mismo color naranja; los distingue el badge numerado.
  const isInCompare = selectionIndex !== null;
  const COMPARE_COLOR = '#d97706';
  const COMPARE_BG = '#fff3e0';
  const stroke = isDragOver
    ? '#0a0'
    : isInCompare
      ? COMPARE_COLOR
      : isSelected
        ? '#1a73e8'
        : '#bbb';
  const strokeWidth = isDragOver || isInCompare ? 3 : isSelected ? 2 : 1;
  const bg = isDragOver
    ? '#eaffea'
    : isInCompare
      ? COMPARE_BG
      : isSelected
        ? '#eaf3ff'
        : '#fff';

  // Refs colgadas: marcar visualmente si la persona tiene padres apuntando
  // a alguien que no existe (parent fue borrado). Triangulito rojo arriba-der.
  const hasDanglingParent =
    (person.fatherId !== null && !personsById.has(person.fatherId)) ||
    (person.motherId !== null && !personsById.has(person.motherId));

  const photoX = x + (BOX_W - PHOTO_SIZE) / 2;
  const photoY = y + 10;

  const ariaLabel = isInCompare
    ? `${person.name} · sel ${(selectionIndex as number) + 1}`
    : person.name;

  return (
    <g
      role="button"
      aria-label={ariaLabel}
      data-person-id={person.id}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <rect
        x={x}
        y={y}
        width={BOX_W}
        height={BOX_H}
        rx={6}
        fill={bg}
        stroke={stroke}
        strokeWidth={strokeWidth}
        style={{ cursor: 'pointer' }}
        onClick={(e) => onSelect(person.id, e.ctrlKey || e.metaKey)}
      />
      {/* Área de foto: click sobre acá abre file picker (no selecciona). */}
      <g
        onClick={(e) => {
          e.stopPropagation();
          fileInputRef.current?.click();
        }}
        style={{ cursor: 'pointer' }}
      >
        {photoUrl ? (
          <image
            href={photoUrl}
            x={photoX}
            y={photoY}
            width={PHOTO_SIZE}
            height={PHOTO_SIZE}
            preserveAspectRatio="xMidYMid slice"
            style={{ borderRadius: 4 }}
          />
        ) : (
          <>
            <rect
              x={photoX}
              y={photoY}
              width={PHOTO_SIZE}
              height={PHOTO_SIZE}
              fill="#fafafa"
              stroke="#ddd"
              strokeDasharray="3 2"
            />
            <text
              x={photoX + PHOTO_SIZE / 2}
              y={photoY + PHOTO_SIZE / 2 + 4}
              textAnchor="middle"
              fontSize={11}
              fill="#aaa"
            >
              + foto
            </text>
          </>
        )}
      </g>
      {/* Nombre debajo de la foto, recortado si es largo. */}
      <text
        x={x + BOX_W / 2}
        y={y + 10 + PHOTO_SIZE + 22}
        textAnchor="middle"
        fontSize={13}
        fontWeight={600}
        fill="#222"
        style={{ pointerEvents: 'none', userSelect: 'none' }}
      >
        {truncate(person.name, 16)}
      </text>
      {/* Indicador de refs colgadas. SVG <text> no soporta `title` prop;
          el tooltip va como child <title> element. */}
      {hasDanglingParent && (
        <text
          x={x + BOX_W - 8}
          y={y + 14}
          textAnchor="end"
          fontSize={14}
          fill="#c00"
        >
          <title>Esta persona tiene padre/madre apuntando a alguien borrado</title>
          ⚠
        </text>
      )}
      {/* Badge numerado esquina superior izquierda cuando el nodo está en
          la lista de comparación. El número refleja el orden de selección
          (1-based: el primer ctrl+click es 1). */}
      {isInCompare && (
        <g pointerEvents="none">
          <rect
            x={x + 4}
            y={y + 4}
            width={20}
            height={20}
            rx={10}
            fill={COMPARE_COLOR}
          />
          <text
            x={x + 4 + 10}
            y={y + 4 + 14}
            textAnchor="middle"
            fontSize={12}
            fontWeight={700}
            fill="#fff"
          >
            {(selectionIndex as number) + 1}
          </text>
        </g>
      )}
      {/* foreignObject hospeda el <input type="file"> oculto. Necesario porque
          los inputs HTML no se renderizan dentro de <svg> directamente. */}
      <foreignObject x={x} y={y} width={1} height={1} style={{ overflow: 'visible' }}>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onUploadPhoto(person.id, f);
            e.target.value = '';
          }}
        />
      </foreignObject>
    </g>
  );
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + '…';
}

// -----------------------------------------
// Panel inferior: detalle de la persona seleccionada con edición de padres,
// upload de foto y borrado. Encapsula la lógica que antes vivía en PersonRow.
// -----------------------------------------

interface PersonDetailPanelProps {
  person: Person;
  persons: Person[];
  personsById: Map<PersonId, Person>;
  photoUrl: string | null;
  onSetParent: (id: PersonId, role: 'father' | 'mother', parentId: PersonId | null) => void;
  onUploadPhoto: (id: PersonId, file: File) => void;
  onDelete: (id: PersonId) => void;
  onClose: () => void;
}

function PersonDetailPanel({
  person,
  persons,
  personsById,
  photoUrl,
  onSetParent,
  onUploadPhoto,
  onDelete,
  onClose,
}: PersonDetailPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const parentCandidates = persons.filter((p) => p.id !== person.id);

  const parentLabel = (id: PersonId | null): string => {
    if (id === null) return '';
    const p = personsById.get(id);
    return p ? p.name : `(borrado: ${id.slice(0, 6)}…)`;
  };
  const fatherDangling = person.fatherId !== null && !personsById.has(person.fatherId);
  const motherDangling = person.motherId !== null && !personsById.has(person.motherId);

  return (
    <div style={panelStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <h3 style={{ margin: 0 }}>{person.name}</h3>
        <button onClick={onClose} style={{ fontSize: 12 }}>cerrar</button>
      </div>
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
        {/* Foto grande clickeable. */}
        <div
          onClick={() => fileInputRef.current?.click()}
          title="Click para cambiar foto"
          style={{
            width: 180,
            height: 180,
            border: '1px dashed #aaa',
            borderRadius: 4,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            background: '#fafafa',
            overflow: 'hidden',
            flexShrink: 0,
          }}
        >
          {photoUrl ? (
            <img
              src={photoUrl}
              alt={person.name}
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />
          ) : (
            <span style={{ color: '#aaa', fontSize: 13 }}>+ foto</span>
          )}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onUploadPhoto(person.id, f);
            e.target.value = '';
          }}
        />

        {/* Dropdowns padre/madre + borrar. */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <label>
            Padre:&nbsp;
            <select
              value={person.fatherId ?? ''}
              onChange={(e) => onSetParent(person.id, 'father', e.target.value || null)}
              style={{ minWidth: 200, color: fatherDangling ? '#c00' : undefined }}
            >
              <option value="">— sin padre —</option>
              {fatherDangling && (
                <option value={person.fatherId!}>{parentLabel(person.fatherId)}</option>
              )}
              {parentCandidates.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Madre:&nbsp;
            <select
              value={person.motherId ?? ''}
              onChange={(e) => onSetParent(person.id, 'mother', e.target.value || null)}
              style={{ minWidth: 200, color: motherDangling ? '#c00' : undefined }}
            >
              <option value="">— sin madre —</option>
              {motherDangling && (
                <option value={person.motherId!}>{parentLabel(person.motherId)}</option>
              )}
              {parentCandidates.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <div>
            <button onClick={() => onDelete(person.id)} style={{ color: '#900' }}>
              ✕ Borrar persona
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// -----------------------------------------
// Geometría: dado el rect (x, y, BOX_W, BOX_H) de dos nodos, devolver el
// segmento "centro-a-centro" recortado contra los bordes de ambos rects.
// Así la línea de comparación termina sobre los bordes y no atraviesa la
// caja. Para nodos en la misma generación (línea horizontal) recorta sobre
// los lados verticales; para nodos en distintas (línea diagonal) recorta
// donde toque cada lado.
//
// Algoritmo: parametrizamos la línea entre los centros con t en [0,1].
// Buscamos el primer t > 0 donde la línea cruza un lado del rect de origen
// (saliendo), y el último t < 1 donde la línea cruza un lado del rect de
// destino (entrando). Si los centros coinciden (caso degenerado por bug
// de posiciones), devolvemos null.
// -----------------------------------------

interface LineSeg { x1: number; y1: number; x2: number; y2: number }

function clipLineBetweenBoxes(
  a: { x: number; y: number },
  b: { x: number; y: number },
  w: number,
  h: number,
): LineSeg | null {
  const cax = a.x + w / 2;
  const cay = a.y + h / 2;
  const cbx = b.x + w / 2;
  const cby = b.y + h / 2;
  const dx = cbx - cax;
  const dy = cby - cay;
  if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) return null;

  // Para un rect (rx, ry, rx+w, ry+h) y la línea (cax+t*dx, cay+t*dy),
  // las intersecciones son t = (rx - cax)/dx, (rx+w - cax)/dx,
  // (ry - cay)/dy, (ry+h - cay)/dy (sólo si dx≠0 / dy≠0).
  // Para la caja de salida (a), buscamos el menor t > 0 que cae en el rect.
  // Para la de entrada (b), buscamos el mayor t < 1 que cae en el rect.

  const intersect = (rx: number, ry: number, originAtStart: boolean) => {
    const candidates: number[] = [];
    if (Math.abs(dx) > 1e-6) {
      candidates.push((rx - cax) / dx);
      candidates.push((rx + w - cax) / dx);
    }
    if (Math.abs(dy) > 1e-6) {
      candidates.push((ry - cay) / dy);
      candidates.push((ry + h - cay) / dy);
    }
    // Filtrar a los que efectivamente caen dentro del rect.
    const valid = candidates.filter((t) => {
      const px = cax + t * dx;
      const py = cay + t * dy;
      return px >= rx - 1e-6 && px <= rx + w + 1e-6 && py >= ry - 1e-6 && py <= ry + h + 1e-6;
    });
    if (valid.length === 0) return null;
    return originAtStart ? Math.max(...valid) : Math.min(...valid);
  };

  const tStart = intersect(a.x, a.y, true);   // mayor t dentro del rect A (salida)
  const tEnd = intersect(b.x, b.y, false);    // menor t dentro del rect B (entrada)
  if (tStart === null || tEnd === null || tStart >= tEnd) return null;

  return {
    x1: cax + tStart * dx,
    y1: cay + tStart * dy,
    x2: cax + tEnd * dx,
    y2: cay + tEnd * dy,
  };
}

// -----------------------------------------
// Label flotante del cosine sobre la línea P1↔P2. Rect blanco redondeado
// con borde naranja, número en monoespaciado adentro. Centrado en el
// midpoint de la línea y un poco desplazado vertical para no pisar la
// línea con el texto.
// -----------------------------------------

interface ComparisonLabelProps {
  x: number;
  y: number;
  cosine: number | null;
  isComputing: boolean;
  /** Si se provee, el label es clickeable y abre el modal de tripleta. */
  onClick?: () => void;
}

function ComparisonLabel({ x, y, cosine, isComputing, onClick }: ComparisonLabelProps) {
  const text = isComputing ? '…' : cosine === null ? '—' : cosine.toFixed(4);
  const w = 70;
  const h = 24;
  const isClickable = onClick !== undefined;
  return (
    <g
      data-testid="cosine-svg-label"
      onClick={onClick}
      style={isClickable ? { cursor: 'pointer' } : undefined}
    >
      <title>{isClickable ? 'Click para ver detalles + agregar tercero' : ''}</title>
      <rect
        x={x - w / 2}
        y={y - h / 2}
        width={w}
        height={h}
        rx={4}
        fill="#fff"
        stroke="#d97706"
        strokeWidth={1.5}
      />
      <text
        x={x}
        y={y + 5}
        textAnchor="middle"
        fontSize={13}
        fontWeight={700}
        fontFamily="monospace"
        fill="#222"
        data-testid="cosine-value"
        pointerEvents="none"
      >
        {text}
      </text>
    </g>
  );
}

// -----------------------------------------
// Estilos compartidos.
// -----------------------------------------

const toolbarStyle: React.CSSProperties = {
  display: 'flex',
  gap: 8,
  alignItems: 'center',
  marginBottom: 12,
  padding: 8,
  background: '#f4f4f4',
  borderRadius: 4,
};

const errorStyle: React.CSSProperties = {
  padding: 10,
  marginBottom: 12,
  background: '#fee',
  color: '#900',
  border: '1px solid #fcc',
  borderRadius: 4,
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
};

const infoStyle: React.CSSProperties = {
  padding: 10,
  marginBottom: 12,
  background: '#eaffea',
  color: '#060',
  border: '1px solid #bde5bd',
  borderRadius: 4,
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  gap: 12,
};

const panelStyle: React.CSSProperties = {
  marginTop: 16,
  padding: 16,
  border: '1px solid #1a73e8',
  borderRadius: 4,
  background: '#f4f8ff',
};
