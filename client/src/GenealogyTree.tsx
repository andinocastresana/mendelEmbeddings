// =========================================
// ID: PHYLOFACE_GENEALOGY_TREE
// VERSION: v3.0
// =========================================
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
  type Comparison,
  type Person,
  type PersonId,
  type Sha256Hex,
  type Tree,
} from './lib/genealogy';
import {
  deleteComparison,
  deletePerson,
  deleteTree,
  getPhoto,
  listComparisons,
  listPersons,
  listTrees,
  putPerson,
  putPhoto,
  putTree,
  saveComparison,
  setPhotoEmbedding,
} from './lib/treeStore';
import { computeTreeLayout } from './lib/treeLayout';
import {
  computeEmbedding,
  cosineSimilarity,
  initFaceLandmarker,
  initOnnxSession,
  loadImage,
} from './lib/pipeline';

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
  const [selectedPersonId, setSelectedPersonId] = useState<PersonId | null>(null);
  // Persona-target del drag-over actual (para highlight visual).
  const [dragOverPersonId, setDragOverPersonId] = useState<PersonId | null>(null);

  // ----- Modo comparación (paso 5) -----
  const [comparisonMode, setComparisonMode] = useState(false);
  const [p1Id, setP1Id] = useState<PersonId | null>(null);
  const [p2Id, setP2Id] = useState<PersonId | null>(null);
  // Resultado del cómputo actual. Null si todavía no se computó (P2 sin elegir
  // o cómputo en curso). Distinto de la lista persistida `comparisons`.
  const [currentCosine, setCurrentCosine] = useState<number | null>(null);
  const [isComputing, setIsComputing] = useState(false);
  // Historial persistido (filtrado por treeId activo).
  const [comparisons, setComparisons] = useState<Comparison[]>([]);

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

  // -------------------------------------
  // Cargar historial de comparaciones cuando cambia el árbol activo.
  // No filtramos por personas borradas: el historial es valioso aún si una
  // de las personas referenciadas dejó de existir (la UI lo refleja).
  // -------------------------------------
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      if (!selectedTreeId) {
        setComparisons([]);
        return;
      }
      try {
        const list = await listComparisons(selectedTreeId);
        if (!cancelled) {
          // Ordenadas más reciente primero — UX habitual de historial.
          list.sort((a, b) => b.computedAt - a.computedAt);
          setComparisons(list);
        }
      } catch (e) {
        if (!cancelled) setError(`Cargando historial de comparaciones: ${(e as Error).message}`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedTreeId]);

  // Helper para resetear la selección viva sin tocar el historial persistido.
  // Lo invocan: toggle off, cambio de árbol, "comparar de nuevo" tras un
  // resultado, y errores fatales de cómputo.
  const resetComparisonSelection = useCallback(() => {
    setP1Id(null);
    setP2Id(null);
    setCurrentCosine(null);
  }, []);

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
  // runComparison(p1, p2, opts): computa cosine + persiste como Comparison.
  // Si `forceRecompute=true`, invalida el cache de embedding de ambas fotos
  // (borra y vuelve a calcular). Sino, usa lo que haya cacheado.
  // -------------------------------------
  const runComparison = useCallback(async (
    person1: Person,
    person2: Person,
    opts: { forceRecompute?: boolean } = {},
  ): Promise<void> => {
    if (!selectedTreeId) return;
    if (!person1.photoSha256) {
      setError(`"${person1.name}" no tiene foto cargada. Arrastrá una sobre el nodo primero.`);
      return;
    }
    if (!person2.photoSha256) {
      setError(`"${person2.name}" no tiene foto cargada. Arrastrá una sobre el nodo primero.`);
      return;
    }
    setIsComputing(true);
    setError(null);
    try {
      const emb1 = await ensureEmbedding(person1.photoSha256, opts.forceRecompute);
      const emb2 = await ensureEmbedding(person2.photoSha256, opts.forceRecompute);
      const cos = cosineSimilarity(emb1, emb2);
      setCurrentCosine(cos);
      const comp = newComparison(
        selectedTreeId,
        person1.id,
        person2.id,
        person1.photoSha256,
        person2.photoSha256,
        cos,
      );
      await saveComparison(comp);
      setComparisons((prev) => [comp, ...prev]);
    } catch (e) {
      setError(`Comparando: ${(e as Error).message}`);
      setCurrentCosine(null);
    } finally {
      setIsComputing(false);
    }
  }, [selectedTreeId, ensureEmbedding]);

  // Click sobre nodo. En modo comparación off → selecciona para detalle.
  // En modo on:
  //   - sin P1 → setea P1
  //   - con P1 sin P2, click sobre otro → setea P2 y dispara cómputo
  //   - con P1 y P2 ya elegidos → trata el click como "P1 nuevo, reset"
  //   - click sobre P1 mientras es P1 → no-op (evita disparar comparación
  //     consigo misma).
  const handleNodeClick = useCallback((id: PersonId) => {
    if (!comparisonMode) {
      setSelectedPersonId(id);
      return;
    }
    if (p1Id === null) {
      setP1Id(id);
      return;
    }
    if (p2Id !== null) {
      setP1Id(id);
      setP2Id(null);
      setCurrentCosine(null);
      return;
    }
    if (id === p1Id) return;
    const person1 = persons.find((p) => p.id === p1Id);
    const person2 = persons.find((p) => p.id === id);
    if (!person1 || !person2) return;
    setP2Id(id);
    void runComparison(person1, person2);
  }, [comparisonMode, p1Id, p2Id, persons, runComparison]);

  const handleToggleComparisonMode = useCallback(() => {
    setComparisonMode((prev) => {
      if (prev) {
        // Al apagar el modo, resetear la selección viva.
        setP1Id(null);
        setP2Id(null);
        setCurrentCosine(null);
      }
      return !prev;
    });
  }, []);

  const handleRecompute = useCallback(() => {
    if (!p1Id || !p2Id) return;
    const person1 = persons.find((p) => p.id === p1Id);
    const person2 = persons.find((p) => p.id === p2Id);
    if (!person1 || !person2) return;
    void runComparison(person1, person2, { forceRecompute: true });
  }, [p1Id, p2Id, persons, runComparison]);

  const handleDeleteComparison = useCallback(async (id: string) => {
    try {
      await deleteComparison(id);
      setComparisons((prev) => prev.filter((c) => c.id !== id));
    } catch (e) {
      setError(`Borrando comparación: ${(e as Error).message}`);
    }
  }, []);

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

      {/* Toolbar de modo comparación */}
      <div style={{ ...toolbarStyle, background: comparisonMode ? '#fff7e0' : '#f4f4f4' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={comparisonMode}
            onChange={handleToggleComparisonMode}
            disabled={!selectedTreeId || persons.length < 2}
          />
          <strong>Modo comparación</strong>
        </label>
        <span style={{ color: '#666', fontSize: 12 }}>
          {!comparisonMode && 'click sobre un nodo abre detalle'}
          {comparisonMode && p1Id === null && '→ click sobre un nodo para elegir P1'}
          {comparisonMode && p1Id !== null && p2Id === null && '→ click sobre otro nodo para elegir P2 y comparar'}
          {comparisonMode && p1Id !== null && p2Id !== null && !isComputing && '✓ comparación lista. Click sobre otro nodo para reiniciar.'}
          {comparisonMode && isComputing && '⏳ computando embeddings…'}
        </span>
        <span style={{ flex: 1 }} />
        {comparisonMode && (p1Id !== null || p2Id !== null) && (
          <button onClick={resetComparisonSelection} disabled={isComputing} style={{ fontSize: 12 }}>
            ↺ reiniciar selección
          </button>
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
          p1Id={comparisonMode ? p1Id : null}
          p2Id={comparisonMode ? p2Id : null}
          onSelect={handleNodeClick}
          onUploadPhoto={handleUploadPhoto}
          onDragOverPerson={setDragOverPersonId}
        />
      )}

      {/* Paneles inferiores: detalle y/o comparación, lado a lado cuando
          ambos están activos. Cada uno se renderea si tiene contenido. */}
      {(selectedPerson || comparisonMode) && (
        <div style={{ display: 'flex', gap: 16, marginTop: 16, flexWrap: 'wrap' }}>
          {selectedPerson && (
            <div style={{ flex: '1 1 360px', minWidth: 320 }}>
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
            </div>
          )}
          {comparisonMode && (
            <div style={{ flex: '1 1 360px', minWidth: 320 }}>
              <ComparisonPanel
                p1={p1Id ? persons.find((p) => p.id === p1Id) ?? null : null}
                p2={p2Id ? persons.find((p) => p.id === p2Id) ?? null : null}
                photoUrls={photoUrls}
                currentCosine={currentCosine}
                isComputing={isComputing}
                comparisons={comparisons}
                personsById={personsById}
                onRecompute={handleRecompute}
                onDeleteComparison={handleDeleteComparison}
              />
            </div>
          )}
        </div>
      )}

      <p style={{ color: '#888', marginTop: 24, fontSize: 12 }}>
        Paso 5 listo: modo comparación on-demand entre dos personas con foto,
        embedding cacheado por SHA-256 en IndexedDB, historial persistido por
        árbol.
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
  /** Si != null, el nodo correspondiente se resalta como P1 (modo comparación). */
  p1Id: PersonId | null;
  /** Si != null, el nodo correspondiente se resalta como P2 (modo comparación). */
  p2Id: PersonId | null;
  onSelect: (id: PersonId) => void;
  onUploadPhoto: (id: PersonId, file: File) => void;
  onDragOverPerson: (id: PersonId | null) => void;
}

function PedigreeSvg({
  persons,
  personsById,
  positions,
  photoUrls,
  selectedPersonId,
  dragOverPersonId,
  p1Id,
  p2Id,
  onSelect,
  onUploadPhoto,
  onDragOverPerson,
}: PedigreeSvgProps) {
  const { pos, viewW, viewH } = positions;

  // Construir lista de líneas de parentesco (padre→hijo y madre→hijo). Solo
  // dibuja la línea si el parent existe en el set (ignora refs colgadas).
  const lines: { x1: number; y1: number; x2: number; y2: number; key: string }[] = [];
  for (const p of persons) {
    const childPos = pos.get(p.id);
    if (!childPos) continue;
    for (const [role, parentId] of [
      ['father', p.fatherId] as const,
      ['mother', p.motherId] as const,
    ]) {
      if (parentId === null) continue;
      const parentPos = pos.get(parentId);
      if (!parentPos) continue; // dangling: parent borrado del set
      // Línea del centro-bottom del padre/madre al centro-top del hijo. Las
      // dos líneas (de father y de mother) terminan en el mismo punto del
      // hijo; visualmente se confunden pero quedan claras al hover.
      lines.push({
        x1: parentPos.x + BOX_W / 2,
        y1: parentPos.y + BOX_H,
        x2: childPos.x + BOX_W / 2,
        y2: childPos.y,
        key: `${parentId}-${role}-${p.id}`,
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
        {/* Líneas primero, así quedan abajo de las cajas. */}
        {lines.map((l) => (
          <line
            key={l.key}
            x1={l.x1}
            y1={l.y1}
            x2={l.x2}
            y2={l.y2}
            stroke="#888"
            strokeWidth={1.5}
          />
        ))}
        {/* Nodos. */}
        {persons.map((p) => {
          const pp = pos.get(p.id);
          if (!pp) return null;
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
              isP1={p.id === p1Id}
              isP2={p.id === p2Id}
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
  isP1: boolean;
  isP2: boolean;
  onSelect: (id: PersonId) => void;
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
  isP1,
  isP2,
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

  // Prioridad visual: dragOver > P2 (verde) > P1 (azul fuerte) > selected (azul claro) > default.
  // P1/P2 ganan a `selected` porque el modo comparación reusa el mismo handler
  // de click; mientras está on, selected no debería tener fuerza visual.
  const stroke = isDragOver
    ? '#0a0'
    : isP2
      ? '#0a8a3a'
      : isP1
        ? '#0044cc'
        : isSelected
          ? '#1a73e8'
          : '#bbb';
  const strokeWidth = isDragOver || isP1 || isP2 ? 3 : isSelected ? 2 : 1;
  const bg = isDragOver
    ? '#eaffea'
    : isP2
      ? '#e0f5e7'
      : isP1
        ? '#dde6ff'
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

  const ariaLabel = isP1
    ? `${person.name} · P1`
    : isP2
      ? `${person.name} · P2`
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
        onClick={() => onSelect(person.id)}
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
      {/* Badge P1/P2 esquina superior izquierda en modo comparación. */}
      {(isP1 || isP2) && (
        <g pointerEvents="none">
          <rect
            x={x + 4}
            y={y + 4}
            width={22}
            height={16}
            rx={3}
            fill={isP2 ? '#0a8a3a' : '#0044cc'}
          />
          <text
            x={x + 4 + 11}
            y={y + 4 + 12}
            textAnchor="middle"
            fontSize={11}
            fontWeight={700}
            fill="#fff"
          >
            {isP1 ? 'P1' : 'P2'}
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
// Panel "Comparación": muestra el resultado del cómputo actual (foto P1,
// foto P2, cosine grande, botón ↻ recompute) + historial persistido del
// árbol. Se renderiza junto al PersonDetailPanel cuando el modo comparación
// está ON; ambos viven en un flex row del componente principal.
//
// Sobre el historial:
//   - Ordenado más reciente primero.
//   - Cada entrada muestra los nombres P1/P2 (o "(borrado)" si la persona
//     ya no existe), el cosine y la fecha relativa.
//   - Marcado "stale" si la photoSha256 actual de la persona difiere del
//     snapshot guardado en la Comparison: la comparación sigue siendo
//     válida para esas fotos, pero esas fotos ya no son las que la persona
//     tiene asignadas hoy.
//   - Botón ✕ para borrar entrada individual.
// -----------------------------------------

interface ComparisonPanelProps {
  p1: Person | null;
  p2: Person | null;
  photoUrls: Map<string, string>;
  currentCosine: number | null;
  isComputing: boolean;
  comparisons: Comparison[];
  personsById: Map<PersonId, Person>;
  onRecompute: () => void;
  onDeleteComparison: (id: string) => void;
}

function ComparisonPanel({
  p1,
  p2,
  photoUrls,
  currentCosine,
  isComputing,
  comparisons,
  personsById,
  onRecompute,
  onDeleteComparison,
}: ComparisonPanelProps) {
  const p1Photo = p1?.photoSha256 ? photoUrls.get(p1.photoSha256) ?? null : null;
  const p2Photo = p2?.photoSha256 ? photoUrls.get(p2.photoSha256) ?? null : null;

  return (
    <div style={comparisonPanelStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <h3 style={{ margin: 0 }}>Comparación</h3>
        <button
          onClick={onRecompute}
          disabled={!p1 || !p2 || isComputing}
          style={{ fontSize: 12 }}
          title="Re-computar ignorando embeddings cacheados"
        >
          ↻ recompute
        </button>
      </div>

      {/* Dos fotos lado a lado + cosine en el medio. */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12 }}>
        <ComparisonSlot label="P1" person={p1} photoUrl={p1Photo} color="#0044cc" />
        <div style={{ flex: '0 0 auto', textAlign: 'center', minWidth: 70 }}>
          <div style={{ fontSize: 11, color: '#666' }}>cosine</div>
          <div
            data-testid="cosine-value"
            style={{
              fontSize: 22,
              fontWeight: 700,
              color: currentCosine === null ? '#bbb' : '#222',
              fontFamily: 'monospace',
            }}
          >
            {isComputing ? '…' : currentCosine === null ? '—' : currentCosine.toFixed(4)}
          </div>
        </div>
        <ComparisonSlot label="P2" person={p2} photoUrl={p2Photo} color="#0a8a3a" />
      </div>

      {/* Historial. */}
      <div>
        <div style={{ fontSize: 12, color: '#666', marginBottom: 6 }}>
          Historial ({comparisons.length})
        </div>
        {comparisons.length === 0 && (
          <p style={{ color: '#aaa', fontSize: 12, margin: 0 }}>
            Sin comparaciones guardadas todavía.
          </p>
        )}
        {comparisons.length > 0 && (
          <ul
            style={{
              listStyle: 'none',
              padding: 0,
              margin: 0,
              maxHeight: 200,
              overflowY: 'auto',
              border: '1px solid #eee',
              borderRadius: 3,
            }}
          >
            {comparisons.map((c) => {
              const cp1 = personsById.get(c.p1Id);
              const cp2 = personsById.get(c.p2Id);
              const p1Stale = cp1 != null && cp1.photoSha256 !== c.p1Sha256;
              const p2Stale = cp2 != null && cp2.photoSha256 !== c.p2Sha256;
              const stale = p1Stale || p2Stale;
              return (
                <li
                  key={c.id}
                  style={{
                    display: 'flex',
                    gap: 8,
                    padding: '4px 8px',
                    borderBottom: '1px solid #f4f4f4',
                    alignItems: 'center',
                    fontSize: 12,
                  }}
                >
                  <span style={{ flex: 1, color: '#333' }}>
                    {cp1 ? cp1.name : <em style={{ color: '#aaa' }}>(borrado)</em>}{' '}
                    ↔ {cp2 ? cp2.name : <em style={{ color: '#aaa' }}>(borrado)</em>}
                    {stale && (
                      <span title="La foto asignada a una de las personas cambió desde este cómputo" style={{ marginLeft: 6, color: '#c80', fontSize: 11 }}>
                        ⚠ stale
                      </span>
                    )}
                  </span>
                  <span style={{ fontFamily: 'monospace', minWidth: 60, textAlign: 'right' }}>
                    {c.cosine.toFixed(4)}
                  </span>
                  <span style={{ color: '#888', fontSize: 11, minWidth: 80, textAlign: 'right' }}>
                    {formatRelative(c.computedAt)}
                  </span>
                  <button
                    onClick={() => onDeleteComparison(c.id)}
                    title="Borrar esta comparación"
                    style={{ fontSize: 11, padding: '0 6px' }}
                  >
                    ✕
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}

interface ComparisonSlotProps {
  label: string;
  person: Person | null;
  photoUrl: string | null;
  color: string;
}

function ComparisonSlot({ label, person, photoUrl, color }: ComparisonSlotProps) {
  return (
    <div style={{ flex: 1, textAlign: 'center', minWidth: 90 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color }}>{label}</div>
      <div
        style={{
          width: 96,
          height: 96,
          margin: '4px auto',
          border: `2px solid ${person ? color : '#ddd'}`,
          borderRadius: 4,
          background: '#fafafa',
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {photoUrl ? (
          <img
            src={photoUrl}
            alt={person?.name ?? label}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        ) : (
          <span style={{ color: '#aaa', fontSize: 11 }}>
            {person ? 'sin foto' : '—'}
          </span>
        )}
      </div>
      <div style={{ fontSize: 12, fontWeight: 500, color: '#333', minHeight: 16 }}>
        {person ? truncate(person.name, 18) : '—'}
      </div>
    </div>
  );
}

// Formato de fecha relativo simple. Para algo más rico habría que sumar
// `Intl.RelativeTimeFormat`, pero no vale la pena por unas líneas de UI.
function formatRelative(ts: number): string {
  const dMs = Date.now() - ts;
  const dSec = Math.floor(dMs / 1000);
  if (dSec < 60) return 'hace seg';
  const dMin = Math.floor(dSec / 60);
  if (dMin < 60) return `hace ${dMin}m`;
  const dHr = Math.floor(dMin / 60);
  if (dHr < 24) return `hace ${dHr}h`;
  const dDay = Math.floor(dHr / 24);
  if (dDay < 30) return `hace ${dDay}d`;
  return new Date(ts).toLocaleDateString();
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

const panelStyle: React.CSSProperties = {
  padding: 16,
  border: '1px solid #1a73e8',
  borderRadius: 4,
  background: '#f4f8ff',
};

const comparisonPanelStyle: React.CSSProperties = {
  padding: 16,
  border: '1px solid #c89000',
  borderRadius: 4,
  background: '#fffaf0',
};
