// =========================================
// ID: PHYLOFACE_GENEALOGY_TREE
// VERSION: v2.0
// =========================================
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
// - El paso 5 (comparación on-demand) extiende este patrón: tras seleccionar
//   P1, un click sobre otro nodo va a disparar la comparación. Por ahora la
//   selección sólo abre el panel.
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
import {
  newPerson,
  newTree,
  wouldCreateCycle,
  type Person,
  type PersonId,
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
} from './lib/treeStore';
import { computeTreeLayout } from './lib/treeLayout';

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
      setSelectedTreeId(tree.id);
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
      setSelectedTreeId(remaining[0]?.id ?? null);
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
            onChange={(e) => setSelectedTreeId(e.target.value || null)}
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
          onSelect={setSelectedPersonId}
          onUploadPhoto={handleUploadPhoto}
          onDragOverPerson={setDragOverPersonId}
        />
      )}

      {/* Panel detalle de la persona seleccionada */}
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
        Paso 4 listo: pedigree SVG + drag-and-drop foto sobre nodo + panel
        inferior con detalle. Paso 5 (comparación on-demand): tras seleccionar
        P1, click sobre otro nodo dispara el cosine.
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

  const stroke = isDragOver ? '#0a0' : isSelected ? '#1a73e8' : '#bbb';
  const strokeWidth = isDragOver ? 2 : isSelected ? 2 : 1;
  const bg = isDragOver ? '#eaffea' : isSelected ? '#eaf3ff' : '#fff';

  // Refs colgadas: marcar visualmente si la persona tiene padres apuntando
  // a alguien que no existe (parent fue borrado). Triangulito rojo arriba-der.
  const hasDanglingParent =
    (person.fatherId !== null && !personsById.has(person.fatherId)) ||
    (person.motherId !== null && !personsById.has(person.motherId));

  const photoX = x + (BOX_W - PHOTO_SIZE) / 2;
  const photoY = y + 10;

  return (
    <g
      role="button"
      aria-label={person.name}
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
  marginTop: 16,
  padding: 16,
  border: '1px solid #1a73e8',
  borderRadius: 4,
  background: '#f4f8ff',
};
