// =========================================
// ID: PHYLOFACE_GENEALOGY_TREE
// VERSION: v1.0
// =========================================
// Vista del Track 2b (Tarea #26) — paso 2 del plan: lista plana de personas
// para validar que la capa de persistencia (lib/treeStore + lib/genealogy)
// funciona end-to-end. Sin SVG todavía (paso 4); sin comparación (paso 5);
// sin export/import (paso 6).
//
// Lo que sí valida en este paso:
//   - Crear / seleccionar / borrar árbol.
//   - Crear / borrar persona dentro del árbol activo.
//   - Asignar foto a persona (input file) → dedup por SHA-256, preview.
//   - Asignar padre/madre vía dropdowns con validación de aciclicidad.
//   - Refrescar página y comprobar que todo persiste.
//
// Diseño:
//   - Una "tree activa" a la vez (selector arriba). Multi-tree existe en el
//     store pero la UI sólo expone una activa para no complicar el MVP.
//   - Dropdowns padre/madre se filtran a personas del mismo árbol; se chequea
//     wouldCreateCycle antes de asignar y se muestra error inline si rechaza.
//   - Si una persona referencia un parentId que ya no existe (porque el padre
//     fue borrado), el dropdown queda en "—" y se marca visualmente.
//   - Object URLs de las fotos se cachean en estado local y se revocan en el
//     cleanup del effect para no leakear memoria al desmontar / cambiar árbol.

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

const LAST_TREE_KEY = 'phyloface-genealogy-last-tree';

export default function GenealogyTree() {
  const [trees, setTrees] = useState<Tree[]>([]);
  const [selectedTreeId, setSelectedTreeId] = useState<string | null>(null);
  const [persons, setPersons] = useState<Person[]>([]);
  const [photoUrls, setPhotoUrls] = useState<Map<string, string>>(new Map());
  const [newPersonName, setNewPersonName] = useState('');
  const [newTreeName, setNewTreeName] = useState('');
  const [error, setError] = useState<string | null>(null);

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
  // Handlers
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
  // Render helpers
  // -------------------------------------

  const personsById = useMemo(() => {
    const m = new Map<PersonId, Person>();
    for (const p of persons) m.set(p.id, p);
    return m;
  }, [persons]);

  // -------------------------------------
  // Render
  // -------------------------------------

  return (
    <div style={{ padding: 20, fontFamily: 'sans-serif', fontSize: 14 }}>
      <h2 style={{ marginTop: 0 }}>Árbol genealógico (Track 2b — MVP paso 2)</h2>

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

      {/* Lista de personas */}
      {!selectedTreeId && (
        <p style={{ color: '#888' }}>Creá un árbol para empezar.</p>
      )}
      {selectedTreeId && persons.length === 0 && (
        <p style={{ color: '#888' }}>Sin personas todavía. Agregá la primera con el botón de arriba.</p>
      )}
      {persons.length > 0 && (
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Foto</th>
              <th style={thStyle}>Nombre</th>
              <th style={thStyle}>Padre</th>
              <th style={thStyle}>Madre</th>
              <th style={thStyle}></th>
            </tr>
          </thead>
          <tbody>
            {persons.map((p) => (
              <PersonRow
                key={p.id}
                person={p}
                persons={persons}
                personsById={personsById}
                photoUrl={p.photoSha256 ? photoUrls.get(p.photoSha256) ?? null : null}
                onSetParent={handleSetParent}
                onUploadPhoto={handleUploadPhoto}
                onDelete={handleDeletePerson}
              />
            ))}
          </tbody>
        </table>
      )}

      <p style={{ color: '#888', marginTop: 24, fontSize: 12 }}>
        Estado del paso 2: persistencia validable. Refrescá la página y todo debería seguir acá. Comparación, layout SVG y export/import vienen en pasos 4-6.
      </p>
    </div>
  );
}

// -----------------------------------------
// Fila de persona — extraída por claridad y para encapsular el file input.
// -----------------------------------------

interface PersonRowProps {
  person: Person;
  persons: Person[];
  personsById: Map<PersonId, Person>;
  photoUrl: string | null;
  onSetParent: (id: PersonId, role: 'father' | 'mother', parentId: PersonId | null) => void;
  onUploadPhoto: (id: PersonId, file: File) => void;
  onDelete: (id: PersonId) => void;
}

function PersonRow({
  person,
  persons,
  personsById,
  photoUrl,
  onSetParent,
  onUploadPhoto,
  onDelete,
}: PersonRowProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Candidatos a padre/madre: todas las personas del árbol menos la persona
  // misma. La validación final de ciclo la hace handleSetParent.
  const parentCandidates = persons.filter((p) => p.id !== person.id);

  // Helper para detectar refs colgadas (padre borrado).
  const parentLabel = (id: PersonId | null): string => {
    if (id === null) return '';
    const p = personsById.get(id);
    return p ? p.name : `(borrado: ${id.slice(0, 6)}…)`;
  };
  const fatherDangling = person.fatherId !== null && !personsById.has(person.fatherId);
  const motherDangling = person.motherId !== null && !personsById.has(person.motherId);

  return (
    <tr style={{ borderBottom: '1px solid #eee' }}>
      <td style={tdStyle}>
        <div
          onClick={() => fileInputRef.current?.click()}
          style={{
            width: 64,
            height: 64,
            border: '1px dashed #aaa',
            borderRadius: 4,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            background: '#fafafa',
            overflow: 'hidden',
          }}
          title="Click para subir foto"
        >
          {photoUrl ? (
            <img
              src={photoUrl}
              alt={person.name}
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />
          ) : (
            <span style={{ color: '#aaa', fontSize: 11 }}>+ foto</span>
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
      </td>
      <td style={tdStyle}>{person.name}</td>
      <td style={tdStyle}>
        <select
          value={person.fatherId ?? ''}
          onChange={(e) => onSetParent(person.id, 'father', e.target.value || null)}
          style={{
            minWidth: 160,
            color: fatherDangling ? '#c00' : undefined,
          }}
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
      </td>
      <td style={tdStyle}>
        <select
          value={person.motherId ?? ''}
          onChange={(e) => onSetParent(person.id, 'mother', e.target.value || null)}
          style={{
            minWidth: 160,
            color: motherDangling ? '#c00' : undefined,
          }}
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
      </td>
      <td style={tdStyle}>
        <button onClick={() => onDelete(person.id)} title="Borrar persona">
          ✕
        </button>
      </td>
    </tr>
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

const tableStyle: React.CSSProperties = {
  width: '100%',
  borderCollapse: 'collapse',
};

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  padding: '8px 6px',
  borderBottom: '2px solid #ccc',
  fontWeight: 600,
};

const tdStyle: React.CSSProperties = {
  padding: '8px 6px',
  verticalAlign: 'middle',
};
