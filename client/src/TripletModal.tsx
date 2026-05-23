// =========================================
// ID: PHYLOFACE_TRIPLET_MODAL
// VERSION: v1.0
// =========================================
// Modal flotante de detalle de comparación. Se abre al clickear el label del
// cosine entre dos nodos en el SVG del árbol (`GenealogyTree.tsx`). Lo que
// hace:
//   1. Muestra los dos rostros del par + el cosine cacheado.
//   2. Permite agregar un tercer rostro elegido entre las personas del árbol
//      con foto. Al agregar, computa los dos cosines extras (A↔C, B↔C) usando
//      el callback `ensureEmbedding` que delega al pipeline browser-only del
//      árbol (mismo `lib/pipeline.ts`).
//   3. Asigna roles inferidos del árbol con override manual. Reglas de
//      inferencia:
//        - Si exactamente uno de los 3 nodos tiene a los otros dos como
//          fatherId y motherId → ese es "Hijo/a"; los otros son Padre/Madre
//          según el campo apuntado.
//        - Sino, default: A=Padre, B=Madre, C=Hijo/a (orden de slots).
//      El usuario puede sobrescribir cada rol vía dropdown
//      (`ROLE_OPTIONS_TRIPLET`).
//   4. Botón "→ abrir en Comparador MVP": escribe
//      `localStorage["phyloface-comparator-prefill"]` con sha256 + roles de
//      los 3 (mapeados a left/child/right según quién tenga rol "Hijo/a") y
//      dispatcha `window.CustomEvent("phyloface-go-to-tab", "comparator")`.
//      `App.tsx` cambia al tab del Comparador; `Comparator.tsx` lee el
//      prefill al montar y precarga los slots.
//
// Decisiones de diseño:
//   - El modal no usa la API nativa `<dialog>`: la implementación visual es
//     un overlay manual (div + flexbox + click sobre backdrop = cerrar)
//     porque <dialog> tiene gotchas de focus/escape en Chromium headless y no
//     vale el costo para un MVP.
//   - El cómputo de los pares extras se hace **dentro del modal**, no en el
//     padre. El padre nos pasa `ensureEmbedding` (helper compartido con el
//     resto del árbol que ya cachea embeddings en IDB) y aquí solo
//     calculamos la similitud coseno.
//   - No hay "fuente de verdad" del cosineByPair del padre: si el par
//     A↔B venía con valor cacheado, se muestra; los nuevos pares (A↔C, B↔C)
//     son locales al modal y NO se inyectan back al cosineByPair del árbol.
//     Esto es intencional: el modal es una vista de detalle de un cómputo
//     "como si" el usuario hubiera seleccionado los 3 nodos en el SVG, pero
//     sin contaminar la selección activa.
//   - El select del tercero filtra personas con `photoSha256 != null` (sin
//     foto no se puede comparar) y excluye al par inicial.

import { useEffect, useMemo, useRef, useState } from 'react';
import { cosineSimilarity } from './lib/pipeline';
import type { Person, PersonId, Sha256Hex } from './lib/genealogy';

// Roles posibles en el modal. Incluye "Hijo/a" (slot central del Comparador
// MVP) además de los de los slots laterales. Si el usuario marca dos como
// "Hijo/a", el handoff toma el primero como child y el otro va a left/right.
// Tipos internos al archivo — no se exportan (el linter de react-refresh
// pide que un módulo de componentes no exporte constantes/tipos junto al
// componente; lo que necesitamos fuera está en `./lib/genealogy`).
const ROLE_OPTIONS_TRIPLET = [
  'Padre', 'Madre', 'Hijo/a',
  'Hermano', 'Hermana', 'Tío', 'Tía', 'Abuelo', 'Abuela', 'Otro',
] as const;
type TripletRole = typeof ROLE_OPTIONS_TRIPLET[number];

interface TripletModalProps {
  /** Persona A del par inicial. */
  a: Person;
  /** Persona B del par inicial. */
  b: Person;
  /** Cosine cacheado del par A↔B (viene del SVG). */
  initialCosine: number;
  /** Candidatos para tercero: todas las personas del árbol con foto, excluyendo A y B. */
  candidates: Person[];
  /** Object URLs por sha256 (creados en el componente padre). */
  photoUrls: Map<string, string>;
  personsById: Map<PersonId, Person>;
  /**
   * Devuelve el embedding ArcFace 512-d de una foto, cacheándolo en IDB.
   * El componente lo invoca para los pares A↔C y B↔C cuando se agrega un
   * tercero. La función ya existe en `GenealogyTree.tsx` para el cómputo
   * normal del árbol; la pasamos como prop para no duplicarla.
   */
  ensureEmbedding: (sha256: Sha256Hex, force?: boolean) => Promise<Float32Array>;
  onClose: () => void;
}

// Inferencia de roles dado el conjunto de hasta 3 personas. Sólo dispara la
// regla "hijo de" cuando ambos padres del candidato a Hijo están en el set.
// Sino, devuelve defaults.
function inferRoles(
  a: Person, b: Person, c: Person | null,
): { aRole: TripletRole; bRole: TripletRole; cRole: TripletRole | null } {
  if (c) {
    // ¿alguno es hijo de los otros dos?
    const isChildOf = (child: Person, p1: Person, p2: Person) =>
      (child.fatherId === p1.id && child.motherId === p2.id) ||
      (child.fatherId === p2.id && child.motherId === p1.id);
    if (isChildOf(a, b, c)) {
      return {
        aRole: 'Hijo/a',
        bRole: a.fatherId === b.id ? 'Padre' : 'Madre',
        cRole: a.fatherId === c.id ? 'Padre' : 'Madre',
      };
    }
    if (isChildOf(b, a, c)) {
      return {
        aRole: b.fatherId === a.id ? 'Padre' : 'Madre',
        bRole: 'Hijo/a',
        cRole: b.fatherId === c.id ? 'Padre' : 'Madre',
      };
    }
    if (isChildOf(c, a, b)) {
      return {
        aRole: c.fatherId === a.id ? 'Padre' : 'Madre',
        bRole: c.fatherId === b.id ? 'Padre' : 'Madre',
        cRole: 'Hijo/a',
      };
    }
    return { aRole: 'Padre', bRole: 'Madre', cRole: 'Hijo/a' };
  }
  // Sólo dos: si uno es hijo del otro, etiqueta básica; sino default.
  if (a.fatherId === b.id || a.motherId === b.id) {
    return { aRole: 'Hijo/a', bRole: a.fatherId === b.id ? 'Padre' : 'Madre', cRole: null };
  }
  if (b.fatherId === a.id || b.motherId === a.id) {
    return { aRole: a.id === b.fatherId ? 'Padre' : 'Madre', bRole: 'Hijo/a', cRole: null };
  }
  return { aRole: 'Padre', bRole: 'Madre', cRole: null };
}

export default function TripletModal({
  a, b, initialCosine, candidates, photoUrls, ensureEmbedding, onClose,
}: TripletModalProps) {
  const [thirdId, setThirdId] = useState<PersonId | null>(null);
  const third = useMemo(
    () => (thirdId ? candidates.find((p) => p.id === thirdId) ?? null : null),
    [thirdId, candidates],
  );

  // Roles inferidos + overrides locales. El useEffect resetea overrides
  // cuando cambia el tercero (re-inferir).
  const inferred = useMemo(() => inferRoles(a, b, third), [a, b, third]);
  const [aRole, setARole] = useState<TripletRole>(inferred.aRole);
  const [bRole, setBRole] = useState<TripletRole>(inferred.bRole);
  const [cRole, setCRole] = useState<TripletRole | null>(inferred.cRole);
  // Re-aplicar inferencia cuando cambia third (sin pisar overrides del usuario
  // si no cambió). Más simple: re-set siempre que cambie. Si el usuario quiere
  // override después de elegir tercero, los toca y listo.
  const lastThirdRef = useRef<PersonId | null>(null);
  useEffect(() => {
    if (lastThirdRef.current !== thirdId) {
      lastThirdRef.current = thirdId;
      setARole(inferred.aRole);
      setBRole(inferred.bRole);
      setCRole(inferred.cRole);
    }
  }, [thirdId, inferred]);

  // Cosines: A↔B viene precargado; los otros dos se computan cuando hay
  // tercero. Map keyed por 'ab' | 'ac' | 'bc'.
  const [cosines, setCosines] = useState<{ ab: number; ac: number | null; bc: number | null }>({
    ab: initialCosine, ac: null, bc: null,
  });
  const [isComputing, setIsComputing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Disparar cómputo de A↔C y B↔C cuando hay tercero. El reset al "sin
  // tercero" se hace en `removeThird()` (handler) para evitar set-state en
  // effect — la regla del linter es estricta. El check de fotos faltantes
  // también va adentro del IIFE async (no en el cuerpo síncrono del effect).
  useEffect(() => {
    if (!third) return;
    let cancelled = false;
    void (async () => {
      if (!a.photoSha256 || !b.photoSha256 || !third.photoSha256) {
        if (!cancelled) setError('Algún nodo no tiene foto cargada — no se puede comparar.');
        return;
      }
      setIsComputing(true);
      setError(null);
      try {
        const eA = await ensureEmbedding(a.photoSha256);
        const eB = await ensureEmbedding(b.photoSha256);
        const eC = await ensureEmbedding(third.photoSha256);
        if (cancelled) return;
        setCosines({
          ab: cosineSimilarity(eA, eB),
          ac: cosineSimilarity(eA, eC),
          bc: cosineSimilarity(eB, eC),
        });
      } catch (e) {
        if (!cancelled) setError(`Computando cosines: ${(e as Error).message}`);
      } finally {
        if (!cancelled) setIsComputing(false);
      }
    })();
    return () => { cancelled = true; };
  }, [third, a, b, initialCosine, ensureEmbedding]);

  const removeThird = () => {
    setThirdId(null);
    setCosines({ ab: initialCosine, ac: null, bc: null });
  };

  // ESC cierra. Click sobre el backdrop también (en el div externo).
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  // Handoff al Comparador MVP. Mapea las 3 personas a slots left/child/right
  // según rol: el que tenga 'Hijo/a' va al center, los otros dos a left/right
  // (preservando el orden original A→left, B→right si ninguno tiene Hijo/a).
  // Si dos personas tienen 'Hijo/a' (raro), la primera (A o B antes que C)
  // se respeta como child; las otras como adultos.
  const handoffToMVP = () => {
    if (!a.photoSha256 || !b.photoSha256) {
      setError('Falta foto en algún nodo.');
      return;
    }
    if (third && !third.photoSha256) {
      setError('El tercero seleccionado no tiene foto.');
      return;
    }
    type SlotKey = 'left' | 'child' | 'right';
    const entries: Array<{ person: Person; role: TripletRole | null }> = [
      { person: a, role: aRole },
      { person: b, role: bRole },
    ];
    if (third) entries.push({ person: third, role: cRole });

    const assignedSlots = new Map<SlotKey, { sha256: string; role?: string }>();

    // Primero asignar Hijo/a (slot 'child'): la primera entry con rol 'Hijo/a'.
    const childIdx = entries.findIndex((e) => e.role === 'Hijo/a');
    if (childIdx !== -1) {
      const c = entries[childIdx];
      assignedSlots.set('child', { sha256: c.person.photoSha256! });
      entries.splice(childIdx, 1);
    }
    // Sino, si hay exactamente 3 entries, falta marcar uno como Hijo/a;
    // por convención el tercero (último) va al child slot. Si hay 2 entries
    // (sin tercero), van a left/right.
    if (childIdx === -1 && entries.length === 3) {
      // mover el último al child
      const last = entries.pop()!;
      assignedSlots.set('child', { sha256: last.person.photoSha256! });
    }
    // Las restantes (1 o 2) van a left/right.
    const sideSlots: SlotKey[] = ['left', 'right'];
    for (let i = 0; i < entries.length && i < sideSlots.length; i++) {
      const e = entries[i];
      assignedSlots.set(sideSlots[i], {
        sha256: e.person.photoSha256!,
        role: e.role ?? undefined,
      });
    }

    const payload = {
      v: 1 as const,
      ts: Date.now(),
      slots: Array.from(assignedSlots, ([slot, info]) => ({
        slot,
        sha256: info.sha256,
        ...(info.role ? { role: info.role } : {}),
      })),
    };
    localStorage.setItem('phyloface-comparator-prefill', JSON.stringify(payload));
    window.dispatchEvent(new CustomEvent('phyloface-go-to-tab', { detail: 'comparator' }));
    onClose();
  };

  // -----------------------------------------
  // Render
  // -----------------------------------------
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Detalle de comparación"
      data-testid="triplet-modal"
      onClick={(e) => {
        // Click sobre backdrop cierra; click en el card interno se propaga
        // y no cierra por el e.stopPropagation del onClick interno.
        if (e.target === e.currentTarget) onClose();
      }}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0, 0, 0, 0.55)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: '#fff',
          borderRadius: 8,
          padding: 24,
          minWidth: 480,
          maxWidth: 900,
          maxHeight: '90vh',
          overflowY: 'auto',
          boxShadow: '0 10px 40px rgba(0, 0, 0, 0.3)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ margin: 0 }}>Detalle de comparación</h3>
          <button onClick={onClose} aria-label="Cerrar" style={{ fontSize: 14 }}>✕ cerrar</button>
        </div>

        {error && (
          <div style={{
            background: '#fee', color: '#900', padding: 8, borderRadius: 4,
            marginBottom: 12, border: '1px solid #fcc', fontSize: 12,
          }}>
            {error}
          </div>
        )}

        {/* Filas de slots: A · (C opcional) · B. Cada slot con foto + nombre + rol dropdown. */}
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', marginBottom: 16 }}>
          <TripletSlot
            label="A"
            person={a}
            photoUrl={a.photoSha256 ? photoUrls.get(a.photoSha256) ?? null : null}
            role={aRole}
            onRoleChange={setARole}
          />
          {third && cRole && (
            <TripletSlot
              label="C"
              person={third}
              photoUrl={third.photoSha256 ? photoUrls.get(third.photoSha256) ?? null : null}
              role={cRole}
              onRoleChange={(r) => setCRole(r)}
              onRemove={removeThird}
            />
          )}
          <TripletSlot
            label="B"
            person={b}
            photoUrl={b.photoSha256 ? photoUrls.get(b.photoSha256) ?? null : null}
            role={bRole}
            onRoleChange={setBRole}
          />
        </div>

        {/* Bloque de cosines: 1 si no hay tercero, 3 si hay. */}
        <div style={{
          background: '#fafafa', borderRadius: 4, padding: 12, marginBottom: 16,
          border: '1px solid #eee',
        }}>
          <div style={{ fontSize: 12, color: '#666', marginBottom: 6 }}>
            Cosine{third ? 's' : ''}
            {isComputing && <span style={{ marginLeft: 8 }}>⏳ computando…</span>}
          </div>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: 13 }}>
            <CosineRow label={`${a.name} ↔ ${b.name}`} value={cosines.ab} />
            {third && <CosineRow label={`${a.name} ↔ ${third.name}`} value={cosines.ac} />}
            {third && <CosineRow label={`${b.name} ↔ ${third.name}`} value={cosines.bc} />}
          </ul>
        </div>

        {/* Selector del tercero o pista cuando ya está. */}
        {!third && (
          <div style={{ marginBottom: 16, fontSize: 13 }}>
            <label>
              + agregar tercero:&nbsp;
              <select
                value=""
                onChange={(e) => setThirdId(e.target.value || null)}
                data-testid="triplet-add-third"
                style={{ minWidth: 200 }}
              >
                <option value="">— elegir persona —</option>
                {candidates.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
            </label>
            {candidates.length === 0 && (
              <span style={{ marginLeft: 8, color: '#888', fontSize: 12 }}>
                (no hay otras personas con foto en este árbol)
              </span>
            )}
          </div>
        )}

        {/* Botón handoff al Comparador MVP. */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button onClick={onClose} style={{ fontSize: 13 }}>
            cerrar
          </button>
          <button
            onClick={handoffToMVP}
            disabled={isComputing}
            data-testid="triplet-handoff-mvp"
            style={{
              fontSize: 13,
              fontWeight: 600,
              padding: '6px 12px',
              background: '#1a73e8',
              color: '#fff',
              border: 'none',
              borderRadius: 4,
              cursor: isComputing ? 'wait' : 'pointer',
            }}
          >
            → abrir en Comparador MVP
          </button>
        </div>
      </div>
    </div>
  );
}

// -----------------------------------------
// Subcomponentes
// -----------------------------------------

interface TripletSlotProps {
  label: string;
  person: Person;
  photoUrl: string | null;
  role: TripletRole | null;
  onRoleChange: (r: TripletRole) => void;
  onRemove?: () => void;
}

function TripletSlot({ label, person, photoUrl, role, onRoleChange, onRemove }: TripletSlotProps) {
  return (
    <div style={{
      flex: 1, textAlign: 'center', minWidth: 120,
      padding: 8, border: '1px solid #ddd', borderRadius: 4, background: '#fff',
    }}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 4, display: 'flex', justifyContent: 'space-between' }}>
        <span>{label}</span>
        {onRemove && (
          <button
            onClick={onRemove}
            style={{ fontSize: 10, padding: '0 6px' }}
            title="Quitar tercero"
          >✕</button>
        )}
      </div>
      <div style={{
        width: 120, height: 120, margin: '0 auto 6px', borderRadius: 4,
        background: '#fafafa', overflow: 'hidden', display: 'flex',
        alignItems: 'center', justifyContent: 'center', border: '1px solid #eee',
      }}>
        {photoUrl ? (
          <img src={photoUrl} alt={person.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        ) : (
          <span style={{ color: '#aaa', fontSize: 12 }}>sin foto</span>
        )}
      </div>
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{person.name}</div>
      <select
        value={role ?? ''}
        onChange={(e) => onRoleChange(e.target.value as TripletRole)}
        style={{ fontSize: 12, width: '100%' }}
      >
        {ROLE_OPTIONS_TRIPLET.map((r) => (
          <option key={r} value={r}>{r}</option>
        ))}
      </select>
    </div>
  );
}

interface CosineRowProps {
  label: string;
  value: number | null;
}

function CosineRow({ label, value }: CosineRowProps) {
  return (
    <li style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '3px 0', borderBottom: '1px dashed #eee',
    }}>
      <span>{label}</span>
      <span style={{ fontFamily: 'monospace', fontWeight: 700 }} data-testid="cosine-modal-value">
        {value === null ? '…' : value.toFixed(4)}
      </span>
    </li>
  );
}
