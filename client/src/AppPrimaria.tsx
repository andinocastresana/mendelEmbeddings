// =========================================
// ID: PHYLOFACE_APP_PRIMARIA
// VERSION: v1.0
// =========================================
// App primaria — "¿A quién se parece?" (Tarea #12, el objetivo final del proyecto,
// ver ARQUITECTURA.md §2.1). Superficie-producto niño-céntrica: tres fotos (Padre ·
// Hijo/a · Madre) → un VEREDICTO interpretable arriba + el desglose visual
// (radar/heatmap/barras por región) abajo.
//
// Qué la distingue del Comparador (MVP): el Comparador es una herramienta genérica
// y flexible (roles configurables, vínculo con el árbol, spikes al lado). Esta es
// la lectura directa del producto: combina el parecido GLOBAL (coseno + posterior
// calibrado de la Tarea #6) con el REGIONAL (reparto P↔M por zona facial) en un
// resumen que responde "cuánto y por qué se parece a cada uno".
//
// Reúso máximo: NO reimplementa motor. El pipeline e2e es `lib/pipeline.ts` (igual
// que el Comparador); el desglose visual es `RegionalScoresPanel` (Tarea #30); la
// calibración es `lib/calibration.ts` (Tarea #6); la síntesis es `lib/verdict.ts`.
// El panel se auto-computa en geométrico (barato, sin GPU) y emite sus scores vía
// `onResults` → el veredicto regional queda sincronizado con lo que muestran las
// barras/radar. Occlusion sigue siendo opt-in dentro del panel.
//
// Privacidad: idéntica al Comparador — las imágenes NUNCA salen del browser
// (inference 100% client-side, sin upload). Sin persistencia: esta vista es
// anónima (el árbol/tripleta es responsabilidad del Comparador/Árbol).

import { useCallback, useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web';
import { FaceLandmarker } from '@mediapipe/tasks-vision';
import {
  computeEmbedding,
  cosineSimilarity,
  initFaceLandmarker,
  initOnnxSession,
  loadImage,
  type PipelineOutput,
} from './lib/pipeline';
import {
  loadCalibration, scoreValue, calibrationWarning,
  type ValueScore,
} from './lib/calibration';
import { getScorer, type RegionalMethod, type RegionalScoresResult } from './lib/regionalScores';
import {
  buildGlobalVerdict, buildRegionalVerdict,
  type GlobalVerdict, type RegionalVerdict, type Side,
} from './lib/verdict';
import RegionalScoresPanel, { type RegionalPanelParent } from './RegionalScoresPanel';

// -----------------------------------------
// Colores coherentes con RegionalScoresPanel (Padre=izquierda azul, Madre=derecha verde).
// -----------------------------------------
const PADRE_COLOR = '#2563eb';
const MADRE_COLOR = '#16a34a';

type SlotKey = 'padre' | 'child' | 'madre';

// El Padre va al slot 'left' del panel; la Madre al 'right'. El veredicto usa
// 'left'/'right' y se traducen acá a las etiquetas Padre/Madre.
const SLOT_TO_SIDE: Record<'padre' | 'madre', Side> = { padre: 'left', madre: 'right' };
const SIDE_LABEL: Record<Side, string> = { left: 'Padre', right: 'Madre' };
const SIDE_COLOR: Record<Side, string> = { left: PADRE_COLOR, right: MADRE_COLOR };

interface SlotState {
  file: File | null;
  previewUrl: string | null;
  result: PipelineOutput | null;
  error: string | null;
  isDraggingOver: boolean;
}

const EMPTY_SLOT: SlotState = {
  file: null, previewUrl: null, result: null, error: null, isDraggingOver: false,
};

const SLOT_META: Record<SlotKey, { label: string; color: string; hint: string }> = {
  padre: { label: 'Padre', color: PADRE_COLOR, hint: 'progenitor 1' },
  child: { label: 'Hijo/a', color: '#444', hint: 'el niño/a a analizar' },
  madre: { label: 'Madre', color: MADRE_COLOR, hint: 'progenitor 2' },
};

export default function AppPrimaria() {
  const [slots, setSlots] = useState<Record<SlotKey, SlotState>>({
    padre: { ...EMPTY_SLOT }, child: { ...EMPTY_SLOT }, madre: { ...EMPTY_SLOT },
  });
  const [running, setRunning] = useState(false);
  const [initStatus, setInitStatus] = useState<'idle' | 'initializing' | 'ready'>('idle');
  const [globalError, setGlobalError] = useState<string | null>(null);

  // Veredicto: parte GLOBAL (la arma esta página) + parte REGIONAL (llega del
  // panel vía onResults). Se resetean al re-analizar.
  const [globalVerdict, setGlobalVerdict] = useState<GlobalVerdict | null>(null);
  const [regionalVerdict, setRegionalVerdict] = useState<RegionalVerdict | null>(null);
  const [calWarning, setCalWarning] = useState<string | null>(null);
  const [calError, setCalError] = useState<string | null>(null);

  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  // Espejo en estado de la sesión ONNX para pasarla al panel sin leer el ref en
  // render (occlusion la usa; geométrico no la necesita). Se setea tras init.
  const [session, setSession] = useState<ort.InferenceSession | null>(null);

  // ---------------------------------------
  // Manejo de slots.
  // ---------------------------------------
  const updateSlot = (key: SlotKey, patch: Partial<SlotState>) =>
    setSlots((prev) => ({ ...prev, [key]: { ...prev[key], ...patch } }));

  const resetVerdict = () => {
    setGlobalVerdict(null);
    setRegionalVerdict(null);
    setGlobalError(null);
  };

  const onPickFile = (key: SlotKey, file: File | null) => {
    setSlots((prev) => {
      const old = prev[key];
      if (old.previewUrl) URL.revokeObjectURL(old.previewUrl);
      return {
        ...prev,
        [key]: {
          file,
          previewUrl: file ? URL.createObjectURL(file) : null,
          result: null, error: null, isDraggingOver: false,
        },
      };
    });
    resetVerdict();
  };

  // ---------------------------------------
  // Drag-and-drop (solo dataTransfer.files; no resuelve URLs externas — privacidad).
  // ---------------------------------------
  const onDragOver = (key: SlotKey) => (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!slots[key].isDraggingOver) updateSlot(key, { isDraggingOver: true });
  };
  const onDragLeave = (key: SlotKey) => (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    updateSlot(key, { isDraggingOver: false });
  };
  const onDrop = (key: SlotKey) => (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    updateSlot(key, { isDraggingOver: false });
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      updateSlot(key, { error: `Tipo no soportado: ${file.type || 'desconocido'}` });
      return;
    }
    onPickFile(key, file);
  };

  // ---------------------------------------
  // Cleanup GPU/WASM al desmontar (mismo patrón que Comparator, Tarea #27).
  // ---------------------------------------
  useEffect(() => {
    return () => {
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
      const sess = sessionRef.current;
      sessionRef.current = null;
      if (sess) void sess.release().catch((e) => console.warn('[AppPrimaria] session.release falló:', e));
    };
  }, []);

  const ensureInit = async () => {
    if (landmarkerRef.current && sessionRef.current) return;
    setInitStatus('initializing');
    const [lm, sess] = await Promise.all([
      landmarkerRef.current ? Promise.resolve(landmarkerRef.current) : initFaceLandmarker(),
      sessionRef.current ? Promise.resolve(sessionRef.current) : initOnnxSession(),
    ]);
    landmarkerRef.current = lm;
    sessionRef.current = sess;
    setSession(sess);
    setInitStatus('ready');
  };

  const processSlot = async (key: SlotKey): Promise<PipelineOutput | null> => {
    const state = slots[key];
    if (!state.file || !state.previewUrl) return null;
    try {
      const { img, imageData } = await loadImage(state.previewUrl);
      const out = await computeEmbedding(img, imageData, landmarkerRef.current!, sessionRef.current!);
      updateSlot(key, { result: out, error: null });
      return out;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      updateSlot(key, { result: null, error: msg });
      console.error(`[AppPrimaria] slot ${key} falló:`, e);
      return null;
    }
  };

  // ---------------------------------------
  // Analizar: corre el pipeline de los slots cargados, calcula los cosines
  // Hijo↔Padre / Hijo↔Madre, los califica (posterior calibrado, relación 'ALL'
  // porque no se conoce el sexo del Hijo/a) y arma el veredicto GLOBAL. El
  // regional llega solo cuando el panel auto-computa y emite onResults.
  // ---------------------------------------
  const onAnalyze = async () => {
    if (!canAnalyze) return;
    setRunning(true);
    resetVerdict();
    try {
      await ensureInit();
      const outChild = await processSlot('child');
      const outPadre = slots.padre.file ? await processSlot('padre') : null;
      const outMadre = slots.madre.file ? await processSlot('madre') : null;
      if (!outChild) {
        setGlobalError('No se pudo procesar la cara del Hijo/a.');
        return;
      }
      const cosPadre = outPadre ? cosineSimilarity(outChild.embedding, outPadre.embedding) : undefined;
      const cosMadre = outMadre ? cosineSimilarity(outChild.embedding, outMadre.embedding) : undefined;

      // Calibración (Tarea #6): cosine → posterior P(parentesco). Si falla la
      // carga, el veredicto sigue mostrando los cosines crudos.
      let scorePadre: ValueScore | undefined;
      let scoreMadre: ValueScore | undefined;
      try {
        const cal = await loadCalibration('KinFaceW-I');
        setCalWarning(calibrationWarning(cal));
        setCalError(null);
        if (cosPadre != null) scorePadre = scoreValue(cal, 'cosine', 'ALL', cosPadre);
        if (cosMadre != null) scoreMadre = scoreValue(cal, 'cosine', 'ALL', cosMadre);
      } catch (e) {
        setCalError(e instanceof Error ? e.message : String(e));
      }

      setGlobalVerdict(buildGlobalVerdict(cosPadre, cosMadre, scorePadre, scoreMadre));
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  // Necesita el Hijo/a + al menos un progenitor.
  const canAnalyze = !!slots.child.file && (!!slots.padre.file || !!slots.madre.file) && !running;

  // Resultados listos para el panel/veredicto.
  const childResult = slots.child.result;
  const padreResult = slots.padre.result;
  const madreResult = slots.madre.result;

  // onResults estable: arma el veredicto regional desde los scores del panel.
  const onRegionalResults = useCallback(
    (method: RegionalMethod, bySide: Partial<Record<Side, RegionalScoresResult>>) => {
      const scorer = getScorer(method);
      setRegionalVerdict(
        buildRegionalVerdict(bySide, {
          method,
          methodLabel: scorer?.label ?? method,
          confidence: scorer?.baseConfidence ?? 'experimental',
        }),
      );
    },
    [],
  );

  const panelParents: RegionalPanelParent[] = [
    padreResult ? { side: SLOT_TO_SIDE.padre, label: 'Padre', result: padreResult } : null,
    madreResult ? { side: SLOT_TO_SIDE.madre, label: 'Madre', result: madreResult } : null,
  ].filter(Boolean) as RegionalPanelParent[];

  return (
    <div style={{ fontFamily: 'monospace', padding: 20, maxWidth: 1100, margin: '0 auto' }}>
      <h1 style={{ borderBottom: '2px solid #333', paddingBottom: 8 }}>
        App primaria — ¿A quién se parece?
      </h1>
      <p style={{ color: '#666', fontSize: 13, lineHeight: 1.5 }}>
        Cargá tres fotos (una cara cada una): el <strong>Hijo/a</strong> en el centro
        y sus dos progenitores a los lados. El análisis corre <strong>100% en tu
        navegador</strong> — las imágenes nunca salen de tu equipo. Devuelve un
        veredicto del parecido <strong>global</strong> (cara completa) y por
        <strong> región</strong> (qué heredó de cada uno).
      </p>

      {/* Tres slots: Padre · Hijo/a · Madre */}
      <div style={{ display: 'flex', gap: 16, marginTop: 16, flexWrap: 'wrap' }}>
        {(['padre', 'child', 'madre'] as const).map((key) => (
          <FaceSlot
            key={key}
            meta={SLOT_META[key]}
            slot={slots[key]}
            onPick={(f) => onPickFile(key, f)}
            onDragOver={onDragOver(key)}
            onDragLeave={onDragLeave(key)}
            onDrop={onDrop(key)}
          />
        ))}
      </div>

      {/* Botón analizar */}
      <div style={{ marginTop: 18 }}>
        <button
          onClick={() => void onAnalyze()}
          disabled={!canAnalyze}
          style={{
            padding: '12px 28px', fontFamily: 'monospace', fontSize: 15, fontWeight: 700,
            cursor: canAnalyze ? 'pointer' : 'not-allowed',
            background: canAnalyze ? '#7c3aed' : '#ccc', color: '#fff',
            border: 'none', borderRadius: 6,
          }}
        >
          {running
            ? (initStatus === 'initializing' ? 'Inicializando modelos…' : 'Analizando…')
            : 'Analizar parecido'}
        </button>
        {!canAnalyze && !running && (
          <span style={{ marginLeft: 12, color: '#999', fontSize: 12 }}>
            Cargá el Hijo/a + al menos un progenitor.
          </span>
        )}
      </div>

      {globalError && (
        <div style={{ background: '#fee', color: '#900', padding: 12, borderRadius: 4, marginTop: 12 }}>
          <strong>Error:</strong> {globalError}
        </div>
      )}

      {/* VEREDICTO */}
      {globalVerdict && (
        <VerdictPanel
          global={globalVerdict}
          regional={regionalVerdict}
          hasCalibration={!calError}
          calWarning={calWarning}
        />
      )}

      {/* Desglose visual reutilizando el panel de scores por región (#30).
          Auto-computa geométrico y emite onResults → alimenta el veredicto. */}
      {childResult && (padreResult || madreResult) && (
        <RegionalScoresPanel
          child={childResult}
          parents={panelParents}
          session={session}
          busy={running}
          autoCompute="geometric"
          onResults={onRegionalResults}
        />
      )}
    </div>
  );
}

// =========================================================
// Veredicto: lectura directa global + regional.
// =========================================================
function VerdictPanel({ global, regional, hasCalibration, calWarning }: {
  global: GlobalVerdict;
  regional: RegionalVerdict | null;
  hasCalibration: boolean;
  calWarning: string | null;
}) {
  const sides: Side[] = ['left', 'right'];
  const present = sides.filter((s) => global.cosine[s] != null);
  const both = present.length === 2;

  const headline = (() => {
    if (!both) {
      const s = present[0];
      return s ? `Comparación con ${SIDE_LABEL[s]}` : 'Sin datos';
    }
    if (global.winner === 'tie') return 'Se parece de forma pareja a ambos';
    return `Se parece más a ${SIDE_LABEL[global.winner === 'left' ? 'left' : 'right']}`;
  })();

  const pct = (x: number | undefined) => (x == null ? '—' : `${Math.round(x * 100)}%`);

  return (
    <div style={{
      border: `2px solid ${both && global.winner !== 'tie' ? SIDE_COLOR[global.winner] : '#999'}`,
      borderRadius: 12, padding: 18, marginTop: 20, background: '#fcfcff',
    }}>
      <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>{headline}</div>
      <div style={{ fontSize: 12, color: '#888', marginBottom: 14 }}>
        Veredicto global por cara completa (coseno de embeddings){hasCalibration ? ' + probabilidad calibrada de parentesco (KinFaceW-I)' : ''}.
      </div>

      {/* Filas global por lado: cosine + posterior + barra */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 16 }}>
        {present.map((s) => {
          const cos = global.cosine[s];
          const post = global.posterior[s];
          const isWinner = both && global.winner === s;
          return (
            <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13 }}>
              <span style={{ width: 64, fontWeight: 700, color: SIDE_COLOR[s] }}>{SIDE_LABEL[s]}</span>
              <div style={{ flex: 1, maxWidth: 360, background: '#eee', borderRadius: 4, height: 16, position: 'relative', overflow: 'hidden' }}>
                <div style={{
                  position: 'absolute', left: 0, top: 0, bottom: 0,
                  width: `${Math.max(0, Math.min(1, post ?? (cos ?? 0))) * 100}%`,
                  background: SIDE_COLOR[s], opacity: 0.85,
                }} />
              </div>
              <span style={{ width: 92, color: '#333' }}>cos {cos != null ? cos.toFixed(4) : '—'}</span>
              {hasCalibration && (
                <span style={{ width: 130, color: '#555' }}>
                  parentesco {pct(post)}{isWinner ? '' : ''}
                  {global.isKin[s] != null && (
                    <span style={{ color: global.isKin[s] ? '#16a34a' : '#b45309', marginLeft: 6 }}>
                      {global.isKin[s] ? '✓' : '·'}
                    </span>
                  )}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Desglose regional: qué heredó de cada uno */}
      <div style={{ borderTop: '1px solid #e5e5e5', paddingTop: 12 }}>
        <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 8 }}>¿Por qué? — herencia por región</div>
        {!regional ? (
          <div style={{ color: '#999', fontSize: 13 }}>Analizando regiones…</div>
        ) : (
          <>
            <InheritedRow label="Padre" color={PADRE_COLOR} groups={regional.inheritedLeft.map((g) => g.label)} />
            <InheritedRow label="Madre" color={MADRE_COLOR} groups={regional.inheritedRight.map((g) => g.label)} />
            {regional.balanced.length > 0 && (
              <InheritedRow label="Equilibrado" color="#888" groups={regional.balanced.map((g) => g.label)} />
            )}
            <div style={{ fontSize: 11, color: '#999', marginTop: 8 }}>
              Según <strong>{regional.methodLabel}</strong> (confiabilidad {regional.confidence}).
              El reparto por región se muestra abajo (radar / heatmap / barras); cambiá el método ahí
              para recalcular el veredicto.
            </div>
          </>
        )}
      </div>

      {calWarning && (
        <div style={{ marginTop: 12, fontSize: 11, color: '#b45309', background: '#fff7ed', padding: 8, borderRadius: 4 }}>
          ⚠ {calWarning}
        </div>
      )}
    </div>
  );
}

function InheritedRow({ label, color, groups }: { label: string; color: string; groups: string[] }) {
  return (
    <div style={{ display: 'flex', gap: 8, fontSize: 13, marginBottom: 4, alignItems: 'baseline' }}>
      <span style={{ width: 96, fontWeight: 700, color }}>Heredó de {label}:</span>
      <span style={{ color: '#333' }}>
        {groups.length > 0 ? groups.join(', ') : <span style={{ color: '#bbb' }}>—</span>}
      </span>
    </div>
  );
}

// =========================================================
// Slot de carga (drag-drop + file). Sin roles ni vínculo: la App primaria tiene
// roles fijos (Padre · Hijo/a · Madre). El preview de la cara alineada vive en el
// panel de abajo, así que acá solo mostramos la foto original.
// =========================================================
function FaceSlot({ meta, slot, onPick, onDragOver, onDragLeave, onDrop }: {
  meta: { label: string; color: string; hint: string };
  slot: SlotState;
  onPick: (file: File | null) => void;
  onDragOver: (e: React.DragEvent<HTMLDivElement>) => void;
  onDragLeave: (e: React.DragEvent<HTMLDivElement>) => void;
  onDrop: (e: React.DragEvent<HTMLDivElement>) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const handleRemove = () => {
    if (fileInputRef.current) fileInputRef.current.value = '';
    onPick(null);
  };
  const dragStyle: React.CSSProperties = slot.isDraggingOver
    ? { background: '#eaf3ff', borderColor: meta.color, borderStyle: 'dashed' } : {};

  return (
    <div
      onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
      style={{
        flex: 1, minWidth: 240, border: `1px solid #ccc`, borderTop: `4px solid ${meta.color}`,
        borderRadius: 6, padding: 12, background: '#fafafa',
        transition: 'background 80ms, border-color 80ms', ...dragStyle,
      }}
    >
      <div style={{ fontWeight: 700, color: meta.color }}>{meta.label}</div>
      <div style={{ fontSize: 11, color: '#999', marginBottom: 8 }}>{meta.hint}</div>

      {!slot.previewUrl && (
        <div style={{
          border: '2px dashed #bbb', borderRadius: 4, padding: '24px 12px', textAlign: 'center',
          fontSize: 12, color: '#777', marginBottom: 8, background: slot.isDraggingOver ? '#dceaff' : '#fff',
        }}>
          arrastrá una imagen acá<br />
          <span style={{ fontSize: 10, color: '#999' }}>o usá el botón ↓</span>
        </div>
      )}

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
        <input
          ref={fileInputRef} type="file" accept="image/*"
          onChange={(e) => onPick(e.target.files?.[0] ?? null)}
          style={{ fontFamily: 'monospace', fontSize: 12 }}
        />
        {slot.previewUrl && (
          <button
            type="button" onClick={handleRemove} title="Quitar imagen"
            style={{
              fontFamily: 'monospace', fontSize: 11, padding: '3px 8px', cursor: 'pointer',
              background: '#fff', color: '#900', border: '1px solid #c99', borderRadius: 3,
            }}
          >✕ Quitar</button>
        )}
      </div>

      {slot.previewUrl && (
        <img
          src={slot.previewUrl} alt={meta.label}
          style={{ maxWidth: '100%', maxHeight: 220, display: 'block', border: '1px solid #ddd' }}
        />
      )}
      {slot.error && (
        <div style={{ marginTop: 10, color: '#900', fontSize: 12, background: '#fee', padding: 8, borderRadius: 3 }}>
          {slot.error}
        </div>
      )}
    </div>
  );
}
