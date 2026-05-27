// =========================================
// ID: PHYLOFACE_APP_PRIMARIA
// VERSION: v1.4
// =========================================
// App primaria — "¿A quién se parece?" (Tarea #12, el objetivo final del proyecto,
// ver ARQUITECTURA.md §2.1). Superficie-producto niño-céntrica: tres fotos (Padre ·
// Hijo/a · Madre) → un VEREDICTO interpretable + el desglose visual por región.
//
// Cambio v1.3 → v1.4 (#31, ajuste pedido por el usuario): el PDF ahora RASTERIZA el
// DOM real del informe (lib/pdfReport v2.0 con html2canvas) para que coincida con lo
// que se ve en pantalla (caras, barras, radar, heatmap, herencia), en vez de un
// layout A4 re-dibujado a mano. Los controles no esenciales para un documento
// estático se marcan con `data-pdf-exclude` y se omiten. El botón "📄 Descargar PDF"
// se movió a la esquina SUPERIOR IZQUIERDA del recuadro (espeja al "🗑️ Limpiar" de
// la derecha). Sigue 100% client-side: las imágenes nunca salen del browser.
//
// Cambio v1.2 → v1.3 (#31): botón "📄 Descargar PDF" en el veredicto → genera el
// informe (caras + global + herencia por región del método mostrado) con jsPDF,
// 100% client-side (lib/pdfReport). Las imágenes nunca salen del browser.
//
// Cambio v1.1 → v1.2 (persistencia local pedida por el usuario): el estado se
// guarda en IndexedDB LOCAL (lib/primariaStore) — fotos + PipelineOutput por slot.
// Al recargar se restauran las fotos + el veredicto GLOBAL (de los embeddings,
// sin inferencia) y el regional GEOMÉTRICO (de los landmarks, sin inferencia);
// occlusion necesita re-analizar (re-init de la sesión ONNX). Botón "Limpiar"
// borra el guardado. Sigue siendo 100% local: las imágenes nunca salen del browser.
//
// Cambio v1.0 → v1.1 (ajustes de apariencia pedidos por el usuario):
//   - Recuadro 1: las 3 fotos originales + el veredicto GLOBAL ("se parece más a X"
//     + coseno/posterior por lado) en un MISMO recuadro (fotos arriba, global
//     debajo). Antes el global vivía en un VerdictPanel separado.
//   - La herencia por región se mudó al recuadro de "Scores por región"
//     (RegionalScoresPanel con prop `showInheritance`), con los dos métodos en
//     SOLAPAS (`methodSelector="tabs"`). El veredicto regional ya no se arma acá.
//
// Qué la distingue del Comparador (MVP): el Comparador es genérico y flexible
// (roles configurables, vínculo con árbol, spikes al lado). Esta es la lectura
// directa del producto. Reúso máximo: NO reimplementa motor. Pipeline e2e =
// lib/pipeline; desglose = RegionalScoresPanel (#30); calibración = lib/calibration
// (#6). Privacidad: imágenes NUNCA salen del browser; sin persistencia.

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
import { buildGlobalVerdict, type GlobalVerdict, type Side } from './lib/verdict';
import {
  savePrimariaState, loadPrimariaState, clearPrimariaState,
  type RestoredSlot, type RegionalCache,
} from './lib/primariaStore';
import { type RegionalMethod, type RegionalScoresResult } from './lib/regionalScores';
import { generateReportPdf } from './lib/pdfReport';
import RegionalScoresPanel, { type RegionalPanelParent } from './RegionalScoresPanel';

// Colores coherentes con RegionalScoresPanel (Padre=izquierda azul, Madre=derecha verde).
const PADRE_COLOR = '#2563eb';
const MADRE_COLOR = '#16a34a';

type SlotKey = 'padre' | 'child' | 'madre';

// El Padre va al slot 'left' del panel; la Madre al 'right'. El veredicto usa
// 'left'/'right' y se traducen acá a Padre/Madre.
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

  // Veredicto GLOBAL (el regional vive ahora dentro del panel). Se resetea al re-analizar.
  const [globalVerdict, setGlobalVerdict] = useState<GlobalVerdict | null>(null);
  const [calWarning, setCalWarning] = useState<string | null>(null);
  const [calError, setCalError] = useState<string | null>(null);

  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  // Espejo en estado de la sesión ONNX para pasarla al panel sin leer el ref en
  // render (occlusion la usa; geométrico no la necesita). Se setea tras init.
  const [session, setSession] = useState<ort.InferenceSession | null>(null);

  // Persistencia de scores regionales (occlusion + geométrico) para sobrevivir
  // recargas: `regionalRef` acumula lo que emite el panel; `restoredRegional`
  // siembra el panel al restaurar; `slotsRef` espeja slots para persistir sin
  // depender del timing de setState.
  const [restoredRegional, setRestoredRegional] = useState<RegionalCache>({});
  const regionalRef = useRef<RegionalCache>({});
  const slotsRef = useRef(slots);
  useEffect(() => { slotsRef.current = slots; });

  // Raíz del informe (lo que se rasteriza al PDF) + flag de "generando" para el botón.
  const reportRef = useRef<HTMLDivElement>(null);
  const [generatingPdf, setGeneratingPdf] = useState(false);

  // ---------------------------------------
  // Manejo de slots.
  // ---------------------------------------
  const updateSlot = (key: SlotKey, patch: Partial<SlotState>) =>
    setSlots((prev) => ({ ...prev, [key]: { ...prev[key], ...patch } }));

  const resetVerdict = () => {
    setGlobalVerdict(null);
    setGlobalError(null);
  };

  // Persiste el estado completo (slots actuales + cache regional) leyendo de refs,
  // sin depender del timing de setState.
  const persistAll = useCallback(() => {
    const s = slotsRef.current;
    const toPersist: Partial<Record<SlotKey, RestoredSlot>> = {};
    (['padre', 'child', 'madre'] as const).forEach((k) => {
      if (s[k].file) toPersist[k] = { file: s[k].file!, result: s[k].result ?? undefined };
    });
    void savePrimariaState(toPersist, regionalRef.current);
  }, []);

  // El panel emite sus scores (geométrico tras auto-cómputo, occlusion tras
  // "Calcular"); los acumulamos y re-persistimos para que sobrevivan recargas.
  const onRegionalResults = useCallback(
    (method: RegionalMethod, bySide: Partial<Record<Side, RegionalScoresResult>>) => {
      regionalRef.current = { ...regionalRef.current, [method]: bySide };
      persistAll();
    },
    [persistAll],
  );

  // Descarga el informe en PDF rasterizando el DOM real del informe (lib/pdfReport
  // v2.0). 100% client-side: html2canvas trabaja sobre el DOM local, las imágenes
  // no salen del browser. Lo no esencial (botones, inputs, solapas) se omite vía
  // `data-pdf-exclude` en el JSX.
  const handleDownloadPdf = async () => {
    if (!reportRef.current || generatingPdf) return;
    setGeneratingPdf(true);
    try {
      await generateReportPdf(reportRef.current);
    } catch (e) {
      console.error('[AppPrimaria] generar PDF falló:', e);
      setGlobalError(e instanceof Error ? `No se pudo generar el PDF: ${e.message}` : 'No se pudo generar el PDF.');
    } finally {
      setGeneratingPdf(false);
    }
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
    regionalRef.current = {}; // cambiar una foto invalida los scores regionales previos
    setRestoredRegional({});  // y la siembra restaurada (era de otras caras)
    // Persistir solo las IMÁGENES (los resultados se re-guardan al volver a analizar).
    const files: Record<SlotKey, File | null> = {
      padre: slots.padre.file, child: slots.child.file, madre: slots.madre.file,
    };
    files[key] = file;
    const toPersist: Partial<Record<SlotKey, RestoredSlot>> = {};
    (['padre', 'child', 'madre'] as const).forEach((k) => { if (files[k]) toPersist[k] = { file: files[k]! }; });
    void savePrimariaState(toPersist, {});
  };

  // Limpiar todo: borra fotos, veredicto y el guardado local.
  const handleClear = () => {
    setSlots((prev) => {
      (['padre', 'child', 'madre'] as const).forEach((k) => {
        if (prev[k].previewUrl) URL.revokeObjectURL(prev[k].previewUrl);
      });
      return { padre: { ...EMPTY_SLOT }, child: { ...EMPTY_SLOT }, madre: { ...EMPTY_SLOT } };
    });
    resetVerdict();
    regionalRef.current = {};
    setRestoredRegional({});
    void clearPrimariaState();
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

  // Arma el veredicto GLOBAL desde los PipelineOutput (coseno + posterior
  // calibrado #6). Usado por onAnalyze y por la restauración al montar.
  const computeGlobalVerdict = async (
    cr: PipelineOutput, pr: PipelineOutput | null, mr: PipelineOutput | null,
  ) => {
    const cosPadre = pr ? cosineSimilarity(cr.embedding, pr.embedding) : undefined;
    const cosMadre = mr ? cosineSimilarity(cr.embedding, mr.embedding) : undefined;
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
  };

  // Restaurar estado persistido al montar (lib/primariaStore). Restaura fotos +
  // resultados; rearma el veredicto global (de embeddings, sin inferencia). El
  // panel rehace el regional GEOMÉTRICO de los landmarks (tampoco infiere).
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const restored = await loadPrimariaState();
      if (!restored || cancelled) return;
      const next: Record<SlotKey, SlotState> = {
        padre: { ...EMPTY_SLOT }, child: { ...EMPTY_SLOT }, madre: { ...EMPTY_SLOT },
      };
      (['padre', 'child', 'madre'] as const).forEach((k) => {
        const r = restored.slots[k];
        if (r) next[k] = {
          file: r.file, previewUrl: URL.createObjectURL(r.file),
          result: r.result ?? null, error: null, isDraggingOver: false,
        };
      });
      if (cancelled) return;
      // Sembrar la cache regional ANTES de montar el panel (mismo batch que setSlots).
      regionalRef.current = restored.regional;
      setRestoredRegional(restored.regional);
      setSlots(next);
      if (next.child.result && (next.padre.result || next.madre.result)) {
        await computeGlobalVerdict(next.child.result, next.padre.result, next.madre.result);
        // Init de la sesión ONNX en BACKGROUND (solo onnx, sin landmarker ni
        // inferencia) para habilitar la solapa Occlusion tras la recarga. No
        // bloquea el display (global + geométrico no la necesitan) ni calienta
        // (crear la sesión no infiere). Si Analizar ya creó una, se descarta esta.
        void initOnnxSession().then((s) => {
          if (cancelled || sessionRef.current) { void s.release().catch(() => {}); return; }
          sessionRef.current = s;
          setSession(s);
        }).catch((e) => console.warn('[AppPrimaria] init ONNX en background falló:', e));
      }
    })();
    return () => { cancelled = true; };
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
  // regional lo arma el panel solo (auto-computa geométrico).
  // ---------------------------------------
  const onAnalyze = async () => {
    if (!canAnalyze) return;
    setRunning(true);
    resetVerdict();
    regionalRef.current = {}; // nuevo análisis → los scores regionales previos no valen
    setRestoredRegional({});  // y la siembra restaurada
    try {
      await ensureInit();
      const outChild = await processSlot('child');
      const outPadre = slots.padre.file ? await processSlot('padre') : null;
      const outMadre = slots.madre.file ? await processSlot('madre') : null;
      if (!outChild) {
        setGlobalError('No se pudo procesar la cara del Hijo/a.');
        return;
      }
      await computeGlobalVerdict(outChild, outPadre, outMadre);

      // Persistir el estado COMPLETO (fotos + resultados) para sobrevivir recargas.
      const toPersist: Partial<Record<SlotKey, RestoredSlot>> = {
        child: { file: slots.child.file!, result: outChild },
      };
      if (slots.padre.file) toPersist.padre = { file: slots.padre.file, result: outPadre ?? undefined };
      if (slots.madre.file) toPersist.madre = { file: slots.madre.file, result: outMadre ?? undefined };
      void savePrimariaState(toPersist, regionalRef.current);
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  // Necesita el Hijo/a + al menos un progenitor.
  const canAnalyze = !!slots.child.file && (!!slots.padre.file || !!slots.madre.file) && !running;

  const childResult = slots.child.result;
  const padreResult = slots.padre.result;
  const madreResult = slots.madre.result;

  const panelParents: RegionalPanelParent[] = [
    padreResult ? { side: SLOT_TO_SIDE.padre, label: 'Padre', result: padreResult } : null,
    madreResult ? { side: SLOT_TO_SIDE.madre, label: 'Madre', result: madreResult } : null,
  ].filter(Boolean) as RegionalPanelParent[];

  // Color del borde del recuadro principal: tinte del ganador global (si hay).
  const boxBorder = globalVerdict && globalVerdict.winner !== 'tie' && globalVerdict.cosine.left != null && globalVerdict.cosine.right != null
    ? SIDE_COLOR[globalVerdict.winner]
    : '#ccc';

  return (
    <div ref={reportRef} style={{ fontFamily: 'monospace', padding: 20, maxWidth: 1100, margin: '0 auto' }}>
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

      {/* RECUADRO 1: fotos originales (arriba) + veredicto global (debajo) */}
      <div style={{ border: `2px solid ${boxBorder}`, borderRadius: 12, padding: 16, marginTop: 12, background: '#fcfcff' }}>
        {/* Cabecera del recuadro (no entra al PDF): descargar PDF a la IZQUIERDA,
            guardado/limpiar a la DERECHA. */}
        <div data-pdf-exclude="1" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, marginBottom: 8 }}>
          <div>
            {globalVerdict && (
              <button
                onClick={() => void handleDownloadPdf()}
                disabled={generatingPdf}
                title="Descargar el informe en PDF (se genera en tu navegador; las imágenes no se suben)"
                style={{
                  fontFamily: 'monospace', fontSize: 13, fontWeight: 700,
                  cursor: generatingPdf ? 'wait' : 'pointer',
                  background: '#fff', color: '#7c3aed', border: '1px solid #c4b5fd',
                  borderRadius: 6, padding: '6px 12px',
                }}
              >
                {generatingPdf ? 'Generando PDF…' : '📄 Descargar PDF'}
              </button>
            )}
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 11, color: '#999' }} title="IndexedDB en este equipo; las imágenes no se suben a ningún lado">
              💾 guardado localmente en este equipo
            </div>
            {(slots.padre.file || slots.child.file || slots.madre.file) && (
              <button
                onClick={handleClear}
                title="Borra las fotos, el veredicto y el guardado local"
                style={{
                  marginTop: 4, fontFamily: 'monospace', fontSize: 12, cursor: 'pointer',
                  background: '#fff', color: '#900', border: '1px solid #c99', borderRadius: 6, padding: '4px 10px',
                }}
              >
                🗑️ Limpiar informe completo
              </button>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
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

        <div data-pdf-exclude="1" style={{ marginTop: 16, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
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
            <span style={{ color: '#999', fontSize: 12 }}>
              Cargá el Hijo/a + al menos un progenitor.
            </span>
          )}
        </div>

        {globalError && (
          <div style={{ background: '#fee', color: '#900', padding: 12, borderRadius: 4, marginTop: 12 }}>
            <strong>Error:</strong> {globalError}
          </div>
        )}

        {/* Veredicto GLOBAL, debajo de las fotos, en el mismo recuadro */}
        {globalVerdict && (
          <GlobalVerdictView
            global={globalVerdict}
            hasCalibration={!calError}
            calWarning={calWarning}
          />
        )}
      </div>

      {/* RECUADRO 2: scores por región (incluye la herencia por región + solapas de método) */}
      {childResult && (padreResult || madreResult) && (
        <RegionalScoresPanel
          child={childResult}
          parents={panelParents}
          session={session}
          busy={running}
          autoCompute="geometric"
          methodSelector="tabs"
          showInheritance
          onResults={onRegionalResults}
          seedResults={restoredRegional}
        />
      )}
    </div>
  );
}

// =========================================================
// Veredicto GLOBAL: cara completa (coseno + posterior calibrado por lado).
// =========================================================
function GlobalVerdictView({ global, hasCalibration, calWarning }: {
  global: GlobalVerdict;
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
  const headlineColor = both && global.winner !== 'tie' ? SIDE_COLOR[global.winner] : '#333';

  return (
    <div style={{ borderTop: '1px solid #e5e5e5', marginTop: 16, paddingTop: 14 }}>
      <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 4, color: headlineColor }}>{headline}</div>
      <div style={{ fontSize: 12, color: '#888', marginBottom: 12 }}>
        Parecido global por cara completa (coseno de embeddings){hasCalibration ? ' + probabilidad calibrada de parentesco (KinFaceW-I)' : ''}.
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {present.map((s) => {
          const cos = global.cosine[s];
          const post = global.posterior[s];
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
                  parentesco {pct(post)}
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

      {calWarning && (
        <div style={{ marginTop: 12, fontSize: 11, color: '#b45309', background: '#fff7ed', padding: 8, borderRadius: 4 }}>
          ⚠ {calWarning}
        </div>
      )}
    </div>
  );
}

// =========================================================
// Slot de carga (drag-drop + file). Roles fijos (Padre · Hijo/a · Madre); sin
// vínculo. El preview de la cara alineada vive en el panel de abajo, así que acá
// sólo mostramos la foto original.
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
        borderRadius: 6, padding: 12, background: '#fff',
        transition: 'background 80ms, border-color 80ms', ...dragStyle,
      }}
    >
      <div style={{ fontWeight: 700, color: meta.color }}>{meta.label}</div>
      <div style={{ fontSize: 11, color: '#999', marginBottom: 8 }}>{meta.hint}</div>

      {!slot.previewUrl && (
        <div data-pdf-exclude="1" style={{
          border: '2px dashed #bbb', borderRadius: 4, padding: '24px 12px', textAlign: 'center',
          fontSize: 12, color: '#777', marginBottom: 8, background: slot.isDraggingOver ? '#dceaff' : '#fafafa',
        }}>
          arrastrá una imagen acá<br />
          <span style={{ fontSize: 10, color: '#999' }}>o usá el botón ↓</span>
        </div>
      )}

      <div data-pdf-exclude="1" style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
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
