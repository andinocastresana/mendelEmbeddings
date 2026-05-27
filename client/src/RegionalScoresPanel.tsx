// =========================================
// ID: PHYLOFACE_COMP_REGIONAL_PANEL
// VERSION: v1.3
// =========================================
// Cambio v1.2 → v1.3 (Tarea #12 — App primaria): refactor ADITIVO, sin cambios
// de comportamiento para callers existentes (Comparador).
//   - Los helpers puros de agregación (DISPLAY_GROUPS, reparto P↔M, groupValues,
//     etc.) se MOVIERON a `lib/regionalAggregate.ts` para compartirlos con el
//     veredicto de la App primaria (lib/verdict.ts) — una sola fuente de verdad,
//     así el resumen nunca diverge de las barras/radar. Acá se importan.
//   - Dos props OPCIONALES nuevas que usa la App primaria; el Comparador no las
//     pasa y queda idéntico:
//       · `autoCompute?`: corre `compute()` de ese método en cuanto las caras
//         están listas, sin esperar el click en "Calcular" (la App primaria pasa
//         'geometric'; occlusion sigue siendo manual por costo).
//       · `onResults?`: emite (method, bySide) cada vez que cambian los scores
//         mostrados, para que el padre arme el veredicto sincronizado.
//
// Cambio v1.1 → v1.2: persistencia IDB de retoque 1. En modo VINCULADO (slots
// atados a personas del árbol, prop `link`), los scores se guardan en el mismo
// registro Comparison del par (campo `regional`, por método) vía treeStore, y se
// REHIDRATAN al montar — así sobreviven recargas y viven "como las comparaciones
// del árbol". En modo anónimo (sin `link`) la cache es solo en memoria.
// =========================================
// Panel de scores por región (Tareas #4/#9/#10/#16). Estructura confirmada con el
// usuario: Hijo/a al centro, un progenitor a cada lado, barras por región
// APUNTANDO hacia el Hijo/a (filas alineadas entre lados para comparar P vs M de un
// vistazo), heatmap opcional sobre la cara del Hijo/a, y un radar fijo abajo con un
// polígono por progenitor.
//
// Desacoplado del algoritmo: usa el contrato `RegionalScorer` (lib/regionalScores)
// vía un selector de método (Geométrico / Occlusion). Importar lib/regionalScorers
// registra los scorers en el registry.
//
// Cambios v1.0 → v1.1 (3 retoques pedidos):
//   1. CACHE POR MÉTODO: los resultados se guardan por método (`resultsByMethod`),
//      así cambiar el radio NO recomputa (occlusion es caro). Se invalida solo si
//      cambian las caras de entrada (referencias de PipelineOutput). [Persistencia
//      IDB "como las comparaciones del árbol" queda pendiente de definir alcance.]
//   2. ESCALA RELATIVA = REPARTO P↔M: por región, share = score / (scoreP + scoreM),
//      de modo que las barras de padre y madre SUMAN 100 por región (responde "¿la
//      región se parece más a P o a M?"). Absoluta = score crudo 0..1 del método.
//   3. PARES COLAPSADOS: las regiones pareadas (cejas, ojos, pómulos, mejillas) se
//      muestran como UNA fila (media del par) con una línea de RANGO dentro de la
//      barra que indica la asimetría izquierda/derecha. 12 regiones → 8 filas/ejes.
//
// Occlusion es pesado (~1 inferencia ONNX por región × progenitor): se dispara con
// "Calcular", no automático.

import { useState, useEffect, useMemo, useRef } from 'react';
import type * as ort from 'onnxruntime-web';
import type { PipelineOutput } from './lib/pipeline';
import { cosineSimilarity } from './lib/pipeline';
import { type RegionName } from './lib/regions';
import {
  getScorer, listScorers, confidenceLabelEs,
  type RegionalMethod, type RegionalScoresResult, type FaceRegionData,
} from './lib/regionalScores';
import {
  DISPLAY_GROUPS, rawScoreMap, perRegionValues, groupValues,
  type Scale, type GroupVal,
} from './lib/regionalAggregate';
import { regionBoxesAligned } from './lib/regionalScorers';
import { newComparison } from './lib/genealogy';
import { getComparisonForPair, saveComparison } from './lib/treeStore';

const LEFT_COLOR = '#2563eb';   // azul — progenitor izquierdo
const RIGHT_COLOR = '#16a34a';  // verde — progenitor derecho
const ALIGNED = 112;
const THUMB = 128;

type Side = 'left' | 'right';

export interface RegionalPanelParent {
  side: Side;
  label: string;
  result: PipelineOutput;
}
/** Vínculo con el árbol (modo tripleta): identifica child + progenitores como
 *  personas con foto. Si está presente, los scores se persisten/rehidratan en
 *  el store de comparaciones. Ausente = modo anónimo (cache solo en memoria). */
export interface RegionalLink {
  treeId: string;
  childPersonId: string;
  childSha256: string;
  parents: Partial<Record<Side, { personId: string; sha256: string }>>;
}
export interface RegionalScoresPanelProps {
  child: PipelineOutput;
  parents: RegionalPanelParent[];   // 1 o 2
  session: ort.InferenceSession | null;
  /** El Comparador todavía está procesando slots (su pipeline usa la MISMA sesión
   *  ONNX que occlusion). Mientras esté true, "Calcular" se deshabilita para no
   *  encolar/solapar runs con el compare inicial. */
  busy?: boolean;
  link?: RegionalLink;
  /** Si está presente, corre `compute()` de ese método en cuanto las caras
   *  están listas, sin esperar el click en "Calcular". La App primaria pasa
   *  'geometric' (barato, sin GPU); occlusion se deja manual por costo. */
  autoCompute?: RegionalMethod;
  /** Se invoca con (method, bySide) cada vez que cambian los scores mostrados.
   *  Lo usa la App primaria para construir el veredicto sincronizado. Pasar un
   *  callback ESTABLE (useCallback) para que no dispare en cada render. */
  onResults?: (method: RegionalMethod, bySide: Partial<Record<Side, RegionalScoresResult>>) => void;
}

// DISPLAY_GROUPS, GroupVal, rawScoreMap, perRegionValues y groupValues se
// movieron a `lib/regionalAggregate.ts` (compartidos con lib/verdict.ts). Ver
// cabecera v1.3.

function toFaceData(p: PipelineOutput): FaceRegionData {
  return { landmarksAligned: p.landmarksAligned, aligned: p.aligned, embedding: p.embedding };
}

// -----------------------------------------
// Cara alineada (112) escalada a THUMB, con overlay opcional de regiones teñidas.
// Las tints vienen por REGIÓN canónica (ambas mitades de un par reciben el valor
// colapsado del grupo, para que el heatmap concuerde con las barras).
// -----------------------------------------
function FaceThumb({ result, color, tints }: {
  result: PipelineOutput;
  color?: string;
  tints?: Map<RegionName, { color: string; alpha: number }>;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const off = document.createElement('canvas');
    off.width = ALIGNED; off.height = ALIGNED;
    off.getContext('2d')!.putImageData(result.aligned, 0, 0);
    ctx.clearRect(0, 0, THUMB, THUMB);
    ctx.drawImage(off, 0, 0, ALIGNED, ALIGNED, 0, 0, THUMB, THUMB);
    if (tints && tints.size > 0) {
      const boxes = regionBoxesAligned(result.landmarksAligned);
      const k = THUMB / ALIGNED;
      for (const [region, tint] of tints) {
        const b = boxes[region];
        if (!b) continue;
        ctx.fillStyle = tint.color;
        ctx.globalAlpha = tint.alpha;
        ctx.fillRect(b.x1 * k, b.y1 * k, (b.x2 - b.x1) * k, (b.y2 - b.y1) * k);
      }
      ctx.globalAlpha = 1;
    }
  }, [result, tints]);
  return (
    <canvas
      ref={ref} width={THUMB} height={THUMB}
      style={{ borderRadius: 8, border: `2px solid ${color ?? '#ccc'}`, display: 'block' }}
    />
  );
}

// -----------------------------------------
// Columna de barras por GRUPO. `anchor` = lado del progenitor (de donde nace la
// barra); crece hacia el Hijo/a. La barra dibuja la media del grupo; si el grupo
// es un par (count>1) y hay asimetría, una línea de rango marca [min,max] (cuánto
// difieren la región izquierda y derecha del par).
// -----------------------------------------
function RegionBars({ groups, color, anchor }: {
  groups: Map<string, GroupVal>;
  color: string;
  anchor: Side;
}) {
  const side = anchor === 'left' ? 'left' : 'right';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3, fontSize: 11, minWidth: 150 }}>
      {DISPLAY_GROUPS.map((g) => {
        const gv = groups.get(g.key);
        const v = gv?.value;
        const pct = v != null && !Number.isNaN(v) ? Math.round(v * 100) : null;
        const fillStyle = anchor === 'left'
          ? { left: 0 as const, width: `${pct ?? 0}%` }
          : { right: 0 as const, width: `${pct ?? 0}%` };
        const hasRange = gv != null && gv.count > 1 && gv.max - gv.min > 0.005;
        const rMin = hasRange ? Math.round(gv!.min * 100) : 0;
        const rMax = hasRange ? Math.round(gv!.max * 100) : 0;
        const label = <span style={{ width: 56, color: '#555', whiteSpace: 'nowrap' }}>{g.labelEs}</span>;
        const val = <span style={{ width: 24, color: '#333', textAlign: anchor === 'left' ? 'right' : 'left' }}>{pct ?? '—'}</span>;
        const bar = (
          <div style={{ flex: 1, background: '#eee', borderRadius: 3, height: 10, position: 'relative', overflow: 'hidden' }}>
            {pct != null && <div style={{ position: 'absolute', top: 0, bottom: 0, background: color, opacity: 0.8, ...fillStyle }} />}
            {hasRange && (
              <span title={`rango L/R: ${rMin}–${rMax}%`}>
                {/* línea de rango: asimetría izquierda/derecha del par */}
                <div style={{ position: 'absolute', top: 4, height: 2, background: 'rgba(0,0,0,0.55)', [side]: `${rMin}%`, width: `${rMax - rMin}%` }} />
                <div style={{ position: 'absolute', top: 2, height: 6, width: 1, background: 'rgba(0,0,0,0.75)', [side]: `${rMin}%` }} />
                <div style={{ position: 'absolute', top: 2, height: 6, width: 1, background: 'rgba(0,0,0,0.75)', [side]: `${rMax}%` }} />
              </span>
            )}
          </div>
        );
        // anchor 'left' (progenitor a la izquierda): [label · bar→ · val(junto al Hijo)]
        // anchor 'right' (progenitor a la derecha): [val(junto al Hijo) · ←bar · label]
        return (
          <div key={g.key} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            {anchor === 'left' ? <>{label}{bar}{val}</> : <>{val}{bar}{label}</>}
          </div>
        );
      })}
      {/* Barra de PROMEDIO: media de los grupos válidos de este lado. Resaltada y
          separada; sin línea de rango (no es un par). En "Reparto P↔M" los dos
          promedios suman ~100; en "Absoluta" es el score medio del progenitor. */}
      {(() => {
        const vals = DISPLAY_GROUPS
          .map((g) => groups.get(g.key)?.value)
          .filter((v): v is number => v != null && !Number.isNaN(v));
        const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
        const pct = avg != null ? Math.round(avg * 100) : null;
        const fillStyle = anchor === 'left'
          ? { left: 0 as const, width: `${pct ?? 0}%` }
          : { right: 0 as const, width: `${pct ?? 0}%` };
        const label = <span style={{ width: 56, color: '#222', fontWeight: 600, whiteSpace: 'nowrap' }}>promedio</span>;
        const val = <span style={{ width: 24, color: '#111', fontWeight: 600, textAlign: anchor === 'left' ? 'right' : 'left' }}>{pct ?? '—'}</span>;
        const bar = (
          <div style={{ flex: 1, background: '#eee', borderRadius: 3, height: 10, position: 'relative', overflow: 'hidden' }}>
            {pct != null && <div style={{ position: 'absolute', top: 0, bottom: 0, background: color, ...fillStyle }} />}
          </div>
        );
        return (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginTop: 4, paddingTop: 5, borderTop: '1px solid #d4d4d4' }}>
            {anchor === 'left' ? <>{label}{bar}{val}</> : <>{val}{bar}{label}</>}
          </div>
        );
      })()}
    </div>
  );
}

// -----------------------------------------
// Radar: un eje por GRUPO (8), un polígono por progenitor.
// -----------------------------------------
function Radar({ left, right }: { left?: Map<string, GroupVal>; right?: Map<string, GroupVal> }) {
  const size = 260, cx = size / 2, cy = size / 2, R = size / 2 - 34;
  const groups = DISPLAY_GROUPS;
  const n = groups.length;
  const ang = (i: number) => (Math.PI * 2 * i) / n - Math.PI / 2;
  const at = (i: number, r: number): [number, number] => [cx + r * Math.cos(ang(i)), cy + r * Math.sin(ang(i))];
  const poly = (vals: Map<string, GroupVal>) =>
    groups.map((g, i) => {
      const v = vals.get(g.key)?.value;
      return at(i, R * (v == null || Number.isNaN(v) ? 0 : Math.max(0, Math.min(1, v)))).join(',');
    }).join(' ');

  return (
    <svg width={size} height={size} role="img" aria-label="Radar de scores por región">
      {[0.25, 0.5, 0.75, 1].map((ring) => (
        <polygon key={ring}
          points={groups.map((_, i) => at(i, R * ring).join(',')).join(' ')}
          fill="none" stroke="#e2e2e2" strokeWidth={1} />
      ))}
      {groups.map((g, i) => {
        const [ex, ey] = at(i, R);
        const [lx, ly] = at(i, R + 16);
        return (
          <g key={g.key}>
            <line x1={cx} y1={cy} x2={ex} y2={ey} stroke="#e2e2e2" strokeWidth={1} />
            <text x={lx} y={ly} fontSize={9} fill="#666" textAnchor="middle" dominantBaseline="middle">{g.labelEs}</text>
          </g>
        );
      })}
      {left && <polygon points={poly(left)} fill={LEFT_COLOR} fillOpacity={0.22} stroke={LEFT_COLOR} strokeWidth={1.5} />}
      {right && <polygon points={poly(right)} fill={RIGHT_COLOR} fillOpacity={0.22} stroke={RIGHT_COLOR} strokeWidth={1.5} />}
    </svg>
  );
}

// =========================================================
// Panel
// =========================================================
type CacheEntry = { bySide: Partial<Record<Side, RegionalScoresResult>> };

export default function RegionalScoresPanel({ child, parents, session, busy = false, link, autoCompute, onResults }: RegionalScoresPanelProps) {
  const [method, setMethod] = useState<RegionalMethod>('geometric');
  const [scale, setScale] = useState<Scale>('relative');
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [computing, setComputing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Cache por método (retoque 1): cambiar el radio no recomputa.
  const [resultsByMethod, setResultsByMethod] = useState<Partial<Record<RegionalMethod, CacheEntry>>>({});

  const scorer = getScorer(method);
  const leftParent = parents.find((p) => p.side === 'left');
  const rightParent = parents.find((p) => p.side === 'right');
  const occlusionNoSession = method === 'occlusion' && !session;

  // Identidad del vínculo: treeId + sha256 de las 3 caras. Cambia si cambian las
  // fotos vinculadas → dispara recarga/limpieza de la cache.
  const linkKey = link
    ? `${link.treeId}|${link.childSha256}|${link.parents.left?.sha256 ?? ''}|${link.parents.right?.sha256 ?? ''}`
    : '';

  // Al cambiar las CARAS de entrada (refs estables de PipelineOutput) o el vínculo:
  //   - modo anónimo (sin link): limpiar la cache.
  //   - modo vinculado: rehidratar desde IDB los scores ya guardados para el par.
  useEffect(() => {
    let cancelled = false;
    setError(null);
    if (!link) { setResultsByMethod({}); return; }
    void (async () => {
      const cache: Partial<Record<RegionalMethod, CacheEntry>> = {};
      try {
        for (const p of parents) {
          const lp = link.parents[p.side];
          if (!lp) continue;
          const comp = await getComparisonForPair(link.treeId, link.childPersonId, lp.personId);
          if (!comp?.regional) continue;
          for (const [m, res] of Object.entries(comp.regional)) {
            const mk = m as RegionalMethod;
            (cache[mk] ??= { bySide: {} }).bySide[p.side] = res;
          }
        }
      } catch (e) {
        console.warn('[RegionalScoresPanel] rehidratar desde IDB falló:', e);
      }
      if (!cancelled) setResultsByMethod(cache);
    })();
    return () => { cancelled = true; };
    // parents se lee dentro pero se recrea por render; las entradas relevantes
    // (refs de result + linkKey) sí están en deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [child, leftParent?.result, rightParent?.result, linkKey]);

  // Persiste los scores recién calculados en el registro Comparison del par
  // (modo vinculado). Upsert: respeta el cosine/registro existente y solo agrega
  // `regional[method]`. Falla en silencio (no debe romper la UI).
  async function persistRegional(m: RegionalMethod, bySide: Partial<Record<Side, RegionalScoresResult>>) {
    if (!link) return;
    try {
      for (const p of parents) {
        const lp = link.parents[p.side];
        const res = bySide[p.side];
        if (!lp || !res) continue;
        const existing = await getComparisonForPair(link.treeId, link.childPersonId, lp.personId);
        const cosine = cosineSimilarity(child.embedding, p.result.embedding);
        const base = existing
          ?? newComparison(link.treeId, link.childPersonId, lp.personId, link.childSha256, lp.sha256, cosine);
        await saveComparison({ ...base, regional: { ...(base.regional ?? {}), [m]: res } });
      }
    } catch (e) {
      console.warn('[RegionalScoresPanel] persistir scores regionales falló:', e);
    }
  }

  async function compute() {
    if (!scorer || parents.length === 0 || occlusionNoSession || busy) return;
    setComputing(true); setError(null);
    try {
      const ctx = { session: session ?? undefined };
      const childData = toFaceData(child);
      const bySide: Partial<Record<Side, RegionalScoresResult>> = {};
      for (const p of parents) {
        bySide[p.side] = await scorer.score(childData, toFaceData(p.result), ctx);
      }
      setResultsByMethod((prev) => ({ ...prev, [method]: { bySide } }));
      await persistRegional(method, bySide);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setComputing(false);
    }
  }

  const shown = resultsByMethod[method];

  // Auto-cómputo opcional (App primaria): corre el método pedido en cuanto las
  // caras están listas, sin esperar el click manual. Solo si no hay resultados
  // para ese método todavía. Occlusion necesita la sesión ONNX; geométrico no.
  useEffect(() => {
    if (!autoCompute || method !== autoCompute) return;
    if (resultsByMethod[autoCompute] || computing || busy || parents.length === 0) return;
    if (autoCompute === 'occlusion' && !session) return;
    void compute();
    // `compute` se omite a propósito de las deps: es un closure recreado en cada
    // render pero estable para lo que importa acá; meterlo dispararía el effect
    // en cada render. Las deps reales (método, cache, busy, caras) sí están.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoCompute, method, resultsByMethod, computing, busy, parents.length, session]);

  // Emite los scores mostrados al padre (App primaria → veredicto). Se dispara al
  // cambiar `shown` o el método. `onResults` debe ser estable (useCallback) para
  // no refirear en cada render del padre.
  useEffect(() => {
    if (onResults && shown) onResults(method, shown.bySide);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shown, method]);

  const rawL = useMemo(() => rawScoreMap(shown?.bySide.left), [shown]);
  const rawR = useMemo(() => rawScoreMap(shown?.bySide.right), [shown]);
  const per = useMemo(() => perRegionValues(rawL, rawR, scale), [rawL, rawR, scale]);
  const leftGroups = useMemo(() => groupValues(per.left), [per]);
  const rightGroups = useMemo(() => groupValues(per.right), [per]);

  // Tints del heatmap sobre el Hijo/a: 2 progenitores → color de quien gana ese
  // grupo, alpha por margen; 1 progenitor → su color, alpha por magnitud. El valor
  // del grupo se aplica a AMBAS regiones del par (concuerda con las barras).
  const tints = useMemo(() => {
    const m = new Map<RegionName, { color: string; alpha: number }>();
    if (!showHeatmap || !shown) return m;
    for (const g of DISPLAY_GROUPS) {
      const l = leftGroups.get(g.key)?.value;
      const rr = rightGroups.get(g.key)?.value;
      const hasL = l != null && !Number.isNaN(l);
      const hasR = rr != null && !Number.isNaN(rr);
      let tint: { color: string; alpha: number } | null = null;
      if (hasL && hasR) {
        const margin = Math.abs((l as number) - (rr as number));
        tint = { color: (l as number) >= (rr as number) ? LEFT_COLOR : RIGHT_COLOR, alpha: Math.min(0.6, 0.15 + margin * 0.6) };
      } else if (hasL) {
        tint = { color: LEFT_COLOR, alpha: Math.min(0.6, 0.1 + (l as number) * 0.5) };
      } else if (hasR) {
        tint = { color: RIGHT_COLOR, alpha: Math.min(0.6, 0.1 + (rr as number) * 0.5) };
      }
      if (tint) for (const region of g.regions) m.set(region, tint);
    }
    return m;
  }, [showHeatmap, shown, leftGroups, rightGroups]);

  const radio = (name: string, checked: boolean, onChange: () => void, label: string, disabled = false) => (
    <label style={{ marginRight: 12, cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.5 : 1 }}>
      <input type="radio" name={name} checked={checked} onChange={onChange} disabled={disabled} /> {label}
    </label>
  );

  return (
    <div style={{ border: '1px solid #ddd', borderRadius: 10, padding: 16, marginTop: 20 }}>
      <h3 style={{ margin: '0 0 10px' }}>Scores por región</h3>

      {/* Controles */}
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8, fontSize: 13, marginBottom: 4 }}>
        <strong>Método:</strong>
        {listScorers().map((s) => {
          const cached = resultsByMethod[s.method] != null;
          return (
            <span key={s.method}>
              {radio('rs-method', method === s.method, () => setMethod(s.method),
                cached ? `${s.label} ✓` : s.label, s.method === 'occlusion' && !session)}
            </span>
          );
        })}
        <span style={{ width: 16 }} />
        <strong>Escala:</strong>
        {radio('rs-scale', scale === 'relative', () => setScale('relative'), 'Reparto P↔M')}
        {radio('rs-scale', scale === 'absolute', () => setScale('absolute'), 'Absoluta')}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 10, fontSize: 13, marginBottom: 10 }}>
        <label style={{ cursor: 'pointer' }}>
          <input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} /> Heatmap sobre el Hijo/a
        </label>
        {scorer && (
          <span style={{ color: '#888' }}>⚠ confiabilidad: {confidenceLabelEs(scorer.baseConfidence)}</span>
        )}
        <button onClick={() => void compute()} disabled={computing || parents.length === 0 || occlusionNoSession || busy}
          style={{ marginLeft: 'auto', padding: '5px 14px', borderRadius: 6, cursor: 'pointer' }}>
          {computing ? 'Calculando…' : shown ? 'Recalcular ▸' : 'Calcular ▸'}
        </button>
      </div>
      {scorer && <p style={{ margin: '0 0 4px', fontSize: 12, color: '#777' }}>{scorer.description}</p>}
      <p style={{ margin: '0 0 12px', fontSize: 12, color: '#999' }}>
        {scale === 'relative'
          ? 'Reparto P↔M: por región, las barras de los dos progenitores suman 100% (¿se parece más a uno o a otro?).'
          : 'Absoluta: score crudo 0..1 del método, independiente por progenitor.'}
        {' '}Los pares (cejas, ojos, pómulos, mejillas) se colapsan en una fila; la línea oscura marca el rango izquierda/derecha.
      </p>
      {occlusionNoSession && <p style={{ color: '#b45309', fontSize: 12 }}>Occlusion necesita el modelo ONNX cargado (compará primero en el comparador).</p>}
      {busy && <p style={{ color: '#b45309', fontSize: 12 }}>Esperá a que termine la comparación de arriba para calcular scores por región.</p>}
      {error && <p style={{ color: '#dc2626', fontSize: 12 }}>Error: {error}</p>}

      {/* Cuerpo: progenitor · barras · Hijo/a · barras · progenitor */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'center', gap: 12, flexWrap: 'wrap' }}>
        {leftParent && (
          <div style={{ textAlign: 'center' }}>
            <FaceThumb result={leftParent.result} color={LEFT_COLOR} />
            <div style={{ fontSize: 12, color: LEFT_COLOR, marginTop: 4 }}>{leftParent.label}</div>
          </div>
        )}
        {leftParent && shown?.bySide.left && <RegionBars groups={leftGroups} color={LEFT_COLOR} anchor="left" />}

        <div style={{ textAlign: 'center' }}>
          <FaceThumb result={child} color="#888" tints={tints} />
          <div style={{ fontSize: 12, color: '#444', marginTop: 4 }}>Hijo/a</div>
        </div>

        {rightParent && shown?.bySide.right && <RegionBars groups={rightGroups} color={RIGHT_COLOR} anchor="right" />}
        {rightParent && (
          <div style={{ textAlign: 'center' }}>
            <FaceThumb result={rightParent.result} color={RIGHT_COLOR} />
            <div style={{ fontSize: 12, color: RIGHT_COLOR, marginTop: 4 }}>{rightParent.label}</div>
          </div>
        )}
      </div>

      {/* Radar fijo */}
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: 14 }}>
        <div style={{ textAlign: 'center' }}>
          <Radar left={leftParent && shown?.bySide.left ? leftGroups : undefined}
                 right={rightParent && shown?.bySide.right ? rightGroups : undefined} />
          <div style={{ fontSize: 12 }}>
            {leftParent && <span style={{ color: LEFT_COLOR, marginRight: 12 }}>▬ {leftParent.label}</span>}
            {rightParent && <span style={{ color: RIGHT_COLOR }}>▬ {rightParent.label}</span>}
          </div>
        </div>
      </div>
      {!shown && !computing && <p style={{ textAlign: 'center', color: '#999', fontSize: 12 }}>Tocá «Calcular» para ver los scores por región.</p>}
    </div>
  );
}
