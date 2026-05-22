// =========================================
// ID: PHYLOFACE_COMPARATOR
// VERSION: v2.2
// =========================================
// Cambio v2.1 → v2.2 (Tarea #27, bugfix calentamiento sostenido):
// - Cleanup explícito de los recursos GPU/WASM al desmontar el componente.
//   `FaceLandmarker.close()` y `InferenceSession.release()` se invocan en el
//   return de un `useEffect([])` dedicado. Sin esto, cambiar de tab (de
//   "Comparador" a "Árbol" o "Spikes") deja el contexto GPU del Face Mesh y
//   la sesión ONNX WebGPU/WASM vivos en el proceso GPU compartido del browser
//   hasta refresh de página, contribuyendo al calentamiento medido en Phase 5
//   del heat-experiment.sh vs Phase 2.
//
// Cambio v2.0 → v2.1:
// - Botón "✕ Quitar" por slot para borrar la imagen cargada.
// - Comparar acepta 2 slots con file (no obliga los 3). Reglas:
//     * Si hay Hijo/a → cosine contra cada adulto presente.
//     * Si NO hay Hijo/a y ambos adultos cargados → cosine adulto ↔ adulto.
//   Sigue sin haber doble-comparación P1↔P2 cuando el Hijo/a está presente
//   (decisión del usuario en v2.0: ese baseline agrega ruido al caso primario).
//
// =========================================
// UI MVP del comparador anónimo browser (Tarea #25, subtarea c).
//
// Cambio v1.0 → v2.0:
// - Tres slots: lateral izquierdo (adulto) · centro (Hijo/a) · lateral derecho
//   (adulto). El caso primario del producto es "niño vs progenitores" (ver
//   Tarea #12 en TAREAS_PENDIENTES.md, App primaria).
// - Cada slot lateral tiene dropdown de rol (Padre/Madre por default;
//   Hermano/a, Tío/a, Abuelo/a, Otro). Si elige "Otro", aparece un input para
//   escribir el rol libremente. El slot central es fijo "Hijo/a".
// - Drag-and-drop por slot: además del input file, se puede arrastrar la
//   imagen desde el filesystem a la zona del slot. Highlight visual al
//   drag-over. Solo soporta `dataTransfer.files` (filesystem); no resuelve
//   URLs externas — drag desde otra pestaña del browser no funciona y es
//   intencional (privacidad: no queremos fetchear nada del afuera).
// - Salida: dos cosines, Hijo/a ↔ slot izquierdo y Hijo/a ↔ slot derecho,
//   cada uno etiquetado con el rol del adulto. Sin comparación P1↔P2.
//
// Garantía de privacidad:
// - Las imágenes NUNCA salen del browser. `URL.createObjectURL(file)` da una
//   URL blob: local al navegador; `loadImage` la lee con un <img> + canvas en
//   memoria; el pipeline corre 100% client-side (MediaPipe WASM + ONNX
//   WebGPU/WASM). No hay upload a ningún servidor.
//
// Estado y costos:
// - El landmarker y la sesión ONNX se instancian una sola vez (lazy, al
//   primer click en Comparar). Cachear es importante: cada init descarga
//   modelos remotos y costó ~varios segundos en los spikes.
// - El cosine se muestra crudo, sin etiqueta semántica (no hay umbral
//   calibrado todavía — Tarea #6 lo va a generar contra KinFaceW).

import { useEffect, useRef, useState } from 'react';
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

// -----------------------------------------
// Slots y roles
// -----------------------------------------
type SlotKey = 'left' | 'child' | 'right';

// Roles predefinidos para los slots laterales. "Otro" abre un input libre.
const ROLE_OPTIONS = [
  'Padre', 'Madre', 'Hermano', 'Hermana',
  'Tío', 'Tía', 'Abuelo', 'Abuela', 'Otro',
] as const;
type Role = typeof ROLE_OPTIONS[number];

interface SlotState {
  file: File | null;
  previewUrl: string | null;       // object URL para mostrar la foto original
  result: PipelineOutput | null;   // embedding + kps + aligned + timings
  error: string | null;
  isDraggingOver: boolean;         // highlight visual durante drag-over
}

const EMPTY_SLOT: SlotState = {
  file: null,
  previewUrl: null,
  result: null,
  error: null,
  isDraggingOver: false,
};

// Etiqueta efectiva del slot a mostrar y a usar en los resultados.
function slotLabel(key: SlotKey, role: Role, roleCustom: string): string {
  if (key === 'child') return 'Hijo/a';
  if (role === 'Otro') return roleCustom.trim() || 'Otro';
  return role;
}

function Comparator() {
  // Estado de los tres slots, indexado por clave.
  const [slots, setSlots] = useState<Record<SlotKey, SlotState>>({
    left: { ...EMPTY_SLOT },
    child: { ...EMPTY_SLOT },
    right: { ...EMPTY_SLOT },
  });

  // Roles de los slots laterales. Por default: izquierda Padre, derecha Madre.
  const [leftRole, setLeftRole] = useState<Role>('Padre');
  const [leftRoleCustom, setLeftRoleCustom] = useState<string>('');
  const [rightRole, setRightRole] = useState<Role>('Madre');
  const [rightRoleCustom, setRightRoleCustom] = useState<string>('');

  const [running, setRunning] = useState(false);
  // Lista de cosines a mostrar. Cada entrada es un par (label, valor).
  // El array dinámico simplifica el caso "comparación de a 2" (solo 1 entry)
  // y el de "Hijo/a vs los dos adultos" (2 entries).
  const [cosines, setCosines] = useState<{ label: string; value: number }[]>([]);
  const [globalError, setGlobalError] = useState<string | null>(null);

  // Sesiones cacheadas entre corridas.
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const [initStatus, setInitStatus] = useState<'idle' | 'initializing' | 'ready'>('idle');

  // Canvases para preview de las caras alineadas (112×112), uno por slot.
  const alignedCanvasRefs = useRef<Record<SlotKey, HTMLCanvasElement | null>>({
    left: null, child: null, right: null,
  });

  // ---------------------------------------
  // Helpers de estado
  // ---------------------------------------
  const updateSlot = (key: SlotKey, patch: Partial<SlotState>) => {
    setSlots((prev) => ({ ...prev, [key]: { ...prev[key], ...patch } }));
  };

  // Manejo de selección de archivo (input file o drop). Liberamos el object
  // URL viejo si lo había para no leakear memoria.
  const onPickFile = (key: SlotKey, file: File | null) => {
    setSlots((prev) => {
      const old = prev[key];
      if (old.previewUrl) URL.revokeObjectURL(old.previewUrl);
      return {
        ...prev,
        [key]: {
          file,
          previewUrl: file ? URL.createObjectURL(file) : null,
          result: null,
          error: null,
          isDraggingOver: false,
        },
      };
    });
    // Invalido resultados (cualquier cambio de input requiere re-comparar).
    setCosines([]);
    setGlobalError(null);
  };

  // ---------------------------------------
  // Drag-and-drop handlers (uno set por slot, instanciado en el render).
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
  // Pinta la ImageData alineada (112×112) escalada 2× sin antialias.
  // ---------------------------------------
  const drawAligned = (canvas: HTMLCanvasElement | null, imgData: ImageData) => {
    if (!canvas) return;
    const scale = 2;
    canvas.width = imgData.width * scale;
    canvas.height = imgData.height * scale;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const off = document.createElement('canvas');
    off.width = imgData.width;
    off.height = imgData.height;
    const offCtx = off.getContext('2d');
    if (!offCtx) return;
    offCtx.putImageData(imgData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  };

  // Cuando un slot recibe `result`, el canvas alineado se monta (porque su
  // render condicional pasa a true). Dibujamos en ese momento, no antes.
  // Effects separados por slot para que exhaustive-deps no pida `slots` entero
  // (cambia con cada drag-over y dispararía el draw innecesariamente).
  const leftResult = slots.left.result;
  const childResult = slots.child.result;
  const rightResult = slots.right.result;
  useEffect(() => {
    if (leftResult) drawAligned(alignedCanvasRefs.current.left, leftResult.aligned);
  }, [leftResult]);
  useEffect(() => {
    if (childResult) drawAligned(alignedCanvasRefs.current.child, childResult.aligned);
  }, [childResult]);
  useEffect(() => {
    if (rightResult) drawAligned(alignedCanvasRefs.current.right, rightResult.aligned);
  }, [rightResult]);

  // ---------------------------------------
  // Cleanup al desmontar: liberar GPU/WASM. Ver cabecera v2.2 (Tarea #27).
  // El effect no depende de nada — el cleanup lee los refs en tiempo de
  // desmontaje, no captura el valor en el render. Usamos `void` sobre
  // release() porque devuelve Promise<void> y no podemos await en cleanup.
  // ---------------------------------------
  useEffect(() => {
    return () => {
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
      const sess = sessionRef.current;
      sessionRef.current = null;
      if (sess) void sess.release().catch((e) => console.warn('[Comparator] session.release falló:', e));
    };
  }, []);

  // ---------------------------------------
  // Lazy init de landmarker + session. Una sola vez por sesión.
  // ---------------------------------------
  const ensureInit = async () => {
    if (landmarkerRef.current && sessionRef.current) return;
    setInitStatus('initializing');
    const [lm, sess] = await Promise.all([
      landmarkerRef.current ? Promise.resolve(landmarkerRef.current) : initFaceLandmarker(),
      sessionRef.current ? Promise.resolve(sessionRef.current) : initOnnxSession(),
    ]);
    landmarkerRef.current = lm;
    sessionRef.current = sess;
    setInitStatus('ready');
  };

  // ---------------------------------------
  // Procesa un slot: carga la imagen, corre el pipeline, guarda resultado.
  // Devuelve el PipelineOutput o null si falló.
  // ---------------------------------------
  const processSlot = async (key: SlotKey): Promise<PipelineOutput | null> => {
    const state = slots[key];
    if (!state.file || !state.previewUrl) return null;
    try {
      const { img, imageData } = await loadImage(state.previewUrl);
      const out = await computeEmbedding(
        img,
        imageData,
        landmarkerRef.current!,
        sessionRef.current!,
      );
      updateSlot(key, { result: out, error: null });
      return out;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      updateSlot(key, { result: null, error: msg });
      console.error(`[Comparator] slot ${key} falló:`, e);
      return null;
    }
  };

  // ---------------------------------------
  // Click handler: corre los slots con file y calcula los cosines aplicables.
  // Reglas: si hay Hijo/a → cosine vs cada adulto presente; si no hay Hijo/a
  // y ambos adultos están cargados → cosine adulto ↔ adulto.
  // ---------------------------------------
  const onCompare = async () => {
    if (!canCompare) return;
    setRunning(true);
    setCosines([]);
    setGlobalError(null);
    try {
      await ensureInit();
      // Solo procesamos los slots que tienen file. processSlot devuelve null
      // si el slot está vacío, sin error — lo aprovechamos.
      const outChild = slots.child.file ? await processSlot('child') : null;
      const outLeft = slots.left.file ? await processSlot('left') : null;
      const outRight = slots.right.file ? await processSlot('right') : null;

      const next: { label: string; value: number }[] = [];
      if (outChild && outLeft) {
        next.push({
          label: `${childLabel} ↔ ${leftLabel}`,
          value: cosineSimilarity(outChild.embedding, outLeft.embedding),
        });
      }
      if (outChild && outRight) {
        next.push({
          label: `${childLabel} ↔ ${rightLabel}`,
          value: cosineSimilarity(outChild.embedding, outRight.embedding),
        });
      }
      if (!outChild && outLeft && outRight) {
        next.push({
          label: `${leftLabel} ↔ ${rightLabel}`,
          value: cosineSimilarity(outLeft.embedding, outRight.embedding),
        });
      }
      setCosines(next);
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  // Habilitar el botón si hay al menos 2 slots con file.
  const filledCount =
    (slots.left.file ? 1 : 0) +
    (slots.child.file ? 1 : 0) +
    (slots.right.file ? 1 : 0);
  const canCompare = filledCount >= 2 && !running;

  const leftLabel = slotLabel('left', leftRole, leftRoleCustom);
  const childLabel = slotLabel('child', 'Padre', '');
  const rightLabel = slotLabel('right', rightRole, rightRoleCustom);

  return (
    <div style={{ fontFamily: 'monospace', padding: '20px', maxWidth: 1200, margin: '0 auto' }}>
      <h1 style={{ borderBottom: '2px solid #333', paddingBottom: 8 }}>
        Comparador anónimo — Hijo/a vs adultos
      </h1>
      <p style={{ color: '#666', fontSize: 13, lineHeight: 1.5 }}>
        Cargá tres fotos (una cara cada una): un hijo/a en el centro, dos
        adultos en los laterales (padre/madre por default, configurable). Se
        corre todo en este browser: Face Mesh detecta los 5 keypoints
        canónicos, alineación 2D al template ArcFace 112×112, embedding ONNX
        (<code>w600k_r50</code>), y se reporta la similitud coseno del hijo/a
        contra cada adulto.
        <br />
        <strong>Las imágenes nunca salen de tu navegador.</strong> No hay
        upload a servidor, no se guarda nada en disco. Podés arrastrar las
        imágenes a los slots o usar el botón.
      </p>

      {/* Tres slots en fila (P1 - Niño - P2). Wrapean en pantallas chicas. */}
      <div style={{ display: 'flex', gap: 16, marginTop: 20, flexWrap: 'wrap' }}>
        <SlotPicker
          slotKey="left"
          label={leftLabel}
          slot={slots.left}
          canvasRefSet={(el) => { alignedCanvasRefs.current.left = el; }}
          onPick={(f) => onPickFile('left', f)}
          onDragOver={onDragOver('left')}
          onDragLeave={onDragLeave('left')}
          onDrop={onDrop('left')}
          roleSelector={
            <RoleSelector
              role={leftRole}
              roleCustom={leftRoleCustom}
              onRoleChange={setLeftRole}
              onCustomChange={setLeftRoleCustom}
            />
          }
        />
        <SlotPicker
          slotKey="child"
          label={childLabel}
          slot={slots.child}
          canvasRefSet={(el) => { alignedCanvasRefs.current.child = el; }}
          onPick={(f) => onPickFile('child', f)}
          onDragOver={onDragOver('child')}
          onDragLeave={onDragLeave('child')}
          onDrop={onDrop('child')}
        />
        <SlotPicker
          slotKey="right"
          label={rightLabel}
          slot={slots.right}
          canvasRefSet={(el) => { alignedCanvasRefs.current.right = el; }}
          onPick={(f) => onPickFile('right', f)}
          onDragOver={onDragOver('right')}
          onDragLeave={onDragLeave('right')}
          onDrop={onDrop('right')}
          roleSelector={
            <RoleSelector
              role={rightRole}
              roleCustom={rightRoleCustom}
              onRoleChange={setRightRole}
              onCustomChange={setRightRoleCustom}
            />
          }
        />
      </div>

      {/* Botón comparar */}
      <div style={{ marginTop: 18 }}>
        <button
          onClick={onCompare}
          disabled={!canCompare}
          style={{
            padding: '10px 24px',
            fontFamily: 'monospace',
            fontSize: 14,
            fontWeight: 600,
            cursor: canCompare ? 'pointer' : 'not-allowed',
            background: canCompare ? '#1a73e8' : '#ccc',
            color: '#fff',
            border: 'none',
            borderRadius: 4,
          }}
        >
          {running
            ? (initStatus === 'initializing' ? 'Inicializando modelos…' : 'Procesando…')
            : 'Comparar'}
        </button>
        {initStatus === 'ready' && !running && (
          <span style={{ marginLeft: 12, color: '#080', fontSize: 12 }}>
            ✓ modelos listos (cacheados para próximas corridas)
          </span>
        )}
      </div>

      {globalError && (
        <div style={{ background: '#fee', color: '#900', padding: 12, borderRadius: 4, marginTop: 12 }}>
          <strong>Error:</strong> {globalError}
        </div>
      )}

      {/* Resultados: una card por par comparado */}
      {cosines.length > 0 && (
        <>
          <div style={{ display: 'flex', gap: 16, marginTop: 24, flexWrap: 'wrap' }}>
            {cosines.map((c) => (
              <CosineCard key={c.label} label={c.label} value={c.value} />
            ))}
          </div>
          <p style={{ fontSize: 11, color: '#888', marginTop: 12 }}>
            Rango teórico: [-1, 1]. Valores cercanos a 1 indican embeddings
            más similares. <em>No hay umbral calibrado todavía</em> (Tarea #6
            lo va a generar contra KinFaceW); estos son los valores crudos de
            similitud entre vectores 512-d.
          </p>
        </>
      )}

      {/* Timings de la última corrida */}
      {(slots.left.result || slots.child.result || slots.right.result) && (
        <details style={{ marginTop: 16, fontSize: 12 }}>
          <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Timings por slot</summary>
          <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 8 }}>
            <thead>
              <tr style={{ background: '#f4f4f4' }}>
                <th style={th}>slot</th>
                <th style={th}>detect (ms)</th>
                <th style={th}>align (ms)</th>
                <th style={th}>preprocess (ms)</th>
                <th style={th}>infer (ms)</th>
              </tr>
            </thead>
            <tbody>
              {(['left', 'child', 'right'] as const).map((k) => {
                const r = slots[k].result;
                if (!r) return null;
                const lbl = k === 'left' ? leftLabel : k === 'right' ? rightLabel : childLabel;
                return (
                  <tr key={k}>
                    <td style={td}>{lbl}</td>
                    <td style={td}>{r.timings.detectMs.toFixed(0)}</td>
                    <td style={td}>{r.timings.alignMs.toFixed(1)}</td>
                    <td style={td}>{r.timings.preprocessMs.toFixed(1)}</td>
                    <td style={td}>{r.timings.inferMs.toFixed(0)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </details>
      )}
    </div>
  );
}

// -----------------------------------------
// Sub-componente: card de resultado de un cosine.
// -----------------------------------------
function CosineCard({ label, value }: { label: string; value: number }) {
  return (
    <div style={{
      flex: 1, minWidth: 240,
      padding: 16, background: '#f4f8ff',
      border: '1px solid #b8d4ff', borderRadius: 4,
    }}>
      <div style={{ fontSize: 13, color: '#555', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 32, fontWeight: 700, color: '#1a73e8' }}>
        {value.toFixed(4)}
      </div>
    </div>
  );
}

// -----------------------------------------
// Sub-componente: dropdown de rol + input "Otro" si corresponde.
// -----------------------------------------
interface RoleSelectorProps {
  role: Role;
  roleCustom: string;
  onRoleChange: (r: Role) => void;
  onCustomChange: (s: string) => void;
}

function RoleSelector({ role, roleCustom, onRoleChange, onCustomChange }: RoleSelectorProps) {
  return (
    <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 8 }}>
      <select
        value={role}
        onChange={(e) => onRoleChange(e.target.value as Role)}
        style={{ fontFamily: 'monospace', fontSize: 12, padding: '3px 6px' }}
      >
        {ROLE_OPTIONS.map((r) => (
          <option key={r} value={r}>{r}</option>
        ))}
      </select>
      {role === 'Otro' && (
        <input
          type="text"
          value={roleCustom}
          onChange={(e) => onCustomChange(e.target.value)}
          placeholder="escribir rol…"
          style={{ fontFamily: 'monospace', fontSize: 12, padding: '3px 6px', flex: 1, minWidth: 0 }}
        />
      )}
    </div>
  );
}

// -----------------------------------------
// Sub-componente: un slot con su input file, drag-zone, preview de la foto
// original, preview de la cara alineada (112×112), dropdown de rol opcional
// y mensaje de error si falla.
// -----------------------------------------
interface SlotPickerProps {
  slotKey: SlotKey;
  label: string;
  slot: SlotState;
  canvasRefSet: (el: HTMLCanvasElement | null) => void;
  onPick: (file: File | null) => void;
  onDragOver: (e: React.DragEvent<HTMLDivElement>) => void;
  onDragLeave: (e: React.DragEvent<HTMLDivElement>) => void;
  onDrop: (e: React.DragEvent<HTMLDivElement>) => void;
  roleSelector?: React.ReactNode;
}

function SlotPicker({
  label, slot, canvasRefSet, onPick,
  onDragOver, onDragLeave, onDrop, roleSelector,
}: SlotPickerProps) {
  const dragStyle: React.CSSProperties = slot.isDraggingOver
    ? { background: '#eaf3ff', borderColor: '#1a73e8', borderStyle: 'dashed' }
    : {};

  // El input type=file no permite setear `value` desde React (seguridad del
  // browser). Para que "Quitar" deje el input vacío y permita re-seleccionar
  // el mismo archivo otra vez, lo limpiamos manualmente vía ref.
  const fileInputRef = useRef<HTMLInputElement>(null);
  const handleRemove = () => {
    if (fileInputRef.current) fileInputRef.current.value = '';
    onPick(null);
  };

  return (
    <div
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      style={{
        flex: 1, minWidth: 260,
        border: '1px solid #ccc', borderRadius: 4, padding: 12,
        background: '#fafafa',
        transition: 'background 80ms, border-color 80ms',
        ...dragStyle,
      }}
    >
      <div style={{ fontWeight: 700, marginBottom: 8 }}>{label}</div>

      {roleSelector}

      {/* Drop zone visible cuando no hay imagen aún */}
      {!slot.previewUrl && (
        <div style={{
          border: '2px dashed #bbb',
          borderRadius: 4,
          padding: '24px 12px',
          textAlign: 'center',
          fontSize: 12,
          color: '#777',
          marginBottom: 8,
          background: slot.isDraggingOver ? '#dceaff' : '#fff',
        }}>
          arrastrá una imagen acá
          <br />
          <span style={{ fontSize: 10, color: '#999' }}>o usá el botón ↓</span>
        </div>
      )}

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={(e) => onPick(e.target.files?.[0] ?? null)}
          style={{ fontFamily: 'monospace', fontSize: 12 }}
        />
        {slot.previewUrl && (
          <button
            type="button"
            onClick={handleRemove}
            title="Quitar imagen de este slot"
            style={{
              fontFamily: 'monospace', fontSize: 11, padding: '3px 8px',
              cursor: 'pointer', background: '#fff', color: '#900',
              border: '1px solid #c99', borderRadius: 3,
            }}
          >
            ✕ Quitar
          </button>
        )}
      </div>

      {slot.previewUrl && (
        <div>
          <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>original:</div>
          <img
            src={slot.previewUrl}
            alt={label}
            style={{ maxWidth: '100%', maxHeight: 240, display: 'block', border: '1px solid #ddd' }}
          />
        </div>
      )}
      {slot.result && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>alineada 112×112 (×2):</div>
          <canvas ref={canvasRefSet} style={{ border: '1px solid #ddd', display: 'block' }} />
        </div>
      )}
      {slot.error && (
        <div style={{ marginTop: 10, color: '#900', fontSize: 12, background: '#fee', padding: 8, borderRadius: 3 }}>
          {slot.error}
        </div>
      )}
    </div>
  );
}

const td: React.CSSProperties = { border: '1px solid #ccc', padding: '6px 8px' };
const th: React.CSSProperties = { border: '1px solid #ccc', padding: '6px 8px', textAlign: 'left' };

export default Comparator;
