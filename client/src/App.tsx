// =========================================
// ID: PHYLOFACE_CLIENT_001
// VERSION: v1.5
// =========================================
// Cambio v1.4 → v1.5 (Tarea #12): se agrega la pestaña "App primaria"
// (PHYLOFACE_APP_PRIMARIA) como PRIMERA y por DEFECTO — es el objetivo final del
// proyecto (ARQUITECTURA §2.1): 3 fotos (Padre·Hijo/a·Madre) → veredicto
// interpretable (global calibrado + herencia por región) reusando el motor del
// Comparador + el panel de scores regionales (#30) + la calibración (#6).
//
// App principal del cliente. Router simple entre:
//   - AppPrimaria       (PHYLOFACE_APP_PRIMARIA): App primaria #12 — ¿A quién se parece?
//   - Comparator        (PHYLOFACE_COMPARATOR): MVP Track 2a — 3 slots (P1 · Hijo · P2)
//   - GenealogyTree     (PHYLOFACE_GENEALOGY_TREE): MVP Track 2b — árbol genealógico (Tarea #26)
//   - CalibrationTab    (PHYLOFACE_CALIBRATION_TAB): Tarea #6 Fase B — histogramas + métricas
//   - Spike ONNX        (PHYLOFACE_SPIKE_001): paridad ONNX Runtime Web vs Python
//   - Spike MediaPipe   (PHYLOFACE_SPIKE_002): paridad MediaPipe Tasks for Web vs Python
//   - Spike Alignment   (PHYLOFACE_SPIKE_003): paridad alineación canónica JS vs Python
//   - Spike Detection   (PHYLOFACE_SPIKE_004): pipeline e2e (detect → align → embed) JS vs Python
//
// Cambio v1.3 → v1.4 (Tarea #6 Fase B): solapa "Calibración" que lee el JSON
// de calibración (servido en public/calibration/) y dibuja los histogramas
// kin vs non-kin + las métricas por relación. El popup que ubica un cosine
// concreto (CalibrationModal) se abre desde el Comparador y el árbol.
//
// Cambio v1.2 → v1.3 (Tarea #26 iter tripleta): App escucha
// CustomEvent("phyloface-go-to-tab", { detail: Tab }) en window. Lo usa
// `GenealogyTree.tsx` cuando el usuario pide "→ abrir en Comparador MVP"
// desde el modal de tripleta. El handoff es asincrónico (el modal escribe
// `localStorage["phyloface-comparator-prefill"]` y dispara el event; el
// Comparator lee el prefill al montar).
//
// Cambio v1.1 → v1.2 (Tarea #26 paso 2): sumar tab "Árbol genealógico" que
// monta el MVP paso 2 (lista plana de personas + persistencia IDB). El SVG
// del pedigree y la comparación on-demand llegan en pasos 4-5.
//
// Cambio v1.0 → v1.1: arranca el comparador anónimo (Tarea #25 subtarea c).
// Los spikes quedan accesibles para reproducir las métricas de paridad y
// debug rápido del pipeline contra fixtures Python.

import { useEffect, useState } from 'react';
import AppPrimaria from './AppPrimaria';
import Comparator from './Comparator';
import GenealogyTree from './GenealogyTree';
import CalibrationTab from './CalibrationTab';
import SpikeOnnx from './SpikeOnnx';
import SpikeMediapipe from './SpikeMediapipe';
import SpikeAlignment from './SpikeAlignment';
import SpikeDetection from './SpikeDetection';

type Tab = 'primary' | 'comparator' | 'genealogy' | 'calibration' | 'onnx' | 'mediapipe' | 'alignment' | 'detection';

const VALID_TABS: Tab[] = ['primary', 'comparator', 'genealogy', 'calibration', 'onnx', 'mediapipe', 'alignment', 'detection'];

function App() {
  // Default = App primaria (#12), el objetivo final del proyecto (ARQUITECTURA §2.1).
  const [tab, setTab] = useState<Tab>('primary');

  // Listener global para cambio de tab vía CustomEvent. Permite que componentes
  // hijos (TripletModal en GenealogyTree) salten al Comparador MVP sin tener
  // que pasar setTab por prop drilling. El handoff completo además escribe
  // `localStorage["phyloface-comparator-prefill"]`; Comparator lo lee al
  // montar.
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (typeof detail === 'string' && (VALID_TABS as string[]).includes(detail)) {
        setTab(detail as Tab);
      }
    };
    window.addEventListener('phyloface-go-to-tab', handler);
    return () => window.removeEventListener('phyloface-go-to-tab', handler);
  }, []);

  const tabStyle = (active: boolean) => ({
    padding: '10px 20px',
    cursor: 'pointer',
    border: '1px solid #ccc',
    borderBottom: active ? '1px solid #fff' : '1px solid #ccc',
    background: active ? '#fff' : '#f4f4f4',
    fontWeight: active ? 700 : 400,
    fontFamily: 'monospace',
    fontSize: 13,
  });

  return (
    <div>
      {/* Barra de tabs */}
      <div style={{
        display: 'flex',
        gap: 4,
        padding: '10px 20px 0',
        borderBottom: '1px solid #ccc',
        marginBottom: -1,
      }}>
        <div style={tabStyle(tab === 'primary')} onClick={() => setTab('primary')}>
          App primaria
        </div>
        <div style={tabStyle(tab === 'comparator')} onClick={() => setTab('comparator')}>
          Comparador (MVP)
        </div>
        <div style={tabStyle(tab === 'genealogy')} onClick={() => setTab('genealogy')}>
          Árbol genealógico
        </div>
        <div style={tabStyle(tab === 'calibration')} onClick={() => setTab('calibration')}>
          Calibración
        </div>
        <div style={tabStyle(tab === 'onnx')} onClick={() => setTab('onnx')}>
          Spike ONNX (embedding)
        </div>
        <div style={tabStyle(tab === 'mediapipe')} onClick={() => setTab('mediapipe')}>
          Spike MediaPipe (landmarks)
        </div>
        <div style={tabStyle(tab === 'alignment')} onClick={() => setTab('alignment')}>
          Spike alignment (warp 112×112)
        </div>
        <div style={tabStyle(tab === 'detection')} onClick={() => setTab('detection')}>
          Spike detección (e2e)
        </div>
      </div>

      {/* Contenido */}
      {tab === 'primary' && <AppPrimaria />}
      {tab === 'comparator' && <Comparator />}
      {tab === 'genealogy' && <GenealogyTree />}
      {tab === 'calibration' && <CalibrationTab />}
      {tab === 'onnx' && <SpikeOnnx />}
      {tab === 'mediapipe' && <SpikeMediapipe />}
      {tab === 'alignment' && <SpikeAlignment />}
      {tab === 'detection' && <SpikeDetection />}
    </div>
  );
}

export default App;
