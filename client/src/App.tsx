// =========================================
// ID: PHYLOFACE_CLIENT_001
// VERSION: v1.1
// =========================================
// App principal del cliente Track 2a. Router simple entre el comparador MVP
// y los spikes de viabilidad que validaron cada pieza del pipeline:
//   - Comparator        (PHYLOFACE_COMPARATOR): MVP — 2 fotos → cosine
//   - Spike ONNX        (PHYLOFACE_SPIKE_001): paridad ONNX Runtime Web vs Python
//   - Spike MediaPipe   (PHYLOFACE_SPIKE_002): paridad MediaPipe Tasks for Web vs Python
//   - Spike Alignment   (PHYLOFACE_SPIKE_003): paridad alineación canónica JS vs Python
//   - Spike Detection   (PHYLOFACE_SPIKE_004): pipeline e2e (detect → align → embed) JS vs Python
//
// Cambio v1.0 → v1.1: arranca el comparador anónimo (Tarea #25 subtarea c).
// Los spikes quedan accesibles para reproducir las métricas de paridad y
// debug rápido del pipeline contra fixtures Python.

import { useState } from 'react';
import Comparator from './Comparator';
import SpikeOnnx from './SpikeOnnx';
import SpikeMediapipe from './SpikeMediapipe';
import SpikeAlignment from './SpikeAlignment';
import SpikeDetection from './SpikeDetection';

type Tab = 'comparator' | 'onnx' | 'mediapipe' | 'alignment' | 'detection';

function App() {
  const [tab, setTab] = useState<Tab>('comparator');

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
        <div style={tabStyle(tab === 'comparator')} onClick={() => setTab('comparator')}>
          Comparador (MVP)
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
      {tab === 'comparator' && <Comparator />}
      {tab === 'onnx' && <SpikeOnnx />}
      {tab === 'mediapipe' && <SpikeMediapipe />}
      {tab === 'alignment' && <SpikeAlignment />}
      {tab === 'detection' && <SpikeDetection />}
    </div>
  );
}

export default App;
