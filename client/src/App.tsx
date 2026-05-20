// =========================================
// ID: PHYLOFACE_CLIENT_001
// VERSION: v1.0
// =========================================
// App principal del cliente Track 2a. Por ahora actúa como **router simple**
// entre los dos spikes de viabilidad:
//   - Spike ONNX        (PHYLOFACE_SPIKE_001): paridad ONNX Runtime Web vs Python
//   - Spike MediaPipe   (PHYLOFACE_SPIKE_002): paridad MediaPipe Tasks for Web vs Python
//
// Cuando el Track 2a real arranque, este App.tsx será reemplazado por el
// comparador real (subir 2 fotos, detectar, alinear, embedding, comparar).
// Por ahora mantiene un toggle tipo tabs para validar las dos puntas
// independientes antes de unificarlas.

import { useState } from 'react';
import SpikeOnnx from './SpikeOnnx';
import SpikeMediapipe from './SpikeMediapipe';

type Tab = 'onnx' | 'mediapipe';

function App() {
  const [tab, setTab] = useState<Tab>('mediapipe');

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
        <div style={tabStyle(tab === 'onnx')} onClick={() => setTab('onnx')}>
          Spike ONNX (embedding)
        </div>
        <div style={tabStyle(tab === 'mediapipe')} onClick={() => setTab('mediapipe')}>
          Spike MediaPipe (landmarks)
        </div>
      </div>

      {/* Contenido */}
      {tab === 'onnx' && <SpikeOnnx />}
      {tab === 'mediapipe' && <SpikeMediapipe />}
    </div>
  );
}

export default App;
