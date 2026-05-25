# mendelEmbeddings

Comparación **interpretable** de parecido facial con embeddings. Dadas las fotos de
un niño/a y sus dos progenitores, estima cuánto se parece a cada uno —a nivel
**global** y **por regiones faciales** (ojos, nariz, boca, contorno…)— para entender
no solo *cuánto* sino *por qué*.

> **Estado: experimental / en desarrollo.** El motor de comparación (Python) y una
> demo web browser-first ya funcionan; varias piezas (features geométricas Nivel A,
> calibración, app primaria) están en curso. El estado fino por componente
> (✅ / 🔄 / ⏳) vive en [`ARQUITECTURA.md`](ARQUITECTURA.md).

## Qué hace

- **Objetivo primario** — App "niño ↔ progenitores": 3 fotos → similitud global ×2 +
  scores por región ×2 + visualización (overlay, radar, heatmap de contribución).
- **Objetivo tangencial** — Reutilizar el mismo motor para comparar grupos grandes
  (p. ej. planteles deportivos): matrices de similitud, boxplots intra-grupo,
  distancias inter-grupo, árboles / grafos.

Ambas líneas comparten el núcleo: comparación de rostros por dos vías
complementarias — **global** (embedding 512-D de todo el rostro) y **por regiones**
(porciones segmentadas con landmarks densos).

## Stack

**Motor (Python):**
- **InsightFace / ArcFace `buffalo_l`** — detección + alineación canónica + embedding
  global `w600k_r50` (512-D). El bundle `buffalo_l` trae detección (SCRFD/`det_10g`),
  reconocimiento (`w600k_r50`), landmarks y `genderage`.
- **MediaPipe Face Mesh** (468 / 478 puntos) — landmarks densos para la segmentación
  por regiones (`phyloface.landmarks`).
- ONNX Runtime, OpenCV, scikit-image / scikit-learn, NumPy.
- Métrica principal: **similitud coseno**; control: **euclídea** (sobre embeddings
  L2-normalizados).

**Cliente web (`client/`):** React 19 + Vite + TypeScript. Inferencia **100% en el
browser** vía `onnxruntime-web` + `@mediapipe/tasks-vision` (mismo `w600k_r50.onnx`
que el motor Python). Las imágenes **nunca salen del navegador**.

Diseño completo del stack web (arquitectura híbrida cliente/servidor, 2 tracks) →
[`ARQUITECTURA.md`](ARQUITECTURA.md) §5.

## Estructura del repo

```
mendelEmbeddings/
├── src/phyloface/        # Motor de comparación (paquete Python)
│   ├── core/             #   io, detector, embedder, comparator, metrics, cache, models…
│   ├── landmarks/        #   MediaPipe Face Mesh
│   ├── regions/          #   segmentación por región (rect / máscara) + geometría
│   ├── benchmark/        #   calibración + protocolo KinFaceW
│   ├── viz/              #   heatmaps, overlays de regiones / landmarks
│   └── app/              #   runners (heatmap pareado, batch por carpeta, build cache)
├── client/               # Demo web (React + Vite + TS); inferencia en el browser
├── scripts/              # Calibración, spikes de paridad JS↔Python, monitoreo de recursos
├── notebooks/            # Notebooks exploratorios (jupytext, ver abajo)
├── tests/smoke/          # Smoke tests versionados
├── data/                 # Datos locales (gitignored, ver "Datos")
├── ARQUITECTURA.md       # Diseño vivo + estado por componente
├── DATASETS.md           # Datasets de validación (en data/input/datasets/)
├── DEVLOG.md             # Historial técnico por commit
└── TAREAS_PENDIENTES.md  # Backlog con IDs
```

El paquete vive bajo `src/`; el nombre de import es `phyloface`.

## Requisitos

- **Python 3.11** con las dependencias de [`requirements.txt`](requirements.txt)
  (`insightface`, `onnxruntime`, `opencv-python`, `scikit-learn`, …) **más
  `mediapipe`** (lo usa `phyloface.landmarks` y hoy no está pineado en
  `requirements.txt`). El entorno de referencia es el conda env `face-sim`.
- **Node ≥ 20.19** para el cliente (lo exige Vite). Hay un [`.nvmrc`](.nvmrc) con
  Node 20 → `nvm use`.

## Instalación

```bash
# Motor Python (conda recomendado; el env de referencia es face-sim)
conda create -n face-sim python=3.11
conda activate face-sim
pip install -r requirements.txt
pip install mediapipe          # landmarks densos (no incluido en requirements.txt)

# Cliente web
cd client
nvm use                        # toma .nvmrc -> Node 20
npm install
```

InsightFace descarga los pesos de `buffalo_l` la primera vez que se usa. Para el
cliente, `w600k_r50.onnx` debe estar en `client/public/models/` (gitignored:
copialo / descargalo localmente).

## Cómo correr

**Cliente web (demo):**

```bash
cd client
nvm use
npm run dev                    # http://localhost:5173
# build estático:
npm run build && npm run preview
```

La demo incluye el comparador 3-slot (niño + progenitores), el árbol genealógico
local (IndexedDB, con import/export) y la pestaña de calibración.

**Scripts del motor (ejemplos):**

```bash
conda activate face-sim
python scripts/run_calibration_kinfacew.py     # calibración de umbrales sobre KinFaceW
python scripts/verify_onnx_web_parity.py       # paridad de embeddings Python ↔ ONNX-web
```

**Smoke tests:** ver [`tests/smoke/`](tests/smoke/) (convención de smoke tests
versionados; cada uno documenta su ejecución).

## Notebooks (jupytext)

Los notebooks de `notebooks/` se versionan como scripts `.py` en formato *percent*,
emparejados con sus `.ipynb` vía **jupytext** (`jupytext.toml`:
`formats = "ipynb,py:percent"`, sin metadata para diffs limpios). El `.ipynb` es la
fuente que editás en Jupyter; el `.py` es lo que se commitea (los `.ipynb` están
gitignored).

```bash
./sync_notebooks.sh            # exporta cada .ipynb no vacío -> .py:percent y lo agrega al staging
```

`sync_notebooks.sh` exporta siempre en sentido `ipynb → py`, ignora notebooks
vacíos y no modifica los `.ipynb`.

## Datos (`data/`, gitignored)

Todo `data/` está fuera de git. Layout esperado:

```
data/
├── input/
│   ├── img/teams/          # selecciones curadas (app de grupos)
│   ├── img/spike_e2e_set/  # set curado para spikes de paridad
│   └── datasets/           # KinFaceW-I, KinFaceW-II, TSKinFace (+ DATASETS.md)
├── cache/faces/            # embeddings cacheados (clave por SHA-256)
└── output/                 # resultados, figuras, JSON de calibración
```

Descripción de los datasets de validación (KinFaceW-I/II, TSKinFace) →
[`DATASETS.md`](DATASETS.md), en `data/input/datasets/`.

## Documentación

| Documento | Qué es |
|-----------|--------|
| [`ARQUITECTURA.md`](ARQUITECTURA.md) | Diseño vivo + estado ✅/🔄/⏳ por componente |
| [`DATASETS.md`](DATASETS.md) | Datasets de validación |
| [`DEVLOG.md`](DEVLOG.md) | Historial técnico por commit |
| [`TAREAS_PENDIENTES.md`](TAREAS_PENDIENTES.md) | Backlog con IDs |
| [`AGENTS.md`](AGENTS.md) · [`AGENTS_HANDOFF.md`](AGENTS_HANDOFF.md) | Convención de trabajo multi-agente |

## Privacidad

En el comparador web (Track 2a) las imágenes se procesan **localmente en el browser**
y nunca se suben a ningún servidor. El refinamiento server-side (Track 2b) es futuro y
estrictamente opt-in con consentimiento.

## Trabajo multi-agente

El repo lo trabajan varios agentes de IA (Claude Code, Codex). La coordinación pasa
por un único canal versionado, `AGENTS_HANDOFF.md`, y las instrucciones comunes viven
en `AGENTS.md`. Si vas a trabajar con un agente acá, empezá por esos dos archivos.
