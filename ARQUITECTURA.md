# ARQUITECTURA — mendelEmbeddings

Documento vivo. Refleja el diseño objetivo del proyecto y el estado actual de cada componente.

Marcado: ✅ implementado · 🔄 parcial · ⏳ pendiente

---

## Objetivo del proyecto

**Objetivo primario.** Construir una aplicación que, dadas tres fotos (un niño y sus dos progenitores), devuelva una representación interpretable de la similitud del niño con cada progenitor:

- una **similitud global** del rostro completo, y
- un **desglose por regiones faciales** (ojos, cejas, nariz, boca, contorno, …),

permitiendo entender *cuánto* y *por qué* se parece a cada uno.

**Objetivo tangencial.** Reutilizar el mismo motor de comparación para análisis de grupos grandes de individuos (por ejemplo selecciones deportivas), generando matrices de similitud, boxplots intra-grupo, distancias entre grupos y visualizaciones tipo árbol / grafo.

Las dos líneas comparten el mismo núcleo: **comparación de rostros individuales** vía dos vías complementarias — comparación global y comparación por porciones.

---

## 1. Motor de comparación

### 1.1 Adquisición e I/O ✅
Lectura RGB con `pathlib`, validaciones, normalización de formato. Implementado en notebooks y en `src/phyloface/`.

### 1.2 Detección y alineación ✅
InsightFace `buffalo_l` para detección + 5-puntos para alineación canónica del embedding global. Expansión configurable de bbox (`face_pad_x/y`). Implementado en `src/phyloface/core/detector.py` y notebooks.

### 1.3 Landmarks densos ✅
**MediaPipe Face Mesh** (468 puntos 3D, 478 con `refine_landmarks`) como fuente primaria para regiones. Migrado al paquete en `phyloface.landmarks` (Tarea #1, Paso 2). Los landmarks de InsightFace (`landmark_2d_106`, `landmark_3d_68`) quedan reservados para alineación interna.

### 1.4 Segmentación por regiones 🔄
- **Regiones rectangulares (v2)** ✅ — bboxes derivadas de landmarks. Migrado a `phyloface.regions.extract_rect` (Tarea #1, Paso 3).
- **Regiones con máscara poligonal (v2 masked)** ✅ — máscaras precisas sobre polígonos faciales. Migrado a `phyloface.regions.extract_masked` (Tarea #1, Paso 4).
- **Lista canónica de regiones** ⏳ — formalizar contrato: ojos izq/der, cejas, nariz, boca, mentón, frente, contorno. (Tarea #2, pendiente.)
- *Deuda histórica*: anotar qué fue "v1" y por qué se descartó. (Tarea #3, pendiente.)

### 1.5 Embeddings
- **Global** ✅ — vector 512D de `recognition (w600k_r50.onnx)`. Re-extracción sobre cara alineada migrada a `phyloface.core.embedder` (Tarea #1, Paso 6b).
- **Por región — Nivel B (visual)** 🔄 — comparación visual por región disponible en `phyloface.comparator_regional` (Tarea #1, Paso 5) usando grayscale + z-score + coseno; aún NO re-aplicamos el modelo de reconocimiento al crop/máscara (eso es lo que falta para Nivel B "real"). (Tarea #5, pendiente.)
- **Por región — Nivel A (geométrico)** ⏳ — distancias entre landmarks, proporciones, ángulos, simetrías. Aún no existe. (Tarea #4, pendiente.)

### 1.6 Métricas y calibración 🔄
- Coseno (principal) + euclídea (control) ✅.
- **Calibración de umbrales con dataset propio** ⏳ — pasar de valores absolutos del modelo a umbrales data-driven validados sobre KinFaceW / TSKinFace.

### 1.7 Caché de embeddings ✅ *(en evolución)*
`src/phyloface/core/cache.py` + `data/cache/faces/`. Falta extender la clave para soportar embeddings por región y versión de regiones.

### 1.8 Registro de modelos / adapters 🔄
`src/phyloface/core/models.py` y `model_validator.py` existen pero son cortos. Falta adapter para `antelopev2` (hoy roto) y para futuros backbones.

### 1.9 Interpretabilidad / explicabilidad ⏳
- **Heatmap por regiones** ⏳ — colorear cada región sobre la cara según su contribución al score global.
- **Occlusion sensitivity** ⏳ — ventana deslizante (12×12 o 16×16, stride 4–6) que tapa zonas de A, recalcula embedding, mide caída del coseno contra B. Es el camino que la charla externa deja recomendado para arrancar.
- **Grad-CAM / saliency** ⏳ — capa posterior, más cara y dependiente del modelo.

### 1.10 Sistema de pesos por región ⏳
Combinación score global + scores regionales con pesos diferenciados (ojos / perioculares > nariz, boca, contorno). Calibrable contra dataset propio.

---

## 2. Aplicaciones

### 2.1 App primaria — Parecido niño ↔ progenitores ⏳
- Input: foto del niño + foto del padre + foto de la madre.
- Output: score global × 2, scores por región × 2, visualización con overlay + radar/spider, heatmap de contribución.
- **Objetivo final del proyecto.**

### 2.2 App secundaria — Comparación pareada dentro de un grupo ✅
`run_pairwise_heatmap.py` + `MultiFaceComparator`. Heatmap N×M operativo.

### 2.3 App secundaria — Comparación entre grupos ⏳
Distancia agregada entre selecciones → grafo / dendrograma. Idea original del slide 1.

---

## 3. Visualización

Primitivas:
- Heatmap N×M ✅
- Overlay de regiones sobre cara ✅
- Detalle por región ✅
- Boxplot intra-grupo ⏳
- Dendrograma ⏳
- Grafo inter-grupos ⏳
- Radar/spider de scores regionales ⏳
- Heatmap de oclusión ⏳

---

## 4. Validación y benchmarking ⏳

Necesario para tener criterio al elegir modelo / métrica / pesos.

### 4.1 Datasets externos disponibles
Ver `data/input/datasets/DATASETS.md` para descripción detallada.

- **KinFaceW-I** — 533 pares uno-a-uno (fotos distintas).
- **KinFaceW-II** — 1000 pares uno-a-uno (misma foto en la mayoría).
- **TSKinFace** — 1015 tríos padre-madre-hijo/a (FM-S 513 + FM-D 502).
- **TSKinFace_SIFT** — features SIFT pre-extraídas (referencia de baseline).
- *No descargados*: FIW (Families In the Wild), KFVW, Family101.

### 4.2 Dataset propio
Las selecciones de fútbol ya curadas en `data/input/img/teams/` para la app secundaria (2.2 / 2.3).

### 4.3 Métricas de éxito
- Accuracy de verificación de parentesco (KinFaceW protocol).
- Separabilidad inter-grupo (objetivo tangencial).
- Estabilidad ante pose, iluminación, edad.

### 4.4 Protocolo de comparación de modelos
Misma evaluación corriendo en `buffalo_l`, `antelopev2`, otros — resultados tabulados.

---

## 5. Stack web (público) 🔄

**Última actualización**: 2026-05-20 (tarde). Tarea #1 cerrada ese mismo día; arrancamos sesión de diseño del stack web.

### 5.1 Snapshot detallado (versionado)

El diseño completo vive como **snapshot inmutable versionado** en `_meta/arquitectura_web/`:

- `v0.1_2026-05-20_arquitectura_web.md` (+ `.pdf` generado) — versión inicial post-Tarea #1.

Cada ajuste de fondo crea una `vN.M_<fecha>_arquitectura_web.md` al lado; las versiones anteriores NO se editan retroactivamente. Esta sección de `ARQUITECTURA.md` solo lleva el resumen del diseño **vigente** + estado de decisiones.

### 5.2 Resumen del diseño vigente (v0.1)

**Arquitectura híbrida cliente/servidor con 2 tracks paralelos**:

| Track | Caso de uso | Stack | Privacidad |
|-------|-------------|-------|------------|
| **1. Vitrina equipos** | Comparar planteles, viz tipo heatmap/dendrograma/boxplot. Datos curados por Diego. | Cliente Vite + React + JSON pre-calculado. Sin server, sin DB, sin auth. | Datos públicos (jugadores profesionales). |
| **2a. Comparador anónimo** | Niño + padres. Inference **100% en browser** vía ONNX Runtime Web + MediaPipe Tasks for Web (mismo modelo `w600k_r50.onnx` que Python). | Cliente Vite + React + onnxruntime-web. Sin server. | Imágenes **nunca salen del browser**. |
| **2b. Refinamiento server** | Registrados opt-in. Cliente envía embeddings ya calculados + imagen → server hace features avanzados (Nivel B regional, occlusion, calibración). | FastAPI + ARQ + Postgres + Redis. Auth managed (Clerk/Supabase). Storage S3-compatible. | Consentimiento explícito + retención máx 90 días + DELETE on demand. |

### 5.3 Estado de decisiones (W1–W8)

| # | Tema | Estado | Resolución |
|---|------|--------|-----------|
| W1 | Framework FE | ✅ Cerrada | React + Vite SPA pura |
| W2 | Auth | ⏸ Diferida (Track 2b) | Clerk o Supabase Auth managed (OAuth + magic link) |
| W3 | Persistencia | ⏸ Diferida (Track 2b) | Postgres (Neon) + Redis (Upstash). En el medio: IndexedDB local en cliente. |
| W4 | Procesamiento pesado | ⏸ Diferida (Track 2b) | ARQ + WebSocket / polling |
| W5 | Imágenes subidas | ✅ Resuelta vía arquitectura | Track 2a: nunca salen del browser. Track 2b: opt-in con consentimiento. |
| W6 | Hosting / deploy | 🟡 Parcial | Hosting estático (CF Pages / Netlify / GH Pages, sin decidir aún) para cliente. Fly.io para server cuando entre Track 2b. |
| W7 | Estructura repo | ✅ Cerrada | Monorepo: `client/`, `scripts/`, `src/phyloface/`, `server/` |
| W8 | Modelos en memoria | ⏸ Diferida (Track 2b) | Warm pool, una instancia del modelo por worker |

### 5.4 Próximo paso inmediato

**Sesión siguiente**: spike de viabilidad del Track 2a — verificar que ONNX Runtime Web carga `w600k_r50.onnx` en el browser y produce embeddings cuasi bit-idénticos a Python. Si falla, todo el plan híbrido se cae y hay que ir a una v0.2 con plan revisado.

Detalle, riesgos identificados, glosario y plan completo → ver `_meta/arquitectura_web/v0.1_2026-05-20_arquitectura_web.md`.

### 5.5 Tareas web en `TAREAS_PENDIENTES.md`

Se crean **a medida** que se necesitan, no como bloque grande adelantado. Esto evita que tareas planificadas hoy queden obsoletas si el spike de v0.1 obliga a una v0.2.

---

## Decisiones de diseño cerradas

| Tema                          | Decisión                                       |
|-------------------------------|------------------------------------------------|
| Backbone global inicial       | InsightFace / ArcFace (`buffalo_l`)            |
| Landmarks densos              | MediaPipe Face Mesh (468 puntos)               |
| Métrica principal             | Similitud coseno                               |
| Métrica de control            | Distancia euclídea                             |
| Estrategia regional           | Dos niveles: A geométrico + B visual           |
| Combinación de scores         | Sistema de pesos (no promedio ingenuo)         |
| Heatmap explicativo inicial   | Occlusion sensitivity (no Grad-CAM)            |
| Modularidad                   | Bloques desacoplados con API estable           |
| Datasets de validación        | KinFaceW-I, KinFaceW-II, TSKinFace             |

---

## Decisiones abiertas

1. **Materialización del Nivel A geométrico** — lista canónica de features (distancias entre qué pares, ángulos de qué triángulos, proporciones).
2. **Forma del score final** de la app primaria — ¿número global + N regionales?, ¿agregado calibrado a probabilidad?, ¿ambos?
3. **Versionado de regiones** — cómo identificar v2 / v2-masked / vN en el caché para no mezclar.
4. **MediaPipe vs InsightFace para landmarks** — confirmar que MediaPipe es el único path para regiones; `landmark_2d_106` / `3d_68` solo para alineación.
5. **Orden de ataque** — sugerido: terminar 1.x del motor + migrar de notebook a paquete + atacar 2.2 robustecida → calibración con KinFaceW → 2.1 → 2.3.
