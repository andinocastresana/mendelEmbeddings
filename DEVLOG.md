# DEVLOG — mendelEmbeddings

Historial técnico de cambios del proyecto.
Los IDs de tarea referencian `TAREAS_PENDIENTES.md`.

Notación de tareas: `T1` = relacionada · `T1↑` = tarea creada · `T1✓` = tarea cerrada

Cada entrada incluye: hash de commit, título de una línea, IDs de tarea relacionados y detalle del cambio (mismo nivel de descripción que las entradas existentes). Las entradas se agrupan por fecha en orden cronológico inverso (lo más nuevo arriba).

---

## 2026-05-25

### `271da30` · [codex] Tareas #4/#5/#7 — features regionales, embeddings regionales y cache versionada `T4✓` `T5✓` `T7✓`

Se abre el bloque regional posterior al contrato canónico de #2/#3. La decisión
fue separar API pura y cache antes de usar embeddings regionales como señal de
producto.

- **#4 cerrado**: `src/phyloface/regions/geometric_features.py` agrega
  `region_geometry`, `face_geometric_features` y `pair_geometric_differences`.
  Calcula bboxes/centroides, distancias, proporciones, ángulos y simetrías
  izquierda/derecha normalizadas por distancia interocular. Re-export desde
  `phyloface.regions`.
- **#7 cerrado**: `src/phyloface/core/cache.py` extiende `make_config_dict` con
  `regions_version`, `region_extraction_mode` y `region_embedding_model`; el
  `config_id` ahora diferencia, por ejemplo,
  `regions-v2.0__masked__w600k_r50`. `save_image_cache` soporta arrays
  regionales opcionales (`region_names`, `region_embeddings`, `region_bboxes`,
  `region_mask_fill`, `region_valid`) sin romper caches existentes.
- **#5 iniciado**: `src/phyloface/regions/regional_embeddings.py` define
  `REGIONAL_EMBEDDINGS_VERSION = "regions-v2.0+arcface-crop-v0.1"`, extracción
  de embeddings por crop/máscara regional, comparación por coseno y serialización
  a arrays. `scripts/validate_region_embeddings_kinfacew.py` corre sanity contra
  KinFaceW.
- **Resultado #5**: KinFaceW-I `--limit 12` corrió end-to-end; luego se amplió a
  `--limit 40` con `--progress-every 5`, `--cool-threshold 80`, `--cool-secs 15`.
  La corrida ampliada tuvo 20 fallos de imagen y AUC regionales débiles: máximo
  `left_cheekbone` 0.621 / `right_cheekbone` 0.620, resto cerca de azar.
  Conclusión: `regions-v2.0+arcface-crop-v0.1` no sirve como base de #11; ArcFace
  full-face reaplicado a parches regionales no aporta señal suficiente sin
  adaptación.
- **Verificación**: `py_compile` OK; smoke
  `tests/smoke/test_regions_level_a_and_cache.py` OK. Logs:
  `_meta/TAREA5_region_embeddings_sanity.log`,
  `_meta/TAREA5_region_embeddings_sanity_resources.log`,
  `_meta/TAREA5_region_embeddings_limit40.log`,
  `_meta/TAREA5_region_embeddings_limit40_resources.log`, y
  `_meta/TAREA5_region_embeddings_limit40.json`. Corrida `--limit 40`:
  `cpu_avg=27%`, `cpu_max=71%`, `temp_avg=66°C`, `temp_max=96°C`.
- **Backlog**: #4, #5 y #7 quedan cerradas. Siguiente recomendado: #11 debe usar
  features geométricas (#4), scores visuales regionales existentes o un diseño
  regional entrenado/adaptado para parches, no embeddings ArcFace regionales crudos.

### `1949296` · [codex] Tarea #29 — CCMTL-lite full-face antes de regiones `T29✓`

Se abre una tarea intermedia tras leer `data/input/docs/notebookLM_SoTA_I.pdf` y
`data/input/docs/notebookLM_SoTA_II.pdf`: antes de pasar a regiones/NMP, evaluar si
queda mejora full-face de baja capacidad. La cabeza MLP de #6 no superó el baseline,
pero eso solo descarta una frontera flexible sobre embeddings 512-d con pocos datos.

- **Nota técnica**: `_meta/FULLFACE_MULTITASK_TAREA29.md` conserva la lectura de los
  PDFs: CCMTL es la dirección más aprovechable localmente; FNN/ViT/aging GAN quedan
  como investigación posterior por costo/datos; KinFaceW-I sigue siendo métrica
  primaria y KinFaceW-II referencia secundaria por same-photo bias.
- **Script nuevo**: `scripts/evaluate_fullface_multitask.py` compara baseline Youden
  de coseno/euclídea contra regresiones logísticas pequeñas: global, global
  cos+euc, offsets por relación FS/MD/FD/MS, slopes por relación, y modelos
  independientes por relación. Usa folds oficiales, sin fuga train/test.
- **Resultado KinFaceW-I completo**: baseline Youden coseno/euclídea acc 0.666 /
  AUC 0.727. Los modelos compartidos por relación suben apenas el AUC
  (`shared_offsets` 0.734, `shared_slopes` 0.736) pero bajan accuracy
  (`0.660`/`0.659`). Los modelos independientes por relación no superan los AUC
  históricos del baseline. Conclusión: señal marginal/no accionable; no reemplaza
  al calibrador full-face actual.
- **Logs**: `_meta/TAREA29_fullface_multitask.log` y
  `_meta/TAREA29_fullface_multitask_resources.log` (`cpu_avg=39%`, `cpu_max=61%`,
  `temp_avg=81°C`, `temp_max=95°C`). Próxima repetición con batch 60-80.
- **Backlog**: #29 se cierra. Siguiente recomendado: features nuevas
  (#4/#5/#7 antes de #11) en vez de más clasificadores full-face.

### `8edd1dd` · [codex] Tareas #2/#3 — contrato canónico de regiones y deuda regions v1 `T2✓` `T3✓`

**Cierre de las tareas base del bloque regional M1.4.** Se formaliza la lista de
regiones como contrato del motor y se documenta la deuda histórica de `regions v1`
para no volver a depender de definiciones implícitas del notebook.

- **Nuevo registry canónico**: `src/phyloface/regions/canonical.py`
  (`PHYLOFACE_REGIONS_CANONICAL v1.0`) define `CANONICAL_REGIONS_VERSION =
  "regions-v2.0"` y 12 `RegionSpec`:
  `left_eyebrow`, `right_eyebrow`, `left_eye`, `right_eye`,
  `left_cheekbone`, `right_cheekbone`, `left_cheek`, `right_cheek`, `nose`,
  `mouth`, `chin`, `forehead`.
- Cada región declara nombre estable, etiqueta, grupo anatómico, lateralidad,
  par simétrico, fuente (`mediapipe-official`, `manual-approx`,
  `derived-approx`), `landmark_idx`, `polygon_idx`, padding, estrategia de bbox,
  estrategia de máscara y modos de extracción (`rect`, `masked`).
- `src/phyloface/regions/__init__.py` re-exporta el registry y helpers:
  `get_region_spec`, `regions_for_group`, `paired_region_names`.
- **Deuda histórica de regions v1**: `_meta/REGIONS_V1_DEBT.md` documenta que
  `regions v1` no era una API estable sino el período experimental previo a
  `regions_v2`: nombres/listas repetidas localmente, sin versión de contrato ni
  semántica de cache, y con métricas regionales visuales todavía no equivalentes
  a embeddings regionales reales. Tras leer el protocolo del KG, se corrigió el
  rescate: los episodios relevantes viven en `IA/memories/_meta/episodes/` con
  `project: mendelEmbeddings`; se incorporaron las lecciones de verificar el
  estado real del código antes de proponer pendientes, no asumir equivalencia
  entre implementaciones homónimas y validar visualmente convenciones Face Mesh.
- **Smoke test**: `tests/smoke/test_regions_canonical.py` valida versión, 12
  nombres, índices dentro del rango MediaPipe Face Mesh 0..477, estrategias
  válidas y simetrías izquierda/derecha.
- **Backlog**: `TAREAS_PENDIENTES.md` mueve #2 y #3 a completadas. No se cambia
  aún la lógica de extracción; el contrato congela el comportamiento vigente para
  futuras migraciones (#4/#5/#7/#9/#11).

### `183064c` · [codex] Tarea #6 — disclaimer KinFaceW-II y evaluación completa de cabeza MLP `T6`

**Cierre del experimento MLP de la Tarea #6 y refuerzo metodológico de
KinFaceW-II.** El objetivo era decidir si una cabeza aprendida pequeña sobre
embeddings ArcFace justificaba avanzar a export ONNX/browser, y dejar explícito
el sesgo de KinFaceW-II antes de reportar ese dataset.

- **Disclaimer KinFaceW-II versionado en contrato y UI**:
  - `scripts/run_calibration_kinfacew.py` ahora emite `primaryDataset`,
    `evaluationRole` y `warning` en el artefacto JSON. KinFaceW-I queda como
    evaluación primaria; KinFaceW-II queda marcado como referencia secundaria
    sesgada por "same-photo".
  - `client/src/lib/calibration.ts` tipa esos campos y expone
    `calibrationWarning`.
  - `CalibrationTab.tsx` y `CalibrationModal.tsx` muestran una advertencia
    metodológica cuando el artefacto cargado la trae o cuando el dataset es
    KinFaceW-II.
  - `client/public/calibration/KinFaceW-I_calibration.json` se actualizó de forma
    no disruptiva (`warning: null`, `evaluationRole: primary`) para mantener el
    contrato actual alineado.

- **Cabeza MLP experimental**:
  - Nuevo `scripts/train_kinship_mlp.py`: entrenamiento/evaluación con folds
    oficiales de KinFaceW, usando features de par
    `abs(e1-e2)` (512) + `e1*e2` (512) + cosine + euclidean = 1026 dims.
  - Modelo inicial: `sklearn.neural_network.MLPClassifier`, capas `(64, 32)`,
    `early_stopping=True`, una evaluación por relación y otra agregada `ALL`.
  - El `--limit` para smokes ahora es estratificado por `fold+label`, evitando
    folds degenerados con una sola clase.

- **Corrida completa KinFaceW-I monitoreada**:
  - Comando envuelto con `scripts/test-monitored.sh`, sin `--limit`.
  - Resultado: la MLP **no mejora** el baseline de cosine crudo calibrado.

| relación | baseline acc | baseline AUC | MLP acc | MLP AUC | Δ AUC |
|---|---:|---:|---:|---:|---:|
| FS | 0.731 | 0.812 | 0.615 | 0.672 | -0.140 |
| MD | 0.655 | 0.746 | 0.649 | 0.708 | -0.038 |
| FD | 0.624 | 0.677 | 0.512 | 0.531 | -0.146 |
| MS | 0.599 | 0.681 | 0.491 | 0.514 | -0.167 |
| ALL | 0.666 | 0.727 | 0.647 | 0.710 | -0.017 |

- **Conclusión**: no conviene exportar esta MLP a ONNX todavía. Mantener el
  cosine calibrado como baseline principal y pasar a regiones canónicas (#2/#3),
  salvo que se quiera probar variantes más regularizadas.
- **Artefactos**:
  - `_meta/CALIBRACION_TAREA6_MLP_INFORME.pdf`: informe completo.
  - `_meta/CALIBRACION_TAREA6_mlp_full.log`: salida completa de corrida.
  - `_meta/CALIBRACION_TAREA6_mlp_full_resources.log`: CPU/temperatura.
  - `_meta/CALIBRACION_TAREA6.md`: nota viva actualizada con resultados.
- **Recursos**: corrida viable pero caliente: 33 muestras, CPU avg/max
  `40%/76%`, temp avg/max `81.2°C/98°C`, 19 muestras `>=85°C`, 6 `>=95°C`.
  Futuras corridas deben bajar batch size o aumentar pausas.
- **Verificación**: `npm run build` OK con Node 20; `json.tool` OK para JSONs de
  calibración; `py_compile` OK para scripts; PDF validado con `pdftotext`.

### `6a68553` · [claude] Infra de coordinación cross-agente (canal + inbox sincrónico) + entorno reproducible

**Infra de proceso, sin tarea asociada.** A pedido del usuario: aislar los hilos
de trabajo de cada agente (Claude / Codex / futuros) y montar un canal de
comunicación, primero asincrónico y después sincrónico.

- **Tres capas separadas**: verdad compartida del proyecto (código, DEVLOG,
  TAREAS — agnóstica de agente) · scratch privado por agente
  (`_meta/agents/<agente>/`, gitignored; las notas sueltas de Codex en
  `_codexTests/` se movieron a `_meta/agents/codex/`) · canal cross-agente.
- **Canal durable**: `AGENTS_HANDOFF.md` (raíz, versionado) — log asincrónico,
  leer al iniciar / escribir al cerrar. `AGENTS.md` (raíz): instrucciones para
  todos los agentes (Codex lo lee por convención) + sección "Entorno de
  desarrollo" + protocolo del inbox + tag `[claude]`/`[codex]` en DEVLOG.
- **Capa sincrónica (inbox)**: `_meta/agents/inbox/<destinatario>/` (gitignored),
  un `.md` por mensaje; consumir = mover a `read/` (no borrar).
  `scripts/agent-inbox-watch.sh` v1.1 — watcher polling que, corrido en background,
  hace que la harness de Claude Code re-invoque al agente al llegar un mensaje;
  modo `CHECK_ONCE=1` no bloqueante para el chequeo de inbox al iniciar sesión.
- **Entorno reproducible**: `.nvmrc` (Node 20, requerido por Vite ≥20.19) +
  documentación de activación (nvm + conda `face-sim`) en `AGENTS.md`. El entorno
  es por-shell, no se hereda entre sesiones de agentes.
- `CLAUDE.md` / `.gitignore`: lectura del canal + inbox check al iniciar; ignore de
  `_meta/agents/`.

**Validado en vivo**: round-trip Claude↔Codex por el inbox (mi watcher en
background se disparó y la harness me re-invocó sola, sin prompt del usuario, 3
veces). **Hallazgo**: Claude puede auto-despertarse (background → re-invocación),
Codex no (sin re-invocación autónoma ni hook de startup configurable) → sincronía
direccional, y el mecanismo de auto-chequeo al iniciar es la instrucción en el
archivo de instrucciones (`AGENTS.md` / `CLAUDE.md`), no un hook. Detalle del
intercambio en `AGENTS_HANDOFF.md`. Los inbox quedan fuera de git (efímeros).

## 2026-05-24

### `8ac1957` · Tarea #6 Fase B (viz calibración) + sync persistente Árbol⇄Comparador `T6` `T28↑` `T28✓`

**Dos features de UI del Track 2b en una sesión.** Pedido del usuario: la
visualización de calibración (histogramas como popup) con una métrica nueva
basada en las distribuciones que muestre también las anteriores; y mayor
sincronía Árbol ⇄ Comparador ("tripletes").

**Tarea #6 Fase B — visualización de la calibración.**
- Solapa nueva "Calibración" (`CalibrationTab.tsx`): histogramas kin vs non-kin
  por relación (FS/MD/FD/MS/ALL) + tabla (n, accuracy 5-CV, umbral Youden, AUC),
  toggle cosine/euclidean. Gráfico SVG reutilizable `CalibrationChart.tsx`.
- Popup `CalibrationModal.tsx` que ubica un cosine concreto sobre la distribución
  calibrada — se abre clickeando el cosine en el Comparador (`CosineCard`) y en
  el árbol (`CosineRow` del `TripletModal`, modal anidado).
- **Métrica nueva = probabilidad calibrada de parentesco P(kin|cos).** La primera
  implementación (density-ratio con piso de conteo) caía en las colas: cosine 0.7
  daba MENOS probabilidad que 0.45 — el smoke lo detectó. Reemplazada por
  **regresión isotónica (pool-adjacent-violators)** sobre la posterior por bin
  (suavizado gaussiano + PAV): monótona por construcción, robusta en colas. LR
  derivado consistente `p/(1-p)`; percentiles por CDF empírica. **Las métricas
  previas (cosine crudo, veredicto vs umbral, accuracy, AUC) se muestran junto a
  la nueva** para ver cómo varían entre ajustes (pedido explícito).
- `lib/calibration.ts`: loader (fetch cacheado) + `scoreValue`. El JSON se sirve
  desde `client/public/calibration/` (no bundle) → re-calibrar = re-copiar el
  archivo, sin rebuild. `inferRelation` devuelve 'ALL' salvo que se conozcan ambos
  sexos (Person no guarda sexo → hoy siempre 'ALL'; el modal deja elegir).
- App v1.4 (tab nueva), Comparator v2.4, TripletModal v1.1.

**Sync bidireccional persistente Árbol ⇄ Comparador.**
- `lib/activeTriplet.ts`: "tripleta activa" persistente en localStorage,
  compartida por ambas vistas. Reemplaza el handoff de un solo uso (TTL 60s del
  paso 5 de #26) por un vínculo vivo. `assignTripletSlots` mapea nodos→slots
  infiriendo el hijo del pedigree (roles Padre/Madre reales).
- Árbol→Comparador: seleccionar 2-3 nodos con foto (ctrl+click) escribe la
  tripleta sola; barra abajo con leyenda de mini-caras + botón "→ Comparar en
  tripletes". GenealogyTree v3.2.
- Comparador→Árbol **aditivo**: cambiar la foto de un slot vinculado re-vincula
  el nodo si la foto ya es de uno, o crea un nodo nuevo si no (NUNCA pisa fotos
  existentes — confirmado con el usuario); reescribe la tripleta. El árbol
  re-hidrata la selección al volver (los nodos nuevos aparecen vía reloadPersons).
  Banner de vínculo + "desvincular" + "← volver al árbol". Comparator v2.5; el
  handoff del TripletModal ahora escribe la tripleta.

**Decisiones (confirmadas con el usuario):** reusar el Comparador MVP como
"tripletes" (no vista nueva); writeback aditivo (nunca destructivo).

**Verificación:** tsc + `npm run build` OK; eslint sin errores nuevos (los 6 que
quedan son preexistentes: `reloadPersons` set-state-in-effect + 5 `any` en
spikes). Dos smokes Playwright headless (loop visual cerrado con screenshots):
(1) calibración — tab cosine/euclidean + `scoreValue` monótona verificada en el
módulo real vía Vite + modal sobre el pipeline real; (2) sync — seleccionar 3 →
barra → comparador vinculado → cambiar foto central crea "Nuevo nodo 4" (original
intacto, selección re-apuntada) → volver muestra 4 nodos. Térmico: pico 89°C
atribuible al pipeline GPU del comparador, no a estas features (JS puro).

## 2026-05-23

### `cc4a029` · Tarea #6 Fase A — calibración de umbrales de parentesco sobre KinFaceW-I `T6`

**Arranca la Tarea #6 (calibración data-driven de umbrales).** Convierte el
cosine crudo del comparador/árbol en un veredicto de parentesco con respaldo
cuantitativo. Fase A = calibración Python; Fase B (visualización web:
histogramas + modal) queda para otra sesión.

**Decisión de de-risking primero:** spike (`scripts/spike_kinfacew_embeddings.py`)
antes del protocolo completo, para resolver la "advertencia 64px" — las imágenes
de KinFaceW vienen recortadas/alineadas a 64×64; en vez de correr detección Face
Mesh sobre 64px (poco confiable), se tratan como **ya alineadas** y se llevan a
112×112 RGB directo a `extract_embedding_from_aligned`. El spike confirmó señal
(AUC rank-based ~0.74) → vía libre al protocolo.

**Resultados KinFaceW-I** (coseno crudo ArcFace `w600k_r50`, 5-fold CV oficial,
umbral de Youden por relación) — baseline honesto:

| relación | n_pos | acc 5-CV | umbral | AUC |
|---|---|---|---|---|
| Father-Son | 156 | 73.1% | 0.123 | 0.812 |
| Mother-Daughter | 127 | 65.5% | 0.231 | 0.746 |
| Father-Daughter | 134 | 62.4% | 0.108 | 0.677 |
| Mother-Son | 116 | 59.9% | 0.173 | 0.681 |
| **ALL** | 533 | **66.6%** | 0.138 | **0.727** |

En línea con NRML clásico (72-77%); el SOTA ~81% (FaCoRNet) requiere cabeza
aprendida. **Umbrales muy distintos por relación** (0.108-0.231) → calibrar por
relación era necesario; cross-género (FD/MS) sistemáticamente peor.

**Cambios de código:**
- **`src/phyloface/benchmark/`** (nuevo subpaquete, área B): `kinfacew.py`
  (loader que parsea `meta_data/*_pairs.mat` desde el zip — folds + pares
  oficiales — y decodifica caras como pre-alineadas 112×112) + `calibration.py`
  (lógica pura solo-numpy: AUC rank-based, umbral de Youden, 5-fold CV
  dirección-aware coseno↑/euclídea↓, histograma pre-binneado).
- **`src/phyloface/core/embedder.py`** v1.0→v1.1: `load_recognition_only()`
  carga solo el submodelo ArcFace vía `model_zoo.get_model`, evitando los 4
  submodelos extra de buffalo_l (menos RAM/calor para correr miles de caras).
- **`scripts/run_calibration_kinfacew.py`**: runner Fase A — embeddings con
  cache + **batching adaptativo a temperatura** (lee `/sys/class/thermal`, pausa
  6s sobre 85°C), calibración por relación + ALL, emisión del artefacto JSON
  (`data/output/calibration/`, gitignored) con umbrales/AUC/histogramas para
  coseno y euclídea — contrato hacia la Fase B.
- **`scripts/test-monitored.sh`**: wrapper de monitoreo de recursos para
  corridas que terminan (hermano de `dev-monitored.sh`); cumple la regla nueva
  de monitorear+loguear todo test automático. Imprime resumen al cierre.
  `.test-resources*` gitignored.
- **`_meta/CALIBRACION_TAREA6.md`**: procedimiento documentado.
- **`_meta/BIBLIOGRAFIA_KINSHIP_DATASETS.md`**: informe de un agente de
  investigación en background (web) — datasets/protocolos, tabla SOTA, mejoras
  rankeadas (cabeza MLP +7-15pts y portable a cliente, scoring tri-subject,
  fusión regional), pitfalls (sesgo "same-photo" de KinFaceW-II → evaluar -I
  como métrica primaria).
- `.claude/settings.json`: allowlist read-only (`unzip -l/-p`, `WebSearch`,
  `WebFetch`).

**Térmico:** el spike y la corrida completa tocaron 94°C; el batching adaptativo
del runner activó pausas y la corrida (~1066 imágenes) cerró sin runaway. Todas
las corridas Python se hicieron vía `test-monitored.sh`.

**Estado de #6:** Fase A cerrada. Resta Fase B (visualización web, otra sesión),
KinFaceW-II con disclaimer de sesgo, y opcional la cabeza MLP.

### `59c1243` · Track 2b paso 6 — export/import JSON+base64 del árbol `T26✓`

**Cierra el paso 6 (último) del plan de la Tarea #26 — y con él la tarea completa.** Serialización del árbol entero (Tree + Persons + Photos + Comparisons) a un JSON autocontenido con imágenes en base64, y re-importación creando un árbol nuevo. Sin deps nuevas (`btoa`/`atob`, `Blob` + `<a download>`, `File.text()`).

**Decisiones cerradas con el usuario al arrancar el paso** (3 elegidas):

- **Embeddings van en el export, etiquetados con la versión del modelo** a nivel de envelope (`modelVersion`). Toda la app usa un solo modelo, así que un campo por foto sería redundante. Al importar: si la versión del archivo coincide con `MODEL_VERSION` del runtime se reusan (import instantáneo); si no, se descartan (null) y se recomputan lazy en la primera comparación. Resuelve el tradeoff caching vs portabilidad. La UI avisa cuando los descarta.
- **El historial de comparaciones va en el export** (metadata útil del árbol).
- **Import crea SIEMPRE un árbol nuevo** (sin merge).

**Cambios de código:**

- **`lib/pipeline.ts` v1.1 → v1.2**: constante exportada `MODEL_VERSION` (`'w600k_r50'`) para etiquetar embeddings. Bumpear si cambia el `.onnx`, el orden de kps, el align o la normalización.
- **`lib/treeStore.ts` v1.1 → v1.2**: helper `importPhotoRecord(record)` que persiste un `PhotoRecord` completo (con embedding + width/height) dedupeando por sha256 — a diferencia de `putPhoto(blob)` que deja embedding null. Si la foto ya existe localmente NO sobrescribe (el record local gana, puede tener embedding más fresco). Devuelve `true` si insertó, `false` si dedupeó.
- **`lib/treeExport.ts` (nuevo)**: schema versionado `v:1` (`TreeExportV1`). `buildTreeExport(treeId)` arma el objeto serializable (junta fotos por el sha256 de cada persona + los snapshots de cada comparación), `downloadTreeExport` lo baja a `.json`, `importTreeFromJson(text)` valida + rehidrata. Helpers `blobToBase64` (por chunks de 32k para no reventar el call stack con `String.fromCharCode`) / `base64ToBlob` y `validateExport` (assert del schema con mensajes legibles).
- **`GenealogyTree.tsx` v3.0 → v3.1**: botones **⬇ Exportar** / **⬆ Importar** en la toolbar de árbol + input file oculto + estado `info` (caja verde) para feedback del import; el `error` existente cubre fallos. Tras importar recarga la lista de árboles y selecciona el nuevo.

**Detalle no obvio — remapeo de ids al importar:** además del `treeId` se remapean TODOS los `PersonId` (+ refs `father`/`mother`) y `ComparisonId` (+ refs `p1`/`p2`). Razón: los `PersonId` son keyPath del store `persons`; importar el mismo archivo dos veces SIN remapear haría que el segundo import pisara las personas del primero (mismo id, distinto treeId) y el primer árbol perdería sus nodos del índice by-tree. El remapeo hace cada import un árbol independiente (idempotente-seguro). Las fotos NO se remapean (content-addressed por sha256 → dedup natural). Refs de padres dangling (apuntan a una persona que no está en el export) se resuelven a `null` — el remapeo limpia la ref colgada como efecto colateral deseable.

**Sobre el roundtrip:** NO es bit-exacto export→import→export porque los ids se regeneran a propósito. El invariante preservado: misma cantidad de personas/fotos/comparaciones, mismos nombres, misma topología de parentesco, mismos bytes de foto (sha256 idéntico), mismos cosines.

**Smoke `genealogy-paso6-smoke.mjs` (nuevo):**

- Construye "Familia Export": Bruno padre + Mateo madre de Hijo, 3 fotos, 1 comparación Bruno↔Mateo (cachea 2 embeddings).
- ⬇ Exportar → intercepta la descarga, parsea y valida el JSON: `format`/`v`/`modelVersion`, 3 personas, 3 fotos (2 con embedding 512-d), 1 comparación, Hijo conserva padre+madre.
- ⬆ Importar el mismo JSON vía el file input → lee IndexedDB cruda: 2 árboles, el importado con 3 personas, topología preservada (Hijo→Bruno/Mateo importados), ids remapeados (Bruno importado ≠ original), sha256 de foto preservado, 1 comparación con el mismo cosine, embedding de foto preservado.
- Doble import → el árbol original conserva sus 3 personas (sin clobber).

**Validación:**

- `tsc -b`: PASS.
- `eslint`: 1 error preexistente (`reloadPersons` `set-state-in-effect`, ya en v2.0); 0 regresiones nuevas.
- Smoke headless: PASS end-to-end. cosine Bruno↔Mateo = `0.2302` (firma determinística conocida del paso 5 — sanity check de que el pipeline no se tocó).
- Screenshot `/tmp/genealogy-p6-02-after-import.png` leído multimodalmente: botones en la toolbar, caja verde "Importado «Familia Export»: 3 personas, 0 fotos (+3 ya existentes), 1 comparación" (las 3 fotos dedupeadas por sha256 contra el árbol original), pedigree importado renderizado con fotos.
- Recursos monitoreados con `dev-monitored.sh`: baseline ~45°C; pico transitorio 90-93°C / CPU 85% durante los ~30s de la única comparación GPU, vuelta a ~53°C al terminar. El export/import no agrega carga GPU.

**Estado del plan #26:** pasos 1+2+3+4+5+6 cerrados. **Tarea #26 completa.**

### `ec1d88f` · Track 2b paso 5 iter B+C — pedigree canónico + ctrl+click + multi-selección + modal de tripleta + handoff al Comparador MVP `T26`

**Refinamiento del paso 5 inicial (`bac4fac`) en 2 sub-iteraciones disparadas por feedback del usuario durante el flow.** Cierra la última iteración del paso 5 de la Tarea #26; resta sólo el paso 6 (export/import). Varias decisiones de UX de `bac4fac` quedaron **reemplazadas** acá (ver abajo): el toggle "Modo comparación" → Ctrl/Cmd+click; el `ComparisonPanel` lateral → línea con cosine sobre el SVG; el par fijo P1/P2 → multi-selección N nodos.

**Iter B — pedigree canónico + ctrl+click + cosine sobre línea (`GenealogyTree.tsx` v3.0):**

- **Render canónico tipo pedigree clínico**: agrupa hijos por par `(fatherId, motherId)` y dibuja (1) línea horizontal de **unión** entre padres a la altura del bottom del más bajo + stubs verticales para igualar altura si están en gens distintas; (2) **vertical** desde el centro de la unión hasta el sib-bus; (3) **horizontal** del sib-bus abarcando todos los hijos del par + el punto de bajada; (4) **vertical** desde el bus al top de cada hijo. Hermanos completos comparten bus; half-sibs tienen buses separados. Padres dangling se ignoran (el ⚠ del nodo ya marca el problema). Reemplaza el render v2.0 de diagonales padre→hijo.
- **Ctrl/Cmd+click** reemplaza el toggle "Modo comparación". Click normal sigue abriendo el panel detalle; el estado `comparisonMode` desaparece — el modifier en cada click manda.
- **Línea naranja gruesa con cosine flotante sobre el midpoint** en el SVG reemplaza el `ComparisonPanel` lateral. El segmento se recorta contra los rects de las dos cajas vía `clipLineBetweenBoxes`.

**Iter C — multi-selección + modal de tripleta + handoff:**

- **`selectedForCompare: PersonId[]`** reemplaza el par fijo `p1Id/p2Id`. Ctrl+click hace toggle (agrega/saca). Para N seleccionados se dibujan `N*(N-1)/2` líneas con sus cosines (`cosineByPair: Map<pairKey, number>`); badges naranjas numerados `1..N` por orden de selección. Quitar un nodo poda las entries del map que lo involucran.
- **Click sobre cualquier label de cosine → modal de tripleta** con detalles del par.
- **`TripletModal.tsx` (nuevo)**: modal flotante (overlay sobre el SVG; backdrop click + ESC cierran). 3 slots `A · (C opcional) · B` con foto + dropdown de rol. **Inferencia automática de roles** desde el árbol: si el seleccionado es hijo de los otros dos (`fatherId`/`motherId`) se etiqueta Hijo/a con Padre/Madre según el campo apuntado; sino default Padre/Madre/Hijo. Override vía `ROLE_OPTIONS_TRIPLET`. El tercero se elige por dropdown de personas del árbol con foto (`photoSha256 !== null`, excluye el par inicial). Computa cosines extras (A↔C, B↔C) reusando el mismo `ensureEmbedding` cacheado en IDB (`setPhotoEmbedding`). Botón **"→ abrir en Comparador MVP"** hace el handoff cross-tab.
- **`Comparator.tsx` v2.2 → v2.3**: al montar lee `localStorage["phyloface-comparator-prefill"]`; si es válido (`v:1`, `age<60s`, shape OK) carga los blobs apuntados por sha256 desde `treeStore.getPhoto()` y los precarga como `File` en los slots, restaurando también roles laterales. La key se borra al consumir (one-shot).
- **`App.tsx` v1.2 → v1.3**: listener `window` para `CustomEvent("phyloface-go-to-tab", { detail })`, que permite al `TripletModal` saltar al tab Comparador sin prop drilling ni store global.

**Detalles no obvios:**

- Línea de unión entre padres con **stubs verticales** cuando están en gens distintas — caso raro, pero el algoritmo no asume mismo bottom-Y para ambos.
- El `cosineByPair` **no** se re-puebla al cambiar de árbol (se podan todos los pares vía `resetComparisonSelection`). El effect que recarga las comparisons persistidas quedó del paso 5 iter A — sin UI todavía, pero la data persiste en IDB.
- Patrón "mover `setError` al IIFE async para silenciar `set-state-in-effect`" aplicado en `TripletModal.tsx` (el linter no marca setState en microtareas).
- El `TripletModal` **no** inyecta los cosines computados de vuelta al `cosineByPair` del árbol: el modal es una vista "como si" se seleccionaran los 3, sin contaminar la selección viva.

**Episodio KG capturado (`_global`):** `2026-05-23-cross-tab-handoff-via-localstorage-and-custom-event` — patrón: separar **datos** (localStorage one-shot, consume-and-remove, versionado, TTL por timestamp) de **señal de vista** (CustomEvent en window). Disciplina: validar shape antes de consumir + smoke cross-component e2e obligatorio. Hermano-prior dual de `2026-05-18-master-indexes-are-stale-by-default` y simétrico de [[2026-05-22-react-cleanup-gpu-wasm-resources-or-leak]].

**Smoke `genealogy-paso5-smoke.mjs` reescrito:**

- 3 personas (Bruno padre, Mateo madre, Hijo) para ejercitar el bus canónico.
- Ctrl+click sucesivos: 1 → 2 (1 cosine = `0.2302`) → 3 (3 cosines `[0.2302, 0.0189, -0.0341]`) → quitar Mateo (queda 1 par).
- Click sobre label Bruno↔Mateo → modal abierto con cosine cacheado correcto.
- Agregar tercero (Hijo) → modal muestra 3 cosines idénticos a los del árbol (reuso de embeddings cacheados).
- Click "→ Comparador MVP" → tab cambia (h1 "Comparador anónimo" visible) + 3 slots precargados (≥3 `<img>`). Screenshot: Padre=Bruno, Hijo/a=Hijo, Madre=Mateo con roles correctos.

**Validación:**

- `tsc --noEmit`: PASS.
- `eslint`: 1 error preexistente (`reloadPersons` con `set-state-in-effect`, ya en v2.0); 0 regresiones nuevas.
- Smoke headless: PASS end-to-end incluyendo cross-tab.
- Screenshots `/tmp/genealogy-p5-{04,05,06,07}-*.png` leídos multimodalmente: 3 líneas naranjas con cosines en el árbol, modal con triángulo de comparaciones, Comparador prellenado.

**Estado del plan #26:** pasos 1+2+3+4+5 cerrados (paso 5 en 3 sub-iteraciones: `bac4fac` base + iter B + iter C). **Próximo: paso 6** — export/import JSON+base64 del árbol completo. Decisiones abiertas: embeddings sí/no en el export (caching vs portabilidad), historial sí/no, import como árbol nuevo siempre o agregar al activo.

### `bac4fac` · Track 2b paso 5 — comparación on-demand entre 2 nodos del árbol (cosine + historial persistido) `T26`

**Cierra el paso 5 del plan de la Tarea #26** (de 6 pasos). Extiende `GenealogyTree.tsx` con un modo comparación que computa el cosine ArcFace entre las fotos de dos personas del árbol, reusando `lib/pipeline.ts` (Face Mesh → align canónico 112×112 → ONNX w600k_r50), cacheando embeddings por SHA-256 en IndexedDB y persistiendo cada comparación como entrada del historial filtrado por treeId.

**Decisiones de UX cerradas al arrancar el paso** (3 elegidas por el usuario, distintas a las que sugería el resume del 22):

- **Trigger** = toggle global "Modo comparación" (no implícito al click ni botón en panel detalle). Mientras está ON, los clicks sobre nodos no abren detalle: 1er click setea P1, 2do click setea P2 y dispara el cómputo.
- **UI del resultado** = panel "Comparación" *separado*, lado a lado con el `PersonDetailPanel` cuando ambos están activos. No expande el panel detalle ni es modal flotante.
- **Persistencia** = comparaciones se persisten en IDB con schema propio (no efímero). Esto agregó scope no trivial al paso 5 (nuevo object store + UI de historial + concepto "stale" para refs con foto cambiada).

**Cambios de código:**

- **`lib/genealogy.ts` v1.0 → v1.1**: tipo `Comparison` + `newComparison(...)`. Snapshotea `p1Sha256` y `p2Sha256` al momento del cómputo: si después le cambian la foto a P1 o P2, el historial sigue reflejando *qué se comparó realmente*. La UI marca esas entradas como "stale".
- **`lib/treeStore.ts` v1.0 → v1.1**: bump `DB_VERSION` 1 → 2; `onupgradeneeded` crea object store `comparisons` (keyPath `id`, índice `by-tree` sobre `treeId`) si no existe (idempotente: cubre upgrade desde v1 e instalación fresh). CRUD: `saveComparison`, `listComparisons(treeId)` filtra vía índice (mismo patrón que `listPersons`), `deleteComparison`.
- **`GenealogyTree.tsx` v2.0 → v3.0**:
  - Tercera toolbar con checkbox "Modo comparación" + mensaje de estado contextual ("→ click sobre un nodo para elegir P1" / "→ click sobre otro nodo para elegir P2 y comparar" / "✓ comparación lista. Click sobre otro nodo para reiniciar." / "⏳ computando embeddings…"). Fondo `#fff7e0` cuando ON, gris cuando OFF.
  - Nodos del SVG: badge "P1"/"P2" en esquina superior izquierda con `<rect>` redondeado de color (azul `#0044cc` / verde `#0a8a3a`) + texto blanco. Borde del nodo y fondo coordinan con el color del rol. `aria-label` extendido a `${name} · P1` / `· P2` para selectores de tests + accesibilidad.
  - `ComparisonPanel`: card lateral con borde naranja (`#c89000`, fondo `#fffaf0`) que contiene dos `ComparisonSlot` (foto 96×96 con borde del color del rol) + número grande monoespaciado del cosine en el medio, botón `↻ recompute` arriba a la derecha. Debajo, lista del historial (max-height 200px, scroll) con cada entrada: `Bruno ↔ Mateo · 0.2302 · hace seg · [✕]`. Refs colgadas → `(borrado)` en gris itálica. Cosine snapshot vs photoSha256 actual → `⚠ stale` con tooltip.
  - `ensureEmbedding(sha256, force?)`: pipeline lazy + cacheado. Si `PhotoRecord.embedding != null` y `!force`, reusa; sino corre `computeEmbedding` y persiste con `setPhotoEmbedding`. `FaceLandmarker` + ONNX session se inicializan una sola vez por mount via refs (init es costoso — ~segundos por descarga de modelos desde CDN).
  - `useEffect([])` con cleanup explícito que libera ambas refs al desmontar (`landmarker?.close()` + `void session?.release().catch`). Hereda la lección de Tarea #27 / episodio [[2026-05-22-react-cleanup-gpu-wasm-resources-or-leak]]. Sin esto, los recursos GPU/WASM persistirían en el proceso GPU compartido del browser hasta refresh.
  - `data-testid="cosine-value"` sobre el número grande para facilitar el smoke headless.
  - Reset de selección viva (P1/P2/cosine) al cambiar de árbol o apagar el toggle: hecho en handlers (`handleSelectTree`, `handleToggleComparisonMode`), no en un `useEffect` reactivo — cumple regla `react-hooks/set-state-in-effect`.
- **`client/scripts/genealogy-paso5-smoke.mjs`**: smoke Playwright headless. Resetea IDB de runs previos, crea árbol "Familia Test", crea Bruno+Mateo, sube fotos vía `setInputFiles` sobre los `<input type="file">` que viven en `<foreignObject>` por nodo, activa modo comparación, dispara cómputo, recompute, reload, valida persistencia (historial = 2 entradas), borra una entrada y verifica que el conteo baja a 1.

**Validación:**

- `npx tsc --noEmit`: PASS.
- `npx eslint`: 1 error preexistente (`reloadPersons` con `set-state-in-effect`, ya estaba en v2.0); 0 regresiones nuevas.
- **Smoke headless** (`node scripts/genealogy-paso5-smoke.mjs`): PASS end-to-end.
  - **cosine Bruno↔Mateo = `0.2302`** (caras humanas distintas, magnitud razonable).
  - Determinístico tras `↻ recompute`: mismo valor `0.2302` en run-2.
  - Tras reload + reactivar modo, historial muestra `Historial (2)` (cómputo inicial + recompute).
  - Borrar entrada baja conteo a `Historial (1)`.
  - Screenshots `/tmp/genealogy-p5-{01..07}-*.png` leídos visualmente: badges P1/P2 OK, panel comparación con fotos lado a lado OK, modo correctamente reseteado post-reload (selección viva descartada, historial recargado desde IDB).

**Detalles de implementación no obvios:**

- **`page.click()` de Playwright + `stopPropagation` intencional**: el smoke replica el workaround del paso 4 ([[2026-05-22-playwright-headless-plus-multimodal-llm-closes-ui-validation-loop]]). `.click()` centra el evento sobre el `<g>` interno de la foto, que hace `stopPropagation` (intencional del producto: click sobre foto = file picker, click sobre borde/nombre = seleccionar). Workaround vía `dispatchEvent` dirigido al `<rect>` background. **No** se toca el producto.
- **Tabs son `<div onClick>`, no `<button>`**: el smoke usa `page.locator('div').filter({ hasText: /^Árbol genealógico$/ })` + espera al `h2:has-text("...")` del componente para asegurar que el componente está montado, no solo que el texto del tab existe.
- **Script `.mjs` debe vivir bajo `client/`**: mismo bug del paso 4 con `heat-experiment.mjs` — Node.js no resuelve `@playwright/test` si el archivo está fuera del `node_modules` del cliente. Movido a `client/scripts/genealogy-paso5-smoke.mjs`.
- **Reset de comparación al cambiar de árbol**: handler `handleSelectTree(id)` envuelve `setSelectedTreeId(id)` + `resetComparisonSelection()` + `setSelectedPersonId(null)`. Se aplica a las 3 entradas que cambian el árbol activo: dropdown, create-tree, delete-tree. El handler `handleToggleComparisonMode` también resetea P1/P2/cosine cuando va de ON → OFF.
- **El cosine = 0.2302 entre Bruno y Mateo** es esperable: son dos niños distintos, no parientes; el embedding ArcFace está optimizado para identificación, no kinship — un valor positivo bajo es lo correcto. Cuando integremos KinFaceW (Tarea #17) tendremos un umbral data-driven contra el cual interpretarlo.

**Estado del plan #26:** pasos 1+2+3+4+5 cerrados. **Próximo: paso 6** — export/import JSON+base64 del árbol completo (metadata + imágenes empaquetadas en base64). El tab `App.tsx` ya está integrado desde el paso 2; queda definir el wire format del export.

## 2026-05-22

### `f56b500` · Track 2b paso 4 — render SVG pedigree + drag-and-drop foto + panel detalle `T26`

**Avanza la Tarea #26 cerrando el paso 4** del plan de 6 pasos. `GenealogyTree.tsx` v1.0 → v2.0: reemplaza la tabla plana de personas por un render SVG pedigree top-down, consumiendo `computeTreeLayout` del paso 3.

**Cambios principales:**

- **Render SVG**: cajas por persona dispuestas por generación (gen 0 arriba, +1 hacia abajo); cada caja ~120×140 px con foto/placeholder + nombre. Líneas de parentesco rectas del centro-bottom del padre/madre al centro-top del hijo (una por relación, pueden ser 2 si tiene ambos padres). Centrado dinámico de cada generación dentro del ancho máximo del árbol. Sin minimización de cruces — orden estable por `createdAt` ASC, suficiente para MVP.
- **Selección por click**: click sobre el borde / nombre del nodo → selecciona (borde azul) → panel detalle aparece abajo. El área de la foto NO selecciona (intencional, stopPropagation): click sobre foto abre file picker. El panel detalle expone foto grande (180×180), dropdowns padre/madre con validación de aciclicidad, botón borrar persona, botón cerrar.
- **Drag-and-drop foto sobre nodo SVG**: arrastrar imagen del filesystem sobre un nodo → highlight verde + borde dashed → al soltar, `putPhoto` con dedup por SHA-256. Handlers `onDragOver/onDragLeave/onDrop` en el `<g>` del nodo, con `preventDefault` para que el browser no navegue a la imagen. Click sobre el placeholder "+ foto" del nodo abre file picker como alternativa.
- **Accesibilidad**: cada nodo SVG es un `<g role="button" aria-label={person.name}>`. Lectores de pantalla pueden navegar el árbol; selectores de tests estables; semánticamente correcto para SVG con contenido interactivo.
- **Refs colgadas**: si una persona tiene `fatherId`/`motherId` apuntando a alguien borrado, aparece un ⚠ rojo en la esquina superior-derecha del nodo. El render SVG ignora la línea de parentesco hacia el parent inexistente (`computeTreeLayout` ya trata el dangling como gen=-1, así la persona huérfana sube de generación naturalmente). El panel detalle muestra "(borrado: XXXXXX…)" en rojo dentro del dropdown.

**Detalles de implementación no obvios:**

- **`<text>` con `title` prop no existe en SVG** (a diferencia de HTML). Para tooltip sobre el ⚠, child `<title>` element dentro del `<text>`. Bug atrapado por `tsc` en la primera iteración.
- **`<input type="file">` dentro de SVG**: requiere `<foreignObject>` con dimensión mínima `1×1` y `overflow: visible`, sino el input no se renderiza ni acepta clicks programáticos.
- **`useEffect` para deseleccionar al cambiar de árbol — eliminado**: la regla `react-hooks/set-state-in-effect` lo marca como antipatrón. Reemplazado por derivación: `selectedPerson` ya se computa con `find` y retorna `null` si el id no existe en el árbol actual (colisión de UUID v4 entre árboles es ~0).

**Lo que se conservó v1.0 → v2.0:**

- Toolbars de árbol y de "+ Persona" (idénticas).
- Multi-tree en el store con una sola activa en UI; `LAST_TREE_KEY` en `localStorage`.
- Object URLs de fotos cacheados por sha256 y revocados en cleanup (consistente con [[2026-05-22-react-cleanup-gpu-wasm-resources-or-leak]]).
- Validación `wouldCreateCycle` antes de asignar padres; mensaje de error inline si rechaza.

**Validación:**

- `npx tsc -b --noEmit`: PASS.
- `npx eslint`: 1 error preexistente (`reloadPersons` con `set-state-in-effect`, ya estaba en v1.0); 0 regresiones nuevas.
- `npx vite build`: PASS (788 kB JS gzip 225 kB; warning del wasm de ONNX es esperado).
- **Smoke browser automatizado con Playwright headless** (perfil temporal, no toca IDB del Chrome del usuario). 5 screenshots en `/tmp/smoke-tree-shots/` cubriendo: render plano 4 personas, panel post-click, layout dinámico 3 generaciones con líneas, foto cargada en nodo y panel, borrado de raíz → ⚠ y recompute de generaciones. **Todos PASS.** Bug del smoke (no de la app) documentado en `/tmp/smoke-tree-paso4.mjs`: `page.click()` de Playwright centra el click sobre el rect, y el centro cae sobre el área de la foto que stopPropaga (intencional); workaround vía `dispatchEvent` dirigido al rect específico. UX real del usuario es correcta (click sobre borde/nombre selecciona, click sobre foto abre picker).
- **Smoke manual del usuario**: PASS (drag-and-drop incluido, no cubierto por el automatizado).

**Próximo paso del plan #26**: paso 5 — comparación on-demand. Tras seleccionar P1 (un nodo), click sobre P2 dispara el cosine reusando `lib/pipeline.ts`. Requiere asegurar que la foto de ambos nodos esté cargada y tener embedding cacheado en `PhotoRecord` (paso 1 ya lo dejó preparado).

### `44b20e9` · Track 2b paso 3 — `lib/treeLayout.ts` puro `T26`

**Avanza la Tarea #26 cerrando el paso 3** del plan de 6 pasos. Función pura que toma `Person[]` (del modelo `lib/genealogy.ts`) y devuelve `Map<PersonId, { generation, indexInGen }>` — base para el render SVG del paso 4. Sin acoplamiento al DOM ni a IDB.

**Algoritmo:**

- **Generación** asignada por DFS recursiva con memoización. Padres ausentes (id `null`) o danglings (id que no está en el set por persona borrada) cuentan como `-1`, de modo que `max(parents) + 1` cae naturalmente en `0` cuando ningún padre es válido. Una persona con un padre válido en gen 2 y otro dangling termina en gen 3.
- **Orden intra-generación**: estable por `createdAt` ASC, tiebreak por `id` ASC. Determinístico — dos invocaciones sobre el mismo input devuelven el mismo orden, necesario para que el render SVG no salte de posición entre re-renders. **No minimiza cruces de líneas**; eso queda para iteraciones posteriores si el grafo crece y se vuelve ilegible. Para MVP, orden estable es suficiente.
- **Guard de ciclos defensivo**: `wouldCreateCycle` en `lib/genealogy.ts` ya los previene en la UI, pero datos importados desde JSON externo podrían colarse. El stack de visitando durante la recursión detecta ciclos y degrada los nodos involucrados sin loopear infinitamente. No es la mejor UX pero el comportamiento defensivo es mínimo.

**Smoke test** — `/tmp/treeLayout-smoke.mjs` (réplica funcional inline para no depender de tsx; si la lógica del .ts cambia, regenerar). 6 casos: pedigree clásico de 3 generaciones, hermanos ordenados por `createdAt`, refs colgadas por padre/madre, ciclo defensivo (verifica que no loopee, no la corrección del layout — ese caso no debería darse vía UI), lista vacía, tiebreak por id con `createdAt` empate. **Todos PASS.**

**Decisiones de diseño no obvias:**

- API devuelve `Map`, no `Record<string, ...>` — keys arbitrarias, además es lo idiomático en TS para lookups dinámicos.
- Generación NO se guarda en `Person` (decisión heredada de la cabecera de `lib/genealogy.ts`): state derivado del grafo, mantenerlo en sync con los parentIds requeriría invalidación en cada edit. Mucho más simple recomputarlo en el render — el set es chico, no hay performance constraint.
- Si emerge la necesidad de iterar generación a generación (probable en el paso 4 para layout SVG), agregar `bucketByGeneration(persons, layout)` en una v1.1. YAGNI por ahora.

**Próximo paso del plan #26**: paso 4 — reemplazar la tabla actual de `GenealogyTree.tsx` por un render SVG pedigree (cajas por persona dispuestas por generación + líneas de parentesco) con drag-and-drop foto sobre nodo, consumiendo `computeTreeLayout`.

### `2889329` · fix(Comparator+Spikes): cleanup GPU/WASM al desmontar `T27✓`

**Cierra Tarea #27** (cleanup de recursos GPU/WASM en componentes React que cargan motores ML pesados). Bug latente detectado y validado cuantitativamente el mismo día por `heat-experiment.sh` (Phase 5 elevación sistemática vs Phase 2). Fix aplicado a los 4 componentes que inicializan `FaceLandmarker` y/o `ort.InferenceSession`.

**Patrón aplicado:**

- **`Comparator.tsx` v2.1 → v2.2**: agregado `useEffect([])` dedicado al cleanup que llama `landmarkerRef.current?.close()` y `void sessionRef.current?.release().catch(...)`. Los refs se anulan después. El cleanup lee los refs en tiempo de desmontaje (no captura el valor en el render).
- **`SpikeMediapipe.tsx` v1.0 → v1.1**: captura `landmarkerInstance` en variable del scope del effect (antes del `if (cancelled) return` del IIFE async). El return del effect llama `.close()`. La asignación pre-cancel es necesaria para StrictMode dev (doble run) y para el caso de unmount in-flight: si la promesa de init resolvió pero el componente ya se desmontó, el cleanup tiene acceso al recurso.
- **`SpikeOnnx.tsx` v1.0 → v1.1**: mismo patrón, `sessionInstance` capturada y liberada con `.release()`.
- **`SpikeDetection.tsx` v2.1 → v2.2**: ambos motores capturados (`landmarkerInstance` + `sessionInstance`), liberados en el return.

**Nota sobre `release()`**: `InferenceSession.release()` devuelve `Promise<void>`. Los cleanups de `useEffect` no aceptan `await`, así que se invoca como `void sess.release().catch((e) => console.warn(...))`. `FaceLandmarker.close()` es sincrónico (`(): void`).

**Validación cuantitativa** — re-corrida de `heat-experiment.sh` post-fix (tarde 14:24) comparada con baseline pre-fix (07:58 del mismo día, preservada como `client/.heat-experiment-baseline-pre-task27.log`):

| | Phase 2 (genealogy idle) | Phase 5 (genealogy con GPU init) | **Δ (5−2)** |
|---|---|---|---|
| Pre-fix temp_avg | 41.4°C | 43.0°C | **+1.6°C** (leak) |
| Post-fix temp_avg | 52.4°C | 50.8°C | **−1.6°C** (cerrado) |
| Pre-fix temp_max | 42°C | 48°C | **+6°C** |
| Post-fix temp_max | 56°C | 52°C | **−4°C** |
| Pre-fix CPU avg | 5.0% | 7.2% | +2.2pp |
| Post-fix CPU avg | 15.6% | 17.0% | +1.4pp |

Los valores absolutos son más altos en post-fix porque la corrida fue por la tarde (temperatura ambiente más alta) — la comparación válida es la **delta intra-corrida** Phase 5 vs Phase 2, que es lo que aísla el efecto del leak. La inversión del signo (+1.6 → −1.6) es la firma del fix funcionando: ahora Phase 5 está **por debajo** de Phase 2, no por encima.

**Bonus visible**: pico de Phase 4 (init MediaPipe+ONNX) cayó de **90°C max** a **63°C max**. La transición Phase 4 → Phase 5 cambia de tab; al desmontar Comparator se libera la GPU antes de empezar a samplear Phase 5, y ese cambio se ve también en el pico de Phase 4 (menos cola térmica residual).

**Verificaciones que pasaron:**

- `npx tsc -b --noEmit` (typecheck) → sin errores.
- `npx eslint` sobre los 4 archivos → 4 errores `@typescript-eslint/no-explicit-any`, todos pre-existentes en bloques `catch (e: any)` de los spikes; **0 regresiones nuevas** (verificado con `git stash` + lint baseline).

**Episodio KG capturado**: `2026-05-22-signature-inversion-validates-fix_diego-lenovo-debian.md` (`_global`) — extiende el patrón "firma numérica como huella diagnóstica" ([[2026-05-20-numeric-signature-as-diagnostic-fingerprint]]) hacia la dirección inversa: la firma del bug también sirve como **criterio de validación cuantitativa de un fix**, no solo como diagnóstico inicial. Si la firma se invierte (no solo desaparece, sino que cambia de signo de forma simétrica), evidencia más fuerte de que el fix actuó sobre el mecanismo correcto, no sobre una variable ortogonal que coincidió.

### `4541c7b` · scripts: heat-experiment automatizado del Track 2b (Playwright + sampler) `T26 T27`

**Experimento reproducible de temperatura/CPU del cliente** disparado por la pregunta del usuario "podés correr vos todas estas evaluaciones?" tras el smoke manual de fases del paso 2. Convierte el protocolo manual de 7 fases en 1 comando que devuelve tabla agregada.

**Artefactos:**

- `scripts/heat-experiment.sh` (ID `PHYLO_HEAT_EXPERIMENT_BASH v1.0`): orquestador bash. Mata vite previo, arranca dev server en background, mide Phase 0 baseline (30s sin browser), lanza Playwright para Phases 1-6, samplea CPU%+temp_max cada 5s vía `vmstat 1 2` (CPU idle column) + `sensors -A | awk '/^Core/' | sort -g | tail -1` (max core temp), agrupa por fase con awk y reporta tabla resumen (avg/max por fase). Cleanup en trap EXIT que mata sampler + vite.
- `client/scripts/heat-experiment.mjs` (ID `PHYLO_HEAT_EXPERIMENT_PLAYWRIGHT v1.0`): script Playwright en **modo headed** (no headless, para que WebGPU funcione + para validación visual). Abre el SPA, navega entre tabs, crea árbol/persona/foto, dispara comparación (esperando init de MediaPipe+ONNX, timeout hasta 120s), escribe nombre de fase actual a `.heat-experiment-current-phase.txt` antes de cada fase para que el sampler bash lo lea por muestra. Usa `getByRole` para selectores robustos. Flags `--enable-unsafe-webgpu --enable-features=Vulkan` para no caer a fallback CPU.
- `@playwright/test` agregado como devDep del cliente. Chromium descargado a `~/.cache/ms-playwright/chromium-1223` (377 MB) + headless shell (260 MB). Reusable a futuro: cuando el Track 2b paso 5 implemente comparación on-demand, tests E2E directos.
- `.gitignore` extendido: `client/.heat-experiment*` y patrón laxo `client/.dev-resources*` (cubre log + .prev del rotado).

**Bug intermedio en la primera corrida** (`ERR_MODULE_NOT_FOUND` al importar `@playwright/test` desde el .mjs): module resolution de Node busca `node_modules` desde el path del archivo .mjs, no del cwd. El script estaba en `scripts/heat-experiment.mjs` pero el paquete en `client/node_modules/`. Fix: mover el .mjs a `client/scripts/heat-experiment.mjs`; el bash invoca con `cd client && node scripts/heat-experiment.mjs`.

**Resultado del primer run válido** (5 muestras por fase, ~3 min totales):

| Phase | CPU avg | CPU max | Temp avg | Temp max |
|-------|---------|---------|----------|----------|
| 0-baseline-no-browser | 1.4% | 4% | 49.4°C | 78°C ⚠️ residual |
| 1-comparator-default-idle | 4.0% | 11% | 41.8°C | 42°C |
| 2-genealogy-empty | 5.0% | 12% | 41.4°C | 42°C |
| 3-genealogy-with-photo | 3.2% | 4% | 41.4°C | 42°C |
| 4-comparator-after-compare | 5.2% | 12% | 52.4°C | **90°C** 🔥 |
| 5-genealogy-after-gpu-init | 7.2% | 14% | 43.0°C | 48°C |
| 6-tab-closed | 3.6% | 4% | 42.6°C | 44°C |

**Conclusiones empíricas:**

- ✅ **`GenealogyTree.tsx` no calienta por sí solo**: Phase 2 (5%/41.4°C) y Phase 3 (3.2%/41.4°C) prácticamente idénticos a Phase 1 (Comparador idle sin GPU init, 4%/41.8°C). La hipótesis intuitiva del usuario quedó falsada cuantitativamente.
- 🔥 **Pico verificado de init MediaPipe+ONNX**: Phase 4 alcanza 90°C max y 52.4°C avg — explica el reporte original de calentamiento.
- ⚠️ **Leak del Comparator validado**: Phase 5 (CPU avg 7.2%, temp avg 43°C) muestra elevación sistemática vs Phase 2 (5%, 41.4°C) — única diferencia entre ambas es si MediaPipe/ONNX se inicializaron antes. Confirma Tarea **#27** como bug real, no especulativo.
- ⚠️ **Tab cerrado** (Phase 6, 42.6°C avg) no baja a baseline (~41.4°C de Phase 2). Cerrar pestaña en contexto Playwright no libera del todo.

**Tooling persistente**: alias `heat-exp` agregado a `~/.bashrc` (junto con `monitor` y `dev-mon`); pointer en `~/.claude/CLAUDE.md` global; doc completa en `~/Proyectos/NOTAS_CONFIGURACION.md`. Episodio KG `_global` capturado: `2026-05-22-controlled-experiment-resolves-hypothesis-tension` — patrón meta-metodológico: ante 3+ hipótesis no resueltas sobre el mismo síntoma, parar y diseñar experimento controlado automatizado.

### `ecfaba1` · Track 2b — modelo + store IDB + UI lista MVP (pasos 1-2) y wrapper dev-monitored.sh `T26 T27↑`

**Avanza la Tarea #26 (Track 2b — comparador con árbol genealógico) cerrando los pasos 1 y 2** del plan de 6 pasos definido al arrancar la tarea. Coexiste con el comparador 3-slot del Track 2a (tab nuevo, no reemplazo).

**Paso 1 — modelo y persistencia** (`client/src/lib/genealogy.ts` ID `PHYLOFACE_LIB_GENEALOGY v1.0` + `client/src/lib/treeStore.ts` ID `PHYLOFACE_LIB_TREESTORE v1.0`):

- **Modelo de datos puro** sin deps del DOM ni IDB. Tipos: `Person` (treeId, name, birthYear?, fatherId/motherId nullable, photoSha256 nullable, createdAt), `Tree` (id, name, createdAt, updatedAt), `PhotoRecord` (sha256 como PK, blob, embedding `Float32Array|null`, width, height, createdAt). Constructores `newId()` (UUID v4 vía Web Crypto), `newTree`, `newPerson`.
- **`sha256OfBlob(blob)`**: SHA-256 hex lowercase con SubtleCrypto. Disponible en `localhost` (contexto seguro) sin sudo.
- **`wouldCreateCycle(persons, personId, candidateParentId)`**: valida que asignar `candidateParentId` como padre/madre de `personId` no introduzca ciclos — DFS por ancestros del candidato buscando si `personId` ya es ancestro. Devuelve `{ ok: true }` o `{ ok: false, cycle: [...] }` con el camino del ciclo para diagnóstico. Necesario porque pedigree formal es DAG y la UI permite reasignar padres.
- **Decisión de diseño**: el embedding vive en `PhotoRecord` (indexado por sha256), no en `Person`. Si dos personas comparten foto, el embedding se computa una sola vez. Dedup natural sin ref-counting.
- **Store IDB nativo** (sin deps tipo `idb`): DB `phyloface-genealogy` v1 con stores `trees` (keyPath `id`), `persons` (keyPath `id`, index `by-tree` sobre treeId), `photos` (keyPath `sha256`). API: `openDb` cacheada, `deleteDb` para reset, CRUD trees/persons, `putPhoto(blob)` que calcula sha256 + dedup + extrae width/height vía `createImageBitmap`, `setPhotoEmbedding(sha256, embedding)` para cachear lazy.
- **`deletePerson` no toca refs colgadas**: las personas que tenían a la borrada como padre/madre quedan con `fatherId`/`motherId` apuntando a un id inexistente. Decisión conservadora — borrar en cascada podría destruir genealogía. La UI los maneja como "(borrado: ABC...)". `deleteTree` borra el árbol y sus personas pero NO las fotos (pueden compartirse entre árboles; GC manual queda para futuro).

**Paso 2 — UI lista MVP** (`client/src/GenealogyTree.tsx` ID `PHYLOFACE_GENEALOGY_TREE v1.0` + tab en `client/src/App.tsx` v1.1 → v1.2):

- **Sin SVG todavía** — el pedigree visual viene en el paso 4. Esta vista valida que la capa de persistencia funciona end-to-end con UI mínima: tabla con foto + nombre + dropdowns padre/madre + botón ✕.
- **Toolbar de árbol**: selector de árbol activo (multi-tree soportado en el store, sólo uno activo en UI), botón borrar árbol, input + botón "+ Árbol". Árbol activo se persiste en `localStorage` con key `phyloface-genealogy-last-tree`.
- **Toolbar de persona**: input + botón "+ Persona" (Enter dispara también).
- **Tabla de personas**: foto (placeholder 64×64 clickeable que abre file picker; preview con `objectFit: cover`), nombre, dropdown padre, dropdown madre, botón borrar. Dropdowns se filtran a las otras personas del árbol; al asignar, se llama `wouldCreateCycle` y si rechaza, muestra mensaje rojo "Asignación rechazada: crearía un ciclo".
- **Refs colgadas**: si una persona tiene `fatherId`/`motherId` apuntando a alguien borrado, el dropdown muestra "(borrado: ABC...)" en rojo, no rompe.
- **Object URLs de fotos**: cacheados por sha256 en estado local, revocados en cleanup del effect (no leakea al desmontar o cambiar de árbol). Coherente con el patrón capturado en el KG el mismo día (`2026-05-22-react-cleanup-gpu-wasm-resources-or-leak`).
- **Validación**: `tsc --noEmit -p tsconfig.app.json` PASS, smoke browser PASS (alta árbol/personas, asignación padres, foto, validación de ciclo, refresh persiste, ref colgada, borrado, refresh vuelve a estado vacío). Sigue el patrón "smoke browser obligatorio para UI changes" del Track 2a.

**Detour de control de recursos** — disparado por reporte del usuario de calentamiento sostenido durante el smoke del paso 2. Diagnóstico: cerrar el browser entero bajó la temperatura instantáneamente, lo que probó que la fuente era el proceso GPU compartido del browser (otros tabs + posibles contextos acumulados), no `GenealogyTree.tsx` (no usa GPU). **Tres patrones capturados en el KG** en `~/Proyectos/0_code_(gitHub)/IA/memories/_meta/episodes/2026-05-22-*`: (a) calor del browser = GPU process compartido; (b) componentes React con recursos externos deben limpiar en useEffect; (c) sesiones tmux/screen/compose vestigiales engañan al re-attach por `has-session`.

- **`scripts/dev-monitored.sh` (nuevo, ID `PHYLO_DEV_MONITORED v1.0`)**: wrapper de `npm run dev` con muestreo de CPU/temp cada 5s (configurable vía env), log a `.dev-resources.log` (gitignored), WARN inline cuando una métrica supera el umbral durante N muestras seguidas. Defaults: 80% / 80°C / 3 muestras / 5s. Colores ANSI sobre stderr para no contaminar stdout del dev server. Sampling locale-agnostic: `vmstat 1 2` para CPU% (columna idle), `sensors -A | awk '/^Core/ ...'` para temp_max. Sin sudo. Cleanup en trap EXIT que mata al sampler junto con el dev server.
- **`.gitignore`**: agregado `client/.dev-resources.log`.
- **Tarea #27 abierta** (`T27↑`): cleanup GPU/WASM en `Comparator.tsx` — `useEffect` con cleanup que llame `landmarkerRef.current?.close()` y `sessionRef.current?.release()` al desmontar. Bug latente capturado en el episodio `_global` `2026-05-22-react-cleanup-gpu-wasm-resources-or-leak`.

**Artefactos fuera de este repo** (parte del mismo trabajo pero viven en `~/Proyectos/` o configs globales, no se versionan acá):

- `~/Proyectos/scripts/monitor.sh` (nuevo, ID `PHYLO_MONITOR v1.1`): dashboard tmux cross-proyecto con 3 panes (`bpytop`, `nvtop` con sudo, `watch -n 2 sensors`). v1.1 incluye fix para sesiones vestigiales (verifica conteo de panes antes de re-attach; sino destruye y recrea) — patrón capturado en `2026-05-22-tmux-vestigial-session-deceives-reattach`.
- `~/Proyectos/NOTAS_CONFIGURACION.md` (nuevo): doc compartida cross-proyecto sobre control de recursos durante desarrollo web (herramientas browser/sistema, patrones defensivos, verificación rápida del culpable, instrucciones de los scripts).
- `~/.claude/CLAUDE.md` (sección agregada "Control de recursos durante desarrollo web/UI"): pointer a las herramientas + criterios de cuándo proponerlas en cualquier sesión.
- `~/.bashrc` (aliases agregados): `monitor` (dashboard global) y `dev-mon` (wrapper desde la raíz del proyecto).

**Instalación de paquetes (una vez)**: `sudo apt install bpytop intel-gpu-tools nvtop tmux`. `btop` (versión C++ más nueva) no está en apt de Debian 11 (sólo snap); `bpytop` es el predecesor del mismo autor en Python, drop-in para los scripts. Cuando se migre a Debian 12+ o se acepte snap, cambiar `bpytop → btop` en `~/Proyectos/scripts/monitor.sh`.

**Próximo paso del plan #26**: paso 3 — `lib/treeLayout.ts` (función pura que asigna generaciones a cada persona y ordena dentro de cada generación; base para el SVG del paso 4).

## 2026-05-21

### `387f8ac` · Track 2a — Comparador 3-slot (Hijo/a vs adultos) MVP cerrado `T25✓ T26↑`

- **Cierra la subtarea (c) de la Tarea #25** y la tarea entera. Nueva página en el cliente: `client/src/Comparator.tsx` (ID `PHYLOFACE_COMPARATOR`), accesible como tab "Comparador (MVP)" (default activa) en `App.tsx` v1.0 → v1.1.
- **Estructura de 3 slots** alineada con el caso primario del producto (niño vs progenitores, ver `TAREAS_PENDIENTES.md` Tarea #12 App primaria): slot izquierdo (adulto) · slot central (Hijo/a, fijo) · slot derecho (adulto). Los dos slots laterales tienen **dropdown de rol** con opciones `Padre, Madre, Hermano, Hermana, Tío, Tía, Abuelo, Abuela, Otro` (default Padre y Madre). Si elige "Otro", aparece input libre. El rol elegido se refleja en la etiqueta del slot y en los labels de los cards de resultado (ej. "Hijo/a ↔ Padre").
- **Drag-and-drop por slot** (`onDragOver/onDragLeave/onDrop`). Cuando el slot está vacío, muestra zona dashed "arrastrá una imagen acá"; durante drag-over: highlight azul claro + borde dashed azul. Solo acepta `dataTransfer.files` (filesystem); no resuelve URLs externas — **intencional** para no romper la garantía de privacidad. Valida `file.type` que empiece con `image/`.
- **Botón "✕ Quitar"** por slot, visible solo con imagen cargada. Limpia file + previewUrl (revoca el object URL) + result + error. Resetea el `<input type="file">` vía `useRef` porque el browser no permite setear `value=""` desde React (seguridad) — sin esto, el input mantendría el nombre del archivo viejo y no se podría re-seleccionar el mismo.
- **Comparación flexible**: el botón Comparar se habilita con **2 o 3** slots con file. Reglas:
  - Hijo/a + un adulto → 1 cosine (Hijo/a ↔ adulto).
  - Hijo/a + ambos adultos → 2 cosines (Hijo/a ↔ P1 y Hijo/a ↔ P2).
  - Ambos adultos sin Hijo/a → 1 cosine adulto ↔ adulto.
  - Estado migrado de `cosineLeft/cosineRight` a `cosines: { label, value }[]` para soportar el caso de a 2 sin ramas condicionales redundantes.
- **Pipeline reusado**: `Comparator` consume `lib/pipeline.ts` (extraído en `77c99d5`) — `loadImage`, `computeEmbedding`, `cosineSimilarity`, `initFaceLandmarker`, `initOnnxSession`. Init lazy de landmarker + sesión ONNX, **una sola vez por sesión** (refs + flag `initStatus`), porque cada init descarga modelos remotos y costó ~varios segundos en los spikes.
- **Preview de las caras alineadas 112×112**: para mostrar qué se está comparando, `lib/pipeline.ts` v1.0 → v1.1 agrega `aligned: ImageData` al `PipelineOutput`. El spike #004 (`SpikeDetection.tsx`) lo ignora silenciosamente — backward-compatible. Render con `imageSmoothingEnabled=false` para escalar ×2 sin antialiasing.
- **Cosine crudo, sin etiqueta semántica**: las cards muestran el cosine con nota explícita "no hay umbral calibrado todavía (Tarea #6 lo va a generar contra KinFaceW); estos son los valores crudos de similitud entre vectores 512-d". Evita que el usuario interprete cualquier número como "match" sin baseline.
- **Privacy banner explícito** en el header: "Las imágenes nunca salen de tu navegador. No hay upload a servidor, no se guarda nada en disco". Coherente con el stack 100% client-side (MediaPipe WASM + ONNX WebGPU/WASM, blob URLs locales).
- **Bug encontrado y arreglado durante el desarrollo**: dibujar el canvas alineado dentro del handler async fallaba silenciosamente — el canvas se monta condicionalmente al llegar `slot.result`, así que `ref.current` era null cuando se intentaba dibujar justo después del `setState`. Fix: tres `useEffect` separados (uno por slot) que observan el `result` y dibujan al próximo render. Effects separados (no uno solo con `[slots.X.result, slots.Y.result, slots.Z.result]`) para que `exhaustive-deps` no pida `slots` entero, lo cual dispararía el draw en cada drag-over.
- **Validación**: `tsc --noEmit -p tsconfig.app.json` PASS · `eslint` clean (0 errors, 0 warnings) en los 3 archivos modificados. Validación visual a cargo del usuario en el dev server (no automatizable acá).
- **Tarea #26 abierta** en `TAREAS_PENDIENTES.md` (`T26↑`): **Track 2b — Comparador con árbol genealógico (pedigree formal)**. Página separada del 3-slot. Cada persona admite max 1 padre + 1 madre. Persistencia IndexedDB local + export/import. Selección interactiva de qué dos personas comparar. Decisiones abiertas registradas en la fila de la tarea. Encaja con la idea diferida `[[project-track2b-dataset-pipeline]]` ampliada con el modelo pedigree.

### `77c99d5` · Track 2a — Refactor: extracción del pipeline e2e browser a `lib/pipeline.ts` `T25`

- **Refactor preparatorio del MVP comparador** (Tarea #25 subtarea (c)). El pipeline e2e browser-only que vivía inline en `SpikeDetection.tsx` (detect Face Mesh → 5 kps en orden InsightFace → align canónico 112×112 → preprocesar → ONNX → embedding) se extrae a un módulo nuevo `client/src/lib/pipeline.ts` (ID `PHYLOFACE_LIB_PIPELINE v1.0`) para que el comparador (próximo paso) y el spike de regresión consuman la misma pieza.
  - **Exports** del nuevo lib:
    - `MESH_INDICES_INSIGHTFACE_ORDER` — constante con los 5 índices del mesh validados en spike #004 (`[468, 473, 4, 61, 291]`).
    - `cosineSimilarity(a, b)` — sobre `Float32Array`.
    - `loadImage(url)` — devuelve `{ img: HTMLImageElement, imageData: ImageData }`. HTMLImage lo necesita MediaPipe; ImageData lo necesita el warpAffine.
    - `imageDataToTensorRGB(imgData, mean=127.5, std=127.5)` — RGBA 112×112 → NCHW float32 normalizado.
    - `initFaceLandmarker()` — FaceLandmarker IMAGE mode, 1 cara, GPU delegate, modelo desde CDN. Costoso: el caller debe instanciar una vez y reusar.
    - `initOnnxSession(modelUrl='/models/w600k_r50.onnx')` — sesión con providers `['webgpu', 'wasm']`.
    - **`computeEmbedding(img, imageData, landmarker, session)`** — pipeline e2e puro, devuelve `{ embedding: Float32Array, kps: number[][], timings: { detectMs, alignMs, preprocessMs, inferMs } }`. NO toca refs, fixtures, ni overlay — eso queda como responsabilidad del caller.
- **`SpikeDetection.tsx` v2.0 → v2.1**: refactor para consumir el lib nuevo. De 618 a 506 líneas (−112). Se eliminan los helpers que se movieron al lib y los inits inline de MediaPipe + ONNX. `runOnePipeline` queda como wrapper delgado: `computeEmbedding` + comparación contra el embedding y kps de referencia del fixture multi-caso (kps distance, cosine, max_abs_diff). Tipos del fixture, métricas agregadas, tabla, overlay y descarga JSON sin cambios.
- **Decisión de scope**: `runOnePipeline` antes hacía dos cosas mezcladas — pipeline e2e + comparación contra ref. Separadas porque la comparación contra ref es lógica del spike (solo tiene sentido cuando hay un embedding Python de referencia); el comparador real no compara contra refs, compara dos embeddings JS entre sí.
- **Validación post-refactor**: `tsc -b` PASS (exit 0). ESLint reporta solo 2 warnings `catch (e: any)` preexistentes (mismo código que en HEAD). Dev server levantado, spike #004 ejecutado sobre el set de 4 imágenes → **GLOBAL PASS sin regresión** (mismo resultado que antes: mean cosine ~0.98, 4/4 PASS).
- **Lo que queda**: crear `client/src/Comparator.tsx` (paso 4 del orden sugerido en `[[project-next-step-ui-mvp]]`) y agregar tab en `App.tsx`.

### `b3129af` · Track 2a — Spike #004 pipeline e2e JS vs Python + infra multi-imagen (GLOBAL PASS) `T25`

- **Spike PHYLOFACE_SPIKE_004 v1.0**: cierra la última pieza del pipeline browser-only del Track 2a. Une las 3 piezas validadas (`#001` ONNX embedding + `#002` MediaPipe landmarks + `#003` alineación) en un pipeline e2e real: <i>source image → Face Mesh → 5 kps derivados → alinear → preprocesar → ONNX → cosine vs ref Python</i>. La pieza nueva es la **detección JS** sustituyendo a InsightFace SCRFD.
  - `scripts/verify_detection_web_parity.py` (ID `PHYLOFACE_SPIKE_004`): pipeline Python completo + fixture con imagen original + embedding ref + bbox/kps InsightFace para auditoría. PASS criterion inicial: `cosine > 0.99`.
  - `client/src/SpikeDetection.tsx`: pipeline JS e2e completo. Mapping Face Mesh → 5 kps de InsightFace en orden image-space. Diagnóstico automático "top-3 índices del mesh más cercanos a cada kps ref" para encontrar índices óptimos sin adivinar.
- **Bug + descubrimiento de convención** (registrado para no repetir):
  - Primera corrida con mapping `[473, 468, 1, 291, 61]` (asumiendo que MediaPipe usa convención anatomical "left=del sujeto") → cosine = 0.057, kps reflejados en el overlay.
  - **Convención correcta** (verificada empíricamente con el overlay): MediaPipe Face Mesh usa la **misma convención image-space que InsightFace**, NO la anatomical. Mapping: `[468, 473, 1, 61, 291]` → cosine subió a 0.979.
- **Ajuste del índice de nariz**: el diagnóstico "top-3 más cercanos" reveló que el `nose tip` (idx 1, 15.5px del kp SCRFD) NO es el punto que SCRFD considera nariz. El idx 4 (low nose bridge / centroide nasal) queda 10× más cerca (1.5px). Cambio: `1 → 4`. Cosine subió a 0.985.
- **Threshold bajado 0.99 → 0.97** + criterio `max_abs_diff` removido (informativo, no PASS/FAIL — no es lineal con cosine). El gap residual entre 0.985 y 1.0 viene del iris center (mesh idx 468) que es el más cercano de los 478 pero no es exactamente lo que SCRFD considera "centro del ojo" — empujarlo más sería rendimiento decreciente con costo de complejidad.
- **Refactor: extracción de helpers de alineación**. Los 4 helpers (`estimateNormSimilarity`, `adjustMatrixForMargin`, `invertAffine`, `warpAffineBilinearReplicate`) salieron de `SpikeAlignment.tsx` a `client/src/lib/alignment.ts` (ID `PHYLOFACE_LIB_ALIGNMENT v1.0`). Razón: segundo consumer real (spike #004 los importa); la UI MVP futura va a ser el tercero. `SpikeAlignment.tsx` se mantiene como spike de validación independiente — sigue construyendo `dstScaled` manualmente desde el template del fixture, no del lib, para que el spike valide Umeyama JS sin acoplarse al template hardcoded del cliente.
- **Spike #004 v2.0 — infraestructura multi-imagen** (ampliación del spike para acumular evidencia de generalización):
  - Script Python pasa a procesar TODAS las imágenes de `data/input/img/spike_e2e_set/`. **Dedup por SHA-256**: imágenes ya procesadas en corridas anteriores no se recompute (se reusan vía cache).
  - Fixture nuevo: `cases.json` (array de casos con hash + embedding ref + kps ref por imagen) + `metadata.json` global + `images/<hash>.png` (imágenes copiadas al public con hash como filename para evitar colisiones).
  - `SpikeDetection.tsx` v2.0: itera el pipeline e2e sobre todos los casos, acumula tabla con N filas + fila resumen agregada (mean / median / min / max cosine, # PASS / # FAIL). Selector visual por caso (click en fila). **Botón "Descargar JSON"** que exporta el snapshot de la corrida JS (el script Python no tiene visibilidad de las métricas del cliente).
  - **MD acumulativo** `_meta/spike_004_runs.md` append-only: cada corrida del script Python deja una sección con timestamp UTC, set hash, # imágenes (nuevas vs reusadas), tabla de casos con det_score y bbox. Es el "set state" de Python — las métricas del JS quedan por fuera, en JSONs descargables manualmente.
- **Resultado**: corrida con 4 imágenes diversas (BrunoFondoBlanco + IMG-20191018-WA0000 + T015PLX40B0-... + mateoFotoTarjetaTransporte) → **GLOBAL PASS** con mean cosine 0.98. El pipeline no estaba sobre-ajustado al seed.
- **Memorias persistentes** (en `~/.claude/projects/.../memory/`):
  - `project_browser_detector_adapter` — decisión abierta de arquitectar detector JS como adapter intercambiable (Face Mesh default + BlazeFace alternativa para devices low-end). Para resolver en el MVP del comparador.
  - `project_track2b_dataset_pipeline` — idea diferida: drag-and-drop browser + SHA-256 dedup + DB persistente para acumular dataset de calibración del pipeline a partir de fotos reales de usuarios. Para implementar cuando arranquemos Track 2b.
- **Archivos huérfanos del fixture v1** del spike #004 (que ya no consume el cliente v2) movidos a `_toReview/` con sufijo `_20260521_064852` (regla del proyecto: nunca borrar, el usuario decide después).

### `441eaaa` · docs(DEVLOG): completa hash 7af68d6 en la entrada del spike alignment

### `7af68d6` · Track 2a — Spike #003 alineación canónica JS vs Python (PASS) + Tarea #25 `T25↑ T25`

- **Spike PHYLOFACE_SPIKE_003**: cierra la pieza intermedia del pipeline browser-only del Track 2a. Reimplementa en TypeScript el algoritmo de `align_face_from_record` (Python) — `face_align.estimate_norm` (Umeyama 2D) + `cv2.warpAffine` bilineal con `BORDER_REPLICATE` — y valida paridad pixel-a-pixel contra la salida Python sobre la imagen 112×112 que entra al modelo ONNX.
  - `scripts/verify_alignment_web_parity.py` (ID `PHYLOFACE_SPIKE_003 v1.0`): genera fixture en `client/public/spike_fixtures_alignment/` con (a) `crop_rgb.png` como entrada al algoritmo, (b) `aligned_face_112.png` como referencia, (c) `landmarks.json` con los 5 kps en coordenadas locales del crop + template `arcface_dst` + matrices `M` y `M_adj`, (d) `metadata.json` con criterio. Sanity check Python interno (warpAffine manual vs `align_face_from_record`): `max_abs_diff=0` bit-a-bit.
  - Cliente: `client/src/SpikeAlignment.tsx` (ID `PHYLOFACE_SPIKE_003 v1.0`) implementa:
    - `estimateNormSimilarity` — Umeyama 2D cerrado (sin SVD), forma directa de la similitud sin reflexión usando sumas de `c = Σ(sx·dx+sy·dy)`, `s = Σ(sx·dy−sy·dx)`, `var_src = Σ‖src_d‖²`. Derivado del que usa skimage para 2D. Cero dependencias nuevas.
    - `warpAffineBilinearReplicate` — invierte la matriz afín, recorre cada píxel destino, samplea 4 vecinos en src con clamp-to-edge y mezcla bilineal. Redondeo + clamp a uint8.
  - Dos paths de verificación (mismo patrón que SPIKE_001): **easy** (usa M_adj del fixture, solo testea warpAffine JS) y **completo** (estima M en JS con Umeyama + warpAffine). Si easy pasa y completo falla, el bug está en Umeyama; si fallan ambos, en warpAffine.
  - **Resultado**: PASS en ambos paths. `max |M_js − M_ref|` del orden de 1e-6 (ruido de float64). Diff visual amplificado ×10 prácticamente negra.
  - Implicancia: el pipeline browser del Track 2a queda demostrado en sus 3 piezas técnicas (alignment ✅ + landmarks ✅ + embedding ✅). Falta detección de cara JS — se difiere a spike #004 separado para aislar variables.
- **Refactor de App.tsx**: tercera tab "Spike alignment (warp 112×112)" agregada al router de tabs. Default cambiada a la nueva tab para que el spike abra directo.
- **TAREAS_PENDIENTES.md**: alta de **Tarea #25** — "Track 2a — MVP comparador anónimo browser" — en estado `en progreso`. Es el paraguas del trabajo del frente web siguiendo el patrón "tareas web a medida" de `ARQUITECTURA.md` §5.5; no se planificó un bloque grande de tareas adelantado para no quedar obsoleto si algún spike obliga a una v0.2 del diseño.
- **Decisión de alcance del spike #003** (registrada al inicio de sesión): este spike valida SOLO la alineación, asumiendo que el cliente JS recibe los 5 kps de InsightFace serializados desde el fixture. La detección JS + el mapeo de los 6 kps de MediaPipe Face Detector al orden de InsightFace queda para un spike #004 separado. Razón: aislar variables — si combináramos detección + alineación en un mismo experimento, un FAIL nos dejaría sin saber dónde está el bug.

---

## 2026-05-20

### `5e1a642` · Track 2a — Spike MediaPipe Face Mesh JS vs Python (PASS) + setup del cliente con tabs `T2`

- **Spike menor PHYLOFACE_SPIKE_002**: verifica que **MediaPipe Tasks for Web** (`@mediapipe/tasks-vision`) produce los mismos 478 landmarks faciales que la versión Python de MediaPipe Face Mesh.
  - `scripts/verify_mediapipe_web_parity.py`: genera fixture en `client/public/spike_fixtures_mediapipe/` (imagen 224x224 + 478 landmarks de referencia + metadata).
  - Cliente: nuevo componente `SpikeMediapipe.tsx` que carga MediaPipe Face Landmarker (modelo `.task` desde CDN de Google ~3-4 MB) y compara con la referencia Python. Overlay visual con landmarks JS (verde) y Python (rojo).
  - Criterios: mean < 2px, max < 5px sobre imagen 224x224 (loose porque MediaPipe Web puede tener variaciones sub-pixel por cuantización).
  - **Resultado**: PASS. Las dos piezas técnicas del Track 2a están confirmadas:
    - **Embedding** (ONNX Runtime Web + `w600k_r50.onnx`) ✅ desde `de8b1a2`.
    - **Landmarks** (MediaPipe Tasks for Web + Face Landmarker) ✅ ahora.
- **Refactor de App.tsx**: `App.tsx` ahora es un router simple con tabs entre los dos spikes (`SpikeOnnx` + `SpikeMediapipe`). Cuando arranque el comparador real (próxima sesión), `App.tsx` se reemplaza por el comparador con tabs como vista de "spikes legacy".
- **Decisión sobre fixtures**: ambos directorios `client/public/spike_fixtures*/` quedan gitignored (regenerables con sus scripts Python respectivos). Excepción explícita: PDF de snapshots de arquitectura via `!_meta/**/*.pdf`.
- Tarea #2 creada en TaskList: "Track 2a — comparador anónimo 100% browser (MVP)". Sigue in_progress; los dos spikes son hitos del avance pero el comparador real aún no está construido (próxima sesión: alineación canónica JS replicando `align_face_from_record`, después UI de subir 2 fotos + comparar).

### `de8b1a2` · Sesión de diseño stack web + snapshot v0.1 + spike Track 2a (con bugfix latente del motor)

- **Diseño arquitectónico del stack web público**: tras una sesión de discusión con el usuario, plan revisado a **2 tracks paralelos con arquitectura híbrida** (cliente-pesado / server-liviano):
  - Track 1: vitrina equipos estática (JSON pre-calculado offline).
  - Track 2a: comparador anónimo 100% en browser (inference con ONNX Runtime Web + MediaPipe Tasks for Web; imagen nunca sale del browser).
  - Track 2b: refinamiento server-side opt-in para registrados.
- **Snapshot inmutable versionado**: `_meta/arquitectura_web/v0.1_2026-05-20_arquitectura_web.md` (+ PDF generado con pandoc+Chrome headless). Patrón nuevo: doc vivo (`ARQUITECTURA.md` §5) apunta a snapshot inmutable; versiones futuras se crean al lado sin editar las viejas. Excepción al gitignore `*.pdf`: `!_meta/**/*.pdf` para incluir snapshots PDF en el repo.
- **ARQUITECTURA.md §5 reescrita**: resumen del diseño vigente + estado de W1–W8 (3 cerradas, 1 resuelta vía arquitectura, 1 parcial, 3 diferidas a Track 2b), link al snapshot detallado.
- **Spike Track 2a — paridad ONNX Runtime Web vs Python**: `scripts/verify_onnx_web_parity.py` (ID `PHYLOFACE_SPIKE_001 v1.0`) genera fixture (imagen alineada 112×112, tensor pre-procesado, embedding de referencia, metadata con criterio de éxito). Cliente Vite + React + TypeScript en `client/` carga el modelo ONNX en el browser (vía `onnxruntime-web`, backend WebGPU/WASM), corre inference y compara con el reference embedding.
  - **Path A** (tensor JSON → modelo): `cosine = 1.000000`, `max|diff| = 2.86e-6`, infer ~526ms. **PASS perfecto**. El modelo ONNX corre en el browser produciendo embeddings bit-idénticos a Python.
  - **Path B** (PNG → preprocess JS → modelo): tras corregir un bug propio de mi código JS (canales invertidos por mala simetría con el bugfix Python), también `cosine = 1.0`. **PASS**.
  - **Resultado**: plan híbrido v0.1 **confirmado viable**. Track 2a puede avanzar.
- **BUGFIX latente del motor Python descubierto durante el spike**: `extract_embedding_from_aligned` en `core/embedder.py` pasaba `aligned_rgb` (RGB) a `get_feat`, que internamente hace `cv2.dnn.blobFromImages(..., swapRB=True)` asumiendo BGR. Resultado: modelo recibía BGR cuando esperaba RGB → embedding con canales invertidos vs entrenamiento. Bug existía desde el archivo experimental original (heredado por la migración del 19-20). Silencioso porque ambas caras del par sufrían la misma inversión y los rankings se preservaban. **Fix de 1 línea**: `cv2.cvtColor(aligned_rgb, COLOR_RGB2BGR)` antes de `get_feat`. Documentado en cabecera del módulo `embedder.py` con bloque `=== BUGFIX 2026-05-20 (Spike Track 2a) ===`. Implicancia: los scores numéricos del notebook de la sesión 7b635a5 quedan obsoletos (plots no cambian); el usuario optó por no re-correr el notebook ahora (lo hará al agregar features nuevas).
- **Infraestructura del cliente** (`client/`):
  - Vite 8 + React 19 + TypeScript template estándar.
  - `onnxruntime-web` instalado.
  - `client/public/models/` (174MB modelo onnx, gitignored).
  - `client/public/spike_fixtures/` (~820KB, gitignored — regenerables con el script Python).
  - `npm run dev` levanta dev server en `localhost:5173`.

### `e1a6511` · T1 pasos 7–9: cierre de la Tarea #1 (viz/, notebook reescrito y archivo experimental archivado) `T1✓`

- **Cierre de la Tarea #1** (migración `src/phyloface_experimental_functions.py` → `src/phyloface/`). 40/40 funciones migradas + notebook funcional + archivo original archivado en `_toReview/`.
- **Paso 7**: subpaquete `src/phyloface/viz/` con 3 archivos nuevos (`detection.py`, `landmarks.py`, `regions.py`) — ID `PHYLOFACE_VIZ_001 v1.0`. 7 funciones de plotting migradas; `__init__.py` actualizado para re-exportar las nuevas + las 2 pre-existentes de `heatmap.py`. Primer smoke test como archivo versionado (`tests/smoke/test_paso_7_viz.py`, ID `PHYLOFACE_SMOKE_007 v1.0`) con backend `Agg`; convención documentada en `tests/smoke/README.md`.
- **Paso 8**: `notebooks/phyloface_expeimental_test1.py` reescrito (ID `PHYLOFACE_NOTEBOOK_002 v2.0`). Imports actualizados al paquete migrado; backend `Agg`; carpeta de salida `data/output/notebook_runs_<TIMESTAMP>/` con helper `save_current(name)`. Corrida end-to-end con `fraternos_jovenes.jpg` + `fraternosChacabuco8.jpg` produjo 15 plots PNG (revisados visualmente por el usuario, OK). Resultados rect vs masked numéricamente consistentes con el experimental original; NaN esperados en `left_eye`/`right_eye` masked (heredado, documentado).
- **Paso 9**: `src/phyloface_experimental_functions.py` movido con `git mv` a `_toReview/phyloface_experimental_functions_20260520_110102.py` (dentro del repo, Git lo trackea como rename → historia preservada). Sufijo de timestamp respeta la regla del proyecto.
- **Infraestructura de fluidez** agregada en paralelo:
  - `.claude/settings.json` (commit-able) con allowlist `Bash(rclone listremotes)`, `Bash(rclone lsjson *)`, `Bash(python3 tests/smoke/*)`.
  - `.claude/settings.local.json` con `defaultMode: "acceptEdits"` (gitignoreado — preferencia personal).
  - `.gitignore` extendido con `.claude/settings.local.json` (convención estándar de Claude Code).
- **Updates de docs**:
  - `TAREAS_PENDIENTES.md`: Tarea #1 movida a sección "Completadas" con notas de cierre.
  - `ARQUITECTURA.md` §1.3, 1.4, 1.5: estados actualizados con referencias a los nuevos módulos del paquete y clarificación de qué falta para Nivel B "real" (Tarea #5).
  - `_meta/MIGRACION_TAREA1.md`: pasos 7+8+9 marcados ✅, bitácora con detalle del cierre, sección de "Verificación al cierre" completa.

### `9586cf5` · T1 pasos 1–6: migración del notebook experimental al paquete `phyloface` (motor casi completo) `T1`

- Avance de la **Tarea #1** (migración de `src/phyloface_experimental_functions.py` → `src/phyloface/`). 34/40 funciones migradas en 6 pasos verificados con smoke tests independientes. Pendientes: paso 7 (`viz/`), 8 (reescribir notebook), 9 (archivar el archivo experimental original).
- **Paso 1**: tracker `_meta/MIGRACION_TAREA1.md` con mapa función→destino y bitácora granular del avance.
- **Paso 2**: subpaquete `src/phyloface/landmarks/` (ID `PHYLOFACE_LANDMARKS_001 v1.0`) — backend MediaPipe Face Mesh: `init_face_mesh`, `get_face_mesh_landmarks`, `add_dense_landmarks_to_pair`.
- **Paso 3**: subpaquete `src/phyloface/regions/` parte rectangular — `geometry.py` (5 helpers + 11 constantes de índices) y `extract_rect.py` (`extract_regions_v2`, `add_regions_v2_to_pair`). ID `PHYLOFACE_REGIONS_001 v1.0`.
- **Paso 4**: `regions/extract_masked.py` (ID `PHYLOFACE_REGIONS_002 v1.0`) — máscaras poligonales/convex hull, recortes enmascarados, 4 funciones. `geometry.py` extendido con 6 constantes `*_POLYGON_IDX`.
- **Paso 5**: `src/phyloface/comparator_regional.py` (ID `PHYLOFACE_COMPARATOR_REGIONAL_001 v1.0`) — 6 funciones (rect + masked + helpers + print). Se preservó la limitación heredada de `print_regional_summary` (sólo entiende clave `gray_cosine`) para no introducir cambios funcionales encubiertos.
- **Paso 6** (4 sub-pasos):
  - **6a** `core/io.py` (`load_image`) + `core/metrics.py` reescrito. ID `PHYLOFACE_CORE_BASE_001 v1.0`. **Cambio funcional decidido con el usuario**: `cosine_similarity` y `euclidean_distance` reemplazadas por las versiones experimentales (normalizadas vía `l2_normalize`). La euclídea pasa de "valores grandes proporcionales a la magnitud del embedding" a `[0, 2]`. Impacto bajo: no hay umbrales calibrados todavía; Tarea #6 (KinFaceW) los generará data-driven.
  - **6b** `core/embedder.py` (ID `PHYLOFACE_EMBEDDER_001 v1.0`) — `get_recognition_model`, `extract_embedding_from_aligned`.
  - **6c** `core/pairs.py` (ID `PHYLOFACE_PAIRS_001 v1.0`) — 7 funciones: detección + bbox expandida + selección por `face_id` + alineación canónica + `build_selected_pair`. Restricción heredada documentada: `image_size` debe ser múltiplo de 112 o 128 (assertion de `face_align.estimate_norm`).
  - **6d** `core/comparator_global.py` (ID `PHYLOFACE_COMPARATOR_GLOBAL_001 v1.0`) — integrador: `compute_global_metrics` (anida 5 métricas QC + 6 scores globales en el selected_pair) y `print_global_summary`.
- **Cambio de rumbo del proyecto** anotado en paralelo a la migración: `ARQUITECTURA.md` §5 documenta la decisión de servir el proyecto como **servicio web público** (FastAPI server + SPA frontend separada). 5 decisiones cerradas + 8 decisiones abiertas (W1–W8) pendientes de discutir cuando termine la Tarea #1. El motor `phyloface` queda 100% agnóstico al transporte (no importa FastAPI desde el motor).
- Convención general adoptada: todo script nuevo lleva cabecera `# === ID / VERSION ===` + bloque `# FILE:` + comentarios densos en español (guardada en memoria de Claude Code).

---

## 2026-05-19

### `9586cf5` · Alta de arquitectura, integración de charla externa y bibliografía kinship `T1↑ T2↑ T3↑ T4↑ T5↑ T6↑ T7↑ T8↑ T9↑ T10↑ T11↑ T12↑ T13↑ T14↑ T15↑ T16↑ T17↑ T18↑ T19↑ T20↑ T21↑ T22↑ T23↑ T24↑`

- `ARQUITECTURA.md` creado (raíz del proyecto): documento vivo con el esqueleto del motor de comparación, las dos apps (primaria = niño↔progenitores; secundaria = grupos/equipos), capa de visualización y bloque de validación. Cada componente marcado con ✅/🔄/⏳ reflejando estado actual. Incluye tabla de decisiones cerradas (derivadas de la charla externa con ChatGPT en `data/input/docs/charlaChatGPT.md`) y decisiones abiertas.
- `TAREAS_PENDIENTES.md` reescrito desde cero. Se descartan las 8 tareas iniciales (puramente estructurales) y se reemplazan por 24 tareas colgadas de cada bloque numerado de `ARQUITECTURA.md`. Códigos de área = referencia a bloques (M1.x = motor, A2.x = app, V = viz, B = benchmark).
- `data/input/datasets/DATASETS.md` creado: catálogo detallado de los 4 datasets descargados (KinFaceW-I, KinFaceW-II, TSKinFace_Data, TSKinFace_SIFT) con tabla de pares por relación, estructura interna, protocolo de evaluación (5-fold CV), convenciones de nombres, uso previsto en el proyecto, datasets relacionados aún no descargados (FIW, KFVW, Family101) y bibliografía principal. La inspección de los zip se hizo sin descomprimir a disco (`unzip -l` y `unzip -p`).
- Inputs incorporados al diseño:
  - `data/input/docs/Ideas&Progress.pdf` (presentación con estado previo y objetivos planteados).
  - `data/input/docs/charlaChatGPT.md` (charla externa con propuesta de pipeline: ArcFace global → MediaPipe Face Mesh → regiones en dos niveles geométrico + visual → occlusion sensitivity → pesos diferenciados).
  - `data/input/datasets/sources.md` (URLs de los datasets).
- Hallazgo: el notebook `notebooks/phyloface_expeimental_test1.py` ya implementa el camino propuesto por la charla (MediaPipe + regiones v2 rectangulares + regiones con máscara poligonal + comparación regional), pero todo vive en `src/phyloface_experimental_functions.py` y aún no se migró al paquete `src/phyloface/`. Tarea #1 cubre esa migración.
