> Generado por un agente de investigación (web) el 2026-05-23 como insumo para la
> Tarea #6. Los números de la tabla SOTA provienen en parte de tablas de papers
> secundarios — verificar contra fuente primaria antes de citarlos como propios.
> Doc de referencia, no contrato de código. Procedimiento de calibración: `CALIBRACION_TAREA6.md`.

# Informe de Investigación Bibliográfica: Verificación de Parentesco Facial para mendelEmbeddings

**Fecha:** 2026-05-23
**Proyecto:** mendelEmbeddings — parentesco facial (KinFaceW-I/II, TSKinFace)
**Alcance:** Datasets, SOTA, mejoras implementables, pitfalls de evaluación

---

## 1. Datasets y Protocolos

### 1.1 KinFaceW-I y KinFaceW-II

**Paper fundacional:** Jiwen Lu et al., "Neighborhood Repulsed Metric Learning for Kinship Verification", CVPR 2012 / TPAMI 2014. ([PubMed](https://pubmed.ncbi.nlm.nih.gov/24356353/), [kinfacew.com](https://www.kinfacew.com/))

**Características del dataset:**

| Propiedad | KinFaceW-I | KinFaceW-II |
|---|---|---|
| Relaciones | FS, FD, MS, MD | FS, FD, MS, MD |
| Pares por relación | 156, 134, 116, 127 (total ~533) | 250 cada una (total 1000) |
| Origen de las fotos | Fotos **distintas** | Misma foto familiar |
| Resolución provista | 64×64 px | 64×64 px |
| Folds de evaluación | 5-fold CV | 5-fold CV |

**Protocolo de evaluación** ([protocol.html](https://www.kinfacew.com/protocol.html)):

El benchmark define tres settings:
- **Unsupervised**: sin información de etiquetas de kinship en entrenamiento.
- **Image-restricted**: se usan sólo los pares de kinship etiquetados del training fold.
- **Image-unrestricted**: además de los pares de kinship, se usa información de identidad para construir pares negativos adicionales.

Los pares positivos (kin) y negativos (no-kin) se generan **dentro de cada fold**, con splits fijos provistos por el benchmark para reproducibilidad. La métrica primaria es **mean accuracy** (accuracy promedio sobre las 4 relaciones) y curvas ROC/AUC. Cada fold tiene pares positivos y negativos balanceados (en KinFaceW-II: 50 positivos y 50 negativos por fold por relación).

**Diferencia crítica entre -I y -II:** KinFaceW-I usa fotos tomadas en diferentes momentos/contextos → más variación realista por iluminación y envejecimiento. KinFaceW-II usa recortes del mismo evento fotográfico → artificialmente más fácil (ver Sección 4 sobre pitfalls).

### 1.2 TSKinFace (Tri-Subject Kinship Face Database)

**Paper fundacional:** Xiaoqian Qin, Xiaoyang Tan, Songcan Chen, "Tri-Subject Kinship Verification: Understanding the Core of A Family", IEEE Transactions on Multimedia, 2015. ([arXiv 1501.02555](https://arxiv.org/abs/1501.02555), [página oficial NUAA](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/TSKinFace.html))

**Características:**

| Propiedad | Valor |
|---|---|
| Relaciones | FM-S (Father-Mother-Son), FM-D (Father-Mother-Daughter) |
| Grupos tri-sujeto | 513 FM-S + 502 FM-D = 1015 total |
| Individuos | 2589 personas |
| Diversidad racial | FM-S: 343 asiáticos + 170 no-asiáticos; FM-D: 331 + 171 |
| Resolución estándar | 64×64 px (también disponibles imágenes originales) |
| Preprocesamiento | Normalización geométrica; patches 16×16 en grilla 7×7 |

**Protocolo de evaluación:**
- 5-fold CV con estratificación por relación
- **Pares positivos**: todos los tríos (padre, madre, hijo/a) con relación real
- **Pares negativos**: combinaciones aleatorias de un niño con dos padres que **no son sus padres biológicos** (constraint explícito)
- Métrica: verification accuracy (binaria: kin / no-kin)
- Tarea: "one-versus-two" — dado un niño y dos padres, determinar si el niño es hijo de ese par

**Línea base histórica:** SVM sobre features concatenadas → 53.4%. Humanos: ~79.9% en FM-S. El paper introduce RSBM (Relative Symmetric Bilinear Model) que alcanza FM-S: 86.4%, FM-D: 84.4% (mean 85.4%), superando DDML (81.0%).

---

## 2. Métodos y SOTA

### 2.1 Familias de métodos

**A) Metric learning clásico (pre-deep learning)**
NRML (Lu et al., 2014): aprende una métrica que repulsa vecinos negativos de pares positivos. DMML/MNRML: extensiones multi-métrica. Features: LBP, HOG, SIFT.

**B) Deep features de reconocimiento off-the-shelf**
Uso de embeddings de modelos entrenados en reconocimiento facial (FaceNet, VGGFace, ArcFace, AdaFace) como features + comparación coseno o pequeño clasificador encima. El paper de baseline 2020 ([arXiv 2006.11739](https://ar5iv.labs.arxiv.org/html/2006.11739)) establece ArcFace (ResNet-101, 512-d) + coseno como baseline moderno, con clasificación por familia como fine-tuning.

**C) Siamese/redes de pares con métrica aprendida**
Arquitecturas Siamese con pérdida de contrastive o triplet. La fusión de features del par (diferencia al cuadrado, diferencia cuadrada, producto) es estándar.

**D) Atención y regiones faciales**
FaCoRNet (2023): cross-attention entre features de dos caras para identificar regiones faciales importantes (ojos, nariz, boca). GKR (2020): grafo estrella para razonamiento relacional. Reasoning Graph Networks (2021): H-RGN jerárquico.

**E) Transformers**
ViT aplicado a KinFaceW-II (2022-2024); cross-attention entre pares de imágenes.

**F) Aprendizaje contrastivo supervisado**
TeamCNU en RFIW2021 usa ArcFace + MLP + contrastive loss, 0.80 acc en kinship verification ([arXiv 2111.00598](https://ar5iv.labs.arxiv.org/html/2111.00598)). KFC (2023): contrastive loss con fairness adversarial ([arXiv 2309.10641](https://arxiv.org/abs/2309.10641)).

**G) Multi-tarea y correlación entre relaciones**
CCMTL (2025): correlación entre tipos de relación (FD y MS son más correlacionadas que FS y MD) modelada con SVDD + aprendizaje compartido. La idea es que los cuatro tipos de parentesco no son independientes.

**H) Tri-subject / scoring padre+madre→hijo**
RSBM (Qin et al., 2015): modelo bilineal simétrico. Deep Fusion Siamese (2020): dos Siamese (padre-hijo, madre-hijo) fusionados con pesos. TeamCNU RFIW2021: 0.84 avg en tri-subject (FMS: 0.86, FMD: 0.82).

**I) Fusión multi-escala / ConvNeXt+ViT**
ConvNeXt-EfficientNet-ViT fusion (2024/2025): 84.85% KinFaceW-I, 91.65% KinFaceW-II.

### 2.2 Tabla SOTA

| Método | Año | Backbone / Features | KinFaceW-I (mean) | KinFaceW-II (mean) | TSKinFace | Notas |
|---|---|---|---|---|---|---|
| NRML | 2014 | LBP/HOG | ~72.8–77.5% | ~72.9–74.7% | n/d | Baselines oficiales (restricted/unrestricted). [kinfacew.com/results](https://www.kinfacew.com/results.html) |
| MNRML | 2014 | LBP | 69.9% | 76.5% | n/d | Extensión multi-métrica de NRML. Tabla FaCoRNet. |
| DMML | ~2017 | LBP/deep | 72.3% | 78.3% | n/d | Discriminative Multi-Metric Learning |
| RSBM (paper TSKinFace) | 2015 | HOG+patches | n/d | n/d | FM-S 86.4%, FM-D 84.4% (mean 85.4%) | Primer SOTA tri-subject. [arXiv 1501.02555](https://arxiv.org/abs/1501.02555) |
| CNN-Basic | ~2018–19 | CNN genérico | 74.8% | 85.3% | n/d | Tabla FaCoRNet |
| CNN-Point | ~2018–19 | CNN con landmarks | 77.5% | 88.4% | n/d | Tabla FaCoRNet |
| D-CBFD | ~2019 | Deep | 78.5% | 78.5% | n/d | Tabla FaCoRNet |
| WGEML | ~2019–20 | Deep | 78.7% | 82.8% | n/d | Tabla FaCoRNet |
| Deep+Tensor ELM | ~2019 | Deep+Tensor | n/d | n/d | FM-S 90.94%, FM-D 91.23% | Mejor resultado reportado en TSKinFace. Búsqueda bibliográfica. |
| GKR (Graph-based Kinship Reasoning) | 2020 | ResNet | 83.85% | 91.75%* | n/d | Star-graph reasoning. [arXiv 2004.10375](https://arxiv.org/abs/2004.10375). *Nota: este número también es atribuido por otras fuentes al método "DFLKV" y "Unified Approach" — hay ambigüedad en la literatura. |
| H-RGN (Hierarchical Reasoning Graph) | 2021 | ResNet | ~83.9% | ~92.8% | n/d | [arXiv 2109.02219](https://arxiv.org/pdf/2109.02219) |
| ArcFace baseline (fine-tuned) | 2020 | ArcFace ResNet-101 | n/d (RFIW: 0.78) | n/d | n/d | Threshold por relación; evaluado en RFIW2020. [arXiv 2006.11739](https://ar5iv.labs.arxiv.org/html/2006.11739) |
| Contrastive (TeamCNU RFIW2021) | 2021 | ArcFace ResNet-101 + MLP | n/d (RFIW: 0.80) | n/d | FMS 0.86, FMD 0.82 | [arXiv 2111.00598](https://ar5iv.labs.arxiv.org/html/2111.00598) |
| Cross-gen feature interaction | 2021 | Deep | 82.71% | 85.83% | n/d | [arXiv 2109.02809](https://arxiv.org/pdf/2109.02809); NRML baseline reportado: ~79.5–82.1% |
| FaCoRNet | 2023 | ArcFace / AdaFace + CI blocks + cross-attn | **81.5%** | **90.6%** | n/d | ICCVW 2023. [arXiv 2304.04546](https://arxiv.org/abs/2304.04546). Mejor en KinFaceW-I FD, MS. |
| Forest Neural Network (FNN) | 2025 | ResNet-18, ERN, VGG + GNN | 82.1% | **93.8%** | n/d | Mejor en KinFaceW-II global. [arXiv 2504.18910](https://arxiv.org/abs/2504.18910) |
| ConvNeXt+EfficientNet+ViT fusion | 2024–25 | Multi-modelo fusion | 84.85% | 91.65% | n/d | Fuente: búsqueda bibliográfica Springer 2025 |
| CCMTL (multi-tarea correlación) | 2025 | LBP/SIFT (no-deep) | 78.2% | 78.8% | n/d | Con features clásicas; incluido como referencia de multi-task. [PMC12419595](https://pmc.ncbi.nlm.nih.gov/articles/PMC12419595/) |

**Nota de honestidad:** Hay ambigüedad en la literatura respecto al número 83.85%/91.75% en KinFaceW-I/II — varias fuentes lo atribuyen a métodos distintos (GKR, "Unified Approach", "DFLKV"). Los números de la tabla de FaCoRNet (paper primario del ICCVW 2023) son los más confiables para los métodos listados. Reportar números sin leer el paper primario implica riesgo de error; marcar columnas con "n/d" donde no se encontró fuente primaria verificable.

---

## 3. Mejoras Implementables para mendelEmbeddings

El stack actual usa ArcFace embeddings 512-d + coseno crudo, sin ningún aprendizaje de parentesco. AUC rank-based empírico: ~0.74 global (KinFaceW-I, tratando imágenes como pre-alineadas), con cross-género (Father-Daughter) el peor (~0.67). Esto es consistente con los baselines históricos de NRML (72–77%) y el comportamiento conocido de features de reconocimiento sin adaptación.

Las mejoras están ordenadas por ratio impacto/esfuerzo estimado, con flag de portabilidad cliente ONNX.

---

### Mejora 1 — Calibración de umbrales por relación y por fold [CLIENTE viable]

**Descripción:** El umbral coseno óptimo varía por tipo de relación (FS/FD/MS/MD). El paper [arXiv 2006.11739](https://ar5iv.labs.arxiv.org/html/2006.11739) demuestra explícitamente que elegir umbrales por separado para cada tipo de kinship mejora resultados. En cross-género (FD, MS) la distribución de scores positivos cae más abajo → el umbral óptimo es menor. Hacer ROC per-relación sobre el training fold y seleccionar el umbral de Youden (máximo J = TPR - FPR) da la Tarea #6 resuelta correctamente.

**Implementación:** Puramente analítica sobre los embeddings actuales, sin reentrenamiento. Python + sklearn en 30 minutos. El umbral calibrado se persiste como JSON por relación.

**Impacto esperado:** Ganar 2–5 puntos de accuracy sin cambiar el modelo. Cross-género (FD) es donde más se gana.

**Portabilidad cliente:** Sí — el umbral es sólo un número flotante por relación tipo. La lógica de decisión es una comparación.

**Esfuerzo:** Muy bajo (horas).

---

### Mejora 2 — Cabeza de metric learning pequeña sobre embeddings ArcFace [CLIENTE viable con exportación ONNX]

**Descripción:** Entrenar una MLP ligera (2 capas, ~512→128→1 o bien aprender una proyección lineal con margen) sobre los embeddings 512-d de ArcFace, usando pares (positivo, negativo) de KinFaceW con contrastive loss o triplet loss. Esto es exactamente lo que hace TeamCNU (2021) y FaCoRNet (2023) con resultados de 0.80 avg en RFIW y 81.5% / 90.6% en KinFaceW.

El modelo aprende a ignorar las dimensiones de identidad que no son heredables y a amplificar las correlaciones de parentesco. La literatura es clara: pasar de coseno crudo (~0.74 AUC) a un modelo de parentesco encima gana ~7–15 puntos de accuracy.

**Arquitectura mínima viable:**
- Input: par de embeddings ArcFace → concatenar features de diferencia: `[|e1-e2|, (e1-e2)^2, e1*e2]` → ~1536-d
- Capa 1: Linear(1536, 256) + BatchNorm + ReLU + Dropout(0.3)
- Capa 2: Linear(256, 64) + ReLU
- Output: Linear(64, 1) + Sigmoid

Entrenamiento: 5-fold CV sobre KinFaceW-I/II con cross-entropy loss. Regularización fuerte (dropout) por el tamaño pequeño del dataset (~500 pares).

**Portabilidad cliente:** Sí — la cabeza MLP tiene ~400K parámetros, exporta a ONNX trivialmente. En el cliente, el ArcFace ya corre vía ONNX Runtime Web; esta cabeza se agrega como segundo modelo ONNX. Latencia adicional: <1ms.

**Esfuerzo:** Medio (1–2 días de entrenamiento + evaluación).

**Nota de riesgo:** Con ~500 pares y 5-fold CV, hay riesgo de sobreajuste. Usar dropout agresivo, early stopping, y reportar el AUC del fold de test (no del training). Validar que el AUC no supere en más de 0.05 al número esperado dado el tamaño del dataset.

---

### Mejora 3 — Scoring tri-subject para pedigree (padre+madre→hijo) [CLIENTE viable]

**Descripción:** Dado que el árbol genealógico tiene tanto padre como madre disponibles, el score de parentesco de un individuo con respecto a sus padres puede combinarse. La literatura (RSBM 2015, Deep Fusion Siamese 2020, TeamCNU 2021) muestra que combinar el score hijo–padre con el score hijo–madre supera a cualquiera de los dos por separado.

**Fórmula simple (sin entrenamiento):** `score_tri = α * cosine(hijo, padre) + (1-α) * cosine(hijo, madre)` donde α se calibra en un validation fold. La literatura sugiere que un hijo puede parecerse más a uno de los padres, de modo que un max o soft-max ponderado supera al promedio simple.

**Fórmula aprendida:** Si se entrena la cabeza MLP (Mejora 2), se puede usar como input la concatenación de `[features_comparación_padre, features_comparación_madre]` → el modelo aprende la ponderación óptima. Esto replica exactamente el setup de tri-subject del RFIW2021 donde TeamCNU logra 0.84 avg.

**Portabilidad cliente:** Sí — es operaciones aritméticas sobre cosenos o un segundo forward pass de la cabeza MLP.

**Esfuerzo:** Bajo si se construye sobre Mejoras 1 y 2. El pedigree ya existe en el árbol genealógico del proyecto.

---

### Mejora 4 — Fusión coseno global + scores regionales [SERVER primario, CLIENTE posible]

**Descripción:** El motor "regional" actual del proyecto ya extrae embeddings por región (ojos, nariz, boca, etc.). FaCoRNet (2023) y GKR (2020) demuestran que ponderar regiones faciales mediante atención mejora la discriminación de parentesco. La fusión simple de scores regionles (media ponderada) puede mejorar el coseno global.

**Implementación:** Calcular coseno por región y combinarlos con pesos aprendidos o fijos. Los pesos pueden calibrarse sobre KinFaceW con regresión logística sobre el vector de scores regionales (features = [cos_ojo_izq, cos_ojo_der, cos_nariz, cos_boca, cos_global]).

**Nota importante sobre los pesos:** Investigación en kinship verification señala que los ojos son altamente heredables, mientras que la boca/piel/textura son más ruidosas. Sin embargo, los pesos óptimos varían por relación (FD vs FS pueden tener perfiles distintos). Lo correcto es aprender pesos separados por relación.

**Portabilidad cliente:** Parcialmente. Si los embeddings regionales ya corren en el cliente (ONNX), la fusión es suma ponderada → trivial. Si el alineamiento regional requiere procesamiento adicional, es server-only.

**Esfuerzo:** Medio-alto. Requiere definir y validar las regiones, extraer embeddings por región en el cliente, y aprender pesos sobre KinFaceW.

---

### Mejora 5 — Fine-tuning del backbone ArcFace con pérdida de parentesco [SERVER only]

**Descripción:** En lugar de sólo añadir una cabeza encima, fine-tuning del backbone ArcFace (las últimas capas) con triplet loss de parentesco. FaCoRNet (2023) hace exactamente esto y obtiene resultados SOTA. La ventaja es que las representaciones internas del backbone se adaptan al espacio de parentesco.

**Advertencia:** Requiere GPU, cuidado extremo de no olvidar el conocimiento de identidad (catastrophic forgetting), y KinFaceW es pequeño para esto. En la práctica la literatura usa frozen backbone + cabeza (Mejora 2) como primer paso, y fine-tuning parcial como segundo paso con learning rate muy bajo.

**Portabilidad cliente:** No directamente. Un backbone ArcFace fine-tuned es el mismo tamaño que el original (~170MB para w600k_r50). El overhead en el cliente sería reemplazar el modelo ONNX base.

**Esfuerzo:** Alto (GPU, varios días de experimentos, riesgo de regresión en reconocimiento).

---

### Mejora 6 — Ponderación por género del par [CLIENTE viable]

**Descripción:** La literatura es consistente: relaciones cross-género (Father-Daughter, Mother-Son) son más difíciles (~0.67–0.72 AUC) que same-gender (Father-Son, Mother-Daughter). Esto sugiere que el sistema debe usar umbrales distintos y posiblemente modelos distintos por tipo de relación. Si se conoce el género de los sujetos (disponible vía el pedigree del árbol), se puede seleccionar el modelo/umbral apropiado.

**Implementación:** El pedigree del árbol ya tiene la relación (padre/madre/hijo). Con detección de género (disponible en InsightFace buffalo_l), el tipo de relación se infiere automáticamente.

**Portabilidad cliente:** Sí — el género se puede inferir client-side con InsightFace ONNX.

**Esfuerzo:** Bajo (es extensión de Mejora 1).

---

### Resumen de prioridades

| Rango | Mejora | Esfuerzo | Impacto esperado | Cliente? |
|---|---|---|---|---|
| 1 | Calibración de umbrales por relación | Muy bajo | +2–5 pts accuracy, inmediato | Sí |
| 2 | Cabeza MLP de metric learning (contrastive) | Medio | +7–15 pts accuracy | Sí (ONNX) |
| 3 | Scoring tri-subject (padre+madre→hijo) | Bajo (sobre mejoras anteriores) | +3–8 pts en tri-subject | Sí |
| 4 | Ponderación por género/relación | Bajo | +2–4 pts cross-género | Sí |
| 5 | Fusión coseno global + regional ponderada | Medio-alto | +3–6 pts, incierto | Parcial |
| 6 | Fine-tuning del backbone | Alto | potencialmente el mayor, pero con riesgo | No |

---

## 4. Pitfalls de Evaluación

### Pitfall 1: Bias "From Same Photo" en KinFaceW-II

**Descripción:** Dawson et al. (2018) demostraron que un clasificador CNN entrenado sólo para detectar si dos caras provienen de la misma fotografía alcanza resultados near-SOTA en benchmarks de kinship verification — **más del 90% de accuracy en al menos uno de los datasets evaluados** (KinFaceW-I, KinFaceW-II, TSKinFace, Cornell KinFace, FIW). ([arXiv 1809.06200](https://arxiv.org/abs/1809.06200), [Oxford VGG publication](https://www.robots.ox.ac.uk/~vgg/publications/2018/Dawson18/dawson18.pdf))

La causa: las caras de pares positivos en KinFaceW-II son recortes de la misma foto familiar. El modelo aprende señales de fondo, iluminación, cámara, ruido de sensor, ropa — no parentesco genético.

**Implicación directa para mendelEmbeddings:** Los resultados en KinFaceW-II son inflados artificialmente. Un AUC de 0.74 en KinFaceW-I (fotos distintas) es un número más honesto que un AUC similar o mayor en KinFaceW-II. **La evaluación primaria debe ser sobre KinFaceW-I.**

**Conclusión del paper:** "It is likely that the fraction of kinship explained by existing kinship models is small" — la señal de parentesco genético propiamente dicha es mucho más débil de lo que muestran los benchmarks.

### Pitfall 2: Resolución 64×64 vs. resolución real de entrada del modelo

El dataset distribuye imágenes a 64×64 px, pero los modelos modernos (ArcFace w600k_r50) fueron entrenados para entrada 112×112 px. Al redimensionar de 64→112 hay artefactos de interpolación. El proyecto trata las imágenes como "pre-alineadas" pero el alineamiento a 64×64 del dataset puede no ser compatible con el formato 112×112 de ArcFace. **Esto puede explicar parte del gap entre el AUC empírico del proyecto (~0.74) y el SOTA reportado con imágenes de mayor resolución.**

**Mitigación:** Usar las imágenes originales de KinFaceW (disponibles bajo pedido en kinfacew.com) y alinear en 112×112 con el propio pipeline de InsightFace, en lugar de usar las imágenes de 64×64 pre-distribuidas.

### Pitfall 3: Leakage si el split no es "family-aware"

**Descripción:** En KinFaceW, los 5 folds están pre-especificados. Si un investigador re-parte los datos aleatoriamente (sin seguir los splits oficiales), individuos de la misma familia pueden aparecer en training y test. Los pares negativos construidos con personas del mismo fold pueden incluir primos/hermanos, que son técnicamente "no kin" en las etiquetas pero comparten más rasgos que strangers — contaminando negativos del test set con casos difíciles.

**Mitigación:** Usar siempre los splits oficiales de 5-fold del benchmark. No generar pares negativos cross-fold.

### Pitfall 4: Desequilibrio positivos/negativos y construcción de negativos

Los negativos se generan aleatoriamente (N pares negativos = N pares positivos por convención). Pero hay N(N-1) pares negativos posibles — sólo se samplea una fracción. Si el sampling de negativos varía entre corridas, los resultados no son reproducibles. Además, algunos negativos aleatorios pueden ser "fáciles" (ethnicidades muy distintas) o "difíciles" (misma etnia, edad similar), sesgando los resultados según qué negativos se sampleen. **Usar los splits fijos del benchmark es la única defensa.**

### Pitfall 5: Reportar accuracy media vs. accuracy por relación

La accuracy media de las 4 relaciones puede ocultar que un modelo es muy bueno en FS y MS pero malo en FD y MD (cross-género). Para mendelEmbeddings, que tiene un foco en familias específicas (el pedigree del árbol), las relaciones cross-género son probablemente más frecuentes. Reportar siempre las 4 relaciones por separado y el mean.

### Pitfall 6: Sobreajuste al dataset en modelos con cabeza entrenada

Con ~500 pares en KinFaceW-I y 5-fold CV, cada training fold tiene ~400 pares positivos. Una MLP con >10K parámetros puede memorizar. Señal de sobreajuste: training accuracy >> test accuracy en el fold. Mitigación: dropout fuerte (p≥0.4), L2 regularización, early stopping, y comparar AUC vs. baseline coseno crudo en el mismo fold.

---

## 5. Referencias (con URLs)

1. **KinFaceW dataset y benchmark oficial** — Lu et al., "Neighborhood Repulsed Metric Learning for Kinship Verification", TPAMI 2014. Sitio: https://www.kinfacew.com/ — Protocolo: https://www.kinfacew.com/protocol.html — Resultados oficiales: https://www.kinfacew.com/results.html

2. **TSKinFace — paper fundacional** — Qin et al., "Tri-Subject Kinship Verification: Understanding the Core of A Family", IEEE TMM 2015. arXiv: https://arxiv.org/abs/1501.02555 — ar5iv HTML: https://ar5iv.labs.arxiv.org/html/1501.02555 — Página oficial NUAA: https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/TSKinFace.html

3. **FaCoRNet — SOTA 2023 en KinFaceW** — Su et al., "Kinship Representation Learning with Face Componential Relation", ICCVW 2023. arXiv: https://arxiv.org/abs/2304.04546 — HTML: https://arxiv.org/html/2304.04546

4. **Comprehensive Review 2022** — "Facial Kinship Verification: A Comprehensive Review and Outlook", IJCV 2022. Springer: https://link.springer.com/article/10.1007/s11263-022-01605-9 — PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9016696/

5. **From Same Photo: Cheating on Visual Kinship Challenges** — Dawson et al., ECCV Workshop 2018. arXiv: https://arxiv.org/abs/1809.06200 — Oxford VGG: https://www.robots.ox.ac.uk/~vgg/publications/2018/Dawson18/dawson18.pdf

6. **Forest Neural Network (FNN) — 2025** — arXiv: https://arxiv.org/abs/2504.18910 — HTML: https://arxiv.org/html/2504.18910v1

7. **Better Baseline for Kinship Recognition (ArcFace)** — "Achieving Better Kinship Recognition Through Better Baseline", 2020. ar5iv: https://ar5iv.labs.arxiv.org/html/2006.11739 — Repo GitHub: https://github.com/vuvko/fitw2020

8. **RFIW 2021 Competition** — "The 5th Recognizing Families in the Wild Data Challenge", 2021. ar5iv: https://ar5iv.labs.arxiv.org/html/2111.00598

9. **Graph-based Kinship Reasoning Network (GKR)** — 2020. arXiv: https://arxiv.org/abs/2004.10375

10. **Reasoning Graph Networks for Kinship Verification (H-RGN)** — 2021. arXiv: https://arxiv.org/pdf/2109.02219

11. **KFC: Fair Contrastive Loss** — 2023. arXiv: https://arxiv.org/abs/2309.10641

12. **Supervised Contrastive + ArcFace for FIW** — 2023. ar5iv: https://ar5iv.labs.arxiv.org/html/2302.09556

13. **CCMTL: Correlation Calculation Multi-Task Learning** — PLoS ONE 2025. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12419595/

14. **Cross-Generation Feature Interaction Learning** — 2021. arXiv: https://arxiv.org/pdf/2109.02809

---

**Nota final sobre incertidumbre:** Varios números en la tabla SOTA provienen de tablas de comparación dentro de papers secundarios, no siempre de los papers originales. El número 83.85% / 91.75% aparece atribuido a múltiples métodos distintos por distintas fuentes — probablemente corresponde al "Unified Approach" (Dahan & Keller 2020 / [arXiv 2009.05871](https://arxiv.org/abs/2009.05871)) y fue luego igualado por GKR y replicado en otros trabajos. Para la Tarea #6 del proyecto, la evaluación honesta requiere: (1) usar splits oficiales del benchmark, (2) reportar KinFaceW-I como métrica primaria (no KinFaceW-II), (3) reportar AUC + accuracy por relación, y (4) comparar contra el baseline coseno crudo propio como referencia de ground truth del stack.
