# Tarea #29 — CCMTL-lite full-face antes de regiones

## Motivacion

Los documentos `data/input/docs/notebookLM_SoTA_I.pdf` y
`data/input/docs/notebookLM_SoTA_II.pdf` apuntan a una direccion razonable del
SoTA: los tipos de parentesco FS/FD/MS/MD no deberian tratarse como problemas
aislados. CCMTL explota correlaciones entre relaciones; Forest Neural Network y
variantes con atencion intentan modelar estructura conjunta.

El MLP completo de Tarea #6 no mejoro el baseline. Resultado resumido sobre
KinFaceW-II: baseline coseno ALL AUC 0.727 vs MLP ALL AUC 0.710; por relacion,
el MLP quedo por debajo del coseno en FS, FD y MS, y solo se acerco en MD. La
lectura mas conservadora es que una cabeza con 1026 features y pocos pares tiene
demasiada capacidad para este dataset.

Antes de pasar a regiones, queda una alternativa full-face mas barata y menos
riesgosa: modelos de baja capacidad sobre scores globales, compartiendo
informacion entre relaciones.

## Hipotesis

Un modelo pequeño con regularizacion puede mejorar o igualar al umbral de coseno
si captura:

- offset distinto por relacion (FS/MD/FD/MS tienen distribuciones diferentes);
- pendiente comun de score full-face;
- interacciones suaves score x relacion, sin aprender una frontera de alta
  dimension sobre el embedding completo.

Si esto no mejora el baseline, refuerza que el siguiente salto debe venir de
features nuevas (regiones, geometria, calidad/edad), no de otro clasificador
full-face.

## Modelos evaluados

El script `scripts/evaluate_fullface_multitask.py` compara:

- `baseline_youden_cosine`: umbral de coseno por Youden J en train folds.
- `baseline_youden_euclidean`: umbral de euclidea L2-normalizada.
- `logreg_global_cosine`: regresion logistica compartida sobre coseno.
- `logreg_global_cosine_euclidean`: compartida sobre coseno + euclidea.
- `logreg_shared_relation_offsets`: coseno + euclidea + offsets por relacion.
- `logreg_shared_relation_slopes`: offsets + interacciones score x relacion.
- `logreg_per_relation_cosine_euclidean`: logistica independiente por relacion.

Todos usan los folds oficiales de KinFaceW, sin fuga train/test.

## Notas sobre los documentos SoTA

Los PDFs son utiles como mapa de investigacion, pero no como especificacion
tecnica cerrada. Puntos a conservar:

- KinFaceW-I debe seguir siendo evaluacion primaria; KinFaceW-II queda como
  referencia secundaria por sesgo same-photo.
- CCMTL es una direccion compatible con nuestro problema porque ataca el
  aislamiento entre tipos de parentesco.
- FNN/ViT/GAN de edad son interesantes, pero demasiado costosos para el siguiente
  paso local con pocos datos.
- La afirmacion de que regiones "superan" al score global todavia no esta
  demostrada en este repo. Las regiones son el siguiente bloque plausible, no un
  resultado ya validado.

## Criterio de decision

- Si algun modelo full-face de baja capacidad supera claramente al baseline en
  KinFaceW-I, se puede considerar exportar un calibrador simple para la web.
- Si empata o pierde, se cierra #29 como evidencia negativa y se priorizan #4/#5/#7
  antes de #11.
- KinFaceW-II no decide la direccion por si solo.

## Resultado KinFaceW-I

Corrida completa:

```bash
TEST_LOG_FILE=_meta/TAREA29_fullface_multitask_resources.log \
MPLCONFIGDIR=/tmp/mpl-codex \
./scripts/test-monitored.sh /home/diego/miniconda3/envs/face-sim/bin/python \
  scripts/evaluate_fullface_multitask.py \
  --dataset KinFaceW-I \
  --batch-size 120 \
  --cool-threshold 88 \
  --out data/output/calibration/KinFaceW-I_fullface_multitask.json
```

Resumen:

| Modelo | Acc 5-fold | AUC / OOF |
|---|---:|---:|
| baseline_youden_cosine | 0.666 | 0.727 |
| baseline_youden_euclidean | 0.666 | 0.727 |
| logreg_global_cosine | 0.659 | 0.726 |
| logreg_global_cosine_euclidean | 0.657 | 0.726 |
| logreg_shared_relation_offsets | 0.660 | 0.734 |
| logreg_shared_relation_slopes | 0.659 | 0.736 |

Modelos independientes por relacion:

| Relacion | Acc 5-fold | AUC OOF |
|---|---:|---:|
| FS | 0.715 | 0.809 |
| MD | 0.670 | 0.746 |
| FD | 0.639 | 0.672 |
| MS | 0.604 | 0.676 |

Lectura: los modelos compartidos con relacion mejoran muy poco el AUC global
(+0.007 a +0.009) pero bajan accuracy respecto al baseline. Los modelos por
relacion no superan los AUC del baseline por relacion ya observado. No hay mejora
clara suficiente para reemplazar el calibrador full-face actual.

Recursos: `_meta/TAREA29_fullface_multitask.log` y
`_meta/TAREA29_fullface_multitask_resources.log`. La corrida completa tuvo
`cpu_avg=39%`, `cpu_max=61%`, `temp_avg=81C`, `temp_max=95C`; para repetirla
conviene bajar `--batch-size` a 60-80 o endurecer el umbral termico.

Decision: cerrar #29 como resultado marginal/no accionable. Siguiente trabajo
recomendado: features nuevas y validacion regional/geometrica (#4/#5/#7 antes de
#11), no mas clasificadores full-face sobre los mismos scores.
