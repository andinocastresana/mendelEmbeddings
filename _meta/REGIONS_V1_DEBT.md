# Regions v1 - deuda historica

Fecha: 2026-05-25

## Resumen

En el proyecto no queda una API versionada llamada literalmente `regions_v1`.
La deuda historica es conceptual: antes de la migracion de la Tarea #1, la
segmentacion regional vivia dentro del notebook/archivo experimental como bloques
sueltos de funciones y constantes. No habia un contrato canonico independiente
para nombres, landmarks, poligonos, version de regiones o semantica de cache.

La migracion dejo ese estado operativo como `regions_v2`:

- `extract_regions_v2`: bbox rectangular + `crop_rgb`.
- `extract_regions_v2_masked`: bbox + mascara + `crop_masked_rgb`.
- `selected_pair["regions_v2"]`: clave historica compartida por ambos paths.

## Que fue "regions v1"

Para efectos del proyecto, "regions v1" se refiere al periodo experimental previo
a `regions_v2`, no a un modulo estable:

- regiones definidas de forma local en el notebook o en
  `src/phyloface_experimental_functions.py`;
- nombres y orden repetidos a mano en extractores, comparadores y visualizadores;
- listas manuales de landmarks sin registry central;
- sin version explicita para cachear o comparar resultados;
- sin separacion clara entre contrato anatomico, estrategia de bbox, estrategia
  de mascara y metrica regional;
- metricas regionales basadas en crops de gris/z-score/coseno, utiles como
  baseline visual pero no equivalentes a embeddings faciales regionales.

## Por que se descarto como base

`regions v1` no era una superficie confiable para construir features nuevas:

- cualquier cambio de nombre o landmarks podia romper visualizaciones/cache sin
  quedar documentado;
- la diferencia entre regiones oficiales de MediaPipe y aproximaciones manuales
  no estaba codificada como contrato;
- frente y menton tenian logica especial, pero esa excepcion no estaba disponible
  para consumidores externos;
- no habia forma robusta de versionar futuros cambios (`v2`, `v2-masked`,
  `v3-periocular`, etc.);
- la comparacion regional historica mezclaba "region visual" con "evidencia de
  parentesco", cuando aun falta validar embeddings regionales reales.

## Decision actual

La Tarea #2 formaliza el contrato en `src/phyloface/regions/canonical.py`.
Ese modulo define `CANONICAL_REGIONS_VERSION = "regions-v2.0"` y una lista
estable de 12 regiones. Por ahora no cambia la extraccion; documenta y congela el
comportamiento vigente para que las tareas siguientes puedan migrar hacia el
registry sin romper compatibilidad.

## Episodios rescatados

El protocolo del KG (`IA/memories/_meta/protocol.md`) aclara que los episodios
del proyecto no viven en `IA/memories/mendelEmbeddings/`, sino en
`IA/memories/_meta/episodes/` con frontmatter `project: mendelEmbeddings`. Se
leyeron e incorporaron los episodios relevantes:

- `2026-05-19-verify-real-code-state-before-proposing-pendings_diego-lenovo-debian.md`:
  el codigo experimental ya tenia MediaPipe Face Mesh, regiones v2, regiones
  enmascaradas, comparacion regional y overlays. Esto confirma que la tarea era
  migrar/formalizar comportamiento existente, no implementar regiones desde cero.
- `2026-05-20-migrating-between-impls-verify-equivalence_diego-lenovo-debian.md`:
  en migraciones entre implementaciones homonimas no hay que asumir equivalencia;
  el nuevo registry debe ser contrato explicito y verificable.
- `2026-05-21-verify-convention-visually-before-numerical-agreement_diego-lenovo-debian.md`:
  las convenciones de MediaPipe Face Mesh deben verificarse visualmente antes de
  confiar en agreement numerico; futuras modificaciones regionales deberian pasar
  por overlays o inspeccion equivalente.

No se escribio un episodio nuevo: el protocolo indica que la captura de episodios
requiere validacion explicita del usuario.
