# Vitrina Mundial 2026 — plan de release

Fecha: 2026-05-27

## Objetivo

Preparar una vitrina web para liberar cuando FIFA confirme las listas finales del
Mundial 2026. El enfoque no es parentesco familiar, sino exploracion de similitud
facial computacional entre jugadores y selecciones del torneo.

Las listas finales oficiales deben tomarse desde FIFA cuando esten confirmadas
(FIFA indica publicacion/confirmacion el 2026-06-02, tras envio de selecciones).
Cobertura objetivo completa: `48 selecciones * 26 jugadores = 1248 jugadores`.

## Costo de calculo esperado

Si ya existen embeddings por jugador, el costo de comparar todo contra todo es
bajo:

- Jugadores completos: `1248`
- Pares jugador-jugador unicos: `1248 * 1247 / 2 = 778128`
- Cruces seleccion-seleccion: `C(48, 2) = 1128`
- Comparaciones jugador-jugador entre selecciones: `1128 * 26 * 26 = 762528`
- Matriz completa `1248 x 1248 float32`: aprox. 6.2 MB

Lo caro no son los cosenos, sino deteccion/embedding/QC y cualquier metodo que
re-embeddee variaciones, especialmente occlusion densa. Recomendacion: pipeline
offline que genere embeddings, landmarks/features regionales, QC, matrices y JSON
estatico para el cliente.

## Funcionalidades candidatas

### MVP vitrina

1. **Matriz de similitud entre selecciones**
   - Grid `48 x 48`.
   - Celda = similitud promedio entre jugadores de dos selecciones.
   - Click abre detalle seleccion A vs seleccion B.

2. **Detalle seleccion vs seleccion**
   - Heatmap `26 x 26`.
   - Top pares mas parecidos.
   - Top pares menos parecidos.
   - Filtros por posicion.
   - Estadisticas: promedio, mediana, maximos y distribucion.

3. **Ranking global de parecidos cruzados**
   - Top N pares mas parecidos del Mundial.
   - Toggle para excluir/incluir pares de la misma seleccion.
   - Tarjetas con headshots, nombres, selecciones y score.

4. **Perfil simple de jugador**
   - Headshot, seleccion, posicion.
   - "Sus dobles del Mundial": top N jugadores mas parecidos.

### Release ampliado

5. **Perfil de seleccion**
   - Grid de los 26 jugadores.
   - Selecciones mas parecidas por promedio.
   - Jugadores externos mas parecidos.
   - Distribucion interna de similitud.

6. **Boxplots intra-seleccion**
   - Distribucion de similitud entre jugadores de una misma seleccion.
   - Comparacion de homogeneidad/diversidad visual entre selecciones.
   - Relacionado con tarea #14.

7. **Dendrograma / arbol de selecciones**
   - Agrupamiento por distancia/similitud promedio entre selecciones.
   - Relacionado con tarea #15.

8. **Grafo de selecciones**
   - Nodos = selecciones.
   - Aristas = similitud alta.
   - Slider de umbral para explorar clusters.

9. **Comparador libre dentro de vitrina**
   - Elegir cualquier jugador A y jugador B.
   - Mostrar score global y, si existe, desglose regional.
   - Sin subir fotos: todo con jugadores precargados.

10. **Vista de cobertura/QC**
    - Cobertura por seleccion.
    - Fotos aceptadas/rechazadas.
    - Warnings de pose/calidad/fuente.
    - Puede empezar como vista interna/debug.

## Lenguaje y riesgos

Evitar lenguaje de etnicidad, raza, origen o inferencias identitarias. La vitrina
debe hablar de "similitud facial computacional entre fotos del torneo" y explicar
limitaciones: pose, iluminacion, edad, barba, calidad de imagen, expresion,
recorte, sesgo del modelo y fuente de imagen.

## Estado actual de datos

Hay un extractor nuevo para retratos estandarizados:

```bash
python scripts/build_transfermarkt_headshot_manifest.py \
  --input data/output/teams/manifest_wikimedia_northamerica2026_all_max8_downloaded.json \
  --output data/output/teams/manifest_transfermarkt_northamerica2026_headshots.json \
  --image-dir data/input/img/teams_players/northamerica2026_transfermarkt \
  --download-images
```

Ultimo resultado local: `259/271` retratos descargados desde Transfermarkt, todos
marcados como `UNREVIEWED_NONPUBLIC_RESEARCH`. Sirven para comparacion/QC local,
no para publicacion hasta resolver permisos/licencia.

Siguiente paso tecnico al retomar: QC facial sobre esos 259 retratos y manifiesto
`accepted/rejected`; despues embeddings y matrices piloto.

## Update 2026-06-26 — QC y payload piloto

Se agregaron dos scripts para cerrar el primer circuito offline:

```bash
PYTHONPATH=src python scripts/qc_transfermarkt_headshots.py \
  --output data/output/teams/manifest_transfermarkt_northamerica2026_headshots_qc.json \
  --include-embeddings

python scripts/build_vitrina_similarity_payload.py \
  --output data/output/teams/vitrina_transfermarkt_northamerica2026_similarity_pilot.json
```

Resultado local del QC Transfermarkt:

- Registros evaluados: `271`.
- Aceptados con embedding: `198`.
- Rechazados: `73`.
- Razones de rechazo: `missing_local_image=12`, `no_face_detected=58`,
  `image_read_error=2`, `not_exactly_one_face=1`.
- Archivos ilegibles detectados:
  - `data/input/img/teams_players/northamerica2026_transfermarkt/iraq/fahad-talib.png`
  - `data/input/img/teams_players/northamerica2026_transfermarkt/ivory-coast/ghislain-konan.png`
- Caso con mas de una cara: Belgium / Maxim De Cuyper.

Resultado local del payload piloto:

- `198` jugadores aceptados.
- `40` selecciones con al menos un jugador aceptado.
- JSON estatico de aprox. `1.3 MB`, con matriz jugador-jugador, matriz
  seleccion-seleccion, stats intra-seleccion y top pares.

Nota: el QC corrio con `face-sim` en CPU (`buffalo_l`, `det_size=640`,
`min_det_score=0.50`). `det_size=1024` empeoro el smoke inicial, por eso no se
uso como default.
