# Candidatos de agrupamiento y representación

Origen: `posibilidades representacion.md`, adaptado al caso de selecciones/jugadores donde una entidad puede tener subgrupos internos que se parecen más a subgrupos de otras entidades que a su propia selección.

## Objetivo

Probar métodos que no fuercen una única estructura plana de puntos. Necesitamos conservar:

- similitud local entre jugadores;
- pertenencia a selección/equipo;
- subgrupos latentes que cruzan selecciones;
- lectura global útil para vitrina.

## Candidatos para probar primero

| Candidato | Qué captura | Estado local | Prueba sugerida |
|---|---|---:|---|
| Grafo kNN + comunidades | Subgrupos cruzados por vecinos externos, sin depender de 2D | Ya viable con `networkx` | Construir kNN jugador-jugador por coseno, correr greedy modularity/Louvain si se instala, resumir composición por selección |
| Clustering aglomerativo | Jerarquía global y dendrogramas | Viable con `sklearn` en `face-sim` | Probar `average`/`complete` sobre distancia coseno; cortar en K y medir mezcla por selección |
| Spectral clustering | Comunidades sobre grafo de afinidad | Viable con `sklearn` en `face-sim` | Usar matriz kNN/coseno y comparar clusters contra kNN attraction |
| MDS / Isomap / Spectral embedding | Representación 2D con más estructura global que t-SNE | Ya existe script para FIFA | Reusar `build_vitrina_embedding_projection.py`, comparar contra radios intra y kNN |
| Grafo multiplex | Combina similitud facial + pertenencia a selección | Implementación propia corta | `W = alpha * W_sim + beta * W_team`; barrer beta y ver cuándo se conserva entidad sin perder subgrupos |
| Embedding bipartito selección-jugador | Modela explícitamente entidades y elementos | Viable con `networkx`, mejor con node2vec | Grafo heterogéneo con aristas jugador-jugador y jugador-selección; proyectar embeddings de nodos |

## Candidatos que requieren instalar paquete

| Candidato | Motivo para probar | Paquete probable |
|---|---|---|
| PaCMAP | Buen balance local/intermedio/global; primera alternativa a UMAP/t-SNE | `pacmap` |
| TriMAP | Preserva relaciones de largo alcance por tríos | `trimap` |
| PHATE | Útil para jerarquías, continuidad y ramas | `phate` |
| HDBSCAN | Clusters de densidad + ruido; bueno para subgrupos no esféricos | `hdbscan` |
| Leiden/Louvain | Comunidades robustas en grafos kNN grandes | `leidenalg`/`igraph` o `python-louvain` |
| Node2Vec/metapath2vec | Embeddings de grafo homogéneo/heterogéneo | `node2vec` o implementación PyG si se escala |

## Orden pragmático

1. kNN jugador-jugador + comunidades con `networkx`, porque no requiere instalar nada y conecta directo con la solapa Atracción kNN.
2. Clustering aglomerativo sobre coseno para tener una línea base jerárquica interpretable.
3. Spectral clustering sobre el mismo grafo kNN para comparar comunidades.
4. Grafo multiplex con barrido `alpha/beta` para inyectar pertenencia a selección sin esconder subgrupos cruzados.
5. Instalar y probar PaCMAP/PHATE si los pasos anteriores no dan una vista estable.

## Métricas de comparación

- Tamaño y estabilidad de clusters.
- Entropía por selección dentro de cada cluster.
- Entropía de clusters dentro de cada selección.
- Porcentaje de vecinos externos por selección.
- Separación intra vs inter en el embedding 2D.
- Concordancia con señales geo/culturales ya calculadas: distancia, idioma, colonialidad.
