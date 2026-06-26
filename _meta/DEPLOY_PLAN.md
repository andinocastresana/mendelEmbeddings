# DEPLOY_PLAN — Publicación estática (vitrina ahora + App primaria futura)

> Estado: **plan de infraestructura**, no ejecutado. Stack consensuado con el
> usuario: **Cloudflare** (Registrar + Pages + R2 cuando toque el modelo).
> Sitio 100% estático, sin backend. Un VPS solo haría falta para la Tarea #32
> ("compartir vía servidor", con upload de imágenes) — **no es este caso**.
>
> La **vista web de vitrina la construye codex** (es su track). Este doc define
> *dónde y cómo se publica*, no la UI. Ver coordinación en `AGENTS_HANDOFF.md`.

---

## §0 — Principios y guardrail de licencias (va primero)

1. **Sitio estático, sin backend.** Vite + React compila a `dist/` y se sirve como
   archivos estáticos. No hay servidor de aplicación.
2. **Las imágenes nunca se suben ni se publican.** Se usan solo localmente para
   inferencia; lo que viaja a la web son **datos precomputados** (scores, matrices).
3. **Guardrail duro de licencias** — la vista de vitrina **NO debe renderizar las
   fotos** de los jugadores. Los retratos Transfermarkt están marcados
   `UNREVIEWED_NONPUBLIC_RESEARCH`: mostrar la foto = redistribuir sin licencia.
   - **Permitido**: nombre, selección, posición, scores de similitud, matrices,
     heatmaps, grafos, rankings, boxplots, dendrogramas.
   - **Para nodos/avatares**: usar un placeholder genérico o las iniciales, nunca la
     foto real.
   - Si en el futuro se quiere mostrar caras, primero sustituir por imágenes con
     licencia trazable (vía Wikimedia/Commons, ya empezado en
     `_meta/VITRINA_EQUIPOS_FUENTES.md`) y registrar atribución.
4. **Saneo del payload antes de publicar.** El JSON crudo
   (`data/output/teams/vitrina_transfermarkt_northamerica2026_similarity_pilot.json`)
   contiene un campo `local_image` con **rutas a archivos locales**. El JSON que se
   publica debe **excluir `local_image`** (y cualquier ruta a archivo) → derivar un
   `vitrina_payload.public.json`. Esto se hace en el build; **no se publica el JSON
   crudo**.

---

## §1 — Tier 1: Vitrina (deploy ahora)

### Qué se publica
- La SPA compilada (`client/dist/`) + el **payload saneado** de vitrina.
- El payload crudo (1.3 MB, schema `phyloface-vitrina-similarity-payload-v0.1`) trae:
  198 jugadores, 40 selecciones, matriz jugador-jugador 198×198, matriz
  selección-selección 40×40, `top_pairs_all` / `top_pairs_cross_team` (100 c/u),
  `intra_team_stats`. **No contiene imágenes ni base64**, solo nombres + scores +
  métricas QC + el campo `local_image` a sanear.

### Dónde vive el payload en el sitio
- El payload está **gitignored** (`.gitignore` → `data/`). No entra al repo.
- Para el deploy, **copiar el payload saneado a `client/public/vitrina/vitrina_payload.public.json`**
  antes de `npm run build`, de modo que se sirva como asset estático junto a la SPA.
- Comando de regeneración + saneo (a documentar/automatizar cuando exista la vista):
  1. Regenerar el payload crudo offline con
     `scripts/build_vitrina_similarity_payload.py` (env `face-sim`).
  2. Sanearlo (drop de `local_image`) → `vitrina_payload.public.json`. Idealmente un
     paso del propio `build_vitrina_similarity_payload.py` con un flag tipo
     `--public-output`, para que el saneo no quede en un script suelto. **Decisión de
     implementación de codex** al construir la vista.
  3. Copiar a `client/public/vitrina/`.

### Configuración de Cloudflare Pages
- **Conectar el repo** de GitHub al proyecto de Pages.
- **Root directory**: `client`
- **Build command**: `npm run build`
- **Build output directory**: `dist` (relativo a `client` → `client/dist`)
- **Node version**: 20 (coincide con `.nvmrc`; setear `NODE_VERSION=20` en Pages si
  hace falta).
- **Headers**: **ninguno especial.** La vitrina solo renderiza datos precomputados;
  no usa onnxruntime-web → no necesita COOP/COEP ni el modelo.

### Separación motor ↔ vitrina (importante para el tamaño)
- Si la vista de vitrina **no importa** `onnxruntime-web` ni el modelo, el bundle del
  tier 1 es **trivial**: JSON 1.3 MB + JS de visualizaciones. Sin los 167 MiB del
  modelo, sin el WASM pesado de ORT.
- **Riesgo a vigilar**: hoy la app es un solo bundle (todas las tabs en `App.tsx`). Si
  la vitrina convive en la misma SPA que la App primaria, Vite podría arrastrar
  `onnxruntime-web` al bundle común. Para el tier 1 puro, conviene **code-splitting**
  (lazy-load de las tabs que usan el motor) o un **entry/proyecto de Pages separado**
  solo-vitrina. Ver §3.

### Dominio
- **Registrar**: Cloudflare Registrar (at-cost) o Porkbun. Si el dominio se registra en
  Cloudflare, el DNS ya queda en CF y el custom domain de Pages es un paso de UI.
- **Custom domain en Pages**: agregar el dominio/subdominio en el proyecto de Pages;
  CF crea el registro CNAME automáticamente si el DNS está en CF (si no, agregar CNAME
  al `*.pages.dev` manualmente).

---

## §2 — Tier 2: App primaria (deploy futuro)

La App primaria corre inferencia ONNX **en el browser** → arrastra dos archivos que
**exceden el límite de 25 MiB/archivo de Cloudflare Pages**:

### a) Modelo `w600k_r50.onnx` (167 MiB) → Cloudflare R2
- Hoy se carga por ruta relativa hardcodeada `/models/w600k_r50.onnx`
  (`client/src/lib/pipeline.ts:220`, también `SpikeOnnx.tsx:230`).
- **No entra en Pages** (167 MiB ≫ 25 MiB). Servirlo desde **R2** (object storage de
  CF; egress gratis hacia Cloudflare, free tier 10 GB) y que el browser lo `fetch`ee
  desde la URL pública de R2.
- **Cambio puntual de código (futuro, no ahora)**: parametrizar la URL del modelo vía
  variable de entorno de Vite:
  ```ts
  // pipeline.ts
  const DEFAULT_MODEL_URL =
    import.meta.env.VITE_MODEL_URL ?? '/models/w600k_r50.onnx';
  ```
  con fallback a la ruta local para dev. En Pages se setea `VITE_MODEL_URL` apuntando a
  R2.
- Costo: ~$0 (free tier).

### b) WASM de ORT (`ort-wasm-simd-threaded.jsep-*.wasm`, 26 MiB)
- También roza/supera 25 MiB. Dos salidas a evaluar:
  1. **Servir los `.wasm` de onnxruntime-web desde R2/CDN** vía
     `ort.env.wasm.wasmPaths = '<url-base-r2>/'`.
  2. Confirmar si Pages acepta el archivo (está justo en el borde de 25 MiB; puede
     fallar). Si entra, no hace falta R2 para el WASM.
- Documentar ambas; decidir al implementar el tier 2.

### c) CORS / CORP (solo si se activa cross-origin isolation)
- ORT hoy usa `['webgpu','wasm']` **sin multi-thread / sin SharedArrayBuffer** → **no
  requiere COOP/COEP**. WebGPU no lo necesita.
- Si en el futuro se quiere WASM **multi-thread** (más rápido), hace falta
  cross-origin isolation: headers `COOP/COEP` en Pages (vía archivo `_headers`) **y**
  que R2 mande `Cross-Origin-Resource-Policy: cross-origin` (o `same-site`) en el
  modelo y los `.wasm`. Solo entonces.

---

## §3 — Un proyecto de Pages vs. dos (decisión a tomar)

| Opción | Cómo | Cuándo conviene |
|--------|------|-----------------|
| **Un proyecto** (SPA completa) | La vitrina es una tab más dentro de la app existente; un solo dominio. | Si se quiere todo bajo una URL y no molesta que el deploy del tier 1 dependa del tier 2 (cuidar code-splitting para no arrastrar el motor a la vitrina). |
| **Dos proyectos** (subdominios) | `vitrina.<dominio>` solo-vitrina (sin motor) y `app.<dominio>` (App primaria + R2). Builds y visibilidad independientes. | Si se quiere publicar la vitrina **ya** sin esperar a resolver el modelo/R2 de la app, y mantener URLs separadas. **Recomendado para arrancar.** |

**Decisión señalada, pendiente del usuario.** Recomendación por defecto: **dos
proyectos** — destraba publicar la vitrina ahora sin acoplarla al tier 2.

---

## §4 — Checklist tier 1 (vitrina)

Precondición: **existe la vista web de vitrina** (la construye codex; ver handoff).

1. [ ] Registrar dominio (Cloudflare Registrar o Porkbun).
2. [ ] Si el registrar no es CF, mover el DNS a Cloudflare (o agregar CNAME al deploy).
3. [ ] Generar payload saneado `vitrina_payload.public.json` (sin `local_image`) y
       copiarlo a `client/public/vitrina/`.
4. [ ] Verificar que la vista de vitrina **no importa** `onnxruntime-web` (o que está
       lazy-split) → el bundle del tier 1 no arrastra el modelo/WASM.
5. [ ] Crear proyecto en Cloudflare Pages, conectar el repo de GitHub.
6. [ ] Build settings: root `client`, build `npm run build`, output `dist`,
       `NODE_VERSION=20`.
7. [ ] Primer deploy a `*.pages.dev`; verificar visualizaciones + que **no se ven
       fotos** de jugadores (guardrail §0).
8. [ ] Conectar custom domain / subdominio.
9. [ ] Verificación final: cargar el sitio público, abrir la vitrina, confirmar
       heatmaps/grafos/rankings y ausencia de fotos.

---

## Apéndice — hechos del repo que aterrizan este plan

- Payload: `data/output/teams/vitrina_transfermarkt_northamerica2026_similarity_pilot.json`
  — 1.3 MB, schema `phyloface-vitrina-similarity-payload-v0.1`, **gitignored**,
  regenerable con `scripts/build_vitrina_similarity_payload.py`.
- Modelo: `client/public/models/w600k_r50.onnx` = **167 MiB**, ruta hardcodeada en
  `client/src/lib/pipeline.ts:220`.
- WASM ORT: `dist/assets/ort-wasm-simd-threaded.jsep-*.wasm` = **26 MiB**.
- Límite Cloudflare Pages: **25 MiB por archivo** (ambos lo superan → R2).
- ORT providers: `['webgpu','wasm']`, sin SharedArrayBuffer → **sin COOP/COEP** hoy.
- Vite: sin `base`, output `dist`, solo plugin react.
- Tabs: `client/src/App.tsx`, tab-state (`useState`), sin router.
- Licencias: retratos Transfermarkt = `UNREVIEWED_NONPUBLIC_RESEARCH`; tier de
  publicación previsto vía Wikimedia/Commons (`NEEDS_COMMONS_REVIEW`) en
  `_meta/VITRINA_EQUIPOS_FUENTES.md`.
