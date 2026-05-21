# Spike #004 — corridas

Append-only. Cada sección registra UNA corrida del script Python
`scripts/verify_detection_web_parity.py`, que es el que mantiene
el fixture multi-imagen del spike e2e.

Solo se registran datos del lado Python (qué imágenes hay en el
set, bbox/det_score detectados). Las métricas del cliente JS
(cosine vs reference) se capturan por separado en el componente
`SpikeDetection.tsx` y se exportan via su botón 'descargar JSON'
si querés persistirlas a mano.

---

## 2026-05-21T09:44:51.897738Z
- **Set dir**: `data/input/img/spike_e2e_set`
- **Imágenes en set**: 1  (nuevas: 1, reusadas via dedup: 0)
- **Set hash agregado**: `a513877fb06f95a0...`
- **Casos**:

| # | hash (16ch) | source filename | det_score | bbox (x1,y1,x2,y2) |
|---|-------------|------------------|-----------|---------------------|
| 1 | `f2d1c10740aefcac` | BrunoFondoBlanco.jpeg | 0.851 | (186, 146, 625, 860) |

---

## 2026-05-21T09:50:14.866699Z
- **Set dir**: `data/input/img/spike_e2e_set`
- **Imágenes en set**: 4  (nuevas: 3, reusadas via dedup: 1)
- **Set hash agregado**: `1ad59aa2f82e4168...`
- **Casos**:

| # | hash (16ch) | source filename | det_score | bbox (x1,y1,x2,y2) |
|---|-------------|------------------|-----------|---------------------|
| 1 | `f2d1c10740aefcac` | BrunoFondoBlanco.jpeg | 0.851 | (186, 146, 625, 860) |
| 2 | `d21698c7752b3324` | IMG-20191018-WA0000.jpg | 0.799 | (147, 176, 337, 487) |
| 3 | `469b2b2841d19de2` | T015PLX40B0-U034J4RTWUT-376597ef9d0d-512.png | 0.913 | (139, 42, 375, 475) |
| 4 | `2f1090ffe1469b91` | mateoFotoTarjetaTransporte.jpeg | 0.855 | (148, 4, 555, 683) |

---
