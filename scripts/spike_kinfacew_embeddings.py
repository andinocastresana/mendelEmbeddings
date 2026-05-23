# =========================================
# ID: PHYLOFACE_SPIKE_KINFACEW_001
# VERSION: v1.0
# =========================================
# Spike de de-risking para la Tarea #6 (calibración de umbrales).
#
# Pregunta que responde (antes de comprometernos al protocolo completo):
#   ¿El pipeline de embedding produce señal de parentesco utilizable sobre las
#   caras de KinFaceW, que son crops 64x64 YA alineados? Es decir: ¿los cosines
#   de pares EMPARENTADOS (padre↔hijo reales) quedan sistemáticamente más altos
#   que los de pares NO emparentados?
#
# Enfoque (resuelve la "advertencia 64px"): NO corremos detección+alineación
# Face Mesh sobre 64px (poco confiable). Tratamos cada imagen como cara ya
# alineada: la pasamos a RGB, resize a 112x112, y la mandamos directo a
# `extract_embedding_from_aligned` (buffalo_l / w600k_r50). Esto valida el
# camino "pre-aligned → embedding" que usará la Fase A.
#
# Métrica de separación: AUC rank-based = P(cos_pos > cos_neg) sobre todos los
# cruces, sin sklearn (Mann-Whitney). AUC≈0.5 → sin señal; >0.5 → señal.
#
# Pares (sin parsear los .mat — la convención de nombres alcanza para el spike):
#   - folder father-son, prefijo fs_xxx_{1,2}: _1 = padre, _2 = hijo.
#   - POSITIVOS: (fs_xxx_1, fs_xxx_2) por cada xxx.
#   - NEGATIVOS: (padre_i, hijo_j) con j != i (permutación), misma relación →
#     composición de género balanceada (negativo "justo").
#
# Lee las imágenes DESDE el zip (zipfile), sin descomprimir a disco.
#
# Ejecución (monitoreada, según regla de recursos):
#   ./scripts/test-monitored.sh python3 scripts/spike_kinfacew_embeddings.py

import sys
import zipfile
from pathlib import Path

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phyloface.core.pairs import init_face_app  # noqa: E402
from phyloface.core.embedder import get_recognition_model, extract_embedding_from_aligned  # noqa: E402
from phyloface.core.metrics import cosine_similarity  # noqa: E402

ZIP_PATH = PROJECT_ROOT / "data/input/datasets/KinFaceW-I.zip"

# folder -> (prefijo, etiqueta legible). Mapeo del ReadMe de KinFaceW-I.
RELATIONS = {
    "father-son": ("fs", "Father-Son (mismo género)"),
    "mother-dau": ("md", "Mother-Daughter (mismo género)"),
    "father-dau": ("fd", "Father-Daughter (cross-género)"),
    "mother-son": ("ms", "Mother-Son (cross-género)"),
}

MAX_PAIRS_PER_RELATION = 60   # cap para acotar runtime del spike (CPU)
INPUT_SIZE = 112
RNG = np.random.default_rng(42)


def decode_aligned_rgb(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    """Lee una imagen del zip y la deja como cara 'pre-alineada' 112x112 RGB."""
    buf = np.frombuffer(zf.read(name), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # imdecode -> BGR
    if bgr is None:
        raise ValueError(f"no pude decodificar {name}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return rgb


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(pos > neg) por conteo de cruces (empates = 0.5). Sin sklearn."""
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return wins / (len(pos) * len(neg))


def list_pairs(zf: zipfile.ZipFile, folder: str, prefix: str):
    """Devuelve lista de (padre_name, hijo_name) según convención de nombres."""
    base = f"KinFaceW-I/images/{folder}/"
    ids = set()
    for n in zf.namelist():
        if n.startswith(base) and n.endswith("_1.jpg"):
            stem = n[len(base):]  # ej. fs_001_1.jpg
            if stem.startswith(prefix + "_"):
                ids.add(stem[: -len("_1.jpg")])  # fs_001
    pairs = []
    for pid in sorted(ids):
        parent = f"{base}{pid}_1.jpg"
        child = f"{base}{pid}_2.jpg"
        if child in zf.namelist():
            pairs.append((parent, child))
    return pairs


def main():
    print(f"Spike KinFaceW #6 — zip: {ZIP_PATH.name}")
    if not ZIP_PATH.exists():
        print(f"FALTA el zip en {ZIP_PATH}")
        sys.exit(1)

    print("Inicializando buffalo_l (CPU)...")
    app = init_face_app(model_name="buffalo_l")
    rec_model = get_recognition_model(app)
    print("rec_model OK\n")

    emb_cache = {}

    def embed(zf, name):
        if name not in emb_cache:
            rgb = decode_aligned_rgb(zf, name)
            emb_cache[name] = extract_embedding_from_aligned(rec_model, rgb)
        return emb_cache[name]

    print(f"{'relación':<34} {'n':>4} {'cos_pos':>16} {'cos_neg':>16} {'AUC':>6}")
    print("-" * 80)

    all_pos, all_neg = [], []
    norm_min = np.inf

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for folder, (prefix, label) in RELATIONS.items():
            pairs = list_pairs(zf, folder, prefix)
            if len(pairs) > MAX_PAIRS_PER_RELATION:
                idx = RNG.choice(len(pairs), MAX_PAIRS_PER_RELATION, replace=False)
                pairs = [pairs[i] for i in sorted(idx)]
            if not pairs:
                print(f"{label:<34} {'0':>4}  (sin pares)")
                continue

            # Positivos: padre_i ↔ hijo_i.
            pos = []
            for parent, child in pairs:
                ep, ec = embed(zf, parent), embed(zf, child)
                norm_min = min(norm_min, np.linalg.norm(ep), np.linalg.norm(ec))
                pos.append(cosine_similarity(ep, ec))

            # Negativos: padre_i ↔ hijo_perm(i), perm sin punto fijo.
            n = len(pairs)
            perm = RNG.permutation(n)
            for k in range(n):
                if perm[k] == k:  # evitar self-match accidental
                    perm[k] = (perm[k] + 1) % n
            neg = []
            for k in range(n):
                ep = embed(zf, pairs[k][0])
                ec = embed(zf, pairs[perm[k]][1])
                neg.append(cosine_similarity(ep, ec))

            pos, neg = np.array(pos), np.array(neg)
            all_pos.extend(pos); all_neg.extend(neg)
            auc = auc_rank(pos, neg)
            print(f"{label:<34} {n:>4} "
                  f"{pos.mean():>7.3f}±{pos.std():.3f} "
                  f"{neg.mean():>7.3f}±{neg.std():.3f} "
                  f"{auc:>6.3f}")

    all_pos, all_neg = np.array(all_pos), np.array(all_neg)
    print("-" * 80)
    print(f"{'GLOBAL':<34} {len(all_pos):>4} "
          f"{all_pos.mean():>7.3f}±{all_pos.std():.3f} "
          f"{all_neg.mean():>7.3f}±{all_neg.std():.3f} "
          f"{auc_rank(all_pos, all_neg):>6.3f}")
    print(f"\nSanity: emb dim={emb_cache[next(iter(emb_cache))].shape[0]}, "
          f"norm_min={norm_min:.2f} (no debe ser ~0), imágenes={len(emb_cache)}")
    print("\nLectura: AUC≈0.5 sin señal · 0.6-0.7 señal débil · >0.7 señal clara. "
          "Cross-género suele < mismo-género.")


if __name__ == "__main__":
    main()
