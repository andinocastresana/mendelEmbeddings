# =========================================
# ID: PHYLOFACE_LANDMARKS_001
# VERSION: v1.0
# =========================================
# Subpaquete: phyloface.landmarks
#
# Reúne todo lo relacionado con la obtención de landmarks faciales densos
# (puntos clave del rostro). Hoy contiene un único backend basado en
# MediaPipe Face Mesh (468 puntos 3D, en este proyecto usados solo en 2D).
#
# Los landmarks densos son la fuente primaria para definir las regiones
# anatómicas (ojos, cejas, nariz, boca, contorno, frente, mentón) que
# después usan los módulos `regions/` y `comparator_regional.py`.
#
# Re-exporta las funciones públicas del backend MediaPipe para que el
# código cliente pueda hacer:
#     from phyloface.landmarks import init_face_mesh, add_dense_landmarks_to_pair
# en vez de tener que conocer el archivo concreto.

from phyloface.landmarks.mediapipe_mesh import (
    init_face_mesh,
    get_face_mesh_landmarks,
    add_dense_landmarks_to_pair,
)

# Lista explícita de símbolos públicos del subpaquete.
# Sirve de contrato: lo que está acá es lo que se promete que no se va a romper.
__all__ = [
    "init_face_mesh",
    "get_face_mesh_landmarks",
    "add_dense_landmarks_to_pair",
]
