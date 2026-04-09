
# -----------------------------------------
# FILE: phyloface/core/detector.py
# -----------------------------------------
from pathlib import Path
import hashlib
import cv2
import numpy as np

from phyloface.core.config import FACE_SIZE


class FaceDetector:
    """
    Encapsula detección de rostros + extracción de datos por imagen.
    Actualmente implementa backend 'insightface'.
    """

    def __init__(
        self,
        library_name: str = "insightface",
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = -1,
        max_faces: int = 11,
    ):
        self.library_name = library_name
        self.model_name = model_name
        self.det_size = det_size
        self.ctx_id = ctx_id
        self.max_faces = max_faces

        if self.library_name == "insightface":
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(name=model_name)
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        else:
            raise ValueError(
                f"Backend no soportado: {library_name}. "
                f"Actualmente solo está implementado 'insightface'."
            )

    @staticmethod
    def compute_file_md5(image_path: Path, chunk_size: int = 1024 * 1024) -> str:
        """
        Hash MD5 del archivo de imagen.
        """
        md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                md5.update(chunk)
        return md5.hexdigest()

    def extract_faces_payload(
        self,
        image_path: Path,
        face_size: tuple[int, int] = FACE_SIZE,
    ) -> dict:
        """
        Devuelve todo lo calculado para una imagen, listo para cachear.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")

        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError(f"No se detectaron caras en: {image_path}")

        faces = sorted(faces, key=lambda f: f.bbox[0])[: self.max_faces]

        h, w = img.shape[:2]

        indices = []
        bboxes = []
        det_scores = []
        embeddings = []
        crops = []
        kps_list = []
        lmk68_list = []
        gender_list = []
        age_list = []

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, face_size)

            indices.append(i)
            bboxes.append([x1, y1, x2, y2])
            det_scores.append(float(getattr(face, "det_score", np.nan)))
            embeddings.append(np.asarray(face.embedding, dtype=np.float32))
            crops.append(crop_resized)

            if hasattr(face, "kps") and face.kps is not None:
                kps_list.append(np.asarray(face.kps, dtype=np.float32))
            else:
                kps_list.append(np.full((5, 2), np.nan, dtype=np.float32))

            if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
                lmk68_list.append(np.asarray(face.landmark_3d_68, dtype=np.float32))
            else:
                lmk68_list.append(np.full((68, 3), np.nan, dtype=np.float32))

            gender_list.append(float(getattr(face, "gender", np.nan)))
            age_list.append(float(getattr(face, "age", np.nan)))

        if len(indices) == 0:
            raise ValueError(
                f"Se detectaron caras pero no se pudieron recortar correctamente en: {image_path}"
            )

        file_stat = image_path.stat()

        payload = {
            "image_path": str(image_path.resolve()),
            "image_name": image_path.name,
            "image_stem": image_path.stem,
            "image_suffix": image_path.suffix.lower(),
            "image_size_bytes": int(file_stat.st_size),
            "image_mtime": float(file_stat.st_mtime),
            "image_md5": self.compute_file_md5(image_path),
            "image_width": int(w),
            "image_height": int(h),
            "n_faces": int(len(indices)),
            "indices": np.asarray(indices, dtype=np.int32),
            "bboxes": np.asarray(bboxes, dtype=np.int32),
            "det_scores": np.asarray(det_scores, dtype=np.float32),
            "embeddings": np.asarray(embeddings, dtype=np.float32),
            "crops": np.asarray(crops, dtype=np.uint8),
            "kps": np.asarray(kps_list, dtype=np.float32),
            "landmark_3d_68": np.asarray(lmk68_list, dtype=np.float32),
            "gender": np.asarray(gender_list, dtype=np.float32),
            "age": np.asarray(age_list, dtype=np.float32),
        }

        return payload

