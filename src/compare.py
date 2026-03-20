from insightface.app import FaceAnalysis
import cv2
import numpy as np


class FaceComparator:
    def __init__(self, det_size=(640, 640), ctx_id=-1):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def get_largest_face(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")

        faces = self.app.get(img)

        if not faces:
            raise ValueError(f"No se detectó ninguna cara en: {image_path}")

        # Elegimos la cara más grande por área
        largest_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        return largest_face

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        return float(np.dot(vec1, vec2))

    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.linalg.norm(vec1 - vec2))

    def compare(self, image1_path: str, image2_path: str):
        face1 = self.get_largest_face(image1_path)
        face2 = self.get_largest_face(image2_path)

        emb1 = face1.embedding
        emb2 = face2.embedding

        cosine = self.cosine_similarity(emb1, emb2)
        euclidean = self.euclidean_distance(emb1, emb2)

        return {
            "cosine_similarity": cosine,
            "euclidean_distance": euclidean
        }


if __name__ == "__main__":
    comparator = FaceComparator(ctx_id=-1)

    result = comparator.compare(
        "data/input/img/BrunoFondoBlanco.jpeg",
        #"data/input/img/Señores_o_gente.jpg"
        "data/input/img/brunoRio.jpg"
    )

    print("Resultado de comparación:")
    print(result)