from insightface.app import FaceAnalysis
import cv2

app = FaceAnalysis(name="buffalo_l")
#app.prepare(ctx_id=0, det_size=(640, 640))#para GPU
app.prepare(ctx_id=-1, det_size=(640, 640))#para GPU

#img = cv2.imread("data/input/img/brunoRio.jpg")
#img = cv2.imread("data/input/img/DiegoSofiaY3Mas.jpg")
#img = cv2.imread("data/input/img/BrunoFondoBlanco.jpeg")
img = cv2.imread("data/input/img/Señores_o_gente.jpg")


faces = app.get(img)

print(f"Caras detectadas: {len(faces)}")

if faces:
    face = faces[0]
    print("Bounding box:", face.bbox)
    print("Embedding shape:", face.embedding.shape)