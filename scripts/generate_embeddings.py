import os
import json
import numpy as np
import face_recognition

DATASET_DIR = "dataset/lfw-deepfunneled"
OUTPUT_DIR = "embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

embeddings = []
metadata = []

print("Scanning dataset...")

for person_name in os.listdir(DATASET_DIR):

    person_dir = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_dir):
        continue

    for image_name in os.listdir(person_dir):

        image_path = os.path.join(person_dir, image_name)

        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) == 0:
                continue

            encoding = face_recognition.face_encodings(image, face_locations)[0]

            embeddings.append(encoding)

            metadata.append({
                "person": person_name,
                "image": image_path
            })

        except Exception as e:
            print("Error:", image_path, e)

print("Images processed:", len(embeddings))
import os, json
import numpy as np
import face_recognition
import cv2

DATA_DIR = "../dataset"
OUT_DIR = "../embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

embs = []
meta = []  # list of {"person": name, "image": filename}

for person in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_dir): continue
    for fname in os.listdir(person_dir):
        path = os.path.join(person_dir, fname)
        img = cv2.imread(path)
        if img is None: continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        if len(boxes) == 0:
            print("no face:", path); continue
        # take the first face
        enc = face_recognition.face_encodings(rgb, boxes)[0]
        embs.append(enc)
        meta.append({"person": person, "image": path})

embs = np.array(embs).astype("float32")
np.save(os.path.join(OUT_DIR, "embeddings.npy"), embs)
with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f)
print("saved", embs.shape)
embeddings = np.array(embeddings).astype("float32")

np.save("embeddings/embeddings.npy", embeddings)

with open("embeddings/metadata.json", "w") as f:
    json.dump(metadata, f)

print("Embeddings saved.")