import json
import numpy as np
import faiss
import face_recognition
import sys

INDEX_PATH = "../embeddings/faiss.index"
METADATA_PATH = "../embeddings/metadata.json"

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load metadata
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

def get_query_embedding(image_path):
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)

    if len(locations) == 0:
        raise Exception("No face detected")

    encoding = face_recognition.face_encodings(image, locations)[0]
    return np.array([encoding]).astype("float32")

def search(image_path, k=5):

    query_embedding = get_query_embedding(image_path)

    distances, indices = index.search(query_embedding, k)

    results = []

    for i, idx in enumerate(indices[0]):
        match = metadata[idx]

        results.append({
            "rank": i + 1,
            "person": match["person"],
            "image": match["image"],
            "distance": float(distances[0][i])
        })

    return results


if __name__ == "__main__":

    query_image = sys.argv[1]

    matches = search(query_image)

    for m in matches:
        print(
            f"{m['rank']} | {m['person']} | "
            f"distance={m['distance']:.4f} | "
            f"{m['image']}"
        )