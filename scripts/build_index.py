import numpy as np
import faiss
import json, os

EMB_FILE = "../embeddings/embeddings.npy"
META_FILE = "../embeddings/metadata.json"
OUT_INDEX = "../embeddings/faiss.index"

embs = np.load(EMB_FILE)  # shape (N, 128)
d = embs.shape[1]

index = faiss.IndexFlatL2(d)   # simple exact search; easy to start with
index.add(embs)
faiss.write_index(index, OUT_INDEX)
print("index built; n=", index.ntotal)