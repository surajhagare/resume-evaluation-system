import faiss
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(BASE_DIR, "vector_db")
os.makedirs(VECTOR_DIR, exist_ok=True)

INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")
META_PATH = os.path.join(VECTOR_DIR, "metadata.pkl")

def load_index():
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    return None, []

def save_index(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
