import faiss
import numpy as np

def init_faiss(embeddings):
    """Creates a FAISS index and adds document vectors."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # Measures Euclidean distance
    index.add(np.array(embeddings).astype('float32'))
    return index

def find_matches(query_vector, index, k=3):
    """Finds the top k most similar text chunks to the query."""
    distances, indices = index.search(np.array(query_vector).astype('float32'), k)
    return indices[0]