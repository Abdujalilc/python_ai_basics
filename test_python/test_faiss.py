import faiss
import numpy as np

d = 128  # Dimension of vectors
index = faiss.IndexFlatL2(d)  # Create FAISS index
print(index.is_trained)  # Should print True
print("FAISS index is trained successfully!")
