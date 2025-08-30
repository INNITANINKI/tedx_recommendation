# precompute_embeddings.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("ted_talks.csv")
df["description"] = df["description"].fillna("")
df["presenterDisplayName"] = df["presenterDisplayName"].fillna("")

# Load faster embedding model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Compute embeddings
embeddings = model.encode(df["description"].tolist(), convert_to_tensor=False)
np.save("ted_embeddings.npy", embeddings)  # Save to disk
print("Embeddings saved successfully!")
