import pandas as pd
import numpy as np
import hdbscan
import joblib
import os

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "knowledge", "gold", "articles_with_embeddings.parquet")
MODEL_PATH = os.path.join(PROJECT_ROOT, "knowledge", "gold", "hdbscan_model.joblib")

# Load embeddings
df = pd.read_parquet(DATA_PATH)
X = np.vstack(df["embedding"].values)

# Fit HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
df["cluster"] = clusterer.fit_predict(X)

# Save model + clustered data
joblib.dump(clusterer, MODEL_PATH)
df.to_parquet(DATA_PATH, index=False)

print(f"✅ Clusters HDBSCAN sauvegardés dans {MODEL_PATH}")
