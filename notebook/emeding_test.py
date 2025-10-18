from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# --------------------------
# 1. Exemple de tickets SAV
# --------------------------
sentences = [
    "Le voyant rouge de ma Freebox clignote",
    "Ma box est en panne totale, plus de connexion",
    "Internet fonctionne bien mais la TV ne marche pas",
    "Je n’ai plus de tonalité sur mon téléphone fixe",
    "Mon débit est très lent depuis 3 jours",
    "Je souhaite changer mon mot de passe Wi-Fi",
    "Le décodeur TV affiche erreur E10",
    "Impossible d'accéder à Netflix via la Freebox",
    "La box redémarre toute seule sans arrêt",
    "Comment activer le contrôle parental sur la box ?"
]

# --------------------------
# 2. Embedding avec e5-base-v2
# --------------------------
model = SentenceTransformer("intfloat/e5-base-v2")
embeddings = model.encode(sentences, convert_to_tensor=False)

# --------------------------
# 3. Clustering KMeans
# --------------------------
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# --------------------------
# 4. Nearest Neighbors
# --------------------------
nn_model = NearestNeighbors(n_neighbors=2, metric="cosine").fit(embeddings)
distances, indices = nn_model.kneighbors(embeddings)

# --------------------------
# 5. PCA pour visualisation
# --------------------------
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# --------------------------
# 6. Plot clusters + KNN
# --------------------------
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for i in range(n_clusters):
    cluster_pts = reduced[cluster_labels == i]
    plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], color=colors[i], label=f"Cluster {i}", alpha=0.7)

# Lignes entre voisins
for idx, (x, y) in enumerate(reduced):
    for neighbor_idx in indices[idx][1:]:  # évite soi-même
        nx, ny = reduced[neighbor_idx]
        plt.plot([x, nx], [y, ny], 'k--', linewidth=0.5, alpha=0.3)

# Annotations
for i, txt in enumerate(sentences):
    plt.annotate(f"{i}", (reduced[i, 0], reduced[i, 1]), fontsize=8)

plt.title("Clustering + KNN sur phrases SAV (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
