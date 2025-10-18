import torch
from sentence_transformers import SentenceTransformer
import os

mots_cles_techniques = [
    "freebox", "box", "voyant", "connexion", "internet", "débit", "redémarre",
    "netflix", "wifi", "téléphone fixe", "décodeur", "télécommande", "erreur",
    "panne", "accès", "clignote", "mot de passe", "contrôle parental"
]

model = SentenceTransformer("dangvantuan/sentence-camembert-large")
embeddings = model.encode(mots_cles_techniques, convert_to_tensor=True, normalize_embeddings=True)

# Chemin de sauvegarde
base_dir = os.path.dirname(os.path.abspath(__file__))  # fichier courant
knowledge_dir = os.path.join(base_dir, "knowledge", "gold")
os.makedirs(knowledge_dir, exist_ok=True)  # création du dossier si besoin

# Chemin final du fichier
save_path = os.path.join(knowledge_dir, "tech_keywords.pt")
# Sauvegarde
torch.save({
    "keywords": mots_cles_techniques,
    "embeddings": embeddings
}, save_path)