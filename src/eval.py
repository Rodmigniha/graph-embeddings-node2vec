import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from .utils import load_karate_club


def load_embeddings(filepath="results/final_embeddings.npy"):
    """Charge les embeddings sauvegardés."""
    return np.load(filepath)


def evaluate_similarity(embeddings, node2idx, node_pairs):
    """Évalue la similarité entre des paires de nœuds."""
    similarities = {}
    for node1, node2 in node_pairs:
        idx1, idx2 = node2idx[node1], node2idx[node2]
        sim = cosine_similarity(embeddings[idx1].reshape(1, -1), embeddings[idx2].reshape(1, -1))
        similarities[(node1, node2)] = sim[0][0]
    
    return similarities


def clustering_quality(embeddings, num_clusters=2):
    """Effectue un clustering K-Means et affiche les résultats."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(label="Cluster")
    plt.title("Clustering des embeddings avec K-Means et t-SNE")
    plt.savefig("results/clustering_visualization.png")
    plt.close()
    
    return labels


if __name__ == "__main__":
    G, _, _ = load_karate_club()
    
    # Charger les embeddings et le dictionnaire node2idx
    embeddings = load_embeddings()
    checkpoint = torch.load("results/best_model.pt")
    node2idx = checkpoint["node2idx"]

    # Évaluation de la similarité entre quelques paires de nœuds
    node_pairs = [(0, 1), (2, 3), (4, 5)]
    similarities = evaluate_similarity(embeddings, node2idx, node_pairs)

    print("\n🔍 Similarité cosinus entre paires de nœuds :")
    for (n1, n2), sim in similarities.items():
        print(f"   - Similarité entre {n1} et {n2} : {sim:.4f}")

    # Évaluation du clustering
    labels = clustering_quality(embeddings, num_clusters=2)
    print("\n Clustering terminé. Visualisation enregistrée dans 'results/clustering_visualization.png'.")
