# 🔍 Graph Mining avec Node2Vec

## 📌 Présentation
Ce projet implémente l'algorithme **Node2Vec** pour générer des embeddings de graphes, utilisés ensuite pour des tâches d'analyse et de clustering. Il repose sur **PyTorch** pour l'entraînement du modèle et **Scikit-learn** pour l'évaluation.

## 📁 Structure du projet
```
├── results
│   ├── best_model.pt
│   ├── embeddings.npy
│   ├── final_embeddings.npy
│   ├── clustering_visualization.png
│
├── src
│   ├── model.py                 # Définition du modèle SkipGram avec négative sampling
│   ├── train.py                 # Script d'entraînement de Node2Vec
│   ├── eval.py                  # Évaluation des embeddings
│   ├── utils.py                 # Fonctions utilitaires (génération de marches aléatoires...)
│   ├── node2vec_training.ipynb  # Notebook d'entraînement et d'analyse
│   ├── __init__.py

```

## 🚀 Installation
### 1️⃣ Cloner le projet
```bash
git clone https://github.com/Rodmigniha/graph-embeddings-node2vec.git
cd graph-mining-node2vec
```

### 2️⃣ Installer les dépendances
Utiliser **Poetry** ou **pip** :
```bash
pip install -r requirements.txt
```

## 🎯 Utilisation
### 1️⃣ Entraîner le modèle
```
Exécuter le notebook node2vec_training.ipynb
```
Cela génère les embeddings et les sauvegarde dans `results/`.

### 2️⃣ Évaluer le modèle
```bash
python -m src.eval
```
Cela affiche les scores de similarité cosinus et génère une visualisation du clustering.

## 📊 Résultats
- 🔹 **Embeddings générés** : `results/final_embeddings.npy`
- 🔹 **Meilleur modèle sauvegardé** : `results/best_model.pt`
- 🔹 **Visualisation du clustering** : `results/clustering_visualization.png`

---

## Auteurs

- **Lina THABET**
- **Rodrigue MIGNIHA**

📧 Contacts :
- linathabet101@gmail.com
- rodrigue.pro2020@gmail.com
- kidam.migniha@gmail.com
