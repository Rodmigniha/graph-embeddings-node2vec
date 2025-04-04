import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from utils import generate_walks, load_karate_club
from model import SkipGramNeg


def batch_generator(walks, node2idx, window_size=5, batch_size=32):
    """Générateur de batchs vectorisés"""
    all_pairs = []
    for walk in walks:
        for i in range(len(walk)):
            target = walk[i]
            context = walk[max(0, i - window_size):i] + walk[i + 1:i + window_size + 1]
            for ctx in context:
                all_pairs.append((node2idx[target], node2idx[ctx]))

    # Découpage en batchs
    for i in range(0, len(all_pairs), batch_size):
        batch = all_pairs[i:i + batch_size]
        yield torch.tensor([x[0] for x in batch], dtype=torch.long), \
              torch.tensor([x[1] for x in batch], dtype=torch.long)


def train_node2vec(G, embed_dim=64, epochs=150, window_size=5, p=1, q=0.5, num_neg=3, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Using device: {device}")
    print(f" Exploration params: p={p} (return), q={q} (exploration)\n")

    walks = generate_walks(G, p=p, q=q, num_walks=20, walk_length=40)
    nodes = list(G.nodes())
    node2idx = {n: i for i, n in enumerate(nodes)}

    degrees = np.array([G.degree(n) for n in nodes])
    noise_dist = torch.tensor(degrees ** 0.75 / (degrees ** 0.75).sum(), dtype=torch.float).to(device)

    model = SkipGramNeg(len(nodes), embed_dim).to(device)
    optimizer = Adam(model.parameters(), lr=0.00025)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    loss_history = []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        count = 0

        for target_idx, ctx_idx in batch_generator(walks, node2idx, window_size, batch_size):
            neg_samples = torch.multinomial(noise_dist, num_neg * len(target_idx), replacement=True).view(len(target_idx), num_neg)

            target_idx, ctx_idx, neg_samples = target_idx.to(device), ctx_idx.to(device), neg_samples.to(device)

            loss = model(target_idx, ctx_idx, neg_samples)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            count += 1

        avg_loss = epoch_loss / max(count, 1)
        scheduler.step()
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'node2idx': node2idx,
                'embeddings': model.in_emb.weight.detach().cpu().numpy()
            }
            torch.save(checkpoint, '../results/best_model.pt')

            np.save('../results/embeddings.npy', model.in_emb.weight.detach().cpu().numpy())

        print(f'Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if optimizer.param_groups[0]['lr'] < 1e-5:
            print("Learning rate too low, stopping training")
            break

    # Vérification avant de charger le modèle
    if os.path.exists('../results/best_model.pt'):
        checkpoint = torch.load('../results/best_model.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to('cpu')
        embeddings = model.in_emb.weight.detach().numpy()
    else:
        print(" No saved model found, returning last computed embeddings.")
        embeddings = model.in_emb.weight.detach().cpu().numpy()

    np.save('../results/final_embeddings.npy', embeddings)

    return embeddings, node2idx, loss_history


