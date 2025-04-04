from collections import defaultdict
import networkx as nx
import numpy as np


def load_karate_club():
    """Charge le graphe Zachary's Karate Club."""
    G = nx.karate_club_graph()
    labels = [G.nodes[n]['club'] for n in G.nodes()]
    label2idx = {'Mr. Hi': 0, 'Officer': 1}
    return G, labels, label2idx

def precompute_transition_probs(G, p=1, q=1):
    """Précalcule toutes les probabilités de transition pour chaque paire (prev, current)."""
    transition_probs = defaultdict(dict)
    
    for node in G.nodes():
        for neighbor in G.neighbors(node):
            if node not in transition_probs:
                transition_probs[node] = {}
            
            # Cas où on revient en arrière (prev -> node -> prev)
            transition_probs[(node, neighbor)][node] = 1 / p
            
            # Cas où on explore un nouveau nœud
            for second_degree_neighbor in G.neighbors(neighbor):
                if second_degree_neighbor == node:
                    continue  # déjà traité
                if not G.has_edge(node, second_degree_neighbor):
                    transition_probs[(node, neighbor)][second_degree_neighbor] = 1 / q
            
            transition_probs[(node, neighbor)][neighbor] = 1
    
    return transition_probs

def biased_random_walk(G, start_node, walk_length, transition_probs, p=1, q=1):
    walk = [start_node]
    
    for _ in range(walk_length - 1):
        current = walk[-1]
        neighbors = list(G.neighbors(current))
        
        if not neighbors:
            restart_node = np.random.choice(list(G.nodes()))
            walk.append(restart_node)
            continue
        
        if len(walk) == 1:
            next_node = np.random.choice(neighbors)
        else:
            prev = walk[-2]
            probs_dict = transition_probs.get((prev, current), {})
            
            if not probs_dict:
                probs = []
                for neighbor in neighbors:
                    if neighbor == prev:
                        prob = 1 / p
                    elif G.has_edge(prev, neighbor):
                        prob = 1
                    else:
                        prob = 1 / q
                    probs.append(prob)
                probs = np.array(probs) / np.sum(probs)
                next_node = np.random.choice(neighbors, p=probs)
            else:
                valid_neighbors = [n for n in neighbors if n in probs_dict]
                if not valid_neighbors:
                    next_node = np.random.choice(neighbors)
                else:
                    probs = np.array([probs_dict[n] for n in valid_neighbors])
                    probs = probs / np.sum(probs)
                    next_node = np.random.choice(valid_neighbors, p=probs)
        
        walk.append(next_node)
    
    return walk


def generate_walks(G, num_walks=10, walk_length=20, p=1, q=1):
    transition_probs = precompute_transition_probs(G, p, q)
    walks = []
    
    for _ in range(num_walks):
        for node in G.nodes():
            walks.append(biased_random_walk(G, node, walk_length, transition_probs, p, q))
    
    return walks