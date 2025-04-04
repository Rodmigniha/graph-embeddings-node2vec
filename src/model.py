import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, embed_dim)
        self.out_emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.in_emb.weight, -0.5/embed_dim, 0.5/embed_dim)
        nn.init.uniform_(self.out_emb.weight, -0.5/embed_dim, 0.5/embed_dim)

    def forward(self, target, context, neg_samples):
        emb_target = self.in_emb(target)
        emb_ctx = self.out_emb(context)
        emb_neg = self.out_emb(neg_samples)
        
        pos = torch.sum(emb_target * emb_ctx, dim=1)
        neg = torch.bmm(emb_neg, emb_target.unsqueeze(2)).squeeze()
        
        pos_loss = F.logsigmoid(pos).mean()
        neg_loss = F.logsigmoid(-neg).mean()
        return -(pos_loss + neg_loss)
