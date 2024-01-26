import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import Block


# device initialization
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---


class Classifier(nn.Module):

    ''' Text Classifier for Sentiment Analysis/Text Classification'''

    def __init__(self, vocab_size, num_classes,n_embd=16, block_size=32, n_head=4, n_layers=4, mask=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, block_size, n_head, mask) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.global_average_pooling = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(n_embd, num_classes)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.global_average_pooling(x.permute(0, 2, 1)).squeeze(-1)
        x = self.ln(x)
        logits = self.linear(x)
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss
    

    def predict(self, idx, block_size=32):
        logits, loss = self(idx)
        return logits