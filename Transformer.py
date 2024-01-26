import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2000)

# parameters
dropout = 0.4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---


class  MultiHead(nn.Module):
    
    '''Multi Head Attention Module'''

    def __init__(self, n_embd, block_size, n_head, head_size, mask=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, block_size, head_size, mask) for _  in  range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out = torch.cat([h(x) for  h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Head(nn.Module):

    ''' Single Head of Self Attention '''
    
    def __init__(self, n_embd, block_size, head_size, mask=True):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size,  bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size,  bias=False)
        self.dropout = nn.Dropout(dropout)
        self.mask =  mask
        if self.mask:
            self.register_buffer('tril', torch.tril((torch.ones(block_size, block_size))))


    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        att = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(C, dtype=torch.float32))
        if self.mask:
            att = att.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        return out

 
class  FeedForward(nn.Module):

    '''A Simple Feed Forward followed by non-linear activation'''

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.net(x)
        return out
    

class Block(nn.Module):

    '''A block consisting of MultiHeads followed by a Feed Forward'''

    def  __init__(self,  n_embd, block_size, n_head, mask=True):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = MultiHead(n_embd, block_size, n_head, head_size, mask)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = self.ln1(x)
        x = x + self.heads(x)
        x = self.ln2(x)
        out = x + self.ff(x)
        return out


class LanguageModel(nn.Module):

    '''The language model that that is trained and used to generate text'''

    def __init__(self, vocab_size, n_embd=16, block_size=32, n_head=4, n_layers=4, mask=True):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, block_size, n_head, mask) for  _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.linear = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets = None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.linear(x)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss
    

    def generate(self, idx, num_tokens=100, block_size=32):
        for _ in range(num_tokens):
            idx_cond =  idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next =  torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
