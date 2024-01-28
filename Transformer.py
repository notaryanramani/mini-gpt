import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2000)

# parameters
dropout = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---


class  MultiHead(nn.Module):
    
    '''Multi Head Attention Module'''

    def __init__(self, n_embd, block_size, n_head, head_size, mask=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, block_size, head_size, mask) for _  in  range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v):
        out = torch.cat([h(q, k, v) for  h in self.heads], dim=-1)
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


    def forward(self, q, k, v):
        B, T, C = q.shape

        k = self.key(k)
        q = self.query(q)
        v = self.value(v)

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
    

class EncoderBlock(nn.Module):

    ''' A encoder block consisting of MultiHeads followed by a Feed Forward without a default mask '''

    def __init__(self, n_embd, block_size, n_head, mask=False):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = MultiHead(n_embd, block_size, n_head, head_size, mask)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = self.ln1(x)
        x = x + self.heads(x, x, x)
        x = self.ln2(x)
        out = x + self.ff(x)
        return out


class DecoderBlock(nn.Module):

    '''A decoder block consisting of MultiHeads followed by a Feed Forward with a default mask'''

    def  __init__(self,  n_embd, block_size, n_head, mask=True):
        super().__init__()
        head_size = n_embd // n_head
        self.mask_heads = MultiHead(n_embd, block_size, n_head, head_size, mask)
        self.heads = MultiHead(n_embd, block_size, n_head, head_size, mask=False)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        

    def forward(self, x, enc_x=None):
        x = self.ln1(x)
        x = self.mask_heads(x, x, x)
        x = self.ln2(x)
        if enc_x is not None:
            x = x + self.heads(x, enc_x, enc_x)
        else:
            x = x + self.heads(x, x, x)
        x = self.ln3(x)
        out = x + self.ff(x)
        
        return out


