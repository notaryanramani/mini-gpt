import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import DecoderBlock
import pandas as pd
import numpy as np
import tiktoken
import time


tokenizer = tiktoken.get_encoding('r50k_base')
# hyperparameters
vocab_size = tokenizer.n_vocab
block_size = 32
learning_rate = 1e-3
steps = 10000
eval_step = steps // 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---


# data-processing
df = pd.read_csv('data/data.csv')
x = df['text'].copy()
y = df['sentiment'].copy()

# input-processing
tokens = []
for x_ in x:
    tokens.append(tokenizer.encode(x_))

# label/target-processing
target_map = {
    'negative' : 0,
    'neutral'  : 1,
    'positive' : 2
}
y = y.map(target_map)

# train-test-split
n = int(len(tokens) * 0.8)
train_x, train_y = tokens[:n], list(y[:n])
val_x, val_y = tokens[n:], list(y[n:])


# data-loader
def data_loader(split='train', block_size=32):
    x_, y_ = (train_x, train_y) if split == 'train' else (val_x, val_y)
    rand_nums = torch.randint(len(x_), (block_size,))
    
    x_ = [x_[i.item()] for i in rand_nums]
    y_ = [y_[i.item()] for i in rand_nums]

    for i, s in enumerate(x_):
        if len(s) < block_size:
            temp = tokenizer.encode(' ' * (block_size - len(s))) + s
            x_[i] = temp
        else:
            x_[i] = s[-block_size:]
    
    x_ = torch.tensor(x_).to(device)
    y_ = torch.tensor(y_).to(device)

    return x_, y_


def get_accuracy(y, y_hat):
    correct = []
    for yi, y_hati in zip(y, y_hat):
        if yi == y_hati:
            correct.append(1)
        else:
            correct.append(0)
    return sum(correct) / len(correct)


# loss-estimator
@torch.no_grad()
def get_metrics(m):
    train_lossi = []
    val_lossi = []
    train_acci = []
    val_acci = []
    m.eval()
    for _ in range(100):
        # train
        x, y = data_loader('train')
        logits, loss = m(x, y)
        y_hat = torch.argmax(logits, dim=1)
        accuracy = get_accuracy(y, y_hat)
        train_lossi.append(loss)
        train_acci.append(accuracy)

        # val
        x, y = data_loader('val')
        logits, loss = m(x, y)
        y_hat = torch.argmax(logits, dim=1)
        accuracy = get_accuracy(y, y_hat)
        val_lossi.append(loss)
        val_acci.append(accuracy)
    
    train_loss = torch.tensor(train_lossi).mean()
    val_loss = torch.tensor(val_lossi).mean()
    train_acc = torch.tensor(train_acci).mean()
    val_acc = torch.tensor(val_acci).mean()
    m.train()
    return train_loss, val_loss, train_acc, val_acc


# model
class Classifier(nn.Module):

    ''' Text Classifier for Sentiment Analysis/Text Classification'''

    def __init__(self, vocab_size, num_classes,n_embd=16, block_size=32, n_head=4, n_layers=4, mask=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock(n_embd, block_size, n_head, mask) for _ in range(n_layers)])
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
    

    @torch.no_grad()
    def predict(self, idx):
        logits, _ = self(idx)
        return logits
    

# model-initialization
c = Classifier(vocab_size, 3)
c.to(device)
optimizer =  torch.optim.AdamW(c.parameters(),  lr=learning_rate)


# model-training

st = time.time()

for step in range(steps):
    x, y = data_loader('train')
    logits, loss = c(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % eval_step == 0:
      train_loss, val_loss, train_acc, val_acc = get_metrics(c)
      print(f'Step {step}:  Train Loss: {train_loss.item():.4f}, Train Accuracy: {train_acc.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_acc.item():.4f}')

et = time.time()

mins = (et - st)   //  60
secs = int((et - st) % 60)

print()
print(f'Time Elasped: {mins} mins {secs} secs')
print()

# inference
print('Prediction for Inference: ')

input = 'It is amazing, what a fabulous day'
idx = tokenizer.encode(input)
idx = tokenizer.encode(' ' * (block_size - len(idx))) + idx
idx = torch.tensor(idx).to(device).view(1, -1)
op = torch.nn.functional.softmax(c.predict(idx), dim=1)
predicted_class = torch.argmax(op[0]).item()
print(f'Input: {input}')
print(f'Output: {predicted_class}')

input = 'This could have been better, I did not like it.'
idx = tokenizer.encode(input)
idx = tokenizer.encode(' ' * (block_size - len(idx))) + idx
idx = torch.tensor(idx).to(device).view(1, -1)
op = torch.nn.functional.softmax(c.predict(idx), dim=1)
predicted_class = torch.argmax(op[0]).item()
print(f'Input: {input}')
print(f'Output: {predicted_class}')