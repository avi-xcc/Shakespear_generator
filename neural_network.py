import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from data_loader import *

batch_size = 1
block_size = 256
max_iters = 50000
eval_interval = 10
learning_rate = 3e-4
device = 'cuda'  # 'cpu'
eval_iters = 20
n_embed = 256
num_heads = 8
num_blocks = 4
vocab_size = len(letter_to_number)
dropout = 0.5

torch.manual_seed(1337)


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, num_heads, n_embed, block_size, dropout_rate=0.5):
        super().__init__()
        self.SA = nn.MultiheadAttention(n_embed, num_heads, dropout=dropout_rate, batch_first=True)
        self.ffwd = FeedForward(n_embed)
        self.lnorm1 = nn.LayerNorm(n_embed)
        self.lnorm2 = nn.LayerNorm(n_embed)

        self.register_buffer('tril', 1 - torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # print(self.tril)
        x_, _ = self.SA(x, x, x, attn_mask=self.tril, is_causal=True, need_weights=False)
        x = self.lnorm1(x + x_)
        x = self.lnorm2(x + self.ffwd(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, num_heads, n_embed, block_size, num_blocks, dropout_rate=0.5):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(num_heads, n_embed, block_size, dropout_rate) for _ in range(num_blocks)])
        self.lnorm = nn.LayerNorm(n_embed)
        self.block_size = block_size
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        # print(tok_emb.size(), pos_emb.size())
        x = tok_emb + pos_emb
        x = x + self.blocks(x)
        x = self.lnorm(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=5)[0:1]
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


train_data = torch.tensor(encode(dataset), dtype=torch.long, device=device)


def get_batch():
    data = train_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


model = LanguageModel(vocab_size, num_heads, n_embed, block_size, num_blocks)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch()
        _, loss = model(X, Y)
        losses[k] = loss.item()
    out['train'] = losses.mean()
    model.train()
    return out


for it in range(max_iters):
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {it}: train loss {losses['train']:.4f}")
        generated = decode(model.generate(context, max_new_tokens=10)[0].tolist())
        print(f"```{generated}```")
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

model.eval()
generated = decode(model.generate(context, max_new_tokens=512)[0].tolist())
print(generated)

torch.save(model, "./shakespear_model")

