import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, n_embd, d_model, n_head, dropout, block_size):
        super(Block, self).__init__()
        self.sa = MultiHeadAttention(n_embd, d_model // n_head, n_head, dropout, block_size)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass for each transformer block."""
        x_res = self.sa(self.ln1(x))
        x = x + x_res
        x_res = self.ffwd(self.ln2(x))
        x = x + x_res
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Forward pass for the feedforward layer."""
        return self.net(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, d_k, n_head, dropout, block_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(n_embd, d_k, dropout, block_size) for _ in range(n_head)]) 
        self.projection = nn.Linear(n_head * d_k, n_head * d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Concatenate outputs from multiple attention heads and project."""
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.projection(self.dropout(out))


class Head(nn.Module):
    def __init__(self, n_embd, d_model, dropout, block_size):
        super(Head, self).__init__()
        self.query = nn.Linear(n_embd, d_model, bias=False)
        self.key = nn.Linear(n_embd, d_model, bias=False)
        self.value = nn.Linear(n_embd, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('causal_mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """Apply scaled dot-product attention."""
        _, seq_len, dims = x.size()
        dk = dims  # Since single head
        q = self.query(x) # (batch_size, seq_len, d_model)
        k = self.key(x) # (batch_size, seq_len, d_model)
        v = self.value(x) # (batch_size, seq_len, d_model)
        key_transpose = k.transpose(1,2)
        attention_scores = q @ key_transpose / torch.sqrt(torch.tensor(dk))
        attention_scores  = attention_scores.masked_fill(self.causal_mask[:seq_len, :seq_len] == 0, float('-inf'))
        attention = F.softmax(attention_scores, dim=-1)
        attention = self.dropout(attention)
        out = attention @ v
        return out
        
    
class TextGeneratorModel(nn.Module):
    def __init__(self, config):
        super(TextGeneratorModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.position = nn.Embedding(config["block_size"], config["n_embd"])
        self.ln = nn.LayerNorm(config["d_model"])
        self.blocks = nn.Sequential(*[Block(config["n_embd"], config["d_model"], config["n_head"], config["dropout"], config["block_size"]) for _ in range(config["n_layer"])])
        self.fc_out = nn.Linear(config["d_model"], config["vocab_size"])

    def forward(self, idx, targets=None):
        B, T = idx.size()
        token_emb = self.embedding(idx)
        pos_emb = self.position(torch.arange(T, device=self.config["device"]))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.fc_out(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    
    def generate_text(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.config["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx