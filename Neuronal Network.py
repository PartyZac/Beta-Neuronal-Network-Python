import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq_len=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# Ejemplo de uso:
vocab_size = 1000      # Número de tokens únicos en tu vocabulario
max_seq_len = 32       # Longitud máxima de la secuencia
model = MiniGPT(vocab_size, max_seq_len=max_seq_len)

# Datos ficticios: batch de 2 secuencias de 10 tokens
x = torch.randint(0, vocab_size, (2, 10))
logits = model(x)
print(logits.shape)  # (2, 10, vocab_size)