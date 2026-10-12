import torch

from config import (
    vocab_size,
    embedding_size,
    num_heads,
    num_blocks,
    sequence_length,
    ffn_hidden_size,
)

from model import MiniGPT

model = MiniGPT(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    num_heads=num_heads,
    num_blocks=num_blocks,
    sequence_length=sequence_length,
    ffn_hidden_size=ffn_hidden_size,
)

tokens = torch.randint(0, vocab_size, (1, 5))

with torch.no_grad():
    logits, kv_cache = model(tokens)

print("Logits shape:", logits.shape)
print("Number of KV layers:", len(kv_cache))
print("K cache shape:", kv_cache[0][0].shape)
