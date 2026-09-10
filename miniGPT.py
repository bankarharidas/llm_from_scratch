from TransformerBlock import TransformerBlock
import torch.nn as nn
import torch


class MiniGPT(nn.Module):

    def __init__(self, vocab_size, embedding_size, num_heads, sequence_length):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)

        self.position_embedding = nn.Embedding(sequence_length, embedding_size)

        self.transformer = TransformerBlock(embedding_size, num_heads)

        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, tokens):

        # Token embedding
        x = self.token_embedding(tokens)

        # Position embedding
        positions = torch.arange(tokens.size(0))

        pos = self.position_embedding(positions)

        # Combine token + position
        x = x + pos

        # Transformer
        x = self.transformer(x)

        # Convert to logits
        logits = self.output_layer(x)

        return logits


vocab_size = 100
embedding_size = 16
num_heads = 4
sequence_length = 10

model = MiniGPT(vocab_size, embedding_size, num_heads, sequence_length)

tokens = torch.tensor([5, 12, 27, 8, 19, 3, 45, 7, 15, 2])

logits = model(tokens)

print("Tokens:", tokens.shape)
print("Logits:", logits.shape)
