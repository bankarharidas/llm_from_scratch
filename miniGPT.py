from TransformerBlock import TransformerBlock
import torch.nn as nn
import torch


class MiniGPT(nn.Module):

    def __init__(
        self, vocab_size, embedding_size, num_heads, num_blocks, sequence_length
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)

        self.position_embedding = nn.Embedding(sequence_length, embedding_size)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embedding_size, num_heads) for _ in range(num_blocks)]
        )

        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, tokens):

        # Token embeddings
        x = self.token_embedding(tokens)

        # Position embeddings
        positions = torch.arange(tokens.size(0), device=tokens.device)

        pos = self.position_embedding(positions)

        # Add token + position information
        x = x + pos

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Convert representations to logits
        logits = self.output_layer(x)

        return logits


vocab_size = 100
embedding_size = 16
num_heads = 4
sequence_length = 10

model = MiniGPT(
    vocab_size=1000, embedding_size=128, num_heads=4, num_blocks=4, sequence_length=128
)

tokens = torch.randint(0, 1000, (20,))
print(tokens)

logits = model(tokens)

print("Tokens:", tokens.shape)
print("Logits:", logits.shape)

total_parameters = sum(p.numel() for p in model.parameters())

print("Total parameters:", total_parameters)
