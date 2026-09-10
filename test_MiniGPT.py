import torch
import torch.nn as nn


class MultHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_head):
        super().__init__()

        self.embedding_size = embedding_size
        self.num_head = num_head
        self.num_dim = embedding_size // num_head

        self.W_Q = nn.Linear(embedding_size, embedding_size)
        self.W_K = nn.Linear(embedding_size, embedding_size)
        self.W_V = nn.Linear(embedding_size, embedding_size)
        self.W_O = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(x.size(0), self.num_head, self.num_dim)
        K = K.view(x.size(0), self.num_head, self.num_dim)
        V = V.view(x.size(0), self.num_head, self.num_dim)

        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)

        scores = Q @ K.transpose(-2, -1)

        mask = torch.tril(torch.ones(x.size(0), x.size(0)))

        scores = scores.masked_fill(mask == 0, float("-inf"))

        weight = torch.softmax(scores, dim=-1)

        head_weight = weight @ V

        head_outputs = head_weight.transpose(0, 1)

        head_outputs = head_outputs.contiguous().view(x.size(0), self.embedding_size)

        output = self.W_O(head_outputs)

        return output


class TransformerBlock(nn.Module):

    def __init__(self, embedding_size, num_head):
        super().__init__()
        self.attention = MultHeadAttention(embedding_size, num_head)

        self.norm1 = nn.LayerNorm(embedding_size)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.ReLU(),
            nn.Linear(embedding_size * 4, embedding_size),
        )

        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        attention_output = self.attention(x)

        x = x + attention_output

        x = self.norm1(x)

        ffn_output = self.ffn(x)

        x = x + ffn_output

        x = self.norm2(x)

        return x


class MiniGPT(nn.Module):

    def __init__(self, vocab_size, embedding_size, num_heads, sequence_length):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)

        self.position_embedding = nn.Embedding(sequence_length, embedding_size)

        self.transformer = TransformerBlock(embedding_size, num_heads)

        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, token):

        # Token embedding
        x = self.token_embedding(token)

        # Position embedding
        positions = torch.arange(token.size(0))

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
