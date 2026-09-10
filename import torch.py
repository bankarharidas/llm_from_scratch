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

        # FIXED ERROR: Swapped dimension order from (x.size(0), self.num_dim, self.num_head)
        # to (x.size(0), self.num_head, self.num_dim) so head splitting is aligned correctly
        Q = Q.view(x.size(0), self.num_head, self.num_dim)
        K = K.view(x.size(0), self.num_head, self.num_dim)
        V = V.view(x.size(0), self.num_head, self.num_dim)

        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)

        # FIXED ERROR: Added scaling factor (sqrt(d_k)) to prevent vanishing gradients
        scores = (Q @ K.transpose(-2, -1)) / (self.num_dim**0.5)

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
            # FIXED ERROR: Input dimension must be embedding_size * 4, output must be embedding_size
            nn.Linear(embedding_size * 4, embedding_size),
        )

        # FIXED TYPO: Changed 'nrom2' to 'norm2'
        self.norm2 = nn.LayerNorm(embedding_size)

    # FIXED ERROR: Renamed method from 'format' to 'forward' so PyTorch calls it during execution
    def forward(self, x):
        attention_output = self.attention(x)

        x = x + attention_output

        x = self.norm1(x)

        ffn_output = self.ffn(x)

        x = x + ffn_output

        x = self.norm2(x)

        return x


class MiniGPT(nn.Module):

    def __init__(self, vocal_size, embedding_size, num_head, sequence_length):
        super().__init__()

        # FIXED ERROR: Parameter order for nn.Embedding is (num_embeddings, embedding_dim); changed from (sequence_length, vocal_size)
        self.token_embedding = nn.Embedding(vocal_size, embedding_size)

        # FIXED ERROR: Changed embedding dimension from vocal_size to embedding_size so it matches token_embedding
        self.position_embedding = nn.Embedding(sequence_length, embedding_size)

        self.transformer = TransformerBlock(embedding_size, num_head)

        self.output_layer = nn.Linear(embedding_size, vocal_size)

    def forward(self, token):

        x = self.token_embedding(token)

        position = torch.arange(token.size(0))

        pos = self.position_embedding(position)

        x = x + pos

        x = self.transformer(x)

        logits = self.output_layer(x)

        return logits


vocal_size = 100
embedding_size = 16
num_head = 4
sequence_length = 10

model = MiniGPT(vocal_size, embedding_size, num_head, sequence_length)

tokens = torch.tensor([5, 12, 27, 8, 19, 3, 45, 7, 15, 2])

logist = model(tokens)

print("tokens:", tokens.shape)
print("logist:", logist.shape)
