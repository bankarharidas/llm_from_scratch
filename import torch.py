import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
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

        score = Q @ K.transpose(-2, -1)

        mask = torch.tril(torch.ones(x.size(0), x.size(0)))

        score = score.masked_fill(mask == 0, float("-inf"))

        weight = torch.softmax(score, dim=-1)

        head_outputs = weight @ V

        head_outputs = head_outputs.transpose(0, 1)

        head_outputs = head_outputs.contiguous().view(x.size(0), self.embedding_size)

        output = self.W_O(head_outputs)

        return output


class transformer(nn.Module):
    def __init__(self, embedding_size, num_head):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_size, num_head)

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
    def __init__(self, vocab_size, embedding_size, num_head, sequence_length):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)

        self.position_embedding = nn.Embedding(sequence_length, embedding_size)

        self.transformer = transformer(embedding_size, num_head)

        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, token):
        x = self.token_embedding(token)

        position = torch.arange(token.size(0))

        pos = self.position_embedding(position)

        x = x + pos

        x = self.transformer(x)

        logit = self.output_layer(x)

        return logit


vocab_size = 100
embedding_size = 16
num_head = 4
sequence_length = 10

model = MiniGPT(vocab_size, embedding_size, num_head, sequence_length)

token = torch.tensor([5, 12, 27, 8, 19, 3, 45, 7, 15, 2])

logit = model(token)

print("Tokens:", token.shape)
print("Logits:", logit.shape)
