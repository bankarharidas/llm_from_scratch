import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Configuration
# =========================

batch_size = 8
sequence_length = 128

embedding_size = 128
num_heads = 4
num_blocks = 4
ffn_hidden_size = 512

learning_rate = 3e-4
num_steps = 10001

# =========================
# Tokenizer
# =========================

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
print("Characters:", len(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}

itos = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return [stoi[ch] for ch in text]


def decode(tokens):
    return "".join(itos[token] for token in tokens)


# =========================
# Dataset
# =========================

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


def get_batch(data, batch_size, sequence_length):

    if len(data) <= sequence_length + 1:
        raise ValueError(
            f"Dataset is too small: "
            f"{len(data)} characters. "
            f"Need more than {sequence_length + 1}."
        )

    starts = torch.randint(0, len(data) - sequence_length - 1, (batch_size,))

    x = torch.stack([data[i : i + sequence_length] for i in starts])

    y = torch.stack([data[i + 1 : i + sequence_length + 1] for i in starts])

    return x, y


print("Vocabulary size:", vocab_size)
print("Data:", data.shape)
print("Train:", train_data.shape)
print("Validation:", val_data.shape)

# =========================
# RMSNorm
# =========================


class RMSNorm(nn.Module):

    def __init__(self, embedding_size, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(embedding_size))

        self.eps = eps

    def forward(self, x):

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return self.weight * (x / rms)


# =========================
# Multi-Head Attention
# =========================


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_size, num_heads):
        super().__init__()

        assert embedding_size % num_heads == 0

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        self.W_Q = nn.Linear(embedding_size, embedding_size)

        self.W_K = nn.Linear(embedding_size, embedding_size)

        self.W_V = nn.Linear(embedding_size, embedding_size)

        self.W_O = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):

        # x: [B, T, C]
        B, T, C = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim)

        K = K.view(B, T, self.num_heads, self.head_dim)

        V = V.view(B, T, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))

        scores = scores.masked_fill(~mask, float("-inf"))

        # Attention probabilities
        weights = torch.softmax(scores, dim=-1)

        # Attention output
        output = weights @ V

        output = output.transpose(1, 2)

        output = output.contiguous().view(B, T, C)

        output = self.W_O(output)

        return output


# =========================
# SwiGLU
# =========================


class SwiGLU(nn.Module):

    def __init__(self, embedding_size, hidden_size):
        super().__init__()

        self.gate = nn.Linear(embedding_size, hidden_size)

        self.up = nn.Linear(embedding_size, hidden_size)

        self.down = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):

        gate = F.silu(self.gate(x))

        up = self.up(x)

        x = gate * up

        x = self.down(x)

        return x


# =========================
# Transformer Block
# =========================


class TransformerBlock(nn.Module):

    def __init__(self, embedding_size, num_heads, ffn_hidden_size):
        super().__init__()

        self.norm1 = RMSNorm(embedding_size)

        self.attention = MultiHeadAttention(embedding_size, num_heads)

        self.norm2 = RMSNorm(embedding_size)

        self.ffn = SwiGLU(embedding_size, ffn_hidden_size)

    def forward(self, x):

        # Attention + residual
        x = x + self.attention(self.norm1(x))

        # FFN + residual
        x = x + self.ffn(self.norm2(x))

        return x


# =========================
# MiniGPT
# =========================


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size,
        num_heads,
        num_blocks,
        sequence_length,
        ffn_hidden_size,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)

        # Position embedding
        self.position_embedding = nn.Embedding(sequence_length, embedding_size)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_size, num_heads, ffn_hidden_size)
                for _ in range(num_blocks)
            ]
        )

        # Final normalization
        self.final_norm = RMSNorm(embedding_size)

        # LM Head
        self.output_layer = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, tokens):

        # tokens: [B, T]
        B, T = tokens.shape

        assert T <= self.sequence_length

        # [B, T] -> [B, T, C]
        x = self.token_embedding(tokens)

        # Position IDs
        positions = torch.arange(T, device=tokens.device)

        # [T, C]
        pos = self.position_embedding(positions)

        # Broadcasting:
        # [B, T, C] + [T, C]
        x = x + pos

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # [B, T, C] -> [B, T, vocab_size]
        logits = self.output_layer(x)

        return logits


# =========================
# Main
# =========================


def main():

    # Create model
    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
        sequence_length=sequence_length,
        ffn_hidden_size=ffn_hidden_size,
    )

    print("Parameters:", sum(p.numel() for p in model.parameters()))

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Optimizer ready!")

    # =========================
    # Training
    # =========================

    model.train()

    best_val_loss = float("inf")

    for step in range(num_steps):

        # Get training batch
        inputs, targets = get_batch(train_data, batch_size, sequence_length)

        # Forward pass
        logits = model(inputs)

        # Loss
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update parameters
        optimizer.step()

        # Check training loss
        if step % 100 == 0:

            print(f"Step {step}: " f"Train Loss {loss.item():.4f}")

        # =========================
        # Validation
        # =========================

        if step % 500 == 0:

            model.eval()

            with torch.no_grad():

                val_inputs, val_targets = get_batch(
                    val_data, batch_size, sequence_length
                )

                val_logits = model(val_inputs)

                val_loss = F.cross_entropy(
                    val_logits.reshape(-1, vocab_size), val_targets.reshape(-1)
                )

            print(f"Step {step}: " f"Validation Loss {val_loss.item():.4f}")

            # Save best model
            if val_loss.item() < best_val_loss:

                best_val_loss = val_loss.item()

                torch.save(model.state_dict(), "best_model.pth")

                print("Best model saved!")

            model.train()
    # # =========================
    # # Save Checkpoint
    # # =========================

    # checkpoint = {
    #     "model_state_dict": model.state_dict(),
    #     "optimizer_state_dict": optimizer.state_dict(),
    #     "step": num_steps,
    #     "vocab_size": vocab_size,
    #     "embedding_size": embedding_size,
    #     "num_heads": num_heads,
    #     "num_blocks": num_blocks,
    #     "sequence_length": sequence_length,
    #     "ffn_hidden_size": ffn_hidden_size,
    #     "chars": chars,
    #     "stoi": stoi,
    #     "itos": itos,
    # }

    # torch.save(checkpoint, "best_model.pth")

    # print("Checkpoint saved!")

    # =========================
    # Text Generation
    # =========================

    def generate(
        model, prompt, max_new_tokens, temperature=1.0, top_k=None, top_p=None
    ):

        model.eval()

        tokens = torch.tensor([encode(prompt)], dtype=torch.long)

        with torch.no_grad():

            for _ in range(max_new_tokens):

                # Keep only the latest context
                tokens_cond = tokens[:, -sequence_length:]

                # Forward pass
                logits = model(tokens_cond)

                # Last position
                next_token_logits = logits[:, -1, :]

                # Temperature
                next_token_logits = next_token_logits / temperature

                # Top-k
                if top_k is not None:

                    values, indices = torch.topk(next_token_logits, top_k)

                    filtered_logits = torch.full_like(next_token_logits, float("-inf"))

                    filtered_logits.scatter_(1, indices, values)

                    next_token_logits = filtered_logits

                # Top-p
                if top_p is not None:

                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )

                    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)

                    cumulative_probabilities = torch.cumsum(
                        sorted_probabilities, dim=-1
                    )

                    remove_tokens = cumulative_probabilities > top_p

                    remove_tokens[:, 1:] = remove_tokens[:, :-1].clone()

                    remove_tokens[:, 0] = False

                    sorted_logits = sorted_logits.masked_fill(
                        remove_tokens, float("-inf")
                    )

                    next_token_logits = torch.full_like(
                        next_token_logits, float("-inf")
                    )

                    next_token_logits.scatter_(1, sorted_indices, sorted_logits)

                # Convert logits to probabilities
                probabilities = torch.softmax(next_token_logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(probabilities, num_samples=1)

                # Add token
                tokens = torch.cat([tokens, next_token], dim=1)

        model.train()

        return decode(tokens[0].tolist())

    # =========================
    # Load Best Model
    # =========================

    model.load_state_dict(torch.load("best_model.pth", weights_only=True))

    model.eval()

    print("Best model loaded!")

    # =========================
    # Generation
    # =========================

    generated_text = generate(
        model,
        prompt="Bequeath",
        max_new_tokens=100,
        temperature=0.1,
        top_k=1,
        top_p=None,
    )

    print()
    print("Generated text:")
    print(generated_text)


# =========================
# Program Entry Point
# =========================

if __name__ == "__main__":
    main()
