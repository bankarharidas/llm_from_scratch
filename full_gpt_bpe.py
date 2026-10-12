import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# =========================
# Configuration
# =========================

batch_size = 8
sequence_length = 128

embedding_size = 128
num_heads = 4
num_blocks = 4
ffn_hidden_size = 512

# =========================
# Learning Rate Schedule
# =========================
num_steps = 10001
learning_rate = 3e-4

warmup_steps = 500
decay_steps = num_steps
min_learning_rate = 3e-5

# =========================
# Tokenizer
# =========================

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = Tokenizer.from_file("tokenizer.json")

vocab_size = tokenizer.get_vocab_size()


def encode(text):

    encoded = tokenizer.encode(text)

    return encoded.ids


def decode(tokens):

    return tokenizer.decode(tokens)


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
# Rotary Positional Embedding
# =========================


def apply_rope(x, start_position=0):

    # x shape:
    # [B, num_heads, T, head_dim]

    B, H, T, D = x.shape

    assert D % 2 == 0

    half_dim = D // 2

    # Frequencies
    theta = 10000.0 ** (
        -torch.arange(0, half_dim, device=x.device, dtype=torch.float32) / half_dim
    )

    # Position indices
    positions = torch.arange(
        start_position, start_position + T, device=x.device, dtype=torch.float32
    )

    # [T, half_dim]
    angles = positions[:, None] * theta[None, :]

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # [1, 1, T, half_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return rotated


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

    def forward(self, x, past_k=None, past_v=None, start_position=0):

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

        # RoPE
        Q = apply_rope(Q, start_position=start_position)

        K = apply_rope(K, start_position=start_position)

        # =========================
        # KV Cache
        # =========================

        if past_k is not None:
            K = torch.cat([past_k, K], dim=2)

        if past_v is not None:
            V = torch.cat([past_v, V], dim=2)

        new_k = K
        new_v = V

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

        return output, new_k, new_v


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

    def forward(self, x, past_k=None, past_v=None, start_position=0):

        # Attention + residual
        attention_output, new_k, new_v = self.attention(
            self.norm1(x), past_k=past_k, past_v=past_v, start_position=start_position
        )

        x = x + attention_output

        # FFN + residual
        x = x + self.ffn(self.norm2(x))

        return x, new_k, new_v


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

    def forward(self, tokens, past_kv=None, start_position=0):

        # tokens: [B, T]
        B, T = tokens.shape

        assert T <= self.sequence_length

        # [B, T] -> [B, T, C]
        x = self.token_embedding(tokens)

        # for block in self.blocks:
        #     x = block(x)

        # Transformer blocks
        new_kv = []

        for layer_index, block in enumerate(self.blocks):

            if past_kv is not None:
                past_k, past_v = past_kv[layer_index]
            else:
                past_k = None
                past_v = None

            x, new_k, new_v = block(
                x, past_k=past_k, past_v=past_v, start_position=start_position
            )

            new_kv.append((new_k, new_v))

        # Final normalization
        x = self.final_norm(x)

        # [B, T, C] -> [B, T, vocab_size]
        logits = self.output_layer(x)

        return logits, new_kv


# =========================
# Learning Rate Scheduler
# =========================


def get_learning_rate(step):

    # Warmup
    if step < warmup_steps:

        return learning_rate * (step + 1) / warmup_steps

    # Decay
    progress = (step - warmup_steps) / (decay_steps - warmup_steps)

    progress = min(max(progress, 0.0), 1.0)

    # Cosine decay
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_learning_rate + (learning_rate - min_learning_rate) * cosine


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

    # model.train()

    # best_val_loss = float("inf")

    # for step in range(num_steps):

    #     # Get training batch
    #     inputs, targets = get_batch(train_data, batch_size, sequence_length)

    #     # Forward pass
    #     logits, _ = model(inputs)

    #     # Loss
    #     loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

    #     # Backpropagation
    #     optimizer.zero_grad()

    #     loss.backward()

    #     # Gradient clipping
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    #     # Update learning rate

    #     current_lr = get_learning_rate(step)

    #     for param_group in optimizer.param_groups:
    #         param_group["lr"] = current_lr

    #     # Update parameters
    #     optimizer.step()

    #     # Training loss
    #     if step % 100 == 0:

    #         print(
    #             f"Step {step}: " f"Train Loss {loss.item():.4f} " f"LR {current_lr:.6f}"
    #         )

    #     # =========================
    #     # Validation
    #     # =========================

    #     if step % 500 == 0:

    #         model.eval()

    #         with torch.no_grad():

    #             val_inputs, val_targets = get_batch(
    #                 val_data, batch_size, sequence_length
    #             )

    #             val_logits, _ = model(val_inputs)

    #             val_loss = F.cross_entropy(
    #                 val_logits.reshape(-1, vocab_size), val_targets.reshape(-1)
    #             )

    #         val_loss_value = val_loss.item()

    #         print(f"Step {step}: " f"Validation Loss {val_loss_value:.4f}")

    #         # =========================
    #         # Save ONLY if better
    #         # =========================

    #         if val_loss_value < best_val_loss:

    #             best_val_loss = val_loss_value

    #             best_checkpoint = {
    #                 "model_state_dict": model.state_dict(),
    #                 "optimizer_state_dict": optimizer.state_dict(),
    #                 "step": step,
    #                 # Model configuration
    #                 "vocab_size": vocab_size,
    #                 "embedding_size": embedding_size,
    #                 "num_heads": num_heads,
    #                 "num_blocks": num_blocks,
    #                 "sequence_length": sequence_length,
    #                 "ffn_hidden_size": ffn_hidden_size,
    #                 # BPE tokenizer
    #                 "tokenizer_file": "tokenizer.json",
    #             }

    #             torch.save(best_checkpoint, "best_model_kv.pth")

    #             print("Best model saved!")

    #         model.train()

    # =========================
    # Load Best Checkpoint
    # =========================

    checkpoint = torch.load("best_model_kv.pth", weights_only=False)

    model = MiniGPT(
        vocab_size=checkpoint["vocab_size"],
        embedding_size=checkpoint["embedding_size"],
        num_heads=checkpoint["num_heads"],
        num_blocks=checkpoint["num_blocks"],
        sequence_length=checkpoint["sequence_length"],
        ffn_hidden_size=checkpoint["ffn_hidden_size"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print("Best checkpoint loaded!")

    print("Best step:", checkpoint["step"])

    # print("Best validation loss:", best_val_loss)

    print("Best model loaded!")

    import time

    # import torch

    prompt = "Bequeath"

    prompt_tokens = torch.tensor([encode(prompt)], dtype=torch.long)

    model.eval()

    print("Prompt token shape:", prompt_tokens.shape)
    print("Model is ready for speed testing!")

    def generate_without_cache(model, prompt_tokens, max_new_tokens):

        tokens = prompt_tokens.clone()

        model.eval()

        with torch.no_grad():

            for _ in range(max_new_tokens):

                # Keep context within the model's limit
                tokens_cond = tokens[:, -sequence_length:]

                # Full forward pass; no KV cache
                logits, _ = model(tokens_cond, past_kv=None, start_position=0)

                # Choose the most likely next token
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                tokens = torch.cat([tokens, next_token], dim=1)

        return tokens

    start_time = time.perf_counter()

    test_output = generate_without_cache(model, prompt_tokens, max_new_tokens=50)

    no_cache_time = time.perf_counter() - start_time

    print("Generated tokens:", 50)
    print("Time without KV cache:", round(no_cache_time, 4), "seconds")
    print("Generated text:")
    print(decode(test_output[0].tolist()))

    def generate_with_cache(model, prompt_tokens, max_new_tokens):

        tokens = prompt_tokens.clone()
        model.eval()

        with torch.no_grad():

            # Process the prompt once and create the cache
            logits, kv_cache = model(tokens, past_kv=None, start_position=0)

            for _ in range(max_new_tokens):

                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                tokens = torch.cat([tokens, next_token], dim=1)

                # Process only the newly generated token
                logits, kv_cache = model(
                    next_token, past_kv=kv_cache, start_position=tokens.shape[1] - 1
                )

        return tokens

    start_time = time.perf_counter()

    cached_output = generate_with_cache(model, prompt_tokens, max_new_tokens=50)

    cache_time = time.perf_counter() - start_time

    print("Generated tokens:", 50)
    print("Time with KV cache:", round(cache_time, 4), "seconds")
    print("Generated text:")
    print(decode(cached_output[0].tolist()))

    # import time

    # Warm-up: allow initial execution overhead to settle
    generate_without_cache(model, prompt_tokens, 10)
    generate_with_cache(model, prompt_tokens, 10)

    # Benchmark settings
    repeats = 5
    max_new_tokens = 50

    no_cache_times = []
    cache_times = []

    # Test without KV cache
    for _ in range(repeats):
        start = time.perf_counter()

        generate_without_cache(model, prompt_tokens, max_new_tokens)

        no_cache_times.append(time.perf_counter() - start)

    # Test with KV cache
    for _ in range(repeats):
        start = time.perf_counter()

        generate_with_cache(model, prompt_tokens, max_new_tokens)

        cache_times.append(time.perf_counter() - start)

    # Calculate averages
    avg_no_cache = sum(no_cache_times) / repeats
    avg_cache = sum(cache_times) / repeats

    print("Without KV cache:", [round(t, 4) for t in no_cache_times])
    print("With KV cache:   ", [round(t, 4) for t in cache_times])

    print("\nAverage without cache:", round(avg_no_cache, 4), "seconds")
    print("Average with cache:   ", round(avg_cache, 4), "seconds")
    print("Speedup:", round(avg_no_cache / avg_cache, 2), "x")
    print("Time saved:", round((avg_no_cache - avg_cache) * 1000, 2), "milliseconds")

    # Step 1: Test the context-window limit

    print("Model context length:", sequence_length)

    # Create a prompt longer than the context window
    long_test_tokens = torch.randint(0, vocab_size, (1, sequence_length + 10))

    print("Test prompt shape:", long_test_tokens.shape)

    # Check whether the model correctly rejects an oversized input
    try:
        with torch.no_grad():
            model(long_test_tokens, past_kv=None, start_position=0)

        print("Model accepted the oversized prompt.")

    except AssertionError:
        print("Model correctly rejected the oversized prompt.")

    # Step 2: Test long-prompt generation with KV cache

    long_test_tokens = torch.randint(0, vocab_size, (1, sequence_length + 10))

    # Trim the prompt to the model's context limit
    long_test_tokens = long_test_tokens[:, -sequence_length:]

    print("Prompt after trimming:", long_test_tokens.shape)

    # Generate 5 tokens
    long_output = generate_with_cache(model, long_test_tokens, max_new_tokens=5)

    print("Output shape:", long_output.shape)
    print("Expected shape:", (1, sequence_length + 5))
    print("Test passed:", long_output.shape == (1, sequence_length + 5))

    # Step 3: Checkpoint save/load test

    # import torch

    test_tokens = torch.tensor([encode("Bequeath")], dtype=torch.long)

    model.eval()

    # Record predictions before saving
    with torch.no_grad():
        logits_before, _ = model(test_tokens)

    # Save a temporary checkpoint
    test_checkpoint_path = "checkpoint_test.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "embedding_size": embedding_size,
            "num_heads": num_heads,
            "num_blocks": num_blocks,
            "sequence_length": sequence_length,
            "ffn_hidden_size": ffn_hidden_size,
        },
        test_checkpoint_path,
    )

    # Create a fresh model with the same architecture
    fresh_model = MiniGPT(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
        sequence_length=sequence_length,
        ffn_hidden_size=ffn_hidden_size,
    )

    # Load saved weights
    checkpoint = torch.load(test_checkpoint_path, map_location="cpu", weights_only=True)

    fresh_model.load_state_dict(checkpoint["model_state_dict"])

    fresh_model.eval()

    # Compare predictions
    with torch.no_grad():
        logits_after, _ = fresh_model(test_tokens)

    print(
        "Predictions match:",
        torch.allclose(logits_before, logits_after, atol=1e-6, rtol=1e-5),
    )

    print("Maximum difference:", (logits_before - logits_after).abs().max().item())

    # Step 4: Tokenizer compatibility test

    # from tokenizers import Tokenizer

    # Load the saved tokenizer from disk
    loaded_tokenizer = Tokenizer.from_file("tokenizer.json")

    test_text = "Bequeath"

    # Encode using the current tokenizer
    original_ids = encode(test_text)

    # Encode using the freshly loaded tokenizer
    loaded_ids = loaded_tokenizer.encode(test_text).ids

    print("Original token IDs:", original_ids)
    print("Loaded token IDs:  ", loaded_ids)

    print("Token IDs match:", original_ids == loaded_ids)

    # Decode using the loaded tokenizer
    decoded_text = loaded_tokenizer.decode(loaded_ids)

    print("Decoded text:", decoded_text)

    # Step 5: Final inference test

    # import torch
    # from tokenizers import Tokenizer

    # Load tokenizer
    inference_tokenizer = Tokenizer.from_file("tokenizer.json")

    # Create a fresh model
    inference_model = MiniGPT(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
        sequence_length=sequence_length,
        ffn_hidden_size=ffn_hidden_size,
    )

    # Load the saved test checkpoint
    checkpoint = torch.load(
        "checkpoint_test.pth", map_location="cpu", weights_only=True
    )

    inference_model.load_state_dict(checkpoint["model_state_dict"])

    inference_model.eval()

    # Encode a prompt using the loaded tokenizer
    prompt = "Bequeath"

    input_ids = inference_tokenizer.encode(prompt).ids

    tokens = torch.tensor([input_ids], dtype=torch.long)

    # Generate five tokens using KV cache
    with torch.no_grad():

        logits, kv_cache = inference_model(tokens, past_kv=None, start_position=0)

        for _ in range(5):

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            tokens = torch.cat([tokens, next_token], dim=1)

            logits, kv_cache = inference_model(
                next_token, past_kv=kv_cache, start_position=tokens.shape[1] - 1
            )

    generated_text = inference_tokenizer.decode(tokens[0].tolist())

    print("Prompt:", prompt)
    print("Generated text:", generated_text)
    print("Total token count:", tokens.shape[1])
    print("Inference test passed:", tokens.shape[1] == len(input_ids) + 5)

    # =========================
    # KV Cache Test
    # =========================

    # prompt = "Bequeath"

    # prompt_tokens = torch.tensor([encode(prompt)], dtype=torch.long)

    # model.eval()

    # with torch.no_grad():

    #     # First pass: full prompt
    #     logits, kv_cache = model(prompt_tokens, past_kv=None, start_position=0)

    #     print("Initial logits shape:", logits.shape)

    #     # Generate one token
    #     next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    #     print("Next token:", next_token.item())

    #     # Second pass: ONLY the new token
    #     logits, kv_cache = model(
    #         next_token, past_kv=kv_cache, start_position=prompt_tokens.shape[1]
    #     )

    #     print("Cached logits shape:", logits.shape)

    #     print("Updated K cache shape:", kv_cache[0][0].shape)

    #     print("Updated V cache shape:", kv_cache[0][1].shape)

    # # KV Cache Correctness Test

    # prompt = "Bequeath"

    # prompt_tokens = torch.tensor([encode(prompt)], dtype=torch.long)

    # model.eval()

    # with torch.no_grad():

    #     # -------------------------
    #     # 1. Cached approach
    #     # -------------------------

    #     prompt_logits, kv_cache = model(prompt_tokens, past_kv=None, start_position=0)

    #     next_token = torch.argmax(prompt_logits[:, -1, :], dim=-1, keepdim=True)

    #     cached_logits, _ = model(
    #         next_token, past_kv=kv_cache, start_position=prompt_tokens.shape[1]
    #     )

    #     cached_result = cached_logits[:, -1, :]

    #     # -------------------------
    #     # 2. Full-context approach
    #     # -------------------------

    #     full_tokens = torch.cat([prompt_tokens, next_token], dim=1)

    #     full_logits, _ = model(full_tokens, past_kv=None, start_position=0)

    #     full_result = full_logits[:, -1, :]

    #     # -------------------------
    #     # 3. Compare
    #     # -------------------------

    #     difference = (full_result - cached_result).abs().max().item()

    #     print("Max difference:", difference)

    #     print(
    #         "Same logits:",
    #         torch.allclose(full_result, cached_result, atol=1e-5, rtol=1e-5),
    #     )

    # =========================
    # Text Generation
    # =========================

    def generate(
        model, prompt, max_new_tokens, temperature=1.0, top_k=None, top_p=None
    ):

        model.eval()

        tokens = torch.tensor([encode(prompt)], dtype=torch.long)

        # Make sure the prompt fits inside the context
        if tokens.shape[1] > sequence_length:
            tokens = tokens[:, -sequence_length:]

        with torch.no_grad():

            # =================================
            # First forward pass
            # =================================

            logits, kv_cache = model(tokens, past_kv=None, start_position=0)

            for _ in range(max_new_tokens):

                # Last position
                next_token_logits = logits[:, -1, :]

                # Temperature
                next_token_logits = next_token_logits / temperature

                # =================================
                # Top-k
                # =================================

                if top_k is not None:

                    values, indices = torch.topk(next_token_logits, top_k)

                    filtered_logits = torch.full_like(next_token_logits, float("-inf"))

                    filtered_logits.scatter_(1, indices, values)

                    next_token_logits = filtered_logits

                # =================================
                # Top-p
                # =================================

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

                # =================================
                # Convert to probabilities
                # =================================

                probabilities = torch.softmax(next_token_logits, dim=-1)

                # =================================
                # Sample next token
                # =================================

                next_token = torch.multinomial(probabilities, num_samples=1)

                # Add token to output
                tokens = torch.cat([tokens, next_token], dim=1)

                # =================================
                # KV CACHE
                # =================================

                # Only send the NEW token to the model
                logits, kv_cache = model(
                    next_token, past_kv=kv_cache, start_position=tokens.shape[1] - 1
                )

        model.train()

        return decode(tokens[0].tolist())

    # # =========================
    # # Generation
    # # =========================

    # generated_text = generate(
    #     model,
    #     prompt="Bequeath",
    #     max_new_tokens=100,
    #     temperature=0.1,
    #     top_k=1,
    #     top_p=None,
    # )

    # print()
    # print("Generated text:")
    # print(generated_text)


# =========================
# Program Entry Point
# =========================

if __name__ == "__main__":
    main()
