# test_kv_cache.py

import torch
from tokenizers import Tokenizer

from config import (
    vocab_size,
    embedding_size,
    num_heads,
    num_blocks,
    sequence_length,
    ffn_hidden_size,
    checkpoint_file,
    tokenizer_file,
)

from model import MiniGPT

# Load tokenizer and trained model
tokenizer = Tokenizer.from_file(tokenizer_file)

model = MiniGPT(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    num_heads=num_heads,
    num_blocks=num_blocks,
    sequence_length=sequence_length,
    ffn_hidden_size=ffn_hidden_size,
)

checkpoint = torch.load(
    checkpoint_file,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.eval()

# Prepare a short prompt
prompt_tokens = torch.tensor(
    [tokenizer.encode("nothing").ids],
    dtype=torch.long,
)

# Ensure the test prompt has at least one token
assert prompt_tokens.shape[1] > 0

# Compare cached and full-context logits
with torch.no_grad():

    # First, process the prompt and build the cache
    prompt_logits, kv_cache = model(
        prompt_tokens,
        past_kv=None,
        start_position=0,
    )

    # Use the most likely next token for a deterministic test
    next_token = torch.argmax(
        prompt_logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    # Cached method: process only the new token
    cached_logits, _ = model(
        next_token,
        past_kv=kv_cache,
        start_position=prompt_tokens.shape[1],
    )

    # Full-context method: recompute prompt plus new token
    full_tokens = torch.cat(
        [prompt_tokens, next_token],
        dim=1,
    )

    full_logits, _ = model(
        full_tokens,
        past_kv=None,
        start_position=0,
    )

    cached_result = cached_logits[:, -1, :]
    full_result = full_logits[:, -1, :]

    max_difference = (cached_result - full_result).abs().max().item()

    matches = torch.allclose(
        cached_result,
        full_result,
        atol=1e-5,
        rtol=1e-5,
    )

print("Prompt shape:", prompt_tokens.shape)
print("Cached logits shape:", cached_result.shape)
print("Maximum difference:", max_difference)
print("KV cache correct:", matches)
