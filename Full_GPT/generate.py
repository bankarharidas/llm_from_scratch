# generate.py

import torch
from tokenizers import Tokenizer

from config import (
    vocab_size,
    embedding_size,
    num_heads,
    num_blocks,
    sequence_length,
    ffn_hidden_size,
    tokenizer_file,
    checkpoint_file,
)

from model import MiniGPT


def load_model():

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

    print("Checkpoint file:", checkpoint_file)
    print("Checkpoint metadata:")

    for key, value in checkpoint.items():
        if key != "model_state_dict":
            print(f"{key}: {value}")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    print("RoPE checkpoint loaded successfully!")
    print("Training step:", checkpoint["step"])
    print("Best validation loss:", checkpoint["best_val_loss"])
    print("Position encoding:", checkpoint["position_encoding"])

    model.eval()

    return model


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
):

    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 or None")

    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1] or None")

    tokens = torch.tensor(
        [tokenizer.encode(prompt).ids],
        dtype=torch.long,
    )

    # Keep the prompt within the context limit.
    tokens = tokens[:, -sequence_length:]

    model.eval()

    with torch.no_grad():

        # Process the prompt once.
        logits, kv_cache = model(
            tokens,
            past_kv=None,
            start_position=0,
        )

        for _ in range(max_new_tokens):

            next_token_logits = logits[:, -1, :] / temperature

            # Top-k filtering.
            if top_k is not None:
                k = min(top_k, next_token_logits.size(-1))

                values, indices = torch.topk(
                    next_token_logits,
                    k,
                    dim=-1,
                )

                filtered_logits = torch.full_like(
                    next_token_logits,
                    float("-inf"),
                )

                filtered_logits.scatter_(
                    -1,
                    indices,
                    values,
                )

                next_token_logits = filtered_logits

            # Top-p (nucleus) filtering.
            if top_p is not None and top_p < 1.0:

                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits,
                    descending=True,
                    dim=-1,
                )

                sorted_probs = torch.softmax(
                    sorted_logits,
                    dim=-1,
                )

                cumulative_probs = torch.cumsum(
                    sorted_probs,
                    dim=-1,
                )

                remove_tokens = cumulative_probs > top_p

                # Keep the first token that crosses the threshold.
                remove_tokens[..., 1:] = remove_tokens[..., :-1].clone()
                remove_tokens[..., 0] = False

                sorted_logits = sorted_logits.masked_fill(
                    remove_tokens,
                    float("-inf"),
                )

                filtered_logits = torch.full_like(
                    next_token_logits,
                    float("-inf"),
                )

                filtered_logits.scatter_(
                    -1,
                    sorted_indices,
                    sorted_logits,
                )

                next_token_logits = filtered_logits

            # Sample the next token.
            probabilities = torch.softmax(
                next_token_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1,
            )

            # Update KV cache using only the new token.
            logits, kv_cache = model(
                next_token,
                past_kv=kv_cache,
                start_position=tokens.shape[1] - 1,
            )

    return tokenizer.decode(tokens[0].tolist())


if __name__ == "__main__":

    tokenizer = Tokenizer.from_file(tokenizer_file)
    model = load_model()

    prompt = input("Enter a prompt: ")

    generated_text = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    print("\nGenerated text:")
    print(generated_text)
