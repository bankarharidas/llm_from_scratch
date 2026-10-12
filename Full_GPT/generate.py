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

    # print("Checkpoint keys:")
    # print(checkpoint["model_state_dict"].keys())

    # print(
    #     "Position embedding present:",
    #     any("position_embedding" in key for key in checkpoint["model_state_dict"]),
    # )

    print("Checkpoint file:", checkpoint_file)
    print("Checkpoint metadata:")

    for key, value in checkpoint.items():
        if key != "model_state_dict":
            print(f"{key}: {value}")

    print(
        "Position embedding shape:",
        checkpoint["model_state_dict"]["position_embedding.weight"].shape,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def generate(model, tokenizer, prompt, max_new_tokens=50):

    tokens = torch.tensor(
        [tokenizer.encode(prompt).ids],
        dtype=torch.long,
    )

    # Keep the prompt within the context limit.
    tokens = tokens[:, -sequence_length:]

    model.eval()

    with torch.no_grad():

        logits, kv_cache = model(
            tokens,
            past_kv=None,
            start_position=0,
        )

        for _ in range(max_new_tokens):

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1,
            )

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
        max_new_tokens=50,
    )

    print("\nGenerated text:")
    print(generated_text)
