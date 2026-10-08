import torch
from GPT import MiniGPT
from Tokenizer import chars, encode, decode, model, sequence_length

vocab_size = len(chars)


# -------------------------
# Model configuration
# -------------------------

# batch_size = 8
# embedding_size = 128
# num_heads = 4
# num_blocks = 4
# sequence_length = 128
# ffn_hidden_size = 512

checkpoint = torch.load("mini_gpt_checkpoint.pth", weights_only=False)

loaded_model = model

# loaded_model = MiniGPT(
#     vocab_size=vocab_size,
#     embedding_size=embedding_size,
#     num_heads=num_heads,
#     num_blocks=num_blocks,
#     sequence_length=sequence_length,
#     ffn_hidden_size=ffn_hidden_size,
# )


def generate(model, prompt, max_new_tokens, temperature=1.0, top_k=None, top_p=None):

    model.eval()

    tokens = torch.tensor([encode(prompt)], dtype=torch.long)

    with torch.no_grad():

        for _ in range(max_new_tokens):

            tokens_cond = tokens[:, -sequence_length:]

            logits = model(tokens_cond)

            next_token_logits = logits[:, -1, :]

            # -------------------------
            # Temperature
            # -------------------------

            next_token_logits = next_token_logits / temperature

            # -------------------------
            # Top-k
            # -------------------------

            if top_k is not None:

                values, indices = torch.topk(next_token_logits, top_k)

                filtered_logits = torch.full_like(next_token_logits, float("-inf"))

                filtered_logits.scatter_(1, indices, values)

                next_token_logits = filtered_logits

            # -------------------------
            # Top-p
            # -------------------------

            if top_p is not None:

                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )

                sorted_probabilities = torch.softmax(sorted_logits, dim=-1)

                cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

                remove_tokens = cumulative_probabilities > top_p

                # Keep the first token that crosses top_p
                remove_tokens[:, 1:] = remove_tokens[:, :-1].clone()

                remove_tokens[:, 0] = False

                sorted_logits = sorted_logits.masked_fill(remove_tokens, float("-inf"))

                next_token_logits = torch.full_like(next_token_logits, float("-inf"))

                next_token_logits.scatter_(1, sorted_indices, sorted_logits)

            # -------------------------
            # Probabilities
            # -------------------------

            probabilities = torch.softmax(next_token_logits, dim=-1)

            # -------------------------
            # Sample
            # -------------------------

            next_token = torch.multinomial(probabilities, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

    generated_text = decode(tokens[0].tolist())

    model.train()

    return generated_text


loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_model.eval()

loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=3e-4)

loaded_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

step = checkpoint["step"]

print("Loaded step:", step)

chars = checkpoint["chars"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

result = generate(
    loaded_model,
    "My clerk hath some good comforts too for you ",
    max_new_tokens=100,
    temperature=0.8,
    top_k=5,
    top_p=0.9,
)

print(result)
