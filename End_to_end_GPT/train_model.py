# import torch
# from GPT import MiniGPT
# from Tokenizer import train_data, val_data
# import torch.nn.functional as F


# def get_batch(data, batch_size, sequence_length):

#     starts = torch.randint(0, len(data) - sequence_length - 1, (batch_size,))

#     x = torch.stack([data[i : i + sequence_length] for i in starts])

#     y = torch.stack([data[i + 1 : i + sequence_length + 1] for i in starts])

#     return x, y


# batch_size = 8
# sequence_length = 32

# x, y = get_batch(train_data, batch_size, sequence_length)


# # -------------------------
# # Model configuration
# # -------------------------

# vocab_size = 17
# embedding_size = 128
# num_heads = 4
# num_blocks = 4
# sequence_length = 128
# ffn_hidden_size = 512

# # -------------------------
# # Create model
# # -------------------------

# model = MiniGPT(
#     vocab_size=vocab_size,
#     embedding_size=embedding_size,
#     num_heads=num_heads,
#     num_blocks=num_blocks,
#     sequence_length=sequence_length,
#     ffn_hidden_size=ffn_hidden_size,
# )

# batch_size = 8

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# num_steps = 1000

# for step in range(num_steps):

#     # Get a fresh batch
#     inputs, targets = get_batch(train_data, batch_size=8, sequence_length=128)

#     # Forward
#     logits = model(inputs)

#     # Loss
#     loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

#     # Backward
#     optimizer.zero_grad()
#     loss.backward()

#     # Gradient clipping
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#     # Update weights
#     optimizer.step()

#     # Print progress
#     if step % 100 == 0:
#         print(f"Step {step}: " f"Loss {loss.item():.4f}")

# model.eval()

# with torch.no_grad():

#     val_inputs, val_targets = get_batch(val_data, batch_size=8, sequence_length=128)

#     val_logits = model(val_inputs)

#     val_loss = F.cross_entropy(
#         val_logits.reshape(-1, vocab_size), val_targets.reshape(-1)
#     )

# print("Validation Loss:", val_loss.item())

# model = model.train()

# print("Training step completed!", model)


# def generate(model, prompt, max_new_tokens, temperature=1.0, top_k=None, top_p=None):

#     model.eval()

#     tokens = torch.tensor([encode(prompt)], dtype=torch.long)

#     with torch.no_grad():

#         for _ in range(max_new_tokens):

#             tokens_cond = tokens[:, -sequence_length:]

#             logits = model(tokens_cond)

#             next_token_logits = logits[:, -1, :]

#             # -------------------------
#             # Temperature
#             # -------------------------

#             next_token_logits = next_token_logits / temperature

#             # -------------------------
#             # Top-k
#             # -------------------------

#             if top_k is not None:

#                 values, indices = torch.topk(next_token_logits, top_k)

#                 filtered_logits = torch.full_like(next_token_logits, float("-inf"))

#                 filtered_logits.scatter_(1, indices, values)

#                 next_token_logits = filtered_logits

#             # -------------------------
#             # Top-p
#             # -------------------------

#             if top_p is not None:

#                 sorted_logits, sorted_indices = torch.sort(
#                     next_token_logits, descending=True
#                 )

#                 sorted_probabilities = torch.softmax(sorted_logits, dim=-1)

#                 cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

#                 remove_tokens = cumulative_probabilities > top_p

#                 # Keep the first token that crosses top_p
#                 remove_tokens[:, 1:] = remove_tokens[:, :-1].clone()

#                 remove_tokens[:, 0] = False

#                 sorted_logits = sorted_logits.masked_fill(remove_tokens, float("-inf"))

#                 next_token_logits = torch.full_like(next_token_logits, float("-inf"))

#                 next_token_logits.scatter_(1, sorted_indices, sorted_logits)

#             # -------------------------
#             # Probabilities
#             # -------------------------

#             probabilities = torch.softmax(next_token_logits, dim=-1)

#             # -------------------------
#             # Sample
#             # -------------------------

#             next_token = torch.multinomial(probabilities, num_samples=1)

#             tokens = torch.cat([tokens, next_token], dim=1)

#     generated_text = decode(tokens[0].tolist())

#     model.train()

#     return generated_text
