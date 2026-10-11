# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tokenizers import Tokenizer
# from GPT import MiniGPT


# def main():
#     # =========================
#     # Configuration
#     # =========================

#     batch_size = 8
#     sequence_length = 128

#     embedding_size = 128
#     num_heads = 4
#     num_blocks = 4
#     ffn_hidden_size = 512

#     # =========================
#     # Learning Rate Schedule
#     # =========================
#     num_steps = 10001
#     learning_rate = 3e-4

#     warmup_steps = 500
#     decay_steps = num_steps
#     min_learning_rate = 3e-5

#     # =========================
#     # Tokenizer
#     # =========================

#     with open("input.txt", "r", encoding="utf-8") as f:
#         text = f.read()

#     tokenizer = Tokenizer.from_file("tokenizer.json")

#     vocab_size = tokenizer.get_vocab_size()

#     def encode(text):

#         encoded = tokenizer.encode(text)

#         return encoded.ids

#     def decode(tokens):

#         return tokenizer.decode(tokens)

#     # =========================
#     # Dataset
#     # =========================

#     data = torch.tensor(encode(text), dtype=torch.long)

#     n = int(0.9 * len(data))

#     train_data = data[:n]
#     val_data = data[n:]

#     def get_batch(data, batch_size, sequence_length):

#         if len(data) <= sequence_length + 1:
#             raise ValueError(
#                 f"Dataset is too small: "
#                 f"{len(data)} characters. "
#                 f"Need more than {sequence_length + 1}."
#             )

#         starts = torch.randint(0, len(data) - sequence_length - 1, (batch_size,))

#         x = torch.stack([data[i : i + sequence_length] for i in starts])

#         y = torch.stack([data[i + 1 : i + sequence_length + 1] for i in starts])

#         return x, y

#     print("Vocabulary size:", vocab_size)
#     print("Data:", data.shape)
#     print("Train:", train_data.shape)
#     print("Validation:", val_data.shape)

#     # Create model
#     model = MiniGPT(
#         vocab_size=vocab_size,
#         embedding_size=embedding_size,
#         num_heads=num_heads,
#         num_blocks=num_blocks,
#         sequence_length=sequence_length,
#         ffn_hidden_size=ffn_hidden_size,
#     )

#     # print("Parameters:", sum(p.numel() for p in model.parameters()))

#     # Optimizer
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#     print("Optimizer ready!")
