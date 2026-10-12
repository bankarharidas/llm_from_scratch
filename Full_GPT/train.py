# train.py

import math
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import (
    vocab_size,
    embedding_size,
    num_heads,
    num_blocks,
    sequence_length,
    ffn_hidden_size,
    data_file,
    tokenizer_file,
)

from model import MiniGPT

# -------------------------
# 1. Device and random seed
# -------------------------

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# -------------------------
# 2. Load tokenizer and data
# -------------------------

tokenizer = Tokenizer.from_file(tokenizer_file)

with open(data_file, "r", encoding="utf-8") as file:
    text = file.read()

token_ids = tokenizer.encode(text).ids

data = torch.tensor(token_ids, dtype=torch.long)

split_index = int(0.9 * len(data))

train_data = data[:split_index]
val_data = data[split_index:]

print("Vocabulary size:", tokenizer.get_vocab_size())
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))

assert (
    tokenizer.get_vocab_size() == vocab_size
), "Tokenizer vocabulary does not match config.py"


# -------------------------
# 3. Batch generator
# -------------------------

batch_size = 8


def get_batch(source):

    starts = torch.randint(0, len(source) - sequence_length - 1, (batch_size,))

    x = torch.stack([source[i : i + sequence_length] for i in starts])

    y = torch.stack([source[i + 1 : i + sequence_length + 1] for i in starts])

    return x.to(device), y.to(device)


# -------------------------
# 4. Create a fresh RoPE model
# -------------------------

model = MiniGPT(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    num_heads=num_heads,
    num_blocks=num_blocks,
    sequence_length=sequence_length,
    ffn_hidden_size=ffn_hidden_size,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

print("Starting from fresh weights. No old checkpoint loaded.")


# -------------------------
# 5. Training settings
# -------------------------

num_steps = 10000
eval_interval = 500
warmup_steps = 500

max_learning_rate = 3e-4
min_learning_rate = 3e-5

best_val_loss = float("inf")
checkpoint_path = "best_model_rope.pth"


def get_learning_rate(step):

    if step < warmup_steps:
        return max_learning_rate * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / (num_steps - warmup_steps)

    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine


@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for name, source in [("train", train_data), ("val", val_data)]:

        losses = []

        for _ in range(20):

            x, y = get_batch(source)

            logits, _ = model(x)

            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


# -------------------------
# 6. Training loop
# -------------------------

model.train()

for step in range(num_steps):

    lr = get_learning_rate(step)

    for group in optimizer.param_groups:
        group["lr"] = lr

    inputs, targets = get_batch(train_data)

    logits, _ = model(inputs)

    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    if step % eval_interval == 0 or step == num_steps - 1:

        losses = estimate_loss()

        print(
            f"Step {step:5d} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f} | "
            f"lr {lr:.2e}"
        )

        if losses["val"] < best_val_loss:

            best_val_loss = losses["val"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "vocab_size": vocab_size,
                    "embedding_size": embedding_size,
                    "num_heads": num_heads,
                    "num_blocks": num_blocks,
                    "sequence_length": sequence_length,
                    "ffn_hidden_size": ffn_hidden_size,
                    "tokenizer_file": tokenizer_file,
                    "position_encoding": "RoPE",
                },
                checkpoint_path,
            )

            print("Saved new best RoPE checkpoint.")

print("\nTraining complete.")
print("Best validation loss:", best_val_loss)
print("Checkpoint:", checkpoint_path)
