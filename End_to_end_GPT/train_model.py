import torch
from GPT import MiniGPT
from Tokenizer import vocab_size, chars, stoi, itos, data, model
import torch.nn.functional as F

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

print("Train:", train_data.shape)
print("Validation:", val_data.shape)


def get_batch(data, batch_size, sequence_length):

    starts = torch.randint(0, len(data) - sequence_length - 1, (batch_size,))

    x = torch.stack([data[i : i + sequence_length] for i in starts])

    y = torch.stack([data[i + 1 : i + sequence_length + 1] for i in starts])

    return x, y


# -------------------------
# Model configuration
# -------------------------

batch_size = 8
embedding_size = 128
num_heads = 4
num_blocks = 4
sequence_length = 128
ffn_hidden_size = 512


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

num_steps = 3001

for step in range(num_steps):

    # Get a fresh batch
    inputs, targets = get_batch(train_data, batch_size, sequence_length)

    # Forward
    logits = model(inputs)

    # Loss
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update weights
    optimizer.step()

    # Print progress
    if step % 100 == 0:
        print(f"Step {step}: " f"Loss {loss.item():.4f}")

model.eval()

with torch.no_grad():

    val_inputs, val_targets = get_batch(val_data, batch_size, sequence_length)

    val_logits = model(val_inputs)

    val_loss = F.cross_entropy(
        val_logits.reshape(-1, vocab_size), val_targets.reshape(-1)
    )

print("Validation Loss:", val_loss.item())

model = model.train()

print("Training step completed!", model)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "step": step,
    "vocab_size": vocab_size,
    "embedding_size": embedding_size,
    "num_heads": num_heads,
    "num_blocks": num_blocks,
    "sequence_length": sequence_length,
    "ffn_hidden_size": ffn_hidden_size,
    "chars": chars,
    "stoi": stoi,
    "itos": itos,
}

torch.save(checkpoint, "mini_gpt_checkpoint.pth")

print("Model saved!")
