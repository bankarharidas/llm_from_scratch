import torch
import torch.nn.functional as F
from GPT import MiniGPT

text = """
hello world
hello cat
hello dog
the cat sat on the mat
the dog sat on the floor
""" * 100

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return [stoi[ch] for ch in text]


def decode(tokens):
    return "".join(itos[token] for token in tokens)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


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

model = MiniGPT(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    num_heads=num_heads,
    num_blocks=num_blocks,
    sequence_length=sequence_length,
    ffn_hidden_size=ffn_hidden_size,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
num_steps = 1000

for step in range(num_steps):
    inputs, targets = get_batch(train_data, batch_size, sequence_length)

    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: Loss {loss.item():.4f}")

model.eval()
with torch.no_grad():
    val_inputs, val_targets = get_batch(val_data, batch_size, sequence_length)
    val_logits = model(val_inputs)
    val_loss = F.cross_entropy(
        val_logits.reshape(-1, vocab_size), val_targets.reshape(-1)
    )

print("Validation Loss:", val_loss.item())
model.train()


def generate(model, prompt, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    model.eval()
    tokens = torch.tensor([encode(prompt)], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            tokens_cond = tokens[:, -sequence_length:]
            logits = model(tokens_cond)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, indices = torch.topk(next_token_logits, top_k)
                filtered_logits = torch.full_like(next_token_logits, float("-inf"))
                filtered_logits.scatter_(1, indices, values)
                next_token_logits = filtered_logits

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
                cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

                remove_tokens = cumulative_probabilities > top_p
                remove_tokens[:, 1:] = remove_tokens[:, :-1].clone()
                remove_tokens[:, 0] = False

                sorted_logits = sorted_logits.masked_fill(remove_tokens, float("-inf"))
                next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                next_token_logits.scatter_(1, sorted_indices, sorted_logits)

            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

    generated_text = decode(tokens[0].tolist())
    model.train()
    return generated_text


print(generate(model, "the ", max_new_tokens=100, temperature=0.8, top_k=5, top_p=0.9))

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
print("Checkpoint saved!")

checkpoint = torch.load("mini_gpt_checkpoint.pth", weights_only=False)

loaded_model = MiniGPT(
    vocab_size=checkpoint["vocab_size"],
    embedding_size=checkpoint["embedding_size"],
    num_heads=checkpoint["num_heads"],
    num_blocks=checkpoint["num_blocks"],
    sequence_length=checkpoint["sequence_length"],
    ffn_hidden_size=checkpoint["ffn_hidden_size"],
)
loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_model.eval()

result = generate(
    loaded_model, "the ", max_new_tokens=100, temperature=0.8, top_k=5, top_p=0.9
)
print(result)
