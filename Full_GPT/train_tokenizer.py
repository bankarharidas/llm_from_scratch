from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# =========================
# Create BPE Tokenizer
# =========================

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

tokenizer.decoder = ByteLevelDecoder()


# =========================
# Trainer
# =========================

trainer = BpeTrainer(vocab_size=8000, min_frequency=2, special_tokens=["<UNK>"])


# =========================
# Train
# =========================

tokenizer.train(["input.txt"], trainer)


# =========================
# Save
# =========================

tokenizer.save("tokenizer.json")

print("Tokenizer trained!")

print("Vocabulary size:", tokenizer.get_vocab_size())


# =========================
# Test
# =========================

test_text = "The transformer model learns language."

encoded = tokenizer.encode(test_text)

print()
print("Original:")
print(test_text)

print()
print("Token IDs:")
print(encoded.ids)

print()
print("Tokens:")
print(encoded.tokens)

print()
print("Decoded:")

print(tokenizer.decode(encoded.ids))
