# config.py

# Tokenizer and data
vocab_size = 8000
tokenizer_file = "tokenizer.json"
data_file = "input.txt"

# Model architecture
embedding_size = 128
num_heads = 4
num_blocks = 4
sequence_length = 128
ffn_hidden_size = 512

# Training settings
batch_size = 8
learning_rate = 3e-4
min_learning_rate = 3e-5
warmup_steps = 500
num_steps = 10000

# Checkpoint
checkpoint_file = "best_model_kv.pth"
