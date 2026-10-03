import torch

# from Tokenizer import chars
from encoder import encode, chars

itos = {i: ch for i, ch in enumerate(chars)}


# print(itos)
def decode(tokens):
    return "".join(itos[token] for token in tokens)
