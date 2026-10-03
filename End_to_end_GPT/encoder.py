# from Tokenizer import chars

text = """
hello world
hello cat
hello dog
the cat sat on the mat
the dog sat on the floor
""" * 100

chars = sorted(set(text))

chars.append("<EOS>")

# print(chars)
# ["\n", " ", "a", "c", "d", "e", "f", "g", "h", "l", "m", "n", "o", "r", "s", "t", "w"]

chars = chars

vocab_size = len(chars)

EOS_TOKEN = vocab_size
vocab_size = vocab_size + 1


def encode_with_eos(text):
    tokens = [stoi[ch] for ch in text]
    tokens.append(EOS_TOKEN)
    return tokens


stoi = {ch: i for i, ch in enumerate(chars)}

encoded = encode_with_eos("hello")


def encode(text):
    return [stoi[ch] for ch in text]


#    {"\n": 0,
#     " ": 1,
#     "a": 2,
#     "c": 3,
#     "d": 4,
#     "e": 5,
#     "f": 6,
#     "g": 7,
#     "h": 8,
#     "l": 9,
#     "m": 10,
#     "n": 11,
#     "o": 12,
#     "r": 13,
#     "s": 14,
#     "t": 15,
#     "w": 16,}
