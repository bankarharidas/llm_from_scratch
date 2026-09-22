import torch

tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

sequence_length = 4

inputs = []
targets = []

for i in range(len(tokens) - sequence_length):
    inputs_chunk = tokens[i : i + 4]
    targets_chunk = tokens[i + 1 : i + 5]

    inputs.append(inputs_chunk)
    targets.append(targets_chunk)

# print(inputs)
# print(targets)
inputs = torch.stack(inputs)
targets = torch.stack(targets)

print(inputs.shape)
print(targets.shape)

batch_size = 2

batch_inputs = inputs[:batch_size]
batch_targets = targets[:batch_size]

print(batch_inputs)
print(batch_targets)
