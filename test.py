import torch

@torch.compile(fullgraph=True)
def print_hi():
    print('hi')

print_hi()
