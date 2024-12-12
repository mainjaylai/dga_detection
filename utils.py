import string
import torch

# 定义字符集
all_letters = string.ascii_letters + string.digits + "-."
n_letters = len(all_letters)

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def label_to_tensor(label):
    return torch.tensor([label], dtype=torch.long) 