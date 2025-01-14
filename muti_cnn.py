import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MultiCNN(nn.Module):
    def __init__(
        self,
        input_length,
        vocab_size,
        embedding_dim,
        num_filters,
        kernel_size,
        hidden_size,
        num_classes,
    ):
        super(MultiCNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size
        )
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(
            num_filters * (input_length - kernel_size + 1), hidden_size
        )
        self.dense2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # 调整维度以适应Conv1d
        x = F.relu(self.conv1d(x))
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x


# 示例用法
# model = MultiCNN(input_length=100, vocab_size=5000, embedding_dim=128, num_filters=64, kernel_size=3, hidden_size=128)
# output = model(torch.randint(0, 5000, (32, 100)))  # 假设批次大小为32，输入长度为100

# 初始化模型、损失函数和优化器
model = MultiCNN(
    input_length=45,
    vocab_size=len("$abcdefghijklmnopqrstuvwxyz0123456789-_.") + 1,
    embedding_dim=128,
    num_filters=64,
    kernel_size=3,
    hidden_size=128,
    num_classes=3,  # 假设有3个类别
)
criterion = nn.CrossEntropyLoss()  # 使用 CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)
