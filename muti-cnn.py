import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

from muti_cnn import MultiCNN


def load_data():
    # 定义列名
    columns = ["domain", "label"]

    # 读取数据
    dga_df = pd.read_csv("test.txt", sep=" ", header=None, names=columns)
    dga_df["label"] = dga_df["label"].apply(lambda x: int(x))

    train_df, test_df = train_test_split(
        dga_df, test_size=0.2, random_state=42, stratify=dga_df["label"]
    )
    return train_df, test_df


def tokenize_domain(domain):

    valid_characters = "$abcdefghijklmnopqrstuvwxyz0123456789-_."
    tokens = {char: idx for idx, char in enumerate(valid_characters)}
    domain = domain.lower()
    domain_encoded = [tokens[char] for char in domain if char in tokens]

    # 如果token长度大于 45，则截取后 45个，小于 45 在前面补上 0
    if len(domain_encoded) > 45:
        domain_encoded = domain_encoded[-45:]
    else:
        domain_encoded = [0] * (45 - len(domain_encoded)) + domain_encoded
    return domain_encoded


# 定义一个自定义数据集类
class DomainDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        domain = self.dataframe.iloc[idx]["domain"]
        label = self.dataframe.iloc[idx]["label"]
        domain_encoded = tokenize_domain(domain)
        return torch.tensor(domain_encoded, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    loss_history = []  # 用于记录每个 epoch 的损失
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()  # 确保输出是 1D
            # labels = labels.float()  # 确保标签是浮点数
            # 将标签保持为 Long 类型
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)  # 记录损失
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_history, marker="o", linestyle="-")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")  # 保存图像为文件
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "multi_cnn_model.pth")
    print("模型已保存为 multi_cnn_model.pth")


def predict(model_path, domain):
    # 加载模型
    model = MultiCNN(
        input_length=45,
        vocab_size=len("$abcdefghijklmnopqrstuvwxyz0123456789-_.") + 1,
        embedding_dim=128,
        num_filters=64,
        kernel_size=3,
        hidden_size=128,
        num_classes=8,  # 确保与训练时的类别数一致
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 处理输入域名
    domain_encoded = tokenize_domain(domain)
    input_tensor = torch.tensor(domain_encoded, dtype=torch.long).unsqueeze(
        0
    )  # 增加批次维度

    # 进行预测
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return predicted_class


def train():
    # 加载数据
    train_df, test_df = load_data()

    # 创建数据集和数据加载器
    train_dataset = DomainDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = MultiCNN(
        input_length=45,
        vocab_size=len("$abcdefghijklmnopqrstuvwxyz0123456789-_.") + 1,
        embedding_dim=128,
        num_filters=64,
        kernel_size=3,
        hidden_size=128,
        num_classes=8,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)


predicted_label = predict("multi_cnn_model.pth", "andpoliticalstatesthe.com")
print(f"预测标签: {predicted_label}")
