import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import string
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 定义特殊符号和字符集（PAD 用于填充，UNK 用于未知字符）
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
valid_chars = string.ascii_letters + string.digits + "-."
all_letters = PAD_TOKEN + valid_chars + UNK_TOKEN
n_letters = len(all_letters)


def letterToIndex(letter):
    """
    将字符映射到对应的索引。如果字符不在 valid_chars 中，则返回 UNK_TOKEN 的索引。
    """
    if letter in valid_chars:
        return all_letters.find(letter)
    else:
        return all_letters.find(UNK_TOKEN)


def lineToIndices(line):
    """
    将字符串转换为字符索引列表的张量（不进行填充）。
    """
    indices = [letterToIndex(letter) for letter in line]
    return torch.tensor(indices, dtype=torch.long)


# 定义数据集类，用于 DataLoader
class DomainDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        domain = row["domain"]
        label = row["label"]
        indices = lineToIndices(domain)
        return indices, label


def collate_fn(batch):
    """
    自定义 collate_fn，用于对批量数据进行填充。
    输入：batch 是一个 list，每个元素为 (indices, label)
    输出：
        padded_sequences: LongTensor，形状 [batch_size, max_seq_len]
        lengths: LongTensor，记录每个序列的真实长度
        labels: LongTensor，形状 [batch_size]
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=0
    )  # 使用 PAD_TOKEN 对应的索引 0
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, lengths, labels


# 定义改进后的 DGA 分类器模型（采用字符嵌入、双向 LSTM 和注意力机制）
class DGAClassifier(nn.Module):
    def __init__(
        self,
        n_letters,
        embedding_dim,
        hidden_size,
        output_size,
        num_layers=2,
        dropout=0.5,
        bidirectional=True,
    ):
        """
        参数说明：
            n_letters: 字符集大小
            embedding_dim: 字符嵌入维度
            hidden_size: LSTM 隐藏层维度
            output_size: 分类类别数（如 2）
            num_layers: LSTM 层数
            dropout: 除最后一层外的 dropout 概率
            bidirectional: 是否使用双向 LSTM
        """
        super(DGAClassifier, self).__init__()
        self.embedding = nn.Embedding(n_letters, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.bidirectional = bidirectional
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Linear(lstm_output_size, 1)
        self.fc = nn.Linear(lstm_output_size, output_size)

    def forward(self, x, lengths=None):
        """
        输入：
            x: LongTensor, [batch_size, seq_len]
            lengths: LongTensor, [batch_size] 表示每个序列的真实长度
        输出：
            log_probs: [batch_size, output_size]
        """
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        if lengths is not None:
            # 按真实长度打包序列，忽略填充部分
            packed_embedded = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed_embedded)
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)

        # 创建 mask（True 表示有效位置）
        batch_size, max_seq_len, _ = lstm_out.size()
        # 若 lengths 为空则默认所有位置有效
        if lengths is None:
            mask = torch.ones(
                batch_size, max_seq_len, dtype=torch.bool, device=x.device
            )
        else:
            mask = torch.arange(max_seq_len, device=x.device).unsqueeze(
                0
            ) < lengths.unsqueeze(1)

        # 计算注意力分数
        attn_scores = self.attention(lstm_out).squeeze(-1)  # [batch_size, seq_len]
        # 将填充部分的注意力分数置为极小值，避免对 softmax 产生影响
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(
            -1
        )  # [batch_size, seq_len, 1]
        # 计算上下文向量
        context = torch.sum(
            lstm_out * attn_weights, dim=1
        )  # [batch_size, lstm_output_size]
        output = self.fc(context)  # [batch_size, output_size]
        log_probs = F.log_softmax(output, dim=1)
        return log_probs


# 定义批量训练的训练器类
class DomainTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.model.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def train_step_batch(self, batch_sequences, lengths, batch_labels):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(batch_sequences, lengths)  # [batch_size, output_size]
        loss = self.criterion(outputs, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return outputs, loss.item()

    def train(self, train_df, batch_size=64, n_epochs=10):
        dataset = DomainDataset(train_df)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        all_losses = []
        best_loss = float("inf")
        n_batches = len(dataloader)

        for epoch in range(n_epochs):
            total_loss = 0.0
            for batch_sequences, lengths, batch_labels in dataloader:
                # 若有必要，将数据移动到设备上（例如 GPU）
                # batch_sequences, lengths, batch_labels = batch_sequences.to(device), lengths.to(device), batch_labels.to(device)
                _, loss = self.train_step_batch(batch_sequences, lengths, batch_labels)
                if torch.isnan(torch.tensor(loss)):
                    print("警告：检测到 NaN 损失值，跳过此批次")
                    continue
                total_loss += loss
            average_loss = total_loss / n_batches
            all_losses.append(average_loss)
            self.scheduler.step(average_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{n_epochs}, Loss: {average_loss:.4f}, Learning Rate: {current_lr:.6f}"
            )
            if average_loss < best_loss:
                best_loss = average_loss
                self.save_model("best_model.pth")
        return all_losses

    def plot_losses(self, losses):
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Training Loss")
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, path="dga_detector.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到: {path}")


# 主函数：读取数据、构建模型、批量训练并保存模型
def main():
    # 定义数据列名
    columns = ["domain", "label"]

    # 读取 DGA 数据（标签 1）和正常域名数据（标签 0）
    dga_df = pd.read_csv(
        "/kaggle/input/dga-detection/all_dga.txt",
        sep=" ",
        header=None,
        names=columns,
    )
    dga_df["label"] = 1
    legit_df = pd.read_csv(
        "/kaggle/input/dga-detection/all_legit.txt", sep=" ", header=None, names=columns
    )
    legit_df["label"] = 0

    # 合并数据并划分训练集和测试集
    all_df = pd.concat([dga_df, legit_df], ignore_index=True)
    train_df, test_df = train_test_split(all_df, test_size=0.2, random_state=42)

    # 模型参数配置
    embedding_dim = 50
    hidden_size = 128
    n_categories = 2
    num_layers = 2
    dropout = 0.5
    bidirectional = True

    # 初始化模型和训练器
    model = DGAClassifier(
        n_letters,
        embedding_dim,
        hidden_size,
        n_categories,
        num_layers,
        dropout,
        bidirectional,
    )
    trainer = DomainTrainer(model, learning_rate=0.001)

    # 批量训练模型
    losses = trainer.train(train_df, batch_size=64, n_epochs=10)

    # 绘制损失曲线
    trainer.plot_losses(losses)

    # 保存最终模型
    trainer.save_model("/kaggle/working/best_model.pth")


if __name__ == "__main__":
    main()
