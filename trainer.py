import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import lineToTensor, label_to_tensor


class DomainTrainer:
    def __init__(self, model, learning_rate=0.003):
        self.model = model
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def train_step(self, category_tensor, line_tensor):
        hidden = self.model.initHidden()
        self.model.zero_grad()

        for i in range(line_tensor.size(0)):
            output, hidden = self.model(line_tensor[i], hidden)

        loss = self.criterion(output, category_tensor)
        loss.backward()
        self.optimizer.step()

        return output, loss.item()

    def train(self, train_df, batch_size=320, n_epochs=10):
        all_losses = []
        n_batches = len(train_df) // batch_size + (
            1 if len(train_df) % batch_size != 0 else 0
        )

        for epoch in range(n_epochs):
            total_loss = 0
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(train_df))
                batch = train_df.iloc[start_idx:end_idx]

                batch_loss = 0
                for index, row in batch.iterrows():
                    domain = row["domain"]
                    label = row["label"]
                    line_tensor = lineToTensor(domain)
                    category_tensor = label_to_tensor(label)
                    _, loss = self.train_step(category_tensor, line_tensor)
                    batch_loss += loss

                # 计算批次平均损失
                batch_avg_loss = batch_loss / len(batch)
                total_loss += batch_avg_loss

            # 计算一个 epoch 的平均损失
            average_loss = total_loss / n_batches
            all_losses.append(average_loss)

            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {average_loss:.4f}")

        return all_losses

    def plot_losses(self, losses):
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Loss")
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, path="dga_detector.pth"):
        # 保存模型
        torch.save(self.model.state_dict(), f"/kaggle/working/{path}")
        print(f"模型已保存到: {path}")
