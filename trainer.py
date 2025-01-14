import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import lineToTensor, label_to_tensor


class DomainTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.model.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def train_step(self, category_tensor, line_tensor):
        hidden = self.model.initHidden()
        self.model.zero_grad()

        for i in range(line_tensor.size(0)):
            output, hidden = self.model(line_tensor[i], hidden)

        loss = self.criterion(output, category_tensor)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return output, loss.item()

    def train(self, train_df, batch_size=320, n_epochs=10):
        all_losses = []
        best_loss = float("inf")
        n_batches = len(train_df) // batch_size + (
            1 if len(train_df) % batch_size != 0 else 0
        )

        for epoch in range(n_epochs):
            total_loss = 0
            self.model.train()

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

                    if torch.isnan(torch.tensor(loss)):
                        print(f"警告：检测到NaN损失值，跳过此样本")
                        continue

                    batch_loss += loss

                batch_avg_loss = batch_loss / len(batch)
                total_loss += batch_avg_loss

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}, Batch {batch_idx}/{n_batches}, Loss: {batch_avg_loss:.4f}"
                    )

            average_loss = total_loss / n_batches
            all_losses.append(average_loss)

            self.scheduler.step(average_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1}/{n_epochs}, Loss: {average_loss:.4f}, "
                f"Learning Rate: {current_lr:.6f}"
            )

            if average_loss < best_loss:
                best_loss = average_loss
                self.save_model("best_model.pth")

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
        torch.save(self.model.state_dict(), f"/kaggle/working/{path}")
        print(f"模型已保存到: {path}")
