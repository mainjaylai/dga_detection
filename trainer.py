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
    
    def train(self, train_df, n_epochs=100):
        all_losses = []
        
        for epoch in range(n_epochs):
            total_loss = 0
            for index, row in train_df.iterrows():
                domain = row['domain']
                label = row['label']
                line_tensor = lineToTensor(domain)
                category_tensor = label_to_tensor(label)
                _, loss = self.train_step(category_tensor, line_tensor)
                total_loss += loss
                
            average_loss = total_loss / len(train_df)
            all_losses.append(average_loss)
            
            if (epoch + 1) % 10 == 0:
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
    
    def save_model(self, path='dga_detector.pth'):
        # 保存模型
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到: {path}") 