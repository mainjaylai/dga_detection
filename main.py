from data_loader import load_data
from model import RNN
from trainer import DomainTrainer
from utils import n_letters

def main():
    # 加载数据
    train_df, test_df = load_data()
    
    # 初始化模型
    n_hidden = 128
    n_categories = 2
    model = RNN(n_letters, n_hidden, n_categories)
    
    # 创建训练器
    trainer = DomainTrainer(model)
    
    # 训练模型
    losses = trainer.train(train_df)
    
    # 绘制损失曲线
    trainer.plot_losses(losses)
    
    # 保存模型
    trainer.save_model()

if __name__ == "__main__":
    main()
