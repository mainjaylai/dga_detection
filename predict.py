import torch
from model import RNN
from utils import n_letters, lineToTensor

class DomainPredictor:
    def __init__(self, model_path='dga_detector.pth'):
        # 初始化模型
        self.n_hidden = 128
        self.n_categories = 2
        self.model = RNN(n_letters, self.n_hidden, self.n_categories)
        
        # 加载训练好的模型参数
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # 设置为评估模式
        
    def predict(self, domain):
        # 将域名转换为张量
        with torch.no_grad():
            line_tensor = lineToTensor(domain)
            hidden = self.model.initHidden()
            
            # 对域名中的每个字符进行预测
            for i in range(line_tensor.size(0)):
                output, hidden = self.model(line_tensor[i], hidden)
            
            # 获取预测结果
            prob = torch.exp(output)
            _, predicted = torch.max(output, 1)
            
            return {
                'is_dga': bool(predicted.item()),
                'confidence': prob[0][predicted.item()].item()
            }

def main():
    # 创建预测器实例
    predictor = DomainPredictor()
    
    # 测试一些域名
    test_domains = [
        'google.com',
        'facebook.com',
        'asd7f6as8df76.com',
        'djf8s7df6g.net'
    ]
    
    print("域名检测结果：")
    print("-" * 50)
    for domain in test_domains:
        result = predictor.predict(domain)
        status = "恶意域名" if result['is_dga'] else "正常域名"
        print(f"域名: {domain}")
        print(f"预测结果: {status}")
        print(f"置信度: {result['confidence']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main() 