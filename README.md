# DGA域名检测系统

[English Version](README.en.md)

## 项目简介
这是一个基于RNN（循环神经网络）的DGA（域名生成算法）检测系统。该系统能够自动识别域名是否为DGA生成的恶意域名。

## 功能特点
- 使用RNN模型进行域名分类
- 支持实时域名检测
- 提供预训练模型
- 可视化训练过程
- 高准确率的检测结果

## 数据集
训练数据集包含两部分：
- DGA生成的恶意域名
- 正常域名

您可以从[Google Drive](https://drive.google.com/drive/folders/162irE43DZLGPOyukwLIFv8mu3vIAGO0p?usp=sharing)下载完整数据集。

## 环境要求
- Python 3.7+
- PyTorch
- pandas
- matplotlib
- scikit-learn

## 安装步骤
1. 克隆仓库
```bash
git clone https://github.com/yourusername/dga-detection.git
cd dga-detection
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 下载数据集并放置在`data`目录下

## 使用方法
1. 训练模型
```bash
python main.py
```

2. 预测域名
```bash
python predict.py
```

## 项目结构
```
.
├── data/               # 数据目录
├── main.py            # 主程序
├── model.py           # 模型定义
├── trainer.py         # 训练器
├── predict.py         # 预测脚本
├── utils.py           # 工具函数
└── data_loader.py     # 数据加载器
```

## 许可证
MIT License

## 贡献指南
欢迎提交Pull Request或Issue。

## 联系方式
如有任何问题，请提交Issue。