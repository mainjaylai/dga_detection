import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    # 定义列名
    columns = ['domain', 'label']
    
    # 读取数据
    dga_df = pd.read_csv('data/all_dga.txt', sep=' ', header=None, names=columns)
    dga_df['label'] = 1
    
    legit_df = pd.read_csv('data/all_legit.txt', sep=' ', header=None, names=columns)
    legit_df['label'] = 0
    
    # 合并数据
    all_df = pd.concat([dga_df, legit_df], ignore_index=True)
    
    # 分割数据集
    train_df, test_df = train_test_split(all_df, test_size=0.2, random_state=42)
    
    return train_df, test_df 