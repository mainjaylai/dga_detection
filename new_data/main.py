# 导入必要的库
import csv


def read_dga_feed(file_path):
    # 打开文件并读取内容
    with open(file_path, mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file, delimiter=",")
        count = 0
        set_dga_classes = set()
        for row in csv_reader:
            # 处理每一行数据
            # print(len(row))
            count += 1
            set_dga_classes.add(row[1])
        print(count)
        print(set_dga_classes)


# 调用函数并传入文件路径
read_dga_feed("/Users/mainjay/Downloads/dga_detection/new_data/dga-feed-high")
