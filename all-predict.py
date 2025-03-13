import json
import torch
import string
import torch.nn as nn
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import torch.nn.functional as F

dga_map = {
    1: "基于时间的生成算法",
    2: "随机字符生成算法",
    3: "基于字典的生成算法",
    4: "基于时间+字典组合",
    5: "随机化生成",
    6: "基于哈希生成",
    7: "基于词汇生成",
    8: "基于重复字母的模式",
}

# 定义字符集
all_letters = string.ascii_letters + string.digits + "-."
n_letters = len(all_letters)


def letterToIndex(letter):
    return all_letters.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


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


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def label_to_tensor(label):
    return torch.tensor([label], dtype=torch.long)


class MultiCNN(nn.Module):
    def __init__(
        self,
        input_length,
        vocab_size,
        embedding_dim,
        num_filters,
        kernel_size,
        hidden_size,
        num_classes,
    ):
        super(MultiCNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size
        )
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(
            num_filters * (input_length - kernel_size + 1), hidden_size
        )
        self.dense2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # 调整维度以适应Conv1d
        x = F.relu(self.conv1d(x))
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x


model = MultiCNN(
    input_length=45,
    vocab_size=len("$abcdefghijklmnopqrstuvwxyz0123456789-_.") + 1,
    embedding_dim=128,
    num_filters=64,
    kernel_size=3,
    hidden_size=128,
    num_classes=8,  # 确保与训练时的类别数一致
)
model.load_state_dict(torch.load("multi_cnn_model.pth"))
model.eval()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class DomainPredictor:
    def __init__(self, model_path="dga_detector.pth"):
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

            if predicted.item() == 0:
                return {
                    "result": 0,
                    "domain": domain,
                }
            else:
                with torch.no_grad():
                    domain_encoded = tokenize_domain(domain)
                    input_tensor = torch.tensor(
                        domain_encoded, dtype=torch.long
                    ).unsqueeze(0)
                    output = model(input_tensor)
                    predicted_class = torch.argmax(output, dim=1).item() + 1
                    return {
                        "result": predicted_class,
                        "domain": domain,
                        "algorithm": dga_map[predicted_class],
                    }


predictor = DomainPredictor()


def process_domains(domain_list):
    try:
        results = []
        for domain in domain_list:
            results.append(predictor.predict(domain))
        return {
            "code": 1,
            "data": results,
        }
    except Exception as e:
        return {
            "code": 0,
            "message": str(e),
        }


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/query":
            query_params = parse_qs(parsed_path.query)
            domains = query_params.get("domain", [None])[0]
            if domains:
                domain_list = domains.split(",")  # 将域名用逗号分开
                result = process_domains(domain_list)
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing domain parameter")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")


if __name__ == "__main__":
    server_address = ("", 8888)
    httpd = HTTPServer(server_address, RequestHandler)
    print("Server running on port 8888...")
    httpd.serve_forever()
