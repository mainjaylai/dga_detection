import json
import torch
import string
import torch.nn as nn
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# 定义字符集
all_letters = string.ascii_letters + string.digits + "-."
n_letters = len(all_letters)


def letterToIndex(letter):
    return all_letters.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def label_to_tensor(label):
    return torch.tensor([label], dtype=torch.long)


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

            return {
                "result": 1 if bool(predicted.item()) else 0,
                "domain": domain,
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
