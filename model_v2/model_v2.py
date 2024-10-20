import torch
from torch import nn

class MyLeNet5(nn.Module):
    def __init__(self):
        super(MyLeNet5, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.Sigmoid = nn.Sigmoid()
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=6, out_channels=120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(12000, 84)  # 修正后的全连接层输入维度
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = self.Sigmoid(self.c1(x))
        x = self.s4(x)
        x = self.Sigmoid(self.c5(x))
        x = self.flatten(x)
        x = self.f6(x)
        x = self.output(x)
        return x

if __name__ == "__main__":
    x = torch.rand([1, 1, 28, 28])
    model = MyLeNet5()
    y = model(x)
