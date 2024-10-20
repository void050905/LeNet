import torch
from torch import nn

class MyLeNet5(nn.Module):
    def __init__(self):
        super(MyLeNet5, self).__init__()
        # 最初的卷积层
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.Sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # 新增卷积层：增加深度
        self.c4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)  # 新的卷积层
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)  # 池化层
        self.c5 = nn.Conv2d(in_channels=32, out_channels=120, kernel_size=3)

        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(480, 84)
        # 新增全连接层：增加模型容量
        self.f7 = nn.Linear(84, 42)  # 新的全连接层，用于在最终输出前减少维度
        self.output = nn.Linear(42, 10)

    def forward(self, x):
        x = self.Sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.Sigmoid(self.c3(x))

        # 通过新增的卷积层传递，并应用Sigmoid激活函数
        x = self.Sigmoid(self.c4(x))  # 新卷积层后的激活函数
        x = self.s4(x)  # 新增的池化层，进一步降低空间维度
        x = self.Sigmoid(self.c5(x))
        x = self.flatten(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.output(x)
        return x

# 测试代码
if __name__ == "__main__":
    x = torch.rand([1, 1, 28, 28])  # 随机输入张量
    model = MyLeNet5()  # 模型实例化
    y = model(x)
