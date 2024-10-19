import gradio as gr
import numpy as np
import torch
from model import MyLeNet5

# 实例化模型
digit_model = MyLeNet5()

# 安全地加载模型权重
weights = torch.load('D:/LeNet/save_model/best_model.pth', weights_only=True)
digit_model.load_state_dict(weights)  # 加载权重
digit_model.eval()  # 设置模型为评估模式

def classify_image(img):
    img_3d = img.reshape(1, 1, 28, 28)  # 处理输入格式
    im_resize = img_3d / 255.0
    im_tensor = torch.tensor(im_resize, dtype=torch.float32)  # 转换为Tensor
    with torch.no_grad():  # 禁用梯度计算
        prediction = digit_model(im_tensor)  # 进行预测
    pred = torch.argmax(prediction, dim=1).item()  # 获取预测结果
    return pred

# 使用gr.Label而不是gr.outputs.Label
label = gr.Label(num_top_classes=3)
interface = gr.Interface(fn=classify_image, inputs="sketchpad", outputs=label)
interface.launch(share=True)
