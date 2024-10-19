import gradio as gr
import numpy as np
import torch
from model import MyLeNet5
import cv2

# 实例化模型
digit_model = MyLeNet5()

# 安全地加载模型权重
weights = torch.load('/home/wangzishan/lenet/LeNet/save_model/best_model.pth')
digit_model.load_state_dict(weights)  # 加载权重
digit_model.eval()  # 设置模型为评估模式

def classify_image(img):
    # 从字典中获取合成图像，这是 RGBA 图像
    composite_image = img['composite']
    # 转换为灰度图，忽略 alpha 通道
    grayscale_image = np.dot(composite_image[..., :3], [0.2989, 0.5870, 0.1140])
    # 确保灰度图像形状为 28x28，这里可能需要调整
    grayscale_image = cv2.resize(grayscale_image, (28, 28), interpolation=cv2.INTER_AREA)
    # 重新整形和缩放像素值
    img_3d = grayscale_image.reshape(1, 1, 28, 28)
    im_resize = img_3d / 255.0
    im_tensor = torch.tensor(im_resize, dtype=torch.float32)
    # 禁用梯度计算
    with torch.no_grad():
        prediction = digit_model(im_tensor)
    # 获取预测结果
    pred = torch.argmax(prediction, dim=1).item()
    return pred


label = gr.Label(num_top_classes=3)
interface = gr.Interface(fn=classify_image, inputs="sketchpad", outputs=label)
interface.launch(share=True)
