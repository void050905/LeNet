import gradio as gr
import torch
from torchvision import transforms
import requests
from PIL import Image

model = torch.hub.load('D:/LeNet/save_model/best_model.pth', 'resnet18', pretrained=True).eval()

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def predict(inp):
  inp = Image.fromarray(inp.astype('uint8'), 'RGB')
  inp = inp.resize((24, 24))
  inp = transforms.ToTensor()(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
  return {labels[i]: float(prediction[i]) for i in range(1000)}

inputs = gr.Image()
outputs = gr.Label(num_top_classes=3)
gr.Interface(fn=predict, inputs=inputs, outputs=outputs).launch()