import torch
import timm
import cv2
from PIL import Image
from torchvision import transforms
import os
import random

MODEL_PATH = "../Models/xception_torch_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfms = transforms.Compose([
    transforms.ToTensor(),
])

def load_model():
    if os.path.exists(MODEL_PATH):
        model = timm.create_model("xception", pretrained=False, num_classes=1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("✅ Model Loaded")
        return model
    else:
        print("⚠️ Dummy Mode")
        return None

def predict_image(img, model):
    if model is None:
        prob = random.uniform(0.3, 0.9)
    else:
        img = cv2.resize(img, (224,224))
        tensor = tfms(Image.fromarray(img)).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(tensor)).item()

    return {
        "label": "FAKE" if prob > 0.7 else "REAL",
        "confidence": prob
    }