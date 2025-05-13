import os
import sys
import torch
import torch.nn as nn
import timm
import gradio as gr
from PIL import Image
import numpy as np
from torchvision import transforms

# ------------------- CONFIG -------------------
IMG_SIZE = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("This software requires an NVIDIA GPU with CUDA support.")
    sys.exit(1)

# Handle executable paths
base_path = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
MODEL_PATHS = [os.path.join(base_path, f"best_model_fold{i}.pth") for i in range(5)]

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------- MODEL -------------------
class BoneAgeRegressor(nn.Module):
    def __init__(self):
        super(BoneAgeRegressor, self).__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=False)
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x.squeeze(1)

def load_models():
    models = []
    for path in MODEL_PATHS:
        model = BoneAgeRegressor().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models.append(model)
    return models

models = load_models()

# ------------------- PREDICTION -------------------
def predict_bone_age(image: Image.Image):
    img = image.convert("RGB")
    img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)

    predictions = []
    for model in models:
        with torch.no_grad():
            pred = model(img_tensor).item()
            predictions.append(pred)

    final_pred = np.mean(predictions)
    final_pred = np.clip(final_pred, 0, 240)
    return f"Predicted Bone Age: {final_pred:.1f} months"

# ------------------- GRADIO UI -------------------
title = "🦴 Bone Age Estimation "
description = (
    "Upload a hand X-ray image to estimate bone age (0–240 months) using an ensemble of CNN models. "
    
)

iface = gr.Interface(
    fn=predict_bone_age,
    inputs=gr.Image(type="pil", label="Upload Hand X-ray Image"),
    outputs=gr.Textbox(label="Prediction"),
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
