# 🦴 Bone Age Estimation System

![Bone Age Estimation Demo](11.png) 

This application estimates bone age (in months) from hand X-ray images using an ensemble of deep learning models. Designed for medical professionals and researchers, it provides quick predictions to assist in pediatric development assessment.

## 🚀 Features
- Ensemble learning with 5 EfficientNetV2 models
- GPU acceleration support (NVIDIA CUDA required)
- Intuitive web interface using Gradio
- Handles images of any aspect ratio
- Prediction range: 0-240 months (20 years)
- Automatic hardware detection (falls back to CPU if GPU unavailable)

## ⚙️ System Requirements
1. **Hardware**:
   - NVIDIA GPU with CUDA support (recommended)
   - Minimum 4GB VRAM
2. **Software**:
   - Windows/Linux/macOS
   - Python 3.8+
   - CUDA 11.x (for GPU acceleration)

## 💻 Installation
```bash
# Clone repository
git clone https://github.com/mhdi002/bone-age-estimation.git
cd bone-age-estimation

# Install dependencies
pip install torch timm gradio pillow numpy
