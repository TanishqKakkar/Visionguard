# 🛡️ VisionGuard — Intelligent CCTV Surveillance System

An AI-powered smart surveillance system for real-time human activity recognition and anomaly detection from CCTV video streams, comparing three hybrid deep learning architectures to find the best accuracy/speed trade-off for real-world deployment.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)

---

## Overview

Traditional CCTV surveillance relies on manual monitoring, causing delayed responses and inconsistent threat detection. VisionGuard automates this with an intelligent video pipeline that understands spatio-temporal human behavior — processing CCTV footage as 16-frame sequences, extracting spatial features, modeling temporal motion, and classifying activity in real time.

## Key Objectives

- Automate human activity recognition from CCTV footage
- Detect abnormal or suspicious activity using temporal modeling
- Compare hybrid deep learning architectures for real-world deployment
- Optimize inference speed without sacrificing accuracy
- Design a scalable, deployment-ready surveillance pipeline

## System Capabilities

- 📹 Video-based human activity recognition
- 🧠 Spatio-temporal feature learning
- 🚨 Anomaly & suspicious activity detection
- ⚡ Real-time optimized inference
- 📊 Comprehensive performance evaluation
- 🌐 Smart-city ready surveillance framework

## Models Compared

| Model | Training Accuracy | Inference Speed | Generalization |
|---|---|---|---|
| ViT + ConvLSTM | 82.20% | Slow | Medium |
| **ConvNeXt + LSTM** ⭐ | **93.16%** | **Fastest** | **Best** |
| EfficientNet3D + ConvLSTM | 95% | Very Slow | Poor (overfits) |

**ConvNeXt + LSTM** emerged as the most stable, accurate, and deployment-ready model — best real-time balance of accuracy and speed.

## System Pipeline

```
CCTV Video → Frame Extraction (16-frame sequences)
  → Preprocessing (Resize, CLAHE, Normalization)
  → Spatial Feature Extraction (CNN / Transformer)
  → Temporal Modeling (LSTM / ConvLSTM)
  → Activity Classification (Softmax)
  → Output: Activity / Anomaly Label
```

## Dataset

- Custom CCTV action recognition dataset
- Sequence length: 16 frames per clip, frame size 224×224
- Split: 80% train / 10% validation / 10% test
- Preprocessing: CLAHE contrast enhancement, brightness normalization, frame stabilization, tensor normalization

## Evaluation Metrics

Accuracy, RMSE, MAE, MAPE, Confusion Matrix, Inference Speed

## Tech Stack

- **Language:** Python
- **Deep Learning & CV:** TensorFlow/Keras, OpenCV
- **Data Handling:** NumPy, Pandas, Matplotlib
- **Model Types:** CNN, Vision Transformer (ViT), LSTM, ConvLSTM, 3D CNN

## Research Contributions

- Designed and evaluated three hybrid deep learning architectures
- Built an optimized end-to-end CCTV activity recognition pipeline
- Analyzed accuracy vs. inference speed trade-offs
- Identified ConvNeXt-LSTM as the best real-world solution
- Provided insights for smart-city surveillance deployment

## Future Enhancements

- Temporal Transformer integration
- Larger, more diverse CCTV datasets
- Advanced anomaly scoring mechanisms
- Real-time alerting dashboards
- Edge-device deployment optimization

## Authors

**Tanishq Kakkar** & **Kartikeya Singh**

## License

MIT License
