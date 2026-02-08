import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import os

# ------------------------------------------------
# Metric Functions
# ------------------------------------------------
def rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_pred, y_true)).item()

def mae(y_true, y_pred):
    return F.l1_loss(y_pred, y_true).item()

def mape(y_true, y_pred):
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    return np.mean(np.abs((y_true_np - y_pred_np) / (y_true_np + 1e-8))) * 100

def accuracy(y_true, y_pred):
    y_pred_class = torch.argmax(y_pred, dim=1)
    correct = (y_pred_class == y_true).sum().item()
    return correct / len(y_true) * 100

# ------------------------------------------------
# Model Evaluation Function
# ------------------------------------------------
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            y_true_list.append(y)
            y_pred_list.append(outputs)

    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)

    # Classification accuracy
    acc = accuracy(y_true, y_pred)
    print(f"\n✅ Accuracy: {acc:.2f}%")

    # For reference, also compute regression-style metrics
    y_true_onehot = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    print(f"✅ RMSE: {rmse(y_true_onehot, F.softmax(y_pred, dim=1)):.4f}")
    print(f"✅ MAE:  {mae(y_true_onehot, F.softmax(y_pred, dim=1)):.4f}")
    print(f"✅ MAPE: {mape(y_true_onehot, F.softmax(y_pred, dim=1)):.2f}%")

# ------------------------------------------------
# Main Script
# ------------------------------------------------
if __name__ == "__main__":
    # 🔧 CHANGE THESE PATHS
    MODEL_PATH = r"D:\cctv\models\efficientnet3d_convlstm_best.pt"
    TEST_DATA_DIR = r"D:\cctv\val"

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Image transforms (must match your training preprocessing)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(TEST_DATA_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    num_classes = len(test_dataset.classes)
    print(f"Detected {num_classes} classes: {test_dataset.classes}")

    # Load your architecture (change if you used something else)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    # Evaluate
    evaluate_model(model, test_loader, device)
