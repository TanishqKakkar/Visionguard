import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error


# ---------------------------------------
# CONFIG
# ---------------------------------------
ROOT = "data balanced"    
SEQ_LEN = 16
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Using device:", device)


# ---------------------------------------
# LOAD SEQUENCE PATHS
# ---------------------------------------
def get_sequences(root):
    X = []
    y = []
    class_names = sorted(os.listdir(root))
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    for cls in class_names:
        cls_path = os.path.join(root, cls)
        for seq in os.listdir(cls_path):
            seq_path = os.path.join(cls_path, seq)

            frames = sorted(os.listdir(seq_path))
            if len(frames) < SEQ_LEN:
                continue

            X.append(seq_path)
            y.append(class_to_idx[cls])

    return X, y, class_names


X, y, class_names = get_sequences(ROOT)
print("Total sequences:", len(X))
print("Classes:", class_names)


# ---------------------------------------
# TRAIN / VAL / TEST SPLITS
# ---------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("Train:", len(X_train))
print("Val:", len(X_val))
print("Test:", len(X_test))


# ---------------------------------------
# DATASET CLASS
# ---------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def load_seq(self, path):
        frames = sorted(os.listdir(path))[:SEQ_LEN]
        imgs = []

        for f in frames:
            img_path = os.path.join(path, f)
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            imgs.append(img)

        imgs = torch.stack(imgs)
        return imgs

    def __getitem__(self, idx):
        seq_path = self.X[idx]
        label = self.y[idx]
        imgs = self.load_seq(seq_path)
        return imgs, label


train_ds = SequenceDataset(X_train, y_train)
val_ds = SequenceDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)


# ---------------------------------------
# MODEL: ConvNeXtTiny + LSTM
# ---------------------------------------
convnext = models.convnext_tiny(pretrained=True)
convnext.classifier = nn.Identity()      
convnext.to(device)
convnext.eval()

for p in convnext.parameters():
    p.requires_grad = False


class ConvNext_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = convnext
        self.lstm = nn.LSTM(768, 256, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        feats = self.cnn(x)
        feats = feats.reshape(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        last = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(last))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = ConvNext_LSTM(len(class_names)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# ---------------------------------------
# TRAINING LOOP (with tqdm)
# ---------------------------------------
train_acc_list, val_acc_list = [], []
train_loss_list, val_loss_list = [], []

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ---------------------- TRAIN ----------------------
    model.train()
    total, correct = 0, 0
    running_loss = 0

    train_pbar = tqdm(train_loader, desc="Training", leave=False)

    for imgs, labels in train_pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)
        correct += (preds.argmax(1) == labels).sum().item()

        train_pbar.set_postfix(loss=loss.item())

    train_acc = correct / total
    train_loss = running_loss / len(train_loader)

    # ---------------------- VALIDATION ----------------------
    model.eval()
    total, correct = 0, 0
    val_loss_sum = 0

    val_pbar = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for imgs, labels in val_pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            val_loss_sum += criterion(preds, labels).item()

            total += labels.size(0)
            correct += (preds.argmax(1) == labels).sum().item()

    val_acc = correct / total
    val_loss = val_loss_sum / len(val_loader)

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    print(f"✅ Epoch {epoch+1}/{EPOCHS} | "
          f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f} | "
          f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")


# ---------------------------------------
# SAVE MODEL
# ---------------------------------------
torch.save(model.state_dict(), "cctv_convnext_lstm.pt")
print("\n✅ Model saved!")


# ---------------------------------------
# TEST SET EVALUATION
# ---------------------------------------
test_ds = SequenceDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

model.eval()
all_labels = []
all_preds = []
test_loss_sum = 0

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)
        loss = criterion(preds, labels)

        test_loss_sum += loss.item()

        all_labels.append(labels.item())
        all_preds.append(preds.argmax(1).item())

# Convert to numpy
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Metrics
test_loss = test_loss_sum / len(test_loader)
test_accuracy = accuracy_score(all_labels, all_preds)
rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
mae = mean_absolute_error(all_labels, all_preds)
mape = np.mean(np.abs((all_labels - all_preds) / (all_labels + 1e-8))) * 100

cm = confusion_matrix(all_labels, all_preds)

print("\n✅ FINAL EVALUATION METRICS")
print("-----------------------------------")
print("Test Accuracy :", test_accuracy)
print("Test Loss     :", test_loss)
print("RMSE          :", rmse)
print("MAE           :", mae)
print("MAPE (%)      :", mape)
print("\nConfusion Matrix:\n", cm)


# ---------------------------------------
# PLOTS
# ---------------------------------------
plt.figure(figsize=(10,4))

# ACC
plt.subplot(1,2,1)
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.title("Accuracy")
plt.legend()

# LOSS
plt.subplot(1,2,2)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.title("Loss")
plt.legend()

plt.show()
