import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path
import warnings
import time
from datetime import timedelta
from tqdm import tqdm
import torchvision.models as models
warnings.filterwarnings('ignore')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✅ Device: {device}\n")

np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# AUGMENTATION
# ============================================================================

class VideoAugmentation:
    @staticmethod
    def augment(frames):
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.85, 1.15)
            frames = frames * alpha
        
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.85, 1.15)
            mean = frames.mean()
            frames = (frames - mean) * factor + mean
        
        if np.random.rand() > 0.6:
            noise = np.random.normal(0, 0.02, frames.shape)
            frames = frames + noise
        
        if np.random.rand() > 0.5:
            frames = np.flip(frames, axis=2).copy()
        
        frames = np.clip(frames, 0, 1)
        return frames


# ============================================================================
# DATASET
# ============================================================================

class OnTheFlyVideoDataset(Dataset):
    def __init__(self, video_paths, labels, sequence_length=8, img_size=224, augment=True):
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.augment = augment
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_frames(self, folder_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        frame_files = []
        for ext in image_extensions:
            frame_files.extend(list(Path(folder_path).glob(f'*{ext}')))
        
        frame_files = sorted(frame_files)
        
        if len(frame_files) == 0:
            return None
        
        if len(frame_files) > self.sequence_length:
            indices = np.linspace(0, len(frame_files) - 1, self.sequence_length, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        else:
            while len(frame_files) < self.sequence_length:
                frame_files.append(frame_files[-1])
        
        frames = []
        for frame_path in frame_files[:self.sequence_length]:
            try:
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    frame = cv2.resize(frame, (self.img_size, self.img_size))
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
            except:
                pass
        
        if len(frames) < self.sequence_length:
            while len(frames) < self.sequence_length:
                frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.float32))
        
        return np.array(frames, dtype=np.float32)
    
    def __getitem__(self, idx):
        frames = self.load_frames(self.video_paths[idx])
        if frames is None:
            frames = np.zeros((self.sequence_length, self.img_size, self.img_size, 3), dtype=np.float32)
        
        if self.augment:
            frames = VideoAugmentation.augment(frames)
        
        frames = np.transpose(frames, (0, 3, 1, 2))
        frames = np.ascontiguousarray(frames)
        
        return torch.FloatTensor(frames), torch.LongTensor([self.labels[idx]])[0]


# ============================================================================
# DATA LOADER
# ============================================================================

class FrameLoader:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"❌ Dataset path not found: {dataset_path}")
        
        all_folders = [f.name for f in self.dataset_path.iterdir() if f.is_dir()]
        self.classes = sorted([f for f in all_folders if f.lower() not in ['train', 'val', 'test']])
        
        if not self.classes:
            raise ValueError(f"❌ No classes found in {dataset_path}")
        
        print(f"📁 Classes found: {self.classes}")
        
        self.video_paths = []
        self.video_labels = []
        self._index_videos()
    
    def _index_videos(self):
        for class_idx, class_name in enumerate(self.classes):
            class_path = self.dataset_path / class_name
            if class_path.exists():
                videos = [v for v in class_path.iterdir() if v.is_dir()]
                print(f"  {class_name}: {len(videos)} sequences")
                for video_folder in videos:
                    self.video_paths.append(video_folder)
                    self.video_labels.append(class_idx)
        
        if not self.video_paths:
            raise ValueError("❌ No sequences found in dataset!")
        
        print(f"📊 Total sequences: {len(self.video_paths)}\n")
    
    def get_splits(self):
        indices = np.arange(len(self.video_paths))
        
        train_idx, test_idx = train_test_split(
            indices, test_size=0.3, random_state=42, 
            stratify=np.array(self.video_labels)
        )
        val_idx, test_idx = train_test_split(
            test_idx, test_size=0.5, random_state=42,
            stratify=np.array([self.video_labels[i] for i in test_idx])
        )
        
        train_paths = [self.video_paths[i] for i in train_idx]
        train_labels = [self.video_labels[i] for i in train_idx]
        
        val_paths = [self.video_paths[i] for i in val_idx]
        val_labels = [self.video_labels[i] for i in val_idx]
        
        test_paths = [self.video_paths[i] for i in test_idx]
        test_labels = [self.video_labels[i] for i in test_idx]
        
        print(f"📊 Split: Train {len(train_paths)} | Val {len(val_paths)} | Test {len(test_paths)}\n")
        
        return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


# ============================================================================
# CONVLSTM CELL
# ============================================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
    
    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i_gate, f_gate, g_gate, o_gate = torch.chunk(gates, 4, dim=1)
        
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        
        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)
        
        return h_new, c_new


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, num_layers=1):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                input_channels if i == 0 else hidden_channels,
                hidden_channels,
                kernel_size
            )
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        batch_size, time_steps, _, h, w = x.shape
        device = x.device
        
        h_states = [torch.zeros(batch_size, self.hidden_channels, h, w, device=device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(batch_size, self.hidden_channels, h, w, device=device) for _ in range(self.num_layers)]
        
        output = []
        
        for t in range(time_steps):
            x_t = x[:, t, :, :, :]
            
            for layer in range(self.num_layers):
                h_states[layer], c_states[layer] = self.cells[layer](x_t, h_states[layer], c_states[layer])
                x_t = h_states[layer]
            
            output.append(h_states[-1].unsqueeze(1))
        
        output = torch.cat(output, dim=1)
        return output, h_states[-1]


# ============================================================================
# EFFICIENTNET 3D + CONVLSTM MODEL
# ============================================================================

class EfficientNet3D_ConvLSTM(nn.Module):
    def __init__(self, seq_len=8, img_size=224, num_classes=5):
        super().__init__()
        
        efficientnet = models.efficientnet_b0(pretrained=True)
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-1])
        
        self.feature_dim = 1280
        
        self.convlstm = ConvLSTM(
            input_channels=1280,
            hidden_channels=128,
            kernel_size=3,
            num_layers=2
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        
        x = x.view(batch_size * seq_len, 3, x.shape[3], x.shape[4])
        x = self.backbone(x)
        
        c, h, w = x.shape[1:]
        x = x.view(batch_size, seq_len, c, h, w)
        
        x, _ = self.convlstm(x)
        x = x[:, -1, :, :, :]
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        x = self.head(x)
        
        return x


# ============================================================================
# TRAINING WITH FP16
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    epoch_start = time.time()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1:3d}/{total_epochs} [TRAIN]", leave=True)
    
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        
        # ✅ FP16 MIXED PRECISION
        with autocast():
            out = model(X)
            loss = criterion(out, y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100*correct/total:.2f}%'})
    
    epoch_time = time.time() - epoch_start
    torch.cuda.empty_cache()
    return total_loss / len(loader), correct / total, epoch_time


def validate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    val_start = time.time()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1:3d}/{total_epochs} [VAL] ", leave=True)
    
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            
            with autocast():
                out = model(X)
                loss = criterion(out, y)
            
            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100*correct/total:.2f}%'})
    
    val_time = time.time() - val_start
    torch.cuda.empty_cache()
    return total_loss / len(loader), correct / total, val_time


# ============================================================================
# MAIN
# ============================================================================

def main():
    DATASET_PATH = r"D:\cctv\data balanced"
    EPOCHS = 50
    BATCH_SIZE = 16  # ✅ INCREASED from 4
    SEQ_LEN = 8  # ✅ REDUCED from 16
    
    print("=" * 80)
    print("🎬 CCTV DETECTION - EfficientNet3D + ConvLSTM (OPTIMIZED)")
    print("=" * 80)
    print(f"⚡ Optimizations: FP16 Mixed Precision + Batch Size 16 + Seq Len 8\n")
    
    print("📦 INDEXING DATASET")
    print("=" * 80)
    
    try:
        loader = FrameLoader(DATASET_PATH)
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = loader.get_splits()
    
    print("🔧 Creating datasets...")
    train_ds = OnTheFlyVideoDataset(train_paths, train_labels, sequence_length=SEQ_LEN, augment=True)
    val_ds = OnTheFlyVideoDataset(val_paths, val_labels, sequence_length=SEQ_LEN, augment=False)
    test_ds = OnTheFlyVideoDataset(test_paths, test_labels, sequence_length=SEQ_LEN, augment=False)
    print("✅ Datasets ready\n")
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"📊 DataLoaders:")
    print(f"   Training batches: {len(train_dl)}")
    print(f"   Validation batches: {len(val_dl)}")
    print(f"   Test batches: {len(test_dl)}\n")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print("=" * 80)
    print("🏗  BUILDING MODEL")
    print("=" * 80)
    
    model = EfficientNet3D_ConvLSTM(seq_len=SEQ_LEN, img_size=224, num_classes=len(loader.classes))
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model on GPU: {next(model.parameters()).is_cuda}")
    print(f"✅ Total parameters: {total_params:,}\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()  # ✅ FP16 SCALER
    
    print("=" * 80)
    print("🎓 TRAINING")
    print("=" * 80 + "\n")
    
    best_acc = 0
    patience_count = 0
    epoch_times = []
    training_start = time.time()
    
    for epoch in range(EPOCHS):
        print(f"\n⏱️  Epoch {epoch+1:3d}/{EPOCHS}")
        print("-" * 80)
        
        train_loss, train_acc, train_time = train_epoch(model, train_dl, criterion, optimizer, scaler, device, epoch, EPOCHS)
        val_loss, val_acc, val_time = validate(model, val_dl, criterion, device, epoch, EPOCHS)
        scheduler.step()
        
        epoch_times.append(train_time + val_time)
        avg_time = np.mean(epoch_times)
        remaining_epochs = EPOCHS - epoch - 1
        eta = avg_time * remaining_epochs
        
        gap = train_acc - val_acc
        status = "✅" if gap < 0.15 else "⚠️" if gap < 0.25 else "🔴"
        
        print(f"\n📊 Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"   Gap: {gap:.4f} {status}")
        print(f"   ⏱️  Train: {train_time:.2f}s | Val: {val_time:.2f}s | Total: {train_time+val_time:.2f}s")
        print(f"   🕐 ETA: {timedelta(seconds=int(eta))}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_count = 0
            torch.save(model.state_dict(), 'models/efficientnet3d_convlstm_best.pt')
            print(f"   ✅ New best accuracy: {val_acc:.4f} → Model saved!")
        else:
            patience_count += 1
            print(f"   ⏳ No improvement. Patience: {patience_count}/15")
        
        if patience_count > 15:
            print(f"\n⏹  Early stopping at epoch {epoch+1}")
            break
        
        torch.cuda.empty_cache()
    
    total_training_time = time.time() - training_start
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n📊 Training Statistics:")
    print(f"   Epochs trained: {epoch+1}/{EPOCHS}")
    print(f"   Average epoch: {np.mean(epoch_times):.2f}s (MUCH FASTER! 🚀)")
    print(f"   Total training time: {timedelta(seconds=int(total_training_time))}")
    print(f"   Best Val Acc: {best_acc:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
