import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import timm
import multiprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Settings
BATCH_SIZE = 32
LR = 3e-4
NUM_EPOCHS = 11
IMG_SIZE = 300
N_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
EARLY_STOPPING_PATIENCE = 10

CSV_PATH = r"C:\Users\mahdi\Downloads\Digital-Hand-Atlas-Train.csv"
IMG_DIR = r"D:\Digital Hand Atlas\Digital Hand Atlas\JPEGimages"

# Dataset
class HandBoneAgeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.ids = self.df['id'].astype(str).values
        self.labels = self.df['boneage'].astype(float).values
        self.img_dir = img_dir
        self.transform = transform
        self.id_to_path = {}
        self._find_image_paths()

    def _find_image_paths(self):
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    img_id = os.path.splitext(file)[0]
                    self.id_to_path[img_id] = img_path

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        label = self.labels[idx]
        img_path = self.id_to_path.get(img_id, None)
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='gray')
        if self.transform:
            img = self.transform(img)
        return img, label

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Model
class BoneAgeRegressor(nn.Module):
    def __init__(self):
        super(BoneAgeRegressor, self).__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True)
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

# Train one epoch
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for imgs, labels in dataloader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE).float()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(dataloader.dataset)

# Validate
def validate(model, dataloader, criterion):
    model.eval()
    val_loss = 0.0
    preds = []
    gts = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).float()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            preds.append(outputs.cpu().numpy())
            gts.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    r2 = r2_score(gts, preds)
    return val_loss / len(dataloader.dataset), r2

# Visualization: Image + True vs Predicted
def plot_predictions_ensemble(models, dataset, num=5, clip_range=(0, 240)):
    for model in models:
        model.eval()

    indices = np.random.choice(len(dataset), num, replace=False)
    fig, axs = plt.subplots(num, 2, figsize=(10, num * 4))

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img_input = img.unsqueeze(0).to(DEVICE)

        preds = []
        for model in models:
            pred = model(img_input).item()
            preds.append(pred)

        final_pred = np.mean(preds)
        final_pred = np.clip(final_pred, *clip_range)  # MAGIC: Clip unrealistic ages

        img_vis = img.cpu().permute(1,2,0).numpy()
        img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_vis = np.clip(img_vis, 0, 1)

        axs[i, 0].imshow(img_vis)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f"GT={label:.0f} / Pred={final_pred:.0f}")

        axs[i, 1].bar(['True', 'Pred'], [label, final_pred], color=['green', 'red'])
        axs[i, 1].set_ylim(0, 250)
        axs[i, 1].set_title("True vs Predicted")

    plt.tight_layout()
    plt.show()

# Plot losses
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.title('Training and Validation Losses')
    plt.show()

# Main
def main():
    print(f"Using {DEVICE}")
    full_dataset = HandBoneAgeDataset(CSV_PATH, IMG_DIR, transform=train_transform)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = 0
    model_paths = []

    for train_idx, val_idx in kf.split(full_dataset):
        print(f"----- Fold {fold+1}/{N_FOLDS} -----")
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = BoneAgeRegressor().to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_val_loss = np.inf
        train_losses = []
        val_losses = []
        patience_counter = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_r2 = validate(model, val_loader, criterion)

            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val R2: {val_r2:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        plot_losses(train_losses, val_losses)
        model_paths.append(f"best_model_fold{fold}.pth")
        fold += 1

    # After all folds: Ensembling
    print("Loading best models for ensembling...")
    models = []
    for path in model_paths:
        model = BoneAgeRegressor().to(DEVICE)
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)

    print("Visualizing ensemble predictions...")
    full_dataset.transform = val_transform
    plot_predictions_ensemble(models, full_dataset, num=5)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
