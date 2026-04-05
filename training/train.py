import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# ===== CONFIG =====
DATA_DIR = "../cropped_faces"
MODEL_SAVE = "../Models/xception_torch_best.pt"
EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# ===== DATASET =====
train_data = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# ===== MODEL =====
model = timm.create_model("xception", pretrained=True, num_classes=1)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ===== TRAIN =====
print("🚀 Training started...")

for epoch in range(EPOCHS):
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# ===== SAVE =====
torch.save(model.state_dict(), MODEL_SAVE)
print("✅ Model saved at:", MODEL_SAVE)