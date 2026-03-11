import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from collections import defaultdict

# -------- SETTINGS --------
DATA_DIR = "dataset"
BATCH_SIZE = 32
EPOCHS = 25
IMAGE_SIZE = 128
MODEL_PATH = "plant_model.pth"

# limit dataset per class for faster training
MAX_IMAGES_PER_CLASS = 400

device = torch.device("cpu")

# -------- TRANSFORMS --------
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# -------- LOAD DATASET --------
dataset = datasets.ImageFolder(DATA_DIR)

print("Classes detected:", dataset.classes)

# -------- LIMIT DATASET SIZE --------
class_count = defaultdict(int)
filtered_samples = []

for path, label in dataset.samples:
    if class_count[label] < MAX_IMAGES_PER_CLASS:
        filtered_samples.append((path, label))
        class_count[label] += 1

dataset.samples = filtered_samples
dataset.targets = [s[1] for s in filtered_samples]

print("Images used for training:", len(dataset.samples))

# -------- SPLIT DATASET --------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -------- MODEL --------
model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# freeze feature layers
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))

model = model.to(device)

# -------- LOSS + OPTIMIZER --------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0005)

best_acc = 0

# -------- TRAIN --------
for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()
    train_loss = 0

    for images, labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print("Training Loss:", train_loss)

    # -------- VALIDATION --------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total

    print("Validation Accuracy:", acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("New best model saved")

print("\nTraining complete")
print("Best Accuracy:", best_acc)
