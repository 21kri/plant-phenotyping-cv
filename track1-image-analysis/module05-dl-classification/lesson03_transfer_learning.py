# Module 5, Lesson 3: transfer learning - teach a pre-trained ResNet-18 to tell OP from LP
# Run from the project root (plant-cv-tutorial), after Lesson 1 has built data/oplp_dataset.
import random
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, models, transforms

torch.manual_seed(0)
random.seed(0)

# ImageNet colour statistics - the pre-trained network expects inputs scaled like this
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Training images get small random changes ("augmentation") so the model sees variety
train_prep = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), norm,
])
# Validation images are always prepared the same way
val_prep = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), norm,
])

train_all = datasets.ImageFolder("data/oplp_dataset", transform=train_prep)
val_all = datasets.ImageFolder("data/oplp_dataset", transform=val_prep)

# --- Split by PHOTO, not by image: both halves of one photo stay on the same side ---
photo_of = [Path(p).stem.rsplit("_", 1)[0] for p, _ in train_all.samples]   # "IMG_0275"
photos = sorted(set(photo_of))
random.shuffle(photos)
val_photos = set(photos[:10])                       # 10 of the 40 photos held back
train_idx = [i for i, p in enumerate(photo_of) if p not in val_photos]
val_idx = [i for i, p in enumerate(photo_of) if p in val_photos]
print(f"train images: {len(train_idx)}   validation images: {len(val_idx)}")

train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_all, train_idx), batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_all, val_idx), batch_size=16)

# --- The model: ResNet-18 pre-trained on ImageNet (1.2 million everyday photos) ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False                         # freeze everything it already knows
model.fc = nn.Linear(model.fc.in_features, 2)       # new last layer: 2 outputs (LP, OP) - the only part we train

loss_fn = nn.CrossEntropyLoss()                     # how wrong the predictions are
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)


def accuracy(loader):
    model.eval()                                    # evaluation mode: no randomness
    correct = total = 0
    with torch.no_grad():                           # no learning here, just predicting
        for x, y in loader:
            pred = model(x).argmax(dim=1)           # the class with the higher score
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


for epoch in range(1, 11):                          # 10 passes over the training images
    model.train()
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()                       # 1. forget last step's corrections
        loss = loss_fn(model(x), y)                 # 2. predict and measure the error
        loss.backward()                             # 3. work out how to change each weight
        optimizer.step()                            # 4. nudge the weights
        total_loss += loss.item() * len(y)
    print(f"epoch {epoch:2d}   training loss {total_loss / len(train_idx):.3f}   "
          f"validation accuracy {accuracy(val_loader):.0%}")

torch.save(model.state_dict(), "outputs/m05_oplp_resnet18.pt")
print("saved outputs/m05_oplp_resnet18.pt")
