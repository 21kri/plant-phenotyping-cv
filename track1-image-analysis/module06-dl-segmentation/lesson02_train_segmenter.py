# Module 6, Lesson 2: train a network to label every pixel as plant or background
# Run from the project root (plant-cv-tutorial), after Lesson 1 has made data/seg_dataset.
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

torch.manual_seed(0)
random.seed(0)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class PlantPairs(torch.utils.data.Dataset):
    """Hands out (image, mask) pairs. A Dataset only needs __len__ and __getitem__."""

    def __init__(self, names, train):
        self.names = names
        self.train = train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = cv2.imread(f"data/seg_dataset/images/{name}.jpg")[:, :, ::-1]   # BGR -> RGB
        mask = cv2.imread(f"data/seg_dataset/masks/{name}.png", 0) > 0        # True = plant

        if self.train:                                   # random 256x256 crop + random mirror
            y = random.randint(0, img.shape[0] - 256)
            x = random.randint(0, img.shape[1] - 256)
            img, mask = img[y:y + 256, x:x + 256], mask[y:y + 256, x:x + 256]
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]  # flip image AND mask together

        img = (img.astype(np.float32) / 255 - MEAN) / STD
        x = torch.from_numpy(img.transpose(2, 0, 1).copy())   # (H, W, 3) -> (3, H, W)
        y = torch.from_numpy(mask.astype(np.int64).copy())    # (H, W) of 0 / 1
        return x, y


# Split by photo: 32 for training, 8 held back
names = sorted(p.stem for p in Path("data/seg_dataset/images").glob("*.jpg"))
random.shuffle(names)
val_names, train_names = sorted(names[:8]), names[8:]
train_loader = torch.utils.data.DataLoader(PlantPairs(train_names, True), batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(PlantPairs(val_names, False), batch_size=4)

# LR-ASPP MobileNetV3: a small, fast segmentation network. Backbone pre-trained on ImageNet.
model = lraspp_mobilenet_v3_large(weights=None, num_classes=2,
                                  weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
loss_fn = nn.CrossEntropyLoss()                 # same loss as Module 5 - now for every pixel
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def plant_iou(loader):
    """IoU = overlap / union of predicted plant and label plant (1.0 = perfect)."""
    model.eval()
    inter = union = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x)["out"].argmax(dim=1)      # (N, H, W): 0 or 1 for every pixel
            inter += ((pred == 1) & (y == 1)).sum().item()
            union += ((pred == 1) | (y == 1)).sum().item()
    return inter / union


for epoch in range(1, 41):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x)["out"], y)            # out: (N, 2, H, W) scores per pixel
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch:2d}   last batch loss {loss.item():.3f}   validation plant IoU {plant_iou(val_loader):.3f}")

torch.save(model.state_dict(), "outputs/m06_segmenter.pt")
print("saved outputs/m06_segmenter.pt")

# Picture of the 8 validation photos: photo | rule mask (label) | network's prediction
model.eval()
rows = []
with torch.no_grad():
    for name in val_names[:4]:
        x, y = PlantPairs([name], False)[0]
        pred = model(x[None])["out"].argmax(dim=1)[0].numpy()
        photo = cv2.imread(f"data/seg_dataset/images/{name}.jpg")
        label = cv2.cvtColor((y.numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        guess = cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        rows.append(np.hstack([photo, label, guess]))
cv2.imwrite("outputs/m06_l02_val_predictions.jpg", np.vstack(rows))
print("saved outputs/m06_l02_val_predictions.jpg  (photo | label | prediction)")
