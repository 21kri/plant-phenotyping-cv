# Shared helpers for Module 5: plant-only masking, training and the outside test.
# Lessons 3 and 4 show these steps in full; later lessons call them from here.
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import datasets, models, transforms

CLASSES = ["LP", "OP"]   # ImageFolder sorts folder names alphabetically
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
VAL_PREP = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), NORM])


def plant_only(img_bgr):
    """Keep green plant pixels, paint everything else black.

    G - B > 40 is the Module 2/3 rule; G > R also rejects yellowish tape labels.
    """
    b = img_bgr[:, :, 0].astype(np.int16)
    g = img_bgr[:, :, 1].astype(np.int16)
    r = img_bgr[:, :, 2].astype(np.int16)
    mask = (((g - b) > 40) & (g > r)).astype(np.uint8) * 255
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, brush)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, brush)
    out = np.zeros_like(img_bgr)
    out[mask > 0] = img_bgr[mask > 0]
    return out


def mask_folder(src, dst):
    """Copy an ImageFolder tree (class sub-folders of .jpg) with plant_only() applied."""
    for path in sorted(Path(src).glob("*/*.jpg")):
        out = Path(dst) / path.parent.name / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), plant_only(cv2.imread(str(path))))


def train(data_dir, model_path, epochs=10, seed=0, color_jitter=None, verbose=True):
    """Lesson 3's transfer learning: frozen ResNet-18, new last layer, split by photo.

    color_jitter: optional transforms.ColorJitter, applied to training images only (Lesson 6).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    steps = [transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip()]
    if color_jitter is not None:
        steps.append(color_jitter)
    train_prep = transforms.Compose(steps + [transforms.ToTensor(), NORM])
    train_all = datasets.ImageFolder(data_dir, transform=train_prep)
    val_all = datasets.ImageFolder(data_dir, transform=VAL_PREP)

    photo_of = [Path(p).stem.rsplit("_", 1)[0] for p, _ in train_all.samples]
    photos = sorted(set(photo_of))
    random.shuffle(photos)
    val_photos = set(photos[:10])
    train_idx = [i for i, p in enumerate(photo_of) if p not in val_photos]
    val_idx = [i for i, p in enumerate(photo_of) if p in val_photos]
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_all, train_idx), batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_all, val_idx), batch_size=16)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                correct += (model(x).argmax(1) == y).sum().item()
        if verbose:
            print(f"epoch {epoch:2d}   validation accuracy {correct / len(val_idx):.0%}")

    model.val_accuracy = correct / len(val_idx)    # last epoch's score, for summaries
    torch.save(model.state_dict(), model_path)
    return model


def load(model_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path))
    return model.eval()


def outside_test(model, test_dir, verbose=True):
    """Lesson 4's check: predict every image in test_dir/<class>/*.jpg. Returns number correct."""
    n_correct = 0
    for path in sorted(Path(test_dir).glob("*/*.jpg")):
        x = VAL_PREP(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        guess = CLASSES[probs.argmax().item()]
        n_correct += guess == path.parent.name
        if verbose:
            mark = "correct" if guess == path.parent.name else "WRONG"
            print(f"{path.name:20s} truth {path.parent.name}   model says {guess}   "
                  f"(LP {probs[0]:.0%}, OP {probs[1]:.0%})   {mark}")
    return n_correct
