# Shared helpers for Module 6 (Lesson 2 shows these steps in full).
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def split(seed=0):
    """Same photo split as Lesson 2: 8 validation photos, the rest for training."""
    random.seed(seed)
    names = sorted(p.stem for p in Path("data/seg_dataset/images").glob("*.jpg"))
    random.shuffle(names)
    return names[8:], sorted(names[:8])


def load_pair(name):
    img = cv2.imread(f"data/seg_dataset/images/{name}.jpg")[:, :, ::-1].astype(np.float32) / 255  # RGB 0..1
    mask = cv2.imread(f"data/seg_dataset/masks/{name}.png", 0) > 0
    return img, mask


def random_lighting(img):
    """Colour augmentation: random overall brightness and a random tint (per-channel gain)."""
    gain = random.uniform(0.5, 1.4) * np.array([random.uniform(0.8, 1.2) for _ in range(3)], np.float32)
    return np.clip(img * gain, 0, 1)


def to_tensor(img_rgb01):
    return torch.from_numpy(((img_rgb01 - MEAN) / STD).transpose(2, 0, 1).copy())


class PlantPairs(torch.utils.data.Dataset):
    def __init__(self, names, lighting_jitter=False):
        self.names, self.lighting_jitter = names, lighting_jitter

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img, mask = load_pair(self.names[i])
        y = random.randint(0, img.shape[0] - 256)
        x = random.randint(0, img.shape[1] - 256)
        img, mask = img[y:y + 256, x:x + 256], mask[y:y + 256, x:x + 256]
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if self.lighting_jitter:
            img = random_lighting(img)
        return to_tensor(img), torch.from_numpy(mask.astype(np.int64).copy())


def new_model():
    return lraspp_mobilenet_v3_large(weights=None, num_classes=2,
                                     weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1)


def train(train_names, epochs=40, lighting_jitter=False, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    loader = torch.utils.data.DataLoader(PlantPairs(train_names, lighting_jitter), batch_size=8, shuffle=True)
    model = new_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss_fn(model(x)["out"], y).backward()
            optimizer.step()
    return model.eval()


def load(path):
    model = new_model()
    model.load_state_dict(torch.load(path))
    return model.eval()


def predict(model, img_rgb01):
    with torch.no_grad():
        return model(to_tensor(img_rgb01)[None])["out"].argmax(dim=1)[0].numpy() == 1


def rule(img_rgb01):
    """The Module 2/3 colour rule, on an RGB 0..1 image."""
    rgb = (img_rgb01 * 255).astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mask = (((g - b) > 40) & (g > r)).astype(np.uint8) * 255
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, brush)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, brush) > 0


def iou(pred, truth):
    return (pred & truth).sum() / max((pred | truth).sum(), 1)
