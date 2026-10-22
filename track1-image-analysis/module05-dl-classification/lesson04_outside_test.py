# Module 5, Lesson 4: test the trained model on photos from a completely different setting
# Run from the project root (plant-cv-tutorial), after Lesson 3 has saved the model.
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision import models, transforms

classes = ["LP", "OP"]                              # same order as ImageFolder used in training

# Rebuild the same network shape, then load the trained weights into it
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("outputs/m05_oplp_resnet18.pt"))
model.eval()

prep = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for path in sorted(Path("data/oplp_outside_test").glob("*/*.jpg")):
    truth = path.parent.name                        # folder name = true class
    x = prep(Image.open(path).convert("RGB")).unsqueeze(0)   # add a batch dimension: (1, 3, 224, 224)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]   # scores -> probabilities that add up to 1
    guess = classes[probs.argmax().item()]
    mark = "correct" if guess == truth else "WRONG"
    print(f"{path.name:20s} truth {truth}   model says {guess}   "
          f"(LP {probs[0]:.0%}, OP {probs[1]:.0%})   {mark}")
