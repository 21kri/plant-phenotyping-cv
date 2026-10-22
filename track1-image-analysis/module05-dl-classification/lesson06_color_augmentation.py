# Module 5, Lesson 6: colour augmentation - random lighting changes so the model can't rely on exact shades
# Run from the project root (plant-cv-tutorial), after Lesson 5 has made the plant-only folders.
# Takes about 3 minutes: it trains 6 models.
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from oplp import outside_test, train

jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05)

# 1. See what it does: one LP image, 8 random "lighting conditions"
img = Image.open("data/oplp_dataset_masked/LP/IMG_0296_full.jpg").convert("RGB")
img = transforms.CenterCrop(400)(img)
torch.manual_seed(1)
examples = [transforms.ToTensor()(jitter(img)) for _ in range(8)]
save_image(examples, "outputs/m05_l06_jitter_examples.png", nrow=4)
print("saved outputs/m05_l06_jitter_examples.png")

# 2. Train with and without colour jitter. Each setting is repeated with 3 different
#    random seeds, because with this little data a single run can be lucky or unlucky.
print("\n                 seed   val acc   outside test (of 2)")
for name, cj in [("no jitter", None), ("colour jitter", jitter)]:
    for seed in (0, 1, 2):
        model = train("data/oplp_dataset_masked", "outputs/m05_tmp.pt", seed=seed,
                      color_jitter=cj, verbose=False)
        score = outside_test(model, "data/oplp_outside_test_masked", verbose=False)
        print(f"{name:15s}  {seed:4d}   {model.val_accuracy:6.0%}   {score}")
