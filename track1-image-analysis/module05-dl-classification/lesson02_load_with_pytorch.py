# Module 5, Lesson 2: load the image folders with PyTorch and look at one batch
# Run from the project root (plant-cv-tutorial), after Lesson 1 has built data/oplp_dataset.
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Every image goes through these steps, in order, each time it is loaded
prep = transforms.Compose([
    transforms.Resize(256),          # shortest side -> 256 px (keeps the shape)
    transforms.CenterCrop(224),      # cut a 224 x 224 square from the middle
    transforms.ToTensor(),           # picture -> tensor of shape (3, 224, 224), values 0..1, RGB order
])

data = datasets.ImageFolder("data/oplp_dataset", transform=prep)
print("classes:", data.classes)                 # folder names, sorted alphabetically
print("class -> number:", data.class_to_idx)
print("images:", len(data))

img, label = data[0]                            # one image and its label number
print("\none image:", img.shape, img.dtype, " min", img.min().item(), " max", img.max().item())
print("its label:", label, "=", data.classes[label])

# A DataLoader hands out images in shuffled groups ("batches")
loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
images, labels = next(iter(loader))             # take the first batch
print("\none batch:", images.shape)              # (16 images, 3 channels, 224, 224)
print("its labels:", labels.tolist())

save_image(images, "outputs/m05_l02_batch.png", nrow=8)   # what the model will actually see
print("saved outputs/m05_l02_batch.png")
