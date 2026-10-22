import sys

import cv2
import numpy
import skimage
import torch
import torchvision

print(f"Python:       {sys.version.split()[0]}")
print(f"NumPy:        {numpy.__version__}")
print(f"OpenCV:       {cv2.__version__}")
print(f"scikit-image: {skimage.__version__}")
print(f"PyTorch:      {torch.__version__}  (threads: {torch.get_num_threads()})")
print(f"torchvision:  {torchvision.__version__}")
print(f"tensor test:  {torch.ones(2, 3).sum().item()} (should be 6.0)")
print("All good!")
