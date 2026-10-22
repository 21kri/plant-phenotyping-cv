# Module 5, Lesson 5: remove the background so the model can only learn from the plants
# Run from the project root (plant-cv-tutorial), after Lesson 1 has built data/oplp_dataset.
from oplp import mask_folder, outside_test, train

# 1. Plant-only copies of the training images AND the outside test images (same rule for both)
mask_folder("data/oplp_dataset", "data/oplp_dataset_masked")
mask_folder("data/oplp_outside_test", "data/oplp_outside_test_masked")
print("made data/oplp_dataset_masked and data/oplp_outside_test_masked")

# 2. Train exactly as in Lesson 3, but on the plant-only images
model = train("data/oplp_dataset_masked", "outputs/m05_oplp_resnet18_masked.pt")

# 3. The real test: the white-room photos, also plant-only
print("\nOutside test (white room, background removed):")
outside_test(model, "data/oplp_outside_test_masked")
