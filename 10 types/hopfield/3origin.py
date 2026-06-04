import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 1. Data Preparation: Ensure consistency with previous scripts
torch.manual_seed(42)
transform = transforms.ToTensor()
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

patterns = []
labels_found = set()

# Iterate through the dataset to extract the same 10 digits
for img, label in mnist_train_full:
    if label not in labels_found:
        # Keep the original 28x28 image format and [0, 1] pixel range for easy plotting
        patterns.append(img.squeeze())
        labels_found.add(label)
    if len(labels_found) == 10:
        break

# 2. Extract the specific digit corresponding to the [6:7] slice
# The list 'patterns' contains 10 elements. Index 6 corresponds to the 7th image.
original_img = patterns[6]

# 3. Plot and save the original image
fig, ax = plt.subplots(figsize=(2, 2))
ax.imshow(original_img, cmap='gray')
ax.axis('off')  # Remove axes for a clean visualization

# Save the original image as a PDF file
save_path = 'original_digit_6.pdf'
plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0)
plt.close(fig)

print(f"Original image successfully saved to: {save_path}")