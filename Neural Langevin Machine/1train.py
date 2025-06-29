import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np  # Only used for saving the final J matrix
from config import  delta_t, N, g, n, N_image, k,T ,n_t,b_size,lambda1,N_model
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, ConcatDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def square_mean_torch(tensor):

    squared = torch.square(tensor)
    return torch.sum(squared) / tensor.numel()

kl_norm = []
phi_prime = lambda x: 1.0 - torch.tanh(x) ** 2

transform = transforms.ToTensor()

desired_nums = [1,3,6]

# Loading the complete MNIST dataset
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#Filter samples of target numbers from the training set
train_mask = torch.isin(mnist_train_full.targets, torch.tensor(desired_nums))
train_indices = train_mask.nonzero().squeeze()
mnist_train_selected = Subset(mnist_train_full, train_indices)

# Filter samples with target numbers in the test set
test_mask = torch.isin(mnist_test_full.targets, torch.tensor(desired_nums))
test_indices = test_mask.nonzero().squeeze()
mnist_test_selected = Subset(mnist_test_full, test_indices)

# Merge the filtered training and test sets
all_data = ConcatDataset([mnist_train_selected, mnist_test_selected])

# Create DataLoader
data_loader = torch.utils.data.DataLoader(all_data, batch_size=N_image, shuffle=False)

# Retrieve the first batch of images
images, _ = next(iter(data_loader))
images = images.to(device)  # Move images to GPU

# Prepare the matrix: shape (N, NUM_IMAGES)
matrix = (images.squeeze(1) * 2 - 1).reshape(N_image, -1).T  # Shape: (784, NUM_IMAGES)





class ColumnDataset(Dataset):
    def __init__(self, X):
        super().__init__()
        self.X = X

    def __len__(self):
        return self.X.size(1)

    def __getitem__(self, idx):
        return self.X[:, idx]


dataset = ColumnDataset(matrix)


dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True)

data_iterator = iter(dataloader)


J = torch.randn(N, N, device=device) * torch.sqrt(torch.tensor((g ** 2) / N, device=device))
J.fill_diagonal_(0)

noise_size = torch.sqrt(torch.tensor(2 * T * delta_t, device=device))


num_cols = matrix.shape[1]

X = torch.randn(N, N_model, n_t + 1, device=device)


for epoch in tqdm(range(n)):
    for batch in dataloader:

        batch = batch.T

        phi_data = torch.tanh(batch)
        h_data = J @ phi_data
        x_h_data = batch - h_data
        data = (1 / (T * (batch.shape[1]))) * x_h_data @ (phi_data.T)




        for j in range(n_t):
            tanh_X = torch.tanh(X[:, :, j])
            h = J @ tanh_X
            delta_h = h - X[:, :, j]
            delta_J = J.T @ delta_h
            X[:, :, j + 1] = (
                    (1 - delta_t) * X[:, :, j] +
                    delta_t * h - delta_t * phi_prime(X[:, :, j]) * delta_J +  noise_size* torch.randn(N,N_model,device=device))
        X[:, :, 0] = X[:, :, -1]
        x_model = X[:, :, -1]
        phi_model = torch.tanh(x_model)
        h_model = J @ phi_model
        x_h_model = x_model - h_model
        model = (1 / (T * (x_model.shape[1]))) * x_h_model @ (phi_model.T)
        J = (1-lambda1)*J - k * (model - data)
    klnorm = square_mean_torch((model - data))
    kl_norm.append(klnorm)




np.save('J_cupy1.npy', J.cpu().numpy())

kl_norm1=kl_norm [-5000:]





plt.figure(figsize=(8, 5))
kl_norm_np = [x.cpu().numpy() if torch.is_tensor(x) else x for x in kl_norm]
plt.plot(kl_norm_np, marker='o', linestyle='-', color='b')
plt.xlabel('n')
plt.ylabel('KL Norm Value')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
kl_norm_np = [x.cpu().numpy() if torch.is_tensor(x) else x for x in kl_norm1]
plt.plot(kl_norm_np, marker='o', linestyle='-', color='b')
plt.xlabel('n')
plt.ylabel('KL Norm Value')
plt.grid(True)
plt.show()








