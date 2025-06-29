import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from config import delta_t, N, T,N_image1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
J = torch.from_numpy(np.load('J_cupy1.npy')).float().to(device)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#set_seed(42)
set_seed(30)
def phi_prime(x):
    return 1.0 - torch.tanh(x) ** 2

def plot_mnist_grid(columns, grid_size=(6, 6), image_size=(28, 28)):
    batch_size = columns.shape[0]
    rows, cols = grid_size
    assert rows * cols == batch_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    for idx in range(batch_size):
        image = (columns[idx].cpu().numpy() + 1) / 2.0
        image = image.reshape(image_size)
        axes[idx].imshow(image, cmap='gray')
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()


a=  10000
b =25000
c = 10

#Initialization
batch_size = 36
Y = torch.zeros(size=(N, batch_size, b + 1), device=device)
Y[:, :, 0] = torch.normal(mean=0.0, std=1.0, size=(N, batch_size), device=device)
noise_size = torch.sqrt(torch.tensor(2 * T * delta_t, device=device))

##Dynamic iteration process
for i in tqdm(range(b), desc="Simulations Progress"):
    tanh_Y = torch.tanh(Y[:, :, i])
    h = J@ tanh_Y
    delta_h = h - Y[:, :, i]
    delta_J = J.T@  delta_h
    Y[:, :, i + 1] = (
            (1 - delta_t) * Y[:, :, i] +
            delta_t * h - delta_t * phi_prime(Y[:, :, i]) * delta_J + noise_size * torch.randn(N,batch_size,device=device))

for i in range(a , b,int((b-a)/c)):
    print(i)
    plot_mnist_grid(torch.tanh(Y[:, :, i].T), grid_size=(6, 6), image_size=(28, 28))











#Filter numbers
transform = transforms.ToTensor()
desired_nums = [1, 3, 6]
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_mask = torch.isin(mnist_train_full.targets, torch.tensor(desired_nums))
train_indices = train_mask.nonzero().squeeze()
mnist_train_selected = Subset(mnist_train_full, train_indices)
test_mask = torch.isin(mnist_test_full.targets, torch.tensor(desired_nums))
test_indices = test_mask.nonzero().squeeze()
mnist_test_selected = Subset(mnist_test_full, test_indices)
all_data_selected = ConcatDataset([mnist_train_selected, mnist_test_selected])
data_loader = DataLoader(all_data_selected, batch_size=N_image1, shuffle=False)
images_for_pca, labels_for_pca = next(iter(data_loader))
actual_num_images = images_for_pca.shape[0]
flattened_images = (images_for_pca.squeeze(1).reshape(actual_num_images, -1) * 2 - 1)
images_np = flattened_images.cpu().numpy()

#pca
pca = PCA(n_components=3)
pca_result = pca.fit_transform(images_np)
labels_np = labels_for_pca.cpu().numpy()
colors_map = {1: 'red', 3: 'green', 6: 'blue'}
processed_new_vectors= torch.tanh(Y[:, -1, a:].T).cpu().numpy()
print(processed_new_vectors.shape)
#Obtain the time series of the last image
for i in range(a , b,int((b-a)/c)):
    image = processed_new_vectors[i-a,:].reshape(28, 28)
    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    #plt.savefig('i.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()




new_pca_result = pca.transform(processed_new_vectors)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.tick_params(axis='both', which='major', labelsize=25)

plot_legend = False
for digit in desired_nums:
    indices = (labels_np == digit)
    plot_legend = True
    ax.scatter(pca_result[indices, 0],
               pca_result[indices, 1],
               pca_result[indices, 2],
               color=colors_map[digit],
               label=f'Digit {digit}',
               alpha=0.3,
               s=50)

trajectory = ax.scatter(new_pca_result[:, 0],new_pca_result[:, 1],new_pca_result[:, 2],color='black',  marker='o',s=1, alpha=0.7,label='Trajectory')


ax.scatter(new_pca_result[10000-a, 0], new_pca_result[10000-a, 1], new_pca_result[10000-a, 2],color='yellow', marker='*', s=300)
ax.scatter(new_pca_result[16000-a, 0], new_pca_result[16000-a, 1], new_pca_result[16000-a, 2],color='blue', marker='*', s=300)
ax.scatter(new_pca_result[19000-a, 0], new_pca_result[19000-a, 1], new_pca_result[19000-a, 2],color='yellow', marker='*', s=300)

ax.scatter(new_pca_result[23500-a, 0], new_pca_result[23500-a, 1], new_pca_result[23500-a, 2],color='blue', marker='*', s=300)

ax.set_xlabel('X', fontsize=25, labelpad=25)
ax.set_ylabel('Y', fontsize=25, labelpad=25)
ax.set_zlabel('Z', fontsize=25, labelpad=25)

ax.legend(fontsize=16, loc='upper right')

plt.grid(True)

plt.tight_layout()

plt.show()












