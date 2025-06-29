import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from config import delta_t, N, T, N_image

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
#Image generation function
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
c = 1

#Initialization
batch_size = 36
Y = torch.zeros(size=(N, batch_size, b + 1), device=device)
Y[:, :, 0] = torch.normal(mean=0.0, std=1.0, size=(N, batch_size), device=device)
noise_size = torch.sqrt(torch.tensor(2 * T * delta_t, device=device))
#Dynamic iteration process
for i in tqdm(range(b), desc="Simulations Progress"):
    tanh_Y = torch.tanh(Y[:, :, i])
    h = J@ tanh_Y
    delta_h = h - Y[:, :, i]
    delta_J = J.T@  delta_h
    Y[:, :, i + 1] = (
            (1 - delta_t) * Y[:, :, i] +
            delta_t * h - delta_t * phi_prime(Y[:, :, i]) * delta_J + noise_size * torch.randn(N,batch_size,device=device))


goal =Y[:, -1, a:].T.cpu().numpy()

#Collect the last point
t=[]
tra2=[]
for i in tqdm(range(15000)):
    t.append(i*0.01)
    tra2.append(goal[i,234])


plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 15


plt.figure(figsize=(6, 3.5))


line = plt.plot(t, tra2 , color='#1f77b4', linewidth=1.5,label='Trajectory1', zorder=3)



plt.xlabel('Time ', fontsize=15, labelpad=5)
plt.ylabel('$Activity\ {x_i}$ ', fontsize=18, labelpad=5)


ax = plt.gca()
ax.grid(True, which='both', linestyle=':', linewidth=0.5,
        color='gray', alpha=0.4, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', direction='in',
               width=0.8, length=4, labelsize=15)


plt.legend(frameon=True, framealpha=1, edgecolor='none',
           facecolor='white', fontsize=10, loc='best')


plt.tight_layout(pad=1.5)


plt.show()


