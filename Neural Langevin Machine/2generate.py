import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
import matplotlib

from config import delta_t, N, T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(30)


def phi_prime(x):
    return 1.0 - torch.tanh(x) ** 2
# # Define the plotting function for drawing images in a 6x6 grid
def plot_mnist_grid(columns, grid_size=(6, 6), image_size=(28, 28)):

    batch_size = columns.shape[0]
    rows, cols = grid_size
    assert rows * cols == batch_size

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for idx in range(batch_size):
        image = (columns[idx].cpu().numpy() + 1) / 2.0  # Convert the range [-1,1] to [0,1]
        image = image.reshape(image_size)
        axes[idx].imshow(image, cmap='gray')
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()


#Added the option to store images

def plot_mnist_grid1(columns, grid_size=(6, 6), image_size=(28, 28)):
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
    plt.savefig("output.pdf", format="pdf")
    plt.show()


J = torch.from_numpy(np.load('J_cupy1.npy')).float().to(device)








print(T)

a=  10000
b = 25000
c = 10





batch_size = 36
#Initialization

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

for i in range(a , b,int((b-a)/c)):
    plot_mnist_grid(torch.tanh(Y[:, :, i].T), grid_size=(6, 6), image_size=(28, 28))


plot_mnist_grid1(torch.tanh(Y[:, :, -1].T), grid_size=(6, 6), image_size=(28, 28))


np.save('Y.npy', torch.tanh(Y[:, :, -1]).cpu())














