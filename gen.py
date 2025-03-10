import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from src.models.VisualTransformerGenerator import Transformer

import yaml
import argparse



parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='config/config.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Paramètres du modèle (doivent correspondre à ceux utilisés lors de l'entraînement)
hidden_d = config['model_parameters']['latent_dim']
n_heads = config['model_parameters']['num_heads']
num_layers = config['model_parameters']['num_layers']
d_ff = config['model_parameters']['d_ffn']
dropout = config['trainer_parameters']['dropout']
n_patches = config['model_parameters']['n_patches']   # Doit être un diviseur de 28 si on travaille sur MNIST
device = "cuda"

# Charger le modèle


# Charger le dataset MNIST pour tester
mnist = torchvision.datasets.MNIST(root="./data", train=False, transform=ToTensor(), download=True)
data_loader = torch.utils.data.DataLoader(mnist, batch_size=1, shuffle=True)

# Charger un échantillon
#image, label = next(iter(data_loader))
image, label = next(iter(data_loader))
image= image.to(device)

# Initialiser le modèle avec les mêmes paramètres que dans ton script
model = Transformer(hidden_d, n_heads, num_layers, d_ff, dropout, n_patches)
model.to(device)
model.load_state_dict(torch.load("./final_motel.pth",map_location=device, weights_only=True), strict=False)
model.eval()  # Mode évaluation

# Passer l'image dans le modèle
with torch.no_grad():
    reconstructed_image = model(image)

# Convertir la sortie du modèle en image (il faut reshaper correctement)
reconstructed_image = reconstructed_image.view(28, 28).cpu().numpy()

# Afficher l'image originale et reconstruite
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
ax[0].set_title("Image originale")
ax[1].imshow(reconstructed_image, cmap='gray')
ax[1].set_title("Image reconstruite")
plt.show()