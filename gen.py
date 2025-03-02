import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from VisualTransformerGenerator import Transformer

# Paramètres du modèle (doivent correspondre à ceux utilisés lors de l'entraînement)
hidden_d = 256
n_heads = 8
num_layers = 6
d_ff = 1024
dropout = 0.1
n_patches = 7  # Doit être un diviseur de 28 si on travaille sur MNIST

# Charger le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(hidden_d, n_heads, num_layers, d_ff, dropout, n_patches).to(device)
model.eval()  # Mettre le modèle en mode évaluation

# Charger une image d'entrée et la convertir en tenseur
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Charger une image de test (ex: une image MNIST)
image_path = "test_image.png"  # Remplace avec ton image
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0).to(device)  # Ajouter une dimension batch

# Générer une sortie à partir du modèle
with torch.no_grad():
    output = model(image_tensor)

# Convertir la sortie en image
output_image = output.view(28, 28).cpu().numpy()

# Afficher l'image générée
plt.imshow(output_image, cmap="gray")
plt.title("Image Générée par le Transformer")
plt.axis("off")
plt.show()
