import torch
import matplotlib.pyplot as plt
from src.models.VisualTransformerGenerator import *

model = Transformer(hidden_d=64, n_heads=8, num_layers=4, d_ff=256, dropout=0.1, n_patches=7)
print(torch.cuda.is_available())
DEVICE = torch.device("cpu")

def generate_image(model, device=DEVICE, latent_dim=784):  
    model.eval()
    
    # Générer un vecteur latent aléatoire
    noise = torch.randn(1, latent_dim,device=DEVICE)  # 1 sample, 784 dimensions

    # Passer le bruit à travers le modèle génératif
    with torch.no_grad():
        generated_image = model.classifier(noise)  # Suppose que le modèle génère une image

    # Reshape en 28x28 si nécessaire
    print(generated_image.shape)
    generated_image = generated_image.view(28, 28).cpu().numpy()

    # Afficher l'image générée
    plt.imshow(generated_image, cmap="gray")
    plt.axis("off")
    plt.show()

# Utilisation
generate_image(model, DEVICE    )
