import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from vit_with_trajectory import ViTWithTrajectory
from trajectory_mia import TrajectoryMIA
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_curve



# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 32
patch_size = 8
num_classes = 10
dim = 64
depth = 6
heads = 8
mlp_dim = 256
epochs = 4
batch_size = 64
"""
MNIST (70k)
├── Target (30k train) → Modèle cible
├── Shadow (30k train) → 5× (25k train + 5k val)
└── Test (10k) → Évaluation
"""

# Chargement des données MNIST
transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalisation pour MNIST
    ]
)

# Chargement des datasets originaux
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

### NOUVELLE ORGANISATION SELON LES CONSIGNES ###

# 1. Division stricte train/test (comme dans le papier original)
# Modèle cible : 30 000 premiers exemples du train
target_train = Subset(train_dataset, indices=range(30_000))

# Données pour shadow models : 30 000 derniers exemples du train
shadow_train = Subset(train_dataset, indices=range(30_000, len(train_dataset)))

# Test set : utilisé uniquement pour les non-membres
test_data = test_dataset  # 10 000 exemples

# 2. Création des DataLoaders
target_loader = DataLoader(target_train, batch_size=batch_size, shuffle=True)

# Pour les shadow models (exemple pour 1 shadow model)
shadow_loader = DataLoader(shadow_train, batch_size=batch_size, shuffle=True)

# 3. Préparation pour l'attaque :
# - Membres : sous-ensemble de shadow_train non vu à l'entraînement (ex: 5 000)
# - Non-membres : sous-ensemble de test_dataset
attack_member_data = Subset(shadow_train, indices=range(5_000))
attack_non_member_data = Subset(test_data, indices=range(5_000))

attack_loader = DataLoader(
    torch.utils.data.ConcatDataset([attack_member_data, attack_non_member_data]),
    batch_size=batch_size,
    shuffle=False,
)

# Initialisation du modèle cible
target_model = ViTWithTrajectory(
    image_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
).to(device)

# Entraînement du modèle cible
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)

print("Training target model...")
target_model.train_model(
    target_loader,
    # shadow_loader,
    None,
    epochs,
    criterion,
    optimizer,
    device,
    save_path="./checkpoints/target",
    model_name="target_model",
)

# 2. RECONFIGURATION DES SHADOW MODELS
num_shadow_models = 5

shadow_models = []


for i in range(num_shadow_models):
    print(f"\nTraining shadow model {i+1}/{num_shadow_models}...")
    model = ViTWithTrajectory(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim
    ).to(device)

    # 3. NOUVELLE STRATÉGIE DE SÉLECTION DES DONNÉES
    # Prend 25k images aléatoires parmi les 30k shadow_train
    indices = np.random.choice(len(shadow_train), 25000, replace=False)
    shadow_subset = Subset(shadow_train, indices)
    shadow_loader = DataLoader(shadow_subset, batch_size=batch_size, shuffle=True)


    # 4. AJOUT DE LA VALIDATION
    val_indices = [x for x in range(len(shadow_train)) if x not in indices][:5000]
    val_loader = DataLoader(Subset(shadow_train, val_indices), batch_size=batch_size)

    model.train_model(
        shadow_loader,
        #shadow_loader,
        val_loader,
        epochs,
        criterion,
        optimizer,
        device,
        save_path=f"./checkpoints/shadow_{i}",
        model_name=f"shadow_model_{i}",
    )
    shadow_models.append(model)

test_loader = DataLoader(
    Subset(test_dataset, indices=range(5000)),  # On prend les premiers 5000 du test set
    batch_size=batch_size,
    shuffle=False
)

    # Après entraînement
mia_results = shadow_models[-1].evaluate_model(  # On utilise le dernier shadow model
    member_loader=val_loader,
    non_member_loader=test_loader,
    criterion=criterion,
    device=device
)


# Calcul des métriques de base
member_loss = np.mean(mia_results['member_losses'])
non_member_loss = np.mean(mia_results['non_member_losses'])
print(f"Perte moyenne membres: {member_loss:.4f}")
print(f"Perte moyenne non-membres: {non_member_loss:.4f}")
print(f"Différence: {non_member_loss - member_loss:.4f}")

# Calcul du seuil optimal 
all_losses = np.concatenate([mia_results['member_losses'], mia_results['non_member_losses']])
labels = np.concatenate([
    np.ones(len(mia_results['member_losses'])),
    np.zeros(len(mia_results['non_member_losses']))
])

fpr, tpr, thresholds = roc_curve(labels, -all_losses)  # Note: on utilise -loss pour avoir > = membre
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = -thresholds[optimal_idx]

print(f"\nSeuil optimal calculé: {optimal_threshold:.4f}")
print(f"Seuil du tuteur: 2.35")

# Validation avec le seuil de 2.35
predictions = (all_losses < 2.35).astype(int)
accuracy = (predictions == labels).mean()
print(f"\nPerformance avec seuil=2.35:")
print(f"- Exactitude: {accuracy:.2%}")
print(f"- Vrais positifs: {(predictions[labels == 1] == 1).mean():.2%}")
print(f"- Faux positifs: {(predictions[labels == 0] == 1).mean():.2%}")