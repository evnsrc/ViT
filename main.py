import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from vit_with_trajectory import ViTWithTrajectory
from trajectory_mia import TrajectoryMIA
import numpy as np
import torch.nn as nn

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 32
patch_size = 8
num_classes = 10
dim = 128
depth = 6
heads = 8
mlp_dim = 256
epochs = 100
batch_size = 64

# Chargement des données CIFAR-10
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Division en ensembles target et shadow
indices = np.arange(len(full_dataset))
np.random.shuffle(indices)

target_indices = indices[:len(indices)//2]
shadow_indices = indices[len(indices)//2:]

target_dataset = Subset(full_dataset, target_indices)
shadow_dataset = Subset(full_dataset, shadow_indices)

# Création des loaders
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
shadow_loader = DataLoader(shadow_dataset, batch_size=batch_size, shuffle=True)

# Division train/test pour l'attaque
target_train_size = int(0.8 * len(target_dataset))
target_train_indices = list(range(target_train_size))  # indices relatifs à target_dataset
target_test_indices = list(range(target_train_size, len(target_dataset)))


# Initialisation du modèle cible
target_model = ViTWithTrajectory(
    image_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim
).to(device)

# Entraînement du modèle cible
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)

print("Training target model...")
target_model.train_model(
    target_loader,
    shadow_loader,
    epochs,
    criterion,
    optimizer,
    device,
    save_path="./checkpoints/target",
    model_name="target_model"
)

# Création de modèles d'ombre
num_shadow_models = 20
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
    
    # Sous-ensemble des données d'ombre pour ce modèle
    shadow_subset = Subset(shadow_dataset, np.random.choice(len(shadow_dataset), len(shadow_dataset)//2, replace=False))
    shadow_sub_loader = DataLoader(shadow_subset, batch_size=batch_size, shuffle=True)
    
    model.train_model(
        shadow_sub_loader,
        target_loader,
        epochs,
        criterion,
        optimizer,
        device,
        save_path=f"./checkpoints/shadow_{i}",
        model_name=f"shadow_model_{i}"
    )
    shadow_models.append(model)
"""
# Préparation des données pour l'attaque
def prepare_attack_data(model, dataset, is_member):
    data = []
    for idx in np.random.choice(len(dataset), min(500, len(dataset)), replace=False):
        x, y = dataset[idx]
        data.append((x, y, is_member))
    return data
print("Preparing attack data...")
# Données cibles (membres et non-membres)
# Créer un Subset correct à partir de target_dataset
target_member_data = prepare_attack_data(target_model, Subset(target_dataset, target_train_indices), 1)
print(f"Target member data size: {len(target_member_data)}")
# Créer un Subset correct à partir de target_dataset
target_non_member_data = prepare_attack_data(target_model, Subset(target_dataset, target_test_indices), 0)


# Données d'ombre (membres et non-membres)
shadow_member_data = []
shadow_non_member_data = []

total_shadow_size = len(shadow_dataset)
split_size = total_shadow_size // num_shadow_models
shadow_splits = [list(range(i * split_size, (i + 1) * split_size)) for i in range(num_shadow_models)]

for i, model in enumerate(shadow_models):
    shadow_train_size = int(0.8 * split_size)
    train_indices = shadow_splits[i][:shadow_train_size]
    subset = Subset(shadow_dataset, train_indices)
    shadow_member_data.extend(prepare_attack_data(model, subset, 1))

    # Données non-membres
    shadow_non_member_data.extend(
        prepare_attack_data(model, Subset(target_dataset, np.random.choice(len(target_dataset), shadow_train_size, replace=False)), 0)
    )

# Combinaison des données
target_data = target_member_data + target_non_member_data
shadow_data = list(zip(shadow_models, shadow_member_data + shadow_non_member_data))

# Entraînement de l'attaque
mia = TrajectoryMIA(target_model, shadow_models)
mia.train_attack_model(target_data, shadow_data)

# Test de l'attaque sur quelques échantillons
test_samples = 5
for i in range(test_samples):
    x, y, _ = target_member_data[i]
    prob = mia.infer_membership(x, y)
    print(f"Sample {i+1} (member) - Predicted membership probability: {prob:.4f}")

for i in range(test_samples):
    x, y, _ = target_non_member_data[i]
    prob = mia.infer_membership(x, y)
    print(f"Sample {i+1} (non-member) - Predicted membership probability: {prob:.4f}")

"""