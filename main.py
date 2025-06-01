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
epochs = 80
batch_size = 64


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
    target_loader,  # Utilise le même loader pour train et val
    # Pour l'attaque MIA, on peut utiliser le même loader pour train et val
    #shadow_loader,
    #None,
    epochs,
    criterion,
    optimizer,
    device,
    save_path="./checkpoints/target",
    model_name="target_model",
)

# 2. RECONFIGURATION DES SHADOW MODELS
num_shadow_models = 6

shadow_models = []

# 1. Shadow models entraînés uniquement sur train (30k shadow_train)
for i in range(num_shadow_models):
    print(f"\nTraining TRAIN-ONLY shadow model {i+1}/{num_shadow_models}...")
    model = ViTWithTrajectory(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim
    ).to(device)
    
    # Prend 25k du train_dataset (80%) + 5k validation (20%)
    indices = np.random.choice(len(shadow_train), 25000, replace=False)
    train_subset = Subset(shadow_train, indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    val_indices = [x for x in range(len(shadow_train)) if x not in indices][:5000]
    val_loader = DataLoader(Subset(shadow_train, val_indices), batch_size=batch_size)
    
    model.train_model(
        train_loader,
        val_loader,
        epochs,
        criterion,
        optimizer,
        device,
        save_path=f"./checkpoints/shadow_train_only_{i}",
        model_name=f"shadow_train_only_{i}"
    )
    shadow_models.append(model)

# 2. Shadow models entraînés uniquement sur test (10k test_data)
for i in range(num_shadow_models):
    print(f"\nTraining TEST-ONLY shadow model {i+1}/{num_shadow_models}...")
    model = ViTWithTrajectory(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim
    ).to(device)
    
    # Prend 8k du test (80%) + 2k validation (20%)
    indices = np.random.choice(len(test_data), 8000, replace=False)
    train_subset = Subset(test_data, indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    val_indices = [x for x in range(len(test_data)) if x not in indices][:2000]
    val_loader = DataLoader(Subset(test_data, val_indices), batch_size=batch_size)
    
    model.train_model(
        train_loader,
        val_loader,
        epochs,
        criterion,
        optimizer,
        device,
        save_path=f"./checkpoints/shadow_test_only_{i}",
        model_name=f"shadow_test_only_{i}"
    )
    shadow_models.append(model)

# 3. Shadow models entraînés sur mix 80% train + 20% test
for i in range(num_shadow_models):
    print(f"\nTraining MIXED shadow model {i+1}/{num_shadow_models}...")
    model = ViTWithTrajectory(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim
    ).to(device)
    
    # 20k du shadow_train (80% de 25k) + 5k du test (20% de 25k)
    train_indices = np.random.choice(len(shadow_train), 20000, replace=False)
    test_indices = np.random.choice(len(test_data), 5000, replace=False)
    
    mixed_train = torch.utils.data.ConcatDataset([
        Subset(shadow_train, train_indices),
        Subset(test_data, test_indices)
    ])
    train_loader = DataLoader(mixed_train, batch_size=batch_size, shuffle=True)
    
    # Validation: 5k du train non utilisés + 2k du test non utilisés
    val_train_indices = [x for x in range(len(shadow_train)) if x not in train_indices][:5000]
    val_test_indices = [x for x in range(len(test_data)) if x not in test_indices][:2000]
    
    mixed_val = torch.utils.data.ConcatDataset([
        Subset(shadow_train, val_train_indices),
        Subset(test_data, val_test_indices)
    ])
    val_loader = DataLoader(mixed_val, batch_size=batch_size)
    
    model.train_model(
        train_loader,
        val_loader,
        epochs,
        criterion,
        optimizer,
        device,
        save_path=f"./checkpoints/shadow_mixed_{i}",
        model_name=f"shadow_mixed_{i}"
    )
    shadow_models.append(model)

test_loader = DataLoader(
    Subset(test_dataset, indices=range(5000)),  # On prend les premiers 5000 du test set
    batch_size=batch_size,
    shuffle=False
)
# Après entraînement
member_losses = []
non_member_losses = []

# Évaluation de chaque shadow modèle
for i in range(num_shadow_models):
    mia_results = shadow_models[i].evaluate_model(
        member_loader=val_loader,
        non_member_loader=test_loader,
        criterion=criterion,
        device=device
    )

    # Collecte des pertes pour chaque modèle
    member_losses.extend(mia_results['member_losses'])
    non_member_losses.extend(mia_results['non_member_losses'])

# Calcul des métriques de base
member_loss = np.mean(member_losses)
non_member_loss = np.mean(non_member_losses)
print(f"Perte moyenne membres: {member_loss:.4f}")
print(f"Perte moyenne non-membres: {non_member_loss:.4f}")
print(f"Différence: {non_member_loss - member_loss:.4f}")

# Calcul du seuil optimal
all_losses = np.concatenate([member_losses, non_member_losses])
labels = np.concatenate([
    np.ones(len(member_losses)),
    np.zeros(len(non_member_losses))
])

fpr, tpr, thresholds = roc_curve(labels, -all_losses)  # Note: on utilise -loss pour avoir >= membre
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = -thresholds[optimal_idx]

print(f"\nSeuil optimal calculé: {optimal_threshold:.4f}")

print(f"\nSeuil optimal calculé: {optimal_threshold:.4f}")

# Validation avec le seuil optimal
predictions = (all_losses < optimal_threshold).astype(int)
accuracy = (predictions == labels).mean()
print(f"\nPerformance avec seuil optimal {optimal_threshold:.4f}")
print(f"- Exactitude: {accuracy:.2%}")
print(f"- Vrais positifs: {(predictions[labels == 1] == 1).mean():.2%}")
print(f"- Faux positifs: {(predictions[labels == 0] == 1).mean():.2%}")



# =============================================
# VISUALISATION ET SAUVEGARDE DES RÉSULTATS
# =============================================

import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os
import pickle

# Création du dossier de résultats si inexistant
os.makedirs('./results', exist_ok=True)

def save_roc_curve(fpr, tpr, auc_score, threshold, filename):
    """Sauvegarde la courbe ROC dans un fichier PNG"""
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    
    # Marquer le seuil optimal
    idx = np.argmin(np.abs(thresholds - threshold))
    plt.plot(fpr[idx], tpr[idx], 'ro', 
             label=f'Optimal Threshold {threshold:.2f}\n(FPR={fpr[idx]:.2f}, TPR={tpr[idx]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Courbe ROC pour l\'attaque Membership Inference')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Sauvegarde en PNG
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nCourbe ROC sauvegardée sous {filename}")

def save_loss_distribution(member_losses, non_member_losses, threshold, filename):
    """Sauvegarde l'histogramme des pertes"""
    plt.figure(figsize=(10, 6))
    
    plt.hist(member_losses, bins=50, alpha=0.5, label="Membres")
    plt.hist(non_member_losses, bins=50, alpha=0.5, label="Non-membres")
    plt.axvline(threshold, color='r', linestyle='--', 
                label=f'Seuil optimal ({threshold:.2f})')
    
    plt.xlabel('Loss')
    plt.ylabel('Fréquence')
    plt.title('Distribution des pertes pour membres/non-membres')
    plt.legend()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histogramme des pertes sauvegardé sous {filename}")

# Calcul de l'AUC
roc_auc = auc(fpr, tpr)

# 1. Sauvegarde de la courbe ROC
save_roc_curve(fpr, tpr, roc_auc, optimal_threshold, './results/roc_curve.png')

# 2. Sauvegarde de l'histogramme des pertes
save_loss_distribution(mia_results['member_losses'], 
                      mia_results['non_member_losses'],
                      optimal_threshold,
                      './results/loss_distribution.png')

# 3. Affichage des métriques dans la console
print("\n" + "="*50)
print("RÉSULTATS COMPLETS DE L'ATTAQUE MIA")
print("="*50)
print(f"- Aire sous la courbe ROC (AUC): {roc_auc:.4f}")
print(f"- Seuil optimal calculé: {optimal_threshold:.4f}")
print(f"- Exactitude au seuil optimal: {accuracy:.2%}")
print(f"- Sensibilité (TPR): {(predictions[labels == 1] == 1).mean():.2%}")
print(f"- Spécificité: {(predictions[labels == 0] == 0).mean():.2%}")
print(f"- Taux de Faux Positifs (FPR): {(predictions[labels == 0] == 1).mean():.2%}")

# 4. Sauvegarde des données brutes
results = {
    'fpr': fpr,
    'tpr': tpr,
    'auc': roc_auc,
    'thresholds': thresholds,
    'optimal_threshold': optimal_threshold,
    'member_losses': mia_results['member_losses'],
    'non_member_losses': mia_results['non_member_losses'],
    'metrics': {
        'accuracy': accuracy,
        'tpr': (predictions[labels == 1] == 1).mean(),
        'fpr': (predictions[labels == 0] == 1).mean()
    }
}

with open('./results/mia_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nDonnées brutes sauvegardées sous ./results/mia_results.pkl")
print("Vous pouvez charger ces données pour analyse ultérieure avec:")
print(">>> import pickle\n>>> with open('./results/mia_results.pkl', 'rb') as f:\n>>>     data = pickle.load(f)")