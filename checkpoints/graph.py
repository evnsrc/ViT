import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Configuration
num_shadows = 12
shadow_prefix = "shadow_"
target_filename = "target_model_losses.csv"
output_image = "all_models_train_losses.png"

plt.figure(figsize=(12, 8))

# Pour déterminer automatiquement les limites Y
min_loss = float('inf')
max_loss = -float('inf')

# Lire et tracer les données pour chaque modèle shadow
num_train_shadow_models = 5
num_test_shadow_models = 5
shadow_models = []


# Crée 5 shadows sur les 60 000 images de train
for i in range(num_train_shadow_models):
    print(f"\nTraining shadow model TRAIN {i+1}/5...")
    model = ViTWithTrajectory(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim
    ).to(device)

    indices = np.random.choice(len(full_train_dataset), len(full_train_dataset) // 5, replace=False)
    shadow_subset = Subset(full_train_dataset, indices)
    shadow_loader = DataLoader(shadow_subset, batch_size=batch_size, shuffle=True)

    model.train_model(
        shadow_loader,
        shadow_loader,
        epochs,
        criterion,
        optimizer,
        device,
        save_path=f"./checkpoints/shadow_train_{i}",
        model_name=f"shadow_model_train_{i}"
    )
    shadow_models.append(model)

# Crée 5 shadows sur les 10 000 images de test
for i in range(num_test_shadow_models):
    print(f"\nTraining shadow model TEST {i+1}/5...")
    model = ViTWithTrajectory(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim
    ).to(device)

    indices = np.random.choice(len(full_test_dataset), len(full_test_dataset) // 5, replace=False)
    shadow_subset = Subset(full_test_dataset, indices)
    shadow_loader = DataLoader(shadow_subset, batch_size=batch_size, shuffle=True)

    model.train_model(
        shadow_loader,
        shadow_loader,
        epochs,
        criterion,
        optimizer,
        device,
        save_path=f"./checkpoints/shadow_test_{i}",
        model_name=f"shadow_model_test_{i}"
    )
    shadow_models.append(model)


# Lire et tracer les données pour le modèle target
if os.path.exists(target_filename):
    try:
        df_target = pd.read_csv(target_filename)
        plt.plot(df_target['Epoch'], df_target['Train Loss'], label='Target', color='black', linewidth=2, linestyle='--')
        
        # Mettre à jour les min/max pour le target
        target_min = df_target['Train Loss'].min()
        target_max = df_target['Train Loss'].max()
        if target_min < min_loss:
            min_loss = target_min
        if target_max > max_loss:
            max_loss = target_max
    except Exception as e:
        print(f"Erreur lors de la lecture de {target_filename}: {e}")

# Ajuster les limites Y avec une marge de 10%
margin = 0.1 * (max_loss - min_loss)
plt.ylim(min_loss - margin, max_loss + margin)

# Ajouter des labels et une légende
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Comparaison des Train Loss entre les modèles Shadow et Target')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Ajuster les marges et sauvegarder
plt.tight_layout()
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé sous {output_image}")