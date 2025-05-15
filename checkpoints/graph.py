import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Configuration
num_shadows = 12
shadow_prefix = "shadow_"
target_filename = "target_model_losses.csv"
output_image = "checkpoints/all_models_train_losses.png"

plt.figure(figsize=(12, 8))

# Pour déterminer automatiquement les limites X et Y
min_epoch = float('inf')
max_epoch = -float('inf')
min_loss = float('inf')
max_loss = -float('inf')

# Lire et tracer les données pour chaque modèle shadow
for i in range(num_shadows):
    filename = f"{shadow_prefix}{i}_losses.csv"
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            plt.plot(df['Epoch'], df['Train Loss'], label=f'Shadow {i}', alpha=0.5)
            
            # Mettre à jour les min/max
            current_min_loss = df['Train Loss'].min()
            current_max_loss = df['Train Loss'].max()
            current_max_epoch = df['Epoch'].max()
            
            if current_min_loss < min_loss:
                min_loss = current_min_loss
            if current_max_loss > max_loss:
                max_loss = current_max_loss
            if current_max_epoch > max_epoch:
                max_epoch = current_max_epoch
        except Exception as e:
            print(f"Erreur lors de la lecture de {filename}: {e}")

# Lire et tracer les données pour le modèle target
if os.path.exists(target_filename):
    try:
        df_target = pd.read_csv(target_filename)
        plt.plot(df_target['Epoch'], df_target['Train Loss'], label='Target', color='black', linewidth=2, linestyle='--')
        
        # Mettre à jour les min/max pour le target
        target_min = df_target['Train Loss'].min()
        target_max = df_target['Train Loss'].max()
        target_max_epoch = df_target['Epoch'].max()
        
        if target_min < min_loss:
            min_loss = target_min
        if target_max > max_loss:
            max_loss = target_max
        if target_max_epoch > max_epoch:
            max_epoch = target_max_epoch
    except Exception as e:
        print(f"Erreur lors de la lecture de {target_filename}: {e}")

# Définir manuellement les limites des axes
#plt.xlim(0, max_epoch)  # De 0 au max des epochs
#plt.ylim(min_loss - 0.1, max_loss + 0.1)  # Avec une petite marge
plt.xlim(0, 100)
plt.ylim(0, 4)

# Ajouter des labels et une légende (inchangé)
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Comparaison des Train Loss entre les modèles Shadow et Target')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Ajuster les marges et sauvegarder (inchangé)
plt.tight_layout()
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé sous {output_image}")