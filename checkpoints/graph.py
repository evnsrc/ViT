import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuration des répertoires à explorer
directories = [f"shadow_test_only_{i}" for i in range(3)] + \
              [f"shadow_train_only_{i}" for i in range(3)] + \
              [f"shadow_mixed_{i}" for i in range(3)] + \
              ["target"]

# Fichiers attendus
file_names = {
    'shadow_mixed_0': 'shadow_mixed_0_losses.csv',
    'shadow_mixed_1': 'shadow_mixed_1_losses.csv',
    'shadow_mixed_2': 'shadow_mixed_2_losses.csv',
    'shadow_train_only_0': 'shadow_train_only_0_losses.csv',
    'shadow_train_only_1': 'shadow_train_only_1_losses.csv',
    'shadow_train_only_2': 'shadow_train_only_2_losses.csv',
    'shadow_test_only_0': 'shadow_test_only_0_losses.csv',
    'shadow_test_only_1': 'shadow_test_only_1_losses.csv',
    'shadow_test_only_2': 'shadow_test_only_2_losses.csv',
    'target': 'target_model_losses.csv'
}

# Association des types de modèle à des couleurs
color_map = {
    'shadow_test_only': 'blue',
    'shadow_train_only': 'green',
    'shadow_mixed': 'orange',
    'target': 'red'
}

# Création du graphique
plt.figure(figsize=(12, 7))

# Chargement et affichage
for dir_name in directories:
    file_path = os.path.join(dir_name, file_names.get(dir_name, ''))
    
    try:
        df = pd.read_csv(file_path)

        # Détection du type de modèle
        if "shadow_test_only" in dir_name:
            color = color_map['shadow_test_only']
        elif "shadow_train_only" in dir_name:
            color = color_map['shadow_train_only']
        elif "shadow_mixed" in dir_name:
            color = color_map['shadow_mixed']
        elif "target" in dir_name:
            color = color_map['target']
        else:
            color = 'black'  # Défaut

        # Affichage
        plt.plot(df['Epoch'], df['Train Loss'],
                 label=dir_name.replace("_", " ").title(),
                 linewidth=2,
                 color=color)

    except FileNotFoundError:
        print(f"Fichier non trouvé : {file_path}")
    except Exception as e:
        print(f" Erreur avec le fichier {file_path}: {str(e)}")

# Finalisation du graphe
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Train Loss', fontsize=12)
plt.title('Comparaison des Train Loss des modèles', fontsize=14, pad=20)
plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Sauvegarde
output_filename = 'comparaison_train_losses_colored.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f" Graphique sauvegardé sous {output_filename}")

plt.close()
