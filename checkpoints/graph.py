import os
import pandas as pd
import matplotlib.pyplot as plt

num_shadow = 20

# Configuration - Génération automatique des noms
shadow_dirs = [f'shadow_{i}' for i in range(num_shadow)]  # shadow_0 à shadow_19
directories = shadow_dirs + ['target']  # Ajoute le target à la fin

file_names = {f'shadow_{i}': f'shadow_model_{i}_losses.csv' for i in range(num_shadow)}
file_names['target'] = 'target_model_losses.csv'

# Créer une nouvelle figure avec une taille adaptée à beaucoup de courbes
plt.figure(figsize=(14, 8))

# Palette de couleurs distinctes
colors = plt.cm.tab20.colors  # Utilise une palette avec 20 couleurs distinctes

# Lire et plotter chaque fichier CSV
for idx, dir_name in enumerate(directories):
    file_path = os.path.join(dir_name, file_names.get(dir_name, ''))
    
    try:
        # Lire le fichier CSV
        df = pd.read_csv(file_path)
        
        # Choisir une couleur - on réutilise les couleurs si plus de 20 modèles
        color = colors[idx % len(colors)]
        
        # Style différent pour le target model
        #linewidth = 3 if dir_name == 'target' else 2
        #linestyle = '-' if dir_name == 'target' else '--'
        linewidth = 0.8
        linestyle = '-'
        
        # Plotter la courbe Train Loss
        plt.plot(df['Epoch'], df['Train Loss'], 
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                label=f'{dir_name.replace("_", " ").title()}')

    except FileNotFoundError:
        print(f"⚠ Fichier non trouvé : {file_path}")
    except Exception as e:
        print(f"Erreur avec le fichier {file_path}: {str(e)}")

# Personnalisation avancée du graphique
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Train Loss', fontsize=12)
plt.title('Comparaison des Train Loss: Shadow Models (0-19) vs Target', fontsize=14, pad=20)
plt.grid(True, linestyle=':', alpha=0.6)

# Légende externe à droite avec défilement si nécessaire
plt.legend(fontsize=9, 
          bbox_to_anchor=(1.05, 1), 
          loc='upper left',
          ncol=1,
          title='Modèles')

# Ajuster les marges
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Réduit la largeur pour faire place à la légende

# Enregistrer la figure
output_filename = 'comparaison_train_losses.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé sous {output_filename}")

# Fermer la figure
plt.close()