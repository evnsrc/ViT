import pickle
import pandas as pd

# Charger les données pickle
with open('./Results/mia_results.pkl', 'rb') as f:
    data = pickle.load(f)

# Vérifie que c'est bien un dictionnaire
if isinstance(data, dict):
    # Convertir en DataFrame
    df = pd.DataFrame(data)
    
    # Sauvegarder en fichier Excel
    df.to_excel('donnees.xlsx', index=False)
    print("Fichier Excel créé avec succès : donnees.xlsx, baby !")
else:
    print("Erreur : les données ne sont pas un dictionnaire, baby !")
