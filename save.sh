#!/bin/bash

# Définition des paramètres
USER="eracat"
SERVER="lille.grid5000.fr"
PLACE=access.grid5000.fr
REMOTE_PATH="lille/ViT/Train0019"
REMOTE_PATH2="lille/ViT/loss.csv"
LOCAL_PATH="/c/Users/evans/OneDrive - IMTBS-TSP/Cours TSP/2A/Cassiopée/Results/"
LOCAL_PATH2="/c/Users/evans/OneDrive - IMTBS-TSP/Cours TSP/2A/Cassiopée/Results/Train0019/"

# Nombre d'heures pendant lesquelles exécuter le script
DURATION=8 
DURATION2=48

for ((i=0; i<DURATION*6; i++)); do
    echo "----- Tentative $(($i+1)) sur $DURATION2 -----"

        echo "Transfert en cours..."
            scp -r "$USER@$PLACE:$REMOTE_PATH" "$LOCAL_PATH"
            scp "$USER@$PLACE:$REMOTE_PATH2" "$LOCAL_PATH2"
        if [ $? -eq 0 ]; then
            echo "Transfert réussi."
        else
            echo "⚠️ Erreur lors du transfert."
        fi

    
    if [ $((i+1)) -lt $DURATION ]; then
        echo "Attente de 10min avant la prochaine tentative..."
        sleep 600  # Attente de 10 min
    fi
done

echo "✅ Transfert terminé après $DURATION minutes."



