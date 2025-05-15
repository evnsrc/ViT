import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn

class TrajectoryMIA:
    def __init__(self, target_model, shadow_models):
        self.target_model = target_model
        self.shadow_models = shadow_models
        self.attack_model = RandomForestClassifier(n_estimators=100)
    
    def extract_features(self, model, x, y, criterion):
        """Extrait les caractéristiques de trajectoire pour un échantillon"""
        features = []
        
        # 1. Perte finale
        with torch.no_grad():
            x = x.to(next(model.parameters()).device)
            #print("x est de type : " + str(type(x)))  
            output = model(x.unsqueeze(0))
            y = torch.tensor(y, device=x.device)

            final_loss = criterion(output, y.unsqueeze(0)).item()
        
        # 2. Statistiques sur l'historique de perte
        if hasattr(model, 'loss_history') and model.loss_history:
            loss_history = np.array(model.loss_history)
            features.extend([
                final_loss,
                np.mean(loss_history),
                np.std(loss_history),
                np.min(loss_history),
                np.max(loss_history),
                np.percentile(loss_history, 25),
                np.percentile(loss_history, 50),
                np.percentile(loss_history, 75),
                (loss_history[-1] - loss_history[0]) / len(loss_history),  # pente moyenne
            ])
        else:
            features.extend([final_loss] + [0]*8)
        
        return np.array(features)
    
    def train_attack_model(self, target_data, shadow_data):
        """Entraîne le modèle d'attaque"""
        X, y = [], []
        
        # Données cibles
        for (x, label, is_member) in target_data:
            features = self.extract_features(self.target_model, x, label, nn.CrossEntropyLoss())
            X.append(features)
            y.append(is_member)
        
        # Données d'ombre
        for shadow_model, (x, label) in zip(self.shadow_models, shadow_data):
            is_member = 1  # ou 0 selon le contexte (shadow model => en général membre)

            features = self.extract_features(shadow_model, x, label, nn.CrossEntropyLoss())
            X.append(features)
            y.append(is_member)
        
        X = np.array(X)
        y = np.array(y)
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entraînement
        self.attack_model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = self.attack_model.predict(X_test)
        y_prob = self.attack_model.predict_proba(X_test)[:, 1]
        
        print(f"Attack Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Attack Model AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    def infer_membership(self, x, y):
        """Infère l'appartenance d'un échantillon"""
        features = self.extract_features(self.target_model, x, y, nn.CrossEntropyLoss())
        prob = self.attack_model.predict_proba(features.reshape(1, -1))[0, 1]
        return prob