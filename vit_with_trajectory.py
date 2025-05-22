import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import os
import csv
import numpy as np

class ViTWithTrajectory(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=1):
        super().__init__()
        # Initialisation originale du ViT
        self.patch_size = patch_size
        self.dim = dim
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        # Pour enregistrer les trajectoires
        self.loss_history = []
        self.train_loss_history = []
        self.val_loss_history = []
    
    def forward(self, img):
        # Transformation des patches
        p = self.patch_size
        patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        
        tokens = self.patch_embedding(patches)
        batch_size, num_patches, _ = tokens.shape
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens += self.pos_embedding[:, :(num_patches + 1)]
        
        encoded = self.transformer(tokens)
        cls_encoded = encoded[:, 0]
        
        return self.mlp_head(cls_encoded)
    
    def train_model(self, train_loader, val_loader, epochs, criterion, optimizer, device, save_path=None, model_name='model'):
        for epoch in range(epochs):
            self.train()
            epoch_train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                outputs = self(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                self.loss_history.append(loss.item())
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            self.train_loss_history.append(avg_train_loss)
            if val_loader is not None:
                avg_val_loss = self.evaluate(val_loader, criterion, device)
                self.val_loss_history.append(avg_val_loss)
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Enregistrement du modèle
            """if save_path:
                os.makedirs(save_path, exist_ok=True)
                model_file = os.path.join(save_path, f"{model_name}_epoch_{epoch+1}.pth")
                torch.save(self.state_dict(), model_file) """

        # Enregistrement des losses après entraînement
        if save_path:
            loss_file = os.path.join(save_path, f"{model_name}_losses.csv")
            with open(loss_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
                for epoch, (train_loss, val_loss) in enumerate(zip(self.train_loss_history, self.val_loss_history), start=1):
                    writer.writerow([epoch, train_loss, val_loss])

    
    def evaluate(self, loader, criterion, device):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                outputs = self(x)
                loss = criterion(outputs, y)
                total_loss += loss.item()
        return total_loss / len(loader)


    def evaluate_model(self, member_loader, non_member_loader, criterion, device):
        """Version modifiée sans seuil prédéfini"""
        self.eval()
        results = {
            'member_losses': [],
            'non_member_losses': [],
            'member_logits': [],
            'non_member_logits': []
        }

        with torch.no_grad():
            # Évaluation membres
            for x, y in member_loader:
                x, y = x.to(device), y.to(device)
                outputs = self(x)
                loss = criterion(outputs, y)
                results['member_losses'].append(loss.item())
                results['member_logits'].append(outputs.cpu().numpy())

            # Évaluation non-membres
            for x, y in non_member_loader:
                x, y = x.to(device), y.to(device)
                outputs = self(x)
                loss = criterion(outputs, y)
                results['non_member_losses'].append(loss.item())
                results['non_member_logits'].append(outputs.cpu().numpy())

        return results  # Retourne les données brutes pour analyse ultérieure