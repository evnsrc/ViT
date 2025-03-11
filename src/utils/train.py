'''
Developed from scratch by Mert Sengul.
Please cite the repo if you readapt.
'''

import torch
from tqdm import tqdm
import numpy as np
import os
#import wandb


class Trainer:
    '''
    Trainer object.
    '''
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        checkpoint,
        experiment,
        device,
        latent_dim,
        d_ffn,
        n_patches,
        num_heads,
        num_layers,
    ):

        super().__init__()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.experiment = experiment
        self.device = device
        self.latent_dim = latent_dim
        self.d_ffn = d_ffn
        self.n_patches = n_patches
        self.num_heads = num_heads
        self.num_layers = num_layers
    
    def fit(self, train_loader, val_loader, epochs):
        torch.manual_seed(42)
        self.model.to(self.device)
        tr_loss=[]
        val_loss=[]
        #self._wandb_initiate()
        for epoch in tqdm(range(epochs)):
            # train
            train_loss = self._train(train_loader)
            tr_loss.append(train_loss)

            #wandb.log({"Training Loss": train_loss})
            # validate
            val_loss = self._validate(val_loader)
            val_loss.append(val_loss)

            #wandb.log({"Validation Loss": val_loss})
            #save model state
            self._save_checkpoint(train_loss, val_loss, epoch)
        return tr_loss, val_loss
            

    #def _wandb_initiate(self):
        #### W&B INIT ###
        #wandb.init(project=self.experiment)
        #wandb.watch(
        #    self.model,
        #    criterion=self.criterion,
        #    log= 'parameters',#Optional[Literal['gradients', 'parameters', 'all']] = "gradients",
        #    log_freq=100,
        #    log_graph=True
        #    )
        ###############

    def _save_checkpoint(self, train_loss, val_loss, epoch):
        path = '{}/{}'.format(self.checkpoint, self.experiment)
        if not os.path.isdir(path):
            os.mkdir(path)
        
        torch.save(self.model.state_dict(), '{}/checkpoint_{}.pth'.format(path, epoch))

    def _train(self, loader):
        _loss = []
        
        self.model.train()

        for images, y in tqdm(loader):            
            self.optimizer.zero_grad()

            #images, _ = data # No need to return
            images = images.to(self.device)

            y = y.to(self.device)
            print(y.shape)  # Doit afficher torch.Size([128])
            #print("test = " + str(y[:5]))    # Vérifie s'il contient des indices de classes (0, 1, 2...)


            reconstructed_image = self.model(images)
            #print("reconstructed_image est de la forme : "+ str(reconstructed_image.shape))
            

            loss = self.criterion(reconstructed_image.view(y.shape[0], -1), y)

            loss.backward()
            self.optimizer.step()

            _loss.append(loss.detach().cpu().item())
            torch.cuda.empty_cache()

        epoch_loss = np.mean(_loss)
        
        return epoch_loss

    def _validate(self, loader):
        _loss = []
       
        self.model.eval()

        with torch.no_grad():
            for images, y in tqdm(loader):   
                #images, _ = data # No need to return
                images = images.to(self.device)
                y = y.to(self.device)
                print("y est de la forme : "+ str(y.shape))
                
                reconstructed_image = self.model(images)
                
                loss = self.criterion(reconstructed_image.view(y.shape[0], -1), y)

                _loss.append(loss.detach().cpu().item())
                torch.cuda.empty_cache()

        epoch_loss = np.mean(_loss)

        return epoch_loss
        

    