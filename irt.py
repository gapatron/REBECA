import os
import pickle
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

sigmoid = nn.Sigmoid()

def forward(Theta1, Theta2, U, E, Ma, Mb, interaction=True):
    if interaction:
        return sigmoid((Theta1(U)*(E@Ma)).sum(1)+Theta2(U).squeeze()+(E@Mb))
    else:
        return sigmoid(Theta2(U).squeeze()+(E@Mb))
        
def train_irt(U,
              E,
              Y, 
              d,
              lr=1e-2,
              wd=1e-4,
              gamma=.999,
              n_epochs=10000, 
              patience=30,
              tol=1e-4,
              scale=1e-5,
              interaction=True,
              val_size=.1,
              device='cpu',
              random_state=42,
              verbose=True):
    """
    Train a model with the given parameters and dataset.

    Parameters:
        U (Tensor): User indices for training.
        E (Tensor): Embedding indices for training.
        Y (Tensor): Binary labels.
        emb_dim (int): Dimension of embeddings.
        d (int): Dimension of latent factors.
        scale (float): Standard deviation for normal initialization.
        lr (float): Learning rate for the optimizer.
        wd (float): Weight decay for the optimizer.
        gamma (float): Learning rate decay factor.
        n_epochs (int): Number of training epochs.
        patience (int): Early stopping patience.
        tol (float): Tolerance for improvement.
        random_state (int): Random seed for reproducibility.
        device (torch.device): Device to use for computation.
    """

    emb_dim = E.shape[1]
    U_train, U_val, Y_train, Y_val, E_train, E_val = train_test_split(U, Y, E, test_size=val_size, random_state=random_state)
    U_train, U_val, Y_train, Y_val, E_train, E_val = U_train.to(device), U_val.to(device), Y_train.to(device), Y_val.to(device), E_train.to(device), E_val.to(device)
    w_dict = {int(u): 1/float((U==u).float().mean()*torch.unique(U).shape[0]) for u in torch.unique(U)}
    w_train = torch.tensor([w_dict[int(u)] for u in U_train]).to(device)
    w_val = torch.tensor([w_dict[int(u)] for u in U_val]).to(device)
    criterion_train = torch.nn.BCELoss(weight=w_train)
    criterion_val = torch.nn.BCELoss(weight=w_val)
    
    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    
    # Define model parameters
    Theta1 = nn.Embedding(num_embeddings=torch.unique(U).shape[0], embedding_dim=d).to(device)
    Theta2 = nn.Embedding(num_embeddings=torch.unique(U).shape[0], embedding_dim=1).to(device)
    Ma = nn.Parameter(torch.normal(0, scale, size=(emb_dim, d), device=device)).to(device)
    Mb = nn.Parameter(torch.normal(0, scale, size=(emb_dim,), device=device)).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam([Theta1.weight, Theta2.weight, Ma, Mb], lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    # Early stopping variables
    patience_counter = 0
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in tqdm(range(n_epochs), desc="Training Progress", disable=not verbose):
        optimizer.zero_grad()
        
        # Forward pass
        Y_hat = forward(Theta1, Theta2, U_train, E_train, Ma, Mb, interaction)
        
        # Compute loss
        loss = criterion_train(Y_hat, Y_train)
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Evaluate model
        with torch.no_grad():
            #train_loss = criterion(forward(Theta1, Theta2, U_train, E_train, Ma, Mb, interaction), Y_train).item()
            val_loss = criterion_val(forward(Theta1, Theta2, U_val, E_val, Ma, Mb, interaction), Y_val).item()
        
        # Early stopping logic
        if val_loss + tol < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            Theta1_copy = nn.Embedding(num_embeddings=torch.unique(U).shape[0], embedding_dim=d)
            Theta2_copy = nn.Embedding(num_embeddings=torch.unique(U).shape[0], embedding_dim=1)
            Theta1_copy.weight = nn.Parameter(Theta1.weight.detach().cpu().clone(), requires_grad=False)
            Theta2_copy.weight = nn.Parameter(Theta2.weight.detach().cpu().clone(), requires_grad=False)
            
            best_weights = {'Theta1':Theta1_copy,
                            'Theta2':Theta2_copy,
                            'Ma':Ma.detach().cpu().clone(),
                            'Mb':Mb.detach().cpu().clone(),
                            'best_val_loss':val_loss}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if verbose:
        print(f"lr {lr} -- d {d} -- wd {wd} -- gamma {gamma} -- val_loss: {best_val_loss}")
    
    return best_weights

class IRT():
    def __init__(self, interaction = True):
        self.interaction = interaction

    def fit(self,
            U,
            E,
            Y,
            ds = [10, 20, 40, 60, 80, 160, 240, 320],
            lrs = [.01,.001,.0001, 0.00001],
            weight_decays = [1e0,1e-2,1e-4,1e-6],
            gammas = [1,.9999,.999],
            val_size=.1,
            random_state=42,
            train_device='cpu',
            verbose=True):

            assert val_size>0 and val_size<1

            self.validation = {'d': [], 'wd': [], 'lr':[], 'gamma':[], 'val_loss': []}
            self.best_val_loss = float('inf')
        
            for lr in tqdm(lrs, disable=not verbose):
                for gamma in tqdm(gammas, disable=not verbose):
                    for d in tqdm(ds, disable=not verbose):
                        for wd in tqdm(weight_decays, disable=not verbose):
                            
                            output = train_irt(U,
                                               E,
                                               Y, 
                                               d=d,
                                               lr=lr,
                                               wd=wd,
                                               gamma=gamma,
                                               interaction=self.interaction,
                                               val_size=val_size,
                                               device=train_device,
                                               random_state=random_state,
                                               verbose=False)

                            self.validation['lr'].append(lr)
                            self.validation['d'].append(d)
                            self.validation['wd'].append(wd)
                            self.validation['gamma'].append(gamma)
                            self.validation['val_loss'].append(output['best_val_loss'])
                            
                            if output['best_val_loss']<self.best_val_loss:
                                self.d = d
                                self.lr = lr
                                self.wd = wd
                                self.gamma = gamma
                                self.Theta1 = output['Theta1']
                                self.Theta2 = output['Theta2']
                                self.Ma = output['Ma']
                                self.Mb = output['Mb']
                                self.best_val_loss = output['best_val_loss']
                
    def predict(self, U_test, E_test):
        scores = forward(self.Theta1, self.Theta2, U_test, E_test, self.Ma, self.Mb, interaction=self.interaction)
        return scores
        
    def score(self, U_test, E_test):
        scores = self.predict(U_test, E_test)
        U = torch.tensor([int(u) for u in U_test])
        output = {}
        #output['overall'] = {'avg':scores.mean().item(),'std_error':float(scores.std().item()/U.shape[0]**.5)}
        for u in torch.unique(U):
            output[int(u)] = {'avg':scores[U==u].mean().item(),'std_error':float(scores[U==u].std().item()/(torch.sum(U==u))**.5)}
        return output

    def save(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump({
                'd':self.d,
                'lr':self.lr,
                'wd':self.wd,
                'gamma':self.gamma,
                'Theta1':self.Theta1,
                'Theta2':self.Theta2,
                'Ma':self.Ma,
                'Mb':self.Mb,
                'best_val_loss':self.best_val_loss,
                'validation':self.validation,
                'interaction':self.interaction
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.d = params['d']
            self.lr = params['lr']
            self.wd = params['wd']
            self.gamma = params['gamma']
            self.Theta1 = params['Theta1']
            self.Theta2 = params['Theta2']
            self.Ma = params['Ma']
            self.Mb = params['Mb']
            self.best_val_loss = params['best_val_loss']
            self.validation = params['validation']
            self.interaction = params['interaction']
            
