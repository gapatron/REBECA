import os
import pickle
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import pandas as pd


sigmoid = nn.Sigmoid()

class NCFModel(nn.Module):
    """Neural Collaborative Filtering Model"""
    def __init__(self, num_users, emb_dim, d, mlp_layers=[128, 64, 32], dropout=0.2):
        super(NCFModel, self).__init__()
        self.num_users = num_users
        self.emb_dim = emb_dim
        self.d = d
        
        # GMF (Generalized Matrix Factorization) components - similar to original IRT
        self.user_embedding_gmf = nn.Embedding(num_users, d)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_transform_gmf = nn.Linear(emb_dim, d)
        self.item_bias = nn.Linear(emb_dim, 1)
        
        # MLP components for capturing nonlinear interactions
        self.user_embedding_mlp = nn.Embedding(num_users, d)
        self.item_transform_mlp = nn.Linear(emb_dim, d)
        
        # MLP layers
        mlp_input_dim = d * 2  # concatenated user and item embeddings
        self.mlp_layers = nn.ModuleList()
        prev_dim = mlp_input_dim
        
        for layer_dim in mlp_layers:
            self.mlp_layers.append(nn.Linear(prev_dim, layer_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            prev_dim = layer_dim
        
        # Final fusion layer
        self.final_layer = nn.Linear(d + mlp_layers[-1], 1)
        
    def forward(self, U, E):
        # GMF component (linear interactions like original IRT)
        user_emb_gmf = self.user_embedding_gmf(U)  # [batch, d]
        item_emb_gmf = self.item_transform_gmf(E)  # [batch, d]
        gmf_output = user_emb_gmf * item_emb_gmf  # Element-wise product [batch, d]
        
        # MLP component (nonlinear interactions)
        user_emb_mlp = self.user_embedding_mlp(U)  # [batch, d]
        item_emb_mlp = self.item_transform_mlp(E)  # [batch, d]
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=1)  # [batch, 2*d]
        
        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
        
        # Combine GMF and MLP outputs
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        
        # Add biases (similar to original IRT)
        user_bias = self.user_bias(U).squeeze()  # [batch]
        item_bias = self.item_bias(E).squeeze()  # [batch]
        
        # Final prediction
        prediction = self.final_layer(combined).squeeze() + user_bias + item_bias
        return sigmoid(prediction)

def forward(Theta1, Theta2, U, E, Ma, Mb, interaction=True):
    """Legacy forward function for backward compatibility"""
    if interaction:
        return sigmoid((Theta1(U)*(E@Ma)).sum(1)+Theta2(U).squeeze()+(E@Mb))
    else:
        return sigmoid(Theta2(U).squeeze()+(E@Mb))

def forward_ncf(model, U, E):
    """NCF forward function"""
    return model(U, E)

def train_ncf(U,
              E,
              Y, 
              d,
              mlp_layers=[128, 64, 32],
              dropout=0.2,
              lr=1e-2,
              wd=1e-4,
              gamma=.999,
              n_epochs=10000, 
              patience=30,
              tol=1e-4,
              val_size=.1,
              device='cpu',
              random_state=42,
              verbose=True):
    """
    Train NCF model with the given parameters and dataset.

    Parameters:
        U (Tensor): User indices for training.
        E (Tensor): Image embeddings for training.
        Y (Tensor): Binary labels.
        d (int): Dimension of latent factors.
        mlp_layers (list): Hidden layer dimensions for MLP component.
        dropout (float): Dropout rate for MLP layers.
        lr (float): Learning rate for the optimizer.
        wd (float): Weight decay for the optimizer.
        gamma (float): Learning rate decay factor.
        n_epochs (int): Number of training epochs.
        patience (int): Early stopping patience.
        tol (float): Tolerance for improvement.
        val_size (float): Validation set size.
        device (torch.device): Device to use for computation.
        random_state (int): Random seed for reproducibility.
        verbose (bool): Whether to print progress.
    """

    emb_dim = E.shape[1]
    num_users = torch.unique(U).shape[0]
    
    U_train, U_val, Y_train, Y_val, E_train, E_val = train_test_split(U, Y, E, test_size=val_size, random_state=random_state)
    U_train, U_val, Y_train, Y_val, E_train, E_val = U_train.to(device), U_val.to(device), Y_train.to(device), Y_val.to(device), E_train.to(device), E_val.to(device)
    
    # Weighted loss for imbalanced data
    w_dict = {int(u): 1/float((U==u).float().mean()*torch.unique(U).shape[0]) for u in torch.unique(U)}
    w_train = torch.tensor([w_dict[int(u)] for u in U_train]).to(device)
    w_val = torch.tensor([w_dict[int(u)] for u in U_val]).to(device)
    criterion_train = torch.nn.BCELoss(weight=w_train)
    criterion_val = torch.nn.BCELoss(weight=w_val)
    
    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    
    # Initialize NCF model
    model = NCFModel(num_users, emb_dim, d, mlp_layers, dropout).to(device)
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    # Early stopping variables
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in tqdm(range(n_epochs), desc="Training NCF", disable=not verbose):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        Y_hat = model(U_train, E_train)
        
        # Compute loss
        loss = criterion_train(Y_hat, Y_train)
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            val_loss = criterion_val(model(U_val, E_val), Y_val).item()
        
        # Early stopping logic
        if val_loss + tol < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if verbose:
        print(f"NCF - lr {lr} -- d {d} -- wd {wd} -- gamma {gamma} -- val_loss: {best_val_loss}")
    
    # Load best model state
    model.load_state_dict(best_model_state)
    model.eval()
    
    return {
        'model': model,
        'best_val_loss': best_val_loss
    }
        
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
            device='cpu',
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
                                               device=device,
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


class NCF():
    """Neural Collaborative Filtering class with same interface as IRT"""
    def __init__(self, mlp_layers=[128, 64, 32], dropout=0.2):
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.model = None

    def fit(self,
            U,
            E,
            Y,
            ds = [10, 20, 40, 60, 80, 160, 240, 320],
            lrs = [.01,.001,.0001, 0.00001],
            weight_decays = [1e0,1e-2,1e-4,1e-6],
            gammas = [1,.9999,.999],
            mlp_layers_options = [[128, 64, 32], [256, 128, 64], [64, 32], [512, 256, 128]],
            dropout_options = [0.1, 0.2, 0.3],
            val_size=.1,
            random_state=42,
            device='cpu',
            verbose=True):

            assert val_size>0 and val_size<1

            self.validation = {'d': [], 'wd': [], 'lr':[], 'gamma':[], 'mlp_layers': [], 'dropout': [], 'val_loss': []}
            self.best_val_loss = float('inf')
        
            for lr in tqdm(lrs, disable=not verbose):
                for gamma in tqdm(gammas, disable=not verbose):
                    for d in tqdm(ds, disable=not verbose):
                        for wd in tqdm(weight_decays, disable=not verbose):
                            for mlp_layers in tqdm(mlp_layers_options, disable=not verbose):
                                for dropout in tqdm(dropout_options, disable=not verbose):
                            
                                    output = train_ncf(U,
                                                       E,
                                                       Y, 
                                                       d=d,
                                                       mlp_layers=mlp_layers,
                                                       dropout=dropout,
                                                       lr=lr,
                                                       wd=wd,
                                                       gamma=gamma,
                                                       val_size=val_size,
                                                       device=device,
                                                       random_state=random_state,
                                                       verbose=False)

                                    self.validation['lr'].append(lr)
                                    self.validation['d'].append(d)
                                    self.validation['wd'].append(wd)
                                    self.validation['gamma'].append(gamma)
                                    self.validation['mlp_layers'].append(mlp_layers)
                                    self.validation['dropout'].append(dropout)
                                    self.validation['val_loss'].append(output['best_val_loss'])
                            
                                    if output['best_val_loss']<self.best_val_loss:
                                        self.d = d
                                        self.lr = lr
                                        self.wd = wd
                                        self.gamma = gamma
                                        self.mlp_layers = mlp_layers
                                        self.dropout = dropout
                                        self.model = output['model']
                                        self.best_val_loss = output['best_val_loss']
                
    def predict(self, U_test, E_test):
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        self.model.eval()
        with torch.no_grad():
            scores = self.model(U_test, E_test)
        return scores
        
    def score(self, U_test, E_test):
        scores = self.predict(U_test, E_test)
        U = torch.tensor([int(u) for u in U_test])
        output = {}
        for u in torch.unique(U):
            output[int(u)] = {'avg':scores[U==u].mean().item(),'std_error':float(scores[U==u].std().item()/(torch.sum(U==u))**.5)}
        return output

    def save(self, path):
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump({
                'd':self.d,
                'lr':self.lr,
                'wd':self.wd,
                'gamma':self.gamma,
                'mlp_layers':self.mlp_layers,
                'dropout':self.dropout,
                'model_state_dict':self.model.state_dict(),
                'model_config': {
                    'num_users': self.model.num_users,
                    'emb_dim': self.model.emb_dim,
                    'd': self.model.d,
                    'mlp_layers': self.mlp_layers,
                    'dropout': self.dropout
                },
                'best_val_loss':self.best_val_loss,
                'validation':self.validation
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.d = params['d']
            self.lr = params['lr']
            self.wd = params['wd']
            self.gamma = params['gamma']
            self.mlp_layers = params['mlp_layers']
            self.dropout = params['dropout']
            self.best_val_loss = params['best_val_loss']
            self.validation = params['validation']
            
            # Recreate model with saved configuration
            config = params['model_config']
            self.model = NCFModel(
                num_users=config['num_users'],
                emb_dim=config['emb_dim'],
                d=config['d'],
                mlp_layers=config['mlp_layers'],
                dropout=config['dropout']
            )
            self.model.load_state_dict(params['model_state_dict'])
            self.model.eval()

    def to(self, device):
        self.model.to(device)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="irt", choices=["irt", "ncf"], help="Model type: irt or ncf")
    parser.add_argument("--interaction", type=bool, default=True)
    parser.add_argument("--usr_threshold", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()  
    
    # Load data
    data_eval = pd.read_csv(f"./data/flickr/processed/test/test_usrthrs_{args.usr_threshold}.csv")
    E_test = torch.load(f"./data/flickr/processed/test/test_ie_usrthrs_{args.usr_threshold}.pt", weights_only=True)
    data_train = pd.read_csv(f"./data/flickr/processed/train/train_usrthrs_{args.usr_threshold}.csv")
    E_train = torch.load(f"./data/flickr/processed/train/train_ie_usrthrs_{args.usr_threshold}.pt", weights_only=True)
    Y_test = (torch.tensor(data_eval.score)>=4).float()
    Y_train = (torch.tensor(data_train.score)>=4).float()
    U_test = torch.tensor(data_eval.worker_id)
    U_train = torch.tensor(data_train.worker_id)
    
    if args.model == "irt":
        model = IRT(interaction=args.interaction)
        model.fit(U_train, E_train, Y_train, device=args.device)
        model.save(f"../data/flickr/evaluation/irt/models/weights/irt/irt_interaction_{args.interaction}_usrthrs_{args.usr_threshold}.pkl")
        model.load(f"../data/flickr/evaluation/irt/models/weights/irt/irt_interaction_{args.interaction}_usrthrs_{args.usr_threshold}.pkl")
        print("IRT Results:", model.score(U_test, E_test))
    elif args.model == "ncf":
        model = NCF()
        model.fit(U_train, E_train, Y_train, device=args.device)
        model.save(f"../data/flickr/evaluation/irt/models/weights/ncf/ncf_usrthrs_{args.usr_threshold}.pkl")
        model.load(f"../data/flickr/evaluation/irt/models/weights/ncf/ncf_usrthrs_{args.usr_threshold}.pkl")
        print("NCF Results:", model.score(U_test, E_test))