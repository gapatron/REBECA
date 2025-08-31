"""
Quick NCF Training with Reduced Hyperparameter Search
Designed to finish in 30-60 minutes instead of days
"""

import torch
import pandas as pd
from irt import NCF
from argparse import ArgumentParser

def quick_ncf_train(usr_threshold=0, device='cuda'):
    """Train NCF with dramatically reduced search space"""
    
    print("🚀 Quick NCF Training (30-60 min instead of days)")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    data_train = pd.read_csv(f"./data/flickr/processed/train/train_usrthrs_{usr_threshold}.csv")
    E_train = torch.load(f"./data/flickr/processed/train/train_ie_usrthrs_{usr_threshold}.pt", weights_only=True)
    Y_train = (torch.tensor(data_train.score) >= 4).float()
    U_train = torch.tensor(data_train.worker_id)
    
    print(f"Training samples: {len(Y_train)}")
    print(f"Users: {len(torch.unique(U_train))}")
    print(f"Positive rate: {Y_train.mean():.3f}")
    
    # Initialize NCF with MUCH smaller search space
    ncf = NCF()
    
    print("\n🔥 Training NCF with reduced hyperparameter search...")
    print("Search space: 2×2×2×1×2×1 = 16 combinations (vs 4,608 original)")
    
    # DRAMATICALLY reduced search space
    ncf.fit(U_train, E_train, Y_train,
            # Reduced from 8 to 2 options
            ds=[60, 160],  
            
            # Reduced from 4 to 2 options  
            lrs=[0.001, 0.0001],
            
            # Reduced from 4 to 2 options
            weight_decays=[1e-2, 1e-4],
            
            # Reduced from 3 to 1 option
            gammas=[0.999],
            
            # Reduced from 4 to 2 options
            mlp_layers_options=[[128, 64, 32], [64, 32]],
            
            # Reduced from 3 to 1 option
            dropout_options=[0.2],
            
            device=device,
            verbose=True)
    
    # Save the model
    import os
    save_path = f"./weights/ncf_quick_usrthrs_{usr_threshold}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ncf.save(save_path)
    
    print(f"\n✅ NCF training completed!")
    print(f"💾 Model saved to: {save_path}")
    print(f"🎯 Best validation loss: {ncf.best_val_loss:.4f}")
    print(f"🔧 Best hyperparameters:")
    print(f"   d: {ncf.d}")
    print(f"   lr: {ncf.lr}")
    print(f"   wd: {ncf.wd}")
    print(f"   mlp_layers: {ncf.mlp_layers}")
    print(f"   dropout: {ncf.dropout}")
    
    return ncf

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--usr_threshold', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    ncf = quick_ncf_train(args.usr_threshold, args.device)
