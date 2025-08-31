"""
Quick NCF vs IRT Comparison Script
Simple test to see if NCF beats the 0.86 AUC baseline
"""

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from irt import IRT, NCF

def quick_evaluate(usr_threshold=0, device='cuda'):
    """Quick evaluation to test NCF performance"""
    
    print("🚀 Quick NCF vs IRT Evaluation")
    print("=" * 40)
    
    # Load data
    print("📊 Loading data...")
    data_eval = pd.read_csv(f"./data/flickr/processed/test/test_usrthrs_{usr_threshold}.csv")
    E_test = torch.load(f"./data/flickr/processed/test/test_ie_usrthrs_{usr_threshold}.pt", weights_only=True)
    data_train = pd.read_csv(f"./data/flickr/processed/train/train_usrthrs_{usr_threshold}.csv")
    E_train = torch.load(f"./data/flickr/processed/train/train_ie_usrthrs_{usr_threshold}.pt", weights_only=True)
    
    Y_test = (torch.tensor(data_eval.score) >= 4).float()
    Y_train = (torch.tensor(data_train.score) >= 4).float()
    U_test = torch.tensor(data_eval.worker_id)
    U_train = torch.tensor(data_train.worker_id)
    
    print(f"Test samples: {len(Y_test)}, Users: {len(torch.unique(U_test))}")
    print(f"Positive rate: {Y_test.mean():.3f}")
    
    results = {}
    
    # Test IRT (baseline)
    print("\n🏁 Testing IRT (Baseline)...")
    irt = IRT(interaction=True)
    
    # Quick hyperparameter search for faster testing
    irt.fit(U_train, E_train, Y_train, 
            ds=[20, 60], 
            lrs=[0.001, 0.0001], 
            weight_decays=[1e-2, 1e-4],
            gammas=[0.999],
            device=device, 
            verbose=False)
    
    irt_preds = irt.predict(U_test, E_test).detach().cpu().numpy()
    irt_fpr, irt_tpr, _ = roc_curve(Y_test.numpy(), irt_preds)
    irt_auc = auc(irt_fpr, irt_tpr)
    
    results['IRT'] = {'auc': irt_auc, 'predictions': irt_preds}
    print(f"✅ IRT AUC: {irt_auc:.4f}")
    
    # Test NCF (our new model)
    print("\n🔥 Testing NCF (Neural Collaborative Filtering)...")
    ncf = NCF()
    
    # Quick hyperparameter search
    ncf.fit(U_train, E_train, Y_train,
            ds=[20, 60],
            lrs=[0.001, 0.0001],
            weight_decays=[1e-2, 1e-4],
            gammas=[0.999],
            mlp_layers_options=[[128, 64, 32], [64, 32]],
            dropout_options=[0.2],
            device=device,
            verbose=False)
    
    ncf_preds = ncf.predict(U_test, E_test).detach().cpu().numpy()
    ncf_fpr, ncf_tpr, _ = roc_curve(Y_test.numpy(), ncf_preds)
    ncf_auc = auc(ncf_fpr, ncf_tpr)
    
    results['NCF'] = {'auc': ncf_auc, 'predictions': ncf_preds}
    print(f"✅ NCF AUC: {ncf_auc:.4f}")
    
    # Comparison
    print("\n" + "=" * 40)
    print("📈 RESULTS SUMMARY")
    print("=" * 40)
    
    target_auc = 0.86
    improvement = ncf_auc - irt_auc
    
    print(f"🎯 Target AUC: {target_auc:.3f}")
    print(f"📊 IRT AUC:    {irt_auc:.4f}")
    print(f"🚀 NCF AUC:    {ncf_auc:.4f}")
    print(f"📈 Improvement: {improvement:+.4f}")
    
    if ncf_auc > target_auc:
        print(f"🎉 SUCCESS! NCF beats baseline by {ncf_auc - target_auc:.4f}")
    else:
        print(f"❌ NCF below target by {target_auc - ncf_auc:.4f}")
        
    if ncf_auc > irt_auc:
        print(f"✅ NCF outperforms IRT by {improvement:.4f}")
    else:
        print(f"❌ IRT still better by {-improvement:.4f}")
    
    # Per-user analysis
    print("\n📊 Per-User Analysis:")
    U_test_np = U_test.numpy()
    Y_test_np = Y_test.numpy()
    
    user_improvements = []
    for u in np.unique(U_test_np):
        user_mask = U_test_np == u
        user_y = Y_test_np[user_mask]
        
        if np.sum(user_y) > 0 and np.sum(1 - user_y) > 0:  # Valid for AUC calculation
            # IRT user AUC
            irt_user_fpr, irt_user_tpr, _ = roc_curve(user_y, irt_preds[user_mask])
            irt_user_auc = auc(irt_user_fpr, irt_user_tpr)
            
            # NCF user AUC
            ncf_user_fpr, ncf_user_tpr, _ = roc_curve(user_y, ncf_preds[user_mask])
            ncf_user_auc = auc(ncf_user_fpr, ncf_user_tpr)
            
            user_improvements.append(ncf_user_auc - irt_user_auc)
    
    user_improvements = np.array(user_improvements)
    print(f"Users where NCF improved: {np.sum(user_improvements > 0)}/{len(user_improvements)} ({100*np.sum(user_improvements > 0)/len(user_improvements):.1f}%)")
    print(f"Average per-user improvement: {user_improvements.mean():+.4f}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--usr_threshold', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    results = quick_evaluate(args.usr_threshold, args.device)
