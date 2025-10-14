#!/usr/bin/env python3
"""
Comprehensive REBECA Study Framework
====================================

This framework systematically tests:
1. Normalization approaches (none, l2_norm, etc.)
2. Architecture variants (transformer vs cross-attention vs new architectures)
3. Training parameters (objectives, schedulers, etc.)
4. User thresholds (0, 50, 100, 150)
5. Automatic image generation after training

The framework is designed to be:
- Robust and error-resistant
- Resumable (can continue from interruptions)
- Well-organized with clear logging
- Comprehensive in tracking all configurations
"""

import torch
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from datetime import datetime
from tqdm import tqdm
import itertools
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import DDPMScheduler
from Datasets import RecommenderUserSampler, EmbeddingsDataset
from prior_models import (
    RebecaDiffusionPrior
)
from train_priors import train_diffusion_prior
from utils import set_seeds, map_embeddings_to_ratings, split_recommender_data
from diffusers import StableDiffusionPipeline
from sampling import sample_user_images
from utils import save_generated_data_to_user_dir

class ComprehensiveREBECAStudy:
    """
    Comprehensive study framework for REBECA experiments.
    """
    
    def __init__(self, base_dir="data/flickr/evaluation/diffusion_priors/models/weights"):
        self.base_dir = Path(base_dir)
        self.study_dir = self.base_dir / f"comprehensive_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.log_file = self.study_dir / "study_log.txt"
        self.results_file = self.study_dir / "results.json"
        self.config_file = self.study_dir / "study_config.json"
        
        # Load or initialize results
        self.results = self._load_results()
        self.completed_experiments = set()
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging to file and console."""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_results(self):
        """Load existing results if available."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                results = json.load(f)
                self.logger.info(f"Loaded {len(results)} existing results")
                return results
        return []
    
    def _save_results(self):
        """Save current results."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def _log_experiment_start(self, experiment_config):
        """Log experiment start."""
        self.logger.info(f"Starting experiment: {experiment_config['experiment_name']}")
        self.logger.info(f"Config: {experiment_config}")
    
    def _log_experiment_complete(self, experiment_config, result):
        """Log experiment completion."""
        self.logger.info(f"Completed experiment: {experiment_config['experiment_name']}")
        self.logger.info(f"Final validation loss: {result['final_val_loss']:.6f}")
    
    def _log_experiment_error(self, experiment_config, error):
        """Log experiment error."""
        self.logger.error(f"Error in experiment {experiment_config['experiment_name']}: {error}")
        self.logger.error(traceback.format_exc())
    
    def create_comprehensive_param_grid(self):
        """
        Create comprehensive parameter grid for all experiments.
        """
        param_grid = {
            # Model architecture selection - now including all new architectures
            'model_type': [
                #'transformer', 
                #'cross_attention', 
                #'cross_attention_large',
                #'cross_attention_per_layer',
                #'direct_cross_attention',
                #'hierarchical_cross_attention',
                'rdp',
                #'adaptive_cross_attention'
            ],
            
            # Architecture parameters (conditional on model type)
            'num_layers': [6, 8],  # For transformer and some cross-attention models
            'num_heads': [4, 8],  # For transformer and some cross-attention models
            'num_tokens': [16, 32],  # For transformer models
            'hidden_dim': [128, 256],
            # Training parameters
            'learning_rate': [1e-4],
            'optimizers': ['adamw'],
            'schedulers': ['reduce_on_plateau'],
            'batch_size': [64],
            'samples_per_user': [100],
            'clip_sample': [False],
            'objective': ['epsilon', 'sample', 'v_prediction'],
            'img_embed_dim': [1024],
            
            # Scheduler parameters (now configurable!)
            'timesteps': [1000],  # Configurable timesteps
            'noise_schedule': ['laplace', "squaredcos_cap_v2"],  # Configurable schedule
            
            # Normalization parameters
            'normalization_type': ['none'],
            
            # User threshold parameters
            'user_threshold': [0]
        }
        
        return param_grid
    
    def apply_normalization_to_embeddings(self, image_features, norm_type):
        """
        Apply normalization to image embeddings.
        """
        if norm_type == 'none':
            return image_features
        
        elif norm_type == 'l2_norm':
            # L2 normalization
            d = image_features.shape[-1]
            norms = image_features.norm(dim=-1, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            return image_features / norms * torch.sqrt(torch.tensor(d))
        
        elif norm_type == 'l2_norm_clamp':
            # L2 normalization with clamping (your current approach)
            d = image_features.shape[-1]
            norms = image_features.norm(dim=-1, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            normalized = image_features / norms * torch.sqrt(torch.tensor(d))
            return torch.clamp(normalized, -3.2, 3.2) / 3.2
        
        elif norm_type == 'standardize':
            # Standardize to mean=0, std=1
            mean = image_features.mean(dim=1, keepdim=True)
            std = image_features.std(dim=1, keepdim=True)
            std = torch.clamp(std, min=1e-8)
            return (image_features - mean) / std
        
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
    
    def filter_users_by_threshold(self, ratings_df, threshold):
        """
        Filter users based on threshold.
        """
        if threshold == 0:
            # No filtering - use all users
            valid_users = ratings_df["worker_id"].unique()
            filtered_ratings_df = ratings_df.copy()
        else:
            # Filter by threshold
            liked_counts = (
                ratings_df[ratings_df["score"] >= 4]
                .groupby("worker_id")["score"]
                .count()
                .reset_index(name="liked_count")
            )
            valid_users = liked_counts[liked_counts["liked_count"] >= threshold]["worker_id"].unique()
            filtered_ratings_df = ratings_df[ratings_df["worker_id"].isin(valid_users)].copy()
        
        # Create user mapping
        worker_mapping = {old_id: new_id for new_id, old_id in enumerate(valid_users)}
        filtered_ratings_df.rename(columns={"worker_id": "old_worker_id"}, inplace=True)
        filtered_ratings_df["worker_id"] = filtered_ratings_df["old_worker_id"].map(worker_mapping)
        
        return filtered_ratings_df, valid_users, worker_mapping
    
    def create_model(self, config,device):
        """
        Create model based on configuration.
        """

        return RebecaDiffusionPrior(
                img_embed_dim=1024,
                num_users=210,
                num_tokens=config["num_tokens"],
                hidden_dim=config['hidden_dim'],
                n_heads=config['num_heads'],
                num_layers=config['num_layers'],
                score_classes=2,
            ).to(device)
        
        
    def create_experiment_name(self, config):
        """
        Create a unique experiment name from configuration.
        """
        name_parts = [
            f"model{config['model_type']}",
            f"num_layers{config.get('num_layers', 'default')}",
            f"num_heads{config.get('num_heads', 'default')}",
            f"hidden_dim{config.get('hidden_dim', 'default')}",
            f"tokens{config.get('num_tokens', 'default')}",
            f"lr{config['learning_rate']}",
            f"opt{config['optimizers']}",
            f"sch{config['schedulers']}",
            f"bs{config['batch_size']}",
            f"ns{config['noise_schedule']}",
            f"ts{config['timesteps']}",
            f"spu{config['samples_per_user']}",
            f"cs{config['clip_sample']}",
            f"obj{config['objective']}",
            f"norm{config['normalization_type']}",
            f"uthr{config['user_threshold']}"
        ]
        return "_".join(name_parts)
    
    def run_single_experiment(self, config, image_features, ratings_df):
        """
        Run a single experiment with the given configuration.
        """
        experiment_name = self.create_experiment_name(config)
        
        # Skip if already completed
        if experiment_name in self.completed_experiments:
            self.logger.info(f"Skipping completed experiment: {experiment_name}")
            return None
        
        self._log_experiment_start(config)
        
        try:
            # Filter users by threshold
            filtered_ratings_df, valid_users, worker_mapping = self.filter_users_by_threshold(
                ratings_df, config['user_threshold']
            )
            
            # Apply normalization
            normalized_features = self.apply_normalization_to_embeddings(
                image_features, config['normalization_type']
            )
            
            # Create expanded features
            expanded_features = map_embeddings_to_ratings(normalized_features, ratings_df)
            
            # Split data
            train_df, val_df, test_df = split_recommender_data(
                ratings_df=filtered_ratings_df,
                val_spu=10,
                test_spu=10,
                seed=42
            )
            
            # Create datasets
            train_dataset = EmbeddingsDataset(
                train_df,
                image_embeddings=expanded_features[train_df.original_index]
            )
            val_dataset = EmbeddingsDataset(
                val_df,
                image_embeddings=expanded_features[val_df.original_index]
            )
            
            # Setup training
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_users = len(valid_users)
            
            # Create model
            model = self.create_model(config, num_users, device)
            
            # Setup optimizer
            if config['optimizers'] == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
            else:
                raise ValueError(f"Unknown optimizer: {config['optimizers']}")
            
            # Setup scheduler
            if config['schedulers'] == 'reduce_on_plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
            elif config['schedulers'] == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            else:
                raise ValueError(f"Unknown scheduler: {config['schedulers']}")
            
            # Setup noise scheduler (now configurable!)
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=config['timesteps'],
                beta_schedule=config['noise_schedule'],
                clip_sample=config['clip_sample'],
                prediction_type=config["objective"]
            )
            
            # Setup dataloaders
            train_user_sampler = RecommenderUserSampler(
                train_df, 
                num_users=num_users, 
                samples_per_user=config['samples_per_user']
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, 
                sampler=train_user_sampler, 
                batch_size=config['batch_size']
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True
            )
            
            # Create save path
            savepath = self.study_dir / f"{experiment_name}.pth"
            
            # Train the model
            train_loss, val_loss = train_diffusion_prior(
                model=model,
                noise_scheduler=noise_scheduler,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_unique_users=num_users,
                objective=config['objective'],
                device=device,
                num_epochs=2001,
                patience=50,
                savepath=str(savepath),
                return_losses=True,
                verbose=False
            )
            
            # Get final validation loss
            final_val_loss = val_loss[-1] if isinstance(val_loss, list) else val_loss
            
            # Clean up training artifacts BEFORE image generation
            model.eval()  # Ensure model is in eval mode
            torch.cuda.empty_cache()  # Clear training cache
            
            # Generate images for evaluation
            image_results = self.generate_evaluation_images(
                model, experiment_name, num_users, device, config
            )
            
            # Store results
            result = {
                'experiment_name': experiment_name,
                'config': config,
                'final_val_loss': final_val_loss,
                'savepath': str(savepath),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'num_users': num_users,
                'image_results': image_results
            }
            
            self.results.append(result)
            self.completed_experiments.add(experiment_name)
            self._save_results()
            
            self._log_experiment_complete(config, result)
            
            # Final cleanup
            del model
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            self._log_experiment_error(config, e)
            return None
    
    def generate_evaluation_images(self, model, experiment_name, num_users, device, config):
        """
        Generate evaluation images for the trained model.
        """
        try:
            # Ensure model is in eval mode and clear any remaining gradients
            model.eval()
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None
            
            # Setup Stable Diffusion pipeline
            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
            pipe.safety_checker = None
            
            # Note: xformers is optional for memory efficiency
            
            # Setup noise scheduler (use the same configuration as training)
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=config['timesteps'],
                beta_schedule=config['noise_schedule'],
                clip_sample=config['clip_sample'],
                prediction_type=config["objective"]
            )
            
            # Select users for generation (first 5 users to reduce memory usage)
            users = list(range(min(5, num_users)))
            
            # Generate images
            gen_data = sample_user_images(
                diffusion_prior_model=model,
                diffusion_pipe=pipe,
                users=users,
                images_per_user=5,  # Fewer images per user to save memory
                noise_scheduler=noise_scheduler,
                guidance_scale=3.0,
                prompt=[""],
                negative_prompt=[""],
                img_embedding_size=1024,
                device=device,
            )
            
            # Save images
            image_dir = self.study_dir / "generated_images" / experiment_name
            save_generated_data_to_user_dir(data=gen_data, dir=str(image_dir))
            
            # Save experiment info for easy reference
            experiment_info = {
                'experiment_name': experiment_name,
                'config': config,
                'num_users': num_users,
                'user_threshold': config['user_threshold'],
                'image_dir': str(image_dir),
                'num_users_generated': len(users),
                'images_per_user': 2,
                'timestamp': datetime.now().isoformat()
            }
            info_file = image_dir / "experiment_info.json"
            info_file.parent.mkdir(parents=True, exist_ok=True)
            with open(info_file, 'w') as f:
                json.dump(experiment_info, f, indent=2)
            
            # Clean up pipeline
            del pipe
            torch.cuda.empty_cache()
            
            return {
                'image_dir': str(image_dir),
                'num_users_generated': len(users),
                'images_per_user': 2
            }
            
        except Exception as e:
            self.logger.error(f"Error generating images for {experiment_name}: {e}")
            return None
    
    def run_comprehensive_study(self):
        """
        Run the comprehensive study.
        """
        self.logger.info("Starting Comprehensive REBECA Study")
        self.logger.info(f"Study directory: {self.study_dir}")
        
        # Load data
        self.logger.info("Loading data...")
        image_features = torch.load("data/flickr/processed/ip-adapters/SD15/sd15_image_embeddings.pt", weights_only=True)
        ratings_df = pd.read_csv("data/flickr/processed/ratings.csv")
        
        # Create parameter grid
        param_grid = self.create_comprehensive_param_grid()
        
        # Generate all combinations
        keys = list(param_grid.keys())
        combinations = list(itertools.product(*param_grid.values()))
        
        self.logger.info(f"Total experiments to run: {len(combinations)}")
        
        # Save study configuration
        study_config = {
            'param_grid': param_grid,
            'total_experiments': len(combinations),
            'start_time': datetime.now().isoformat(),
            'study_dir': str(self.study_dir)
        }
        with open(self.config_file, 'w') as f:
            json.dump(study_config, f, indent=2)
        
        # Run experiments
        for i, combo in enumerate(tqdm(combinations, desc="Running experiments")):
            config = dict(zip(keys, combo))
            config['experiment_name'] = self.create_experiment_name(config)
            
            result = self.run_single_experiment(config, image_features, ratings_df)
            
            # Save progress every 10 experiments
            if (i + 1) % 10 == 0:
                self._save_results()
                self.logger.info(f"Progress: {i+1}/{len(combinations)} experiments completed")
        
        # Create final summary
        self.create_study_summary()
        
        self.logger.info("Comprehensive study completed!")
        return self.results
    
    def create_study_summary(self):
        """
        Create a comprehensive summary of the study results.
        """
        if not self.results:
            self.logger.warning("No results to summarize")
            return
        
        # Create summary report
        summary_file = self.study_dir / "study_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Comprehensive REBECA Study Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total experiments: {len(self.results)}\n")
            f.write(f"Completed: {len([r for r in self.results if r['final_val_loss'] is not None])}\n\n")
            
            # Best performing configuration
            best_result = min(self.results, key=lambda x: x['final_val_loss'])
            f.write(f"Best configuration:\n")
            f.write(f"  Experiment: {best_result['experiment_name']}\n")
            f.write(f"  Validation loss: {best_result['final_val_loss']:.6f}\n")
            f.write(f"  Config: {best_result['config']}\n\n")
            
            # Summary statistics
            val_losses = [r['final_val_loss'] for r in self.results if r['final_val_loss'] is not None]
            f.write(f"Validation loss statistics:\n")
            f.write(f"  Mean: {np.mean(val_losses):.6f}\n")
            f.write(f"  Std: {np.std(val_losses):.6f}\n")
            f.write(f"  Min: {np.min(val_losses):.6f}\n")
            f.write(f"  Max: {np.max(val_losses):.6f}\n\n")
            
            # Results by different parameters
            self._write_parameter_analysis(f)
        
        self.logger.info(f"Study summary saved to: {summary_file}")
    
    def _write_parameter_analysis(self, f):
        """
        Write analysis of results by different parameters.
        """
        f.write("Results by Parameter:\n")
        f.write("-" * 30 + "\n\n")
        
        # Analysis by model type
        f.write("By Model Type:\n")
        model_types = set(r['config']['model_type'] for r in self.results)
        for model_type in sorted(model_types):
            type_results = [r for r in self.results if r['config']['model_type'] == model_type]
            type_losses = [r['final_val_loss'] for r in type_results if r['final_val_loss'] is not None]
            if type_losses:
                f.write(f"  {model_type}: {np.mean(type_losses):.6f} ± {np.std(type_losses):.6f} (n={len(type_losses)})\n")
        f.write("\n")
        
        # Analysis by normalization type
        f.write("By Normalization Type:\n")
        norm_types = set(r['config']['normalization_type'] for r in self.results)
        for norm_type in sorted(norm_types):
            type_results = [r for r in self.results if r['config']['normalization_type'] == norm_type]
            type_losses = [r['final_val_loss'] for r in type_results if r['final_val_loss'] is not None]
            if type_losses:
                f.write(f"  {norm_type}: {np.mean(type_losses):.6f} ± {np.std(type_losses):.6f} (n={len(type_losses)})\n")
        f.write("\n")
        
        # Analysis by objective
        f.write("By Objective:\n")
        objectives = set(r['config']['objective'] for r in self.results)
        for obj in sorted(objectives):
            obj_results = [r for r in self.results if r['config']['objective'] == obj]
            obj_losses = [r['final_val_loss'] for r in obj_results if r['final_val_loss'] is not None]
            if obj_losses:
                f.write(f"  {obj}: {np.mean(obj_losses):.6f} ± {np.std(obj_losses):.6f} (n={len(obj_losses)})\n")
        f.write("\n")
        
        # Analysis by user threshold
        f.write("By User Threshold:\n")
        thresholds = set(r['config']['user_threshold'] for r in self.results)
        for thresh in sorted(thresholds):
            thresh_results = [r for r in self.results if r['config']['user_threshold'] == thresh]
            thresh_losses = [r['final_val_loss'] for r in thresh_results if r['final_val_loss'] is not None]
            if thresh_losses:
                f.write(f"  {thresh}: {np.mean(thresh_losses):.6f} ± {np.std(thresh_losses):.6f} (n={len(thresh_losses)})\n")
        f.write("\n")
        
        # Analysis by noise schedule
        f.write("By Noise Schedule:\n")
        schedules = set(r['config']['noise_schedule'] for r in self.results)
        for schedule in sorted(schedules):
            schedule_results = [r for r in self.results if r['config']['noise_schedule'] == schedule]
            schedule_losses = [r['final_val_loss'] for r in schedule_results if r['final_val_loss'] is not None]
            if schedule_losses:
                f.write(f"  {schedule}: {np.mean(schedule_losses):.6f} ± {np.std(schedule_losses):.6f} (n={len(schedule_losses)})\n")

def main():
    """
    Main function to run the comprehensive study.
    """
    print("Comprehensive REBECA Study")
    print("=" * 40)
    
    # Set random seed
    set_seeds(42)
    
    # Create and run study
    study = ComprehensiveREBECAStudy()
    results = study.run_comprehensive_study()
    
    print(f"\nStudy completed!")
    print(f"Results saved to: {study.study_dir}")

if __name__ == "__main__":
    main() 