"""
NPE training and inference for Lotka-Volterra model.
"""
import torch
from typing import Optional, Dict, Any, Tuple
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sbi.inference import SNPE
from sbi.utils import posterior_nn

from .utils import create_lv_prior
from .data_generation import LVDataGenerator


class LVInference:
    """Neural Posterior Estimation for Lotka-Volterra model."""
    
    def __init__(self, 
                 use_summary_stats: bool = False,
                 device: str = 'cpu',
                 seed: Optional[int] = None):
        """
        Initialize NPE inference.
        
        Parameters:
        -----------
        use_summary_stats : bool
            Whether using summary statistics or full trajectories
        device : str
            Device for training ('cpu' or 'cuda')
        seed : int, optional
            Random seed
        """
        self.use_summary_stats = use_summary_stats
        self.device = device
        
        if seed is not None:
            torch.manual_seed(seed)
            
        self.prior = create_lv_prior()
        self.inference = None
        self.posterior = None
        
    def setup_inference(self, x_dim: int, **kwargs):
        """
        Set up SBI inference object.
        
        Parameters:
        -----------
        x_dim : int
            Dimension of observations
        **kwargs : additional arguments for SNPE
        """
        # Default neural network configuration
        neural_net_kwargs = {
            'hidden_features': 128,
            'num_transforms': 5,
            'embedding_net': torch.nn.Identity(),
        }
        neural_net_kwargs.update(kwargs.get('neural_net_kwargs', {}))
        
        # Create neural posterior estimator
        neural_posterior = posterior_nn(
            model='nsf',  # Neural Spline Flow
            **neural_net_kwargs
        )
        
        self.inference = SNPE(
            prior=self.prior,
            density_estimator=neural_posterior,
            device=self.device,
            **{k: v for k, v in kwargs.items() if k != 'neural_net_kwargs'}
        )
        
    def train(self, 
              theta: torch.Tensor, 
              x: torch.Tensor,
              training_batch_size: int = 512,
              learning_rate: float = 1e-4,
              max_num_epochs: int = 100,
              validation_fraction: float = 0.1,
              stop_after_epochs: int = 20,
              **kwargs) -> Dict[str, Any]:
        """
        Train neural posterior estimator.
        
        Parameters:
        -----------
        theta : torch.Tensor of shape (n_samples, 4)
            Parameter vectors
        x : torch.Tensor of shape (n_samples, x_dim)
            Observations
        training_batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate
        max_num_epochs : int
            Maximum training epochs
        validation_fraction : float
            Fraction of data for validation
        stop_after_epochs : int
            Early stopping patience
        **kwargs : additional training arguments
            
        Returns:
        --------
        training_info : dict
            Training information and losses
        """
        if self.inference is None:
            self.setup_inference(x_dim=x.shape[1])
            
        print(f"Training NPE with {len(theta)} samples...")
        print(f"Parameter shape: {theta.shape}")
        print(f"Observation shape: {x.shape}")
        
        # Add training data
        self.inference = self.inference.append_simulations(theta, x)
        
        # Train
        training_info = self.inference.train(
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            max_num_epochs=max_num_epochs,
            validation_fraction=validation_fraction,
            stop_after_epochs=stop_after_epochs,
            show_train_summary=True,
            **kwargs
        )
        
        # Build posterior
        self.posterior = self.inference.build_posterior()
        
        return training_info
    
    def sample_posterior(self, 
                        x_obs: torch.Tensor,
                        num_samples: int = 1000,
                        **kwargs) -> torch.Tensor:
        """
        Sample from posterior given observed data.
        
        Parameters:
        -----------
        x_obs : torch.Tensor
            Observed data
        num_samples : int
            Number of posterior samples
        **kwargs : additional sampling arguments
            
        Returns:
        --------
        samples : torch.Tensor of shape (num_samples, 4)
            Posterior samples
        """
        if self.posterior is None:
            raise RuntimeError("Must train model before sampling")
            
        return self.posterior.sample((num_samples,), x=x_obs, **kwargs)
    
    def log_prob(self, theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of parameters given observations.
        
        Parameters:
        -----------
        theta : torch.Tensor
            Parameter vectors
        x_obs : torch.Tensor
            Observed data
            
        Returns:
        --------
        log_prob : torch.Tensor
            Log probabilities
        """
        if self.posterior is None:
            raise RuntimeError("Must train model before computing log prob")
            
        return self.posterior.log_prob(theta, x=x_obs)
    
    def save_model(self, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save trained model.
        
        Parameters:
        -----------
        filepath : str
            Output filepath
        metadata : dict, optional
            Additional metadata
        """
        if self.posterior is None:
            raise RuntimeError("No trained model to save")
            
        data = {
            'posterior': self.posterior,
            'inference': self.inference,
            'use_summary_stats': self.use_summary_stats,
            'metadata': metadata or {}
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Saved model to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'LVInference':
        """
        Load trained model.
        
        Parameters:
        -----------
        filepath : str
            Model filepath
            
        Returns:
        --------
        inference_obj : LVInference
            Loaded inference object
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        obj = cls(use_summary_stats=data['use_summary_stats'])
        obj.posterior = data['posterior']
        obj.inference = data['inference']
        
        return obj
    
    def plot_posterior_samples(self, 
                              samples: torch.Tensor,
                              true_theta: Optional[torch.Tensor] = None,
                              figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot posterior samples.
        
        Parameters:
        -----------
        samples : torch.Tensor
            Posterior samples
        true_theta : torch.Tensor, optional
            True parameter values (for validation)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        param_names = ['α (prey birth)', 'β (predation)', 'δ (pred. birth)', 'γ (pred. death)']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        samples_np = samples.numpy()
        
        for i, (ax, name) in enumerate(zip(axes, param_names)):
            # Histogram
            ax.hist(samples_np[:, i], bins=50, alpha=0.7, density=True, color='skyblue')
            
            # True value if provided
            if true_theta is not None:
                ax.axvline(true_theta[i].item(), color='red', linestyle='--', 
                          linewidth=2, label='True value')
                ax.legend()
                
            ax.set_xlabel(name)
            ax.set_ylabel('Density')
            ax.set_title(f'Posterior: {name}')
            
        plt.tight_layout()
        return fig
    
    def plot_pairwise(self, 
                     samples: torch.Tensor,
                     true_theta: Optional[torch.Tensor] = None,
                     figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
        """
        Plot pairwise posterior relationships.
        
        Parameters:
        -----------
        samples : torch.Tensor
            Posterior samples
        true_theta : torch.Tensor, optional
            True parameter values
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        param_names = ['α', 'β', 'δ', 'γ']
        samples_np = samples.numpy()
        
        fig, axes = plt.subplots(4, 4, figsize=figsize)
        
        for i in range(4):
            for j in range(4):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histograms
                    ax.hist(samples_np[:, i], bins=30, alpha=0.7, color='skyblue')
                    if true_theta is not None:
                        ax.axvline(true_theta[i].item(), color='red', linestyle='--')
                    ax.set_title(param_names[i])
                    
                elif i > j:
                    # Lower triangle: scatter plots
                    ax.scatter(samples_np[:, j], samples_np[:, i], 
                             alpha=0.3, s=1, color='skyblue')
                    if true_theta is not None:
                        ax.scatter(true_theta[j].item(), true_theta[i].item(), 
                                 color='red', s=50, marker='x')
                    ax.set_xlabel(param_names[j])
                    ax.set_ylabel(param_names[i])
                    
                else:
                    # Upper triangle: turn off
                    ax.axis('off')
                    
        plt.tight_layout()
        return fig