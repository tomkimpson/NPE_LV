"""
NPE training and inference for TEIRV viral dynamics model.
"""
import torch
from typing import Optional, Dict, Any, Tuple
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn

from .teirv_utils import create_teirv_prior, visualize_teirv_trajectory
from .teirv_data_generation import TEIRVDataGenerator


class TEIRVInference:
    """Neural Posterior Estimation for TEIRV viral dynamics model."""
    
    def __init__(self, 
                 observation_type: str = 'rt_pcr',
                 device: str = 'cpu',
                 seed: Optional[int] = None):
        """
        Initialize TEIRV NPE inference.
        
        Parameters:
        -----------
        observation_type : str
            Type of observations ('rt_pcr' or 'full_trajectory')
        device : str
            Device for training ('cpu' or 'cuda')
        seed : int, optional
            Random seed
        """
        self.observation_type = observation_type
        self.device = device
        
        if seed is not None:
            torch.manual_seed(seed)
            
        self.prior = create_teirv_prior()
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
        # Default neural network configuration for TEIRV
        neural_net_kwargs = {
            'hidden_features': 256,  # Larger network for more complex problem
            'num_transforms': 8,     # More transforms for better expressivity
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
              learning_rate: float = 5e-4,  # Slightly higher for TEIRV
              max_num_epochs: int = 150,    # More epochs for complex problem
              validation_fraction: float = 0.15,
              stop_after_epochs: int = 25,
              **kwargs) -> Dict[str, Any]:
        """
        Train neural posterior estimator.
        
        Parameters:
        -----------
        theta : torch.Tensor of shape (n_samples, 6)
            Parameter vectors [β, π, δ, φ, ρ, V₀]
        x : torch.Tensor of shape (n_samples, x_dim)
            Observations (RT-PCR or full trajectory)
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
            
        print(f"Training TEIRV NPE with {len(theta)} samples...")
        print(f"Parameter shape: {theta.shape}")
        print(f"Observation shape: {x.shape}")
        print(f"Observation type: {self.observation_type}")
        
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
            Observed RT-PCR data or full trajectory
        num_samples : int
            Number of posterior samples
        **kwargs : additional sampling arguments
            
        Returns:
        --------
        samples : torch.Tensor of shape (num_samples, 6)
            Posterior samples [β, π, δ, φ, ρ, V₀]
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
            'observation_type': self.observation_type,
            'metadata': metadata or {}
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Saved TEIRV model to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TEIRVInference':
        """
        Load trained model.
        
        Parameters:
        -----------
        filepath : str
            Model filepath
            
        Returns:
        --------
        inference_obj : TEIRVInference
            Loaded inference object
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        obj = cls(observation_type=data['observation_type'])
        obj.posterior = data['posterior']
        obj.inference = data['inference']
        
        return obj
    
    def plot_posterior_samples(self, 
                              samples: torch.Tensor,
                              true_theta: Optional[torch.Tensor] = None,
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot posterior samples for TEIRV parameters.
        
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
        param_names = ['β (infection)', 'π (virion prod.)', 'δ (cell clear.)', 
                      'φ (interferon)', 'ρ (reversion)', 'V₀ (initial)']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        samples_np = samples.numpy()
        
        for i, (ax, name) in enumerate(zip(axes, param_names)):
            # Histogram
            ax.hist(samples_np[:, i], bins=50, alpha=0.7, density=True, color='teal')
            
            # True value if provided
            if true_theta is not None:
                ax.axvline(true_theta[i].item(), color='orange', linestyle='--', 
                          linewidth=2, label='True value')
                ax.legend()
                
            ax.set_xlabel(name)
            ax.set_ylabel('Density')
            ax.set_title(f'Posterior: {name}')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        return fig
    
    def plot_pairwise(self, 
                     samples: torch.Tensor,
                     true_theta: Optional[torch.Tensor] = None,
                     figsize: Tuple[int, int] = (12, 12)) -> plt.Figure:
        """
        Plot pairwise posterior relationships for TEIRV parameters.
        
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
        param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
        samples_np = samples.numpy()
        
        fig, axes = plt.subplots(6, 6, figsize=figsize)
        
        for i in range(6):
            for j in range(6):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histograms
                    ax.hist(samples_np[:, i], bins=30, alpha=0.7, color='teal')
                    if true_theta is not None:
                        ax.axvline(true_theta[i].item(), color='orange', linestyle='--')
                    ax.set_title(param_names[i])
                    
                elif i > j:
                    # Lower triangle: scatter plots
                    ax.scatter(samples_np[:, j], samples_np[:, i], 
                             alpha=0.3, s=1, color='teal')
                    if true_theta is not None:
                        ax.scatter(true_theta[j].item(), true_theta[i].item(), 
                                 color='orange', s=50, marker='x')
                    ax.set_xlabel(param_names[j])
                    ax.set_ylabel(param_names[i])
                    
                else:
                    # Upper triangle: turn off
                    ax.axis('off')
                    
        plt.tight_layout()
        return fig
    
    def plot_corner(self,
                   samples: torch.Tensor,
                   true_theta: Optional[torch.Tensor] = None,
                   **corner_kwargs) -> plt.Figure:
        """
        Create corner plot for TEIRV parameters.
        
        Parameters:
        -----------
        samples : torch.Tensor
            Posterior samples
        true_theta : torch.Tensor, optional
            True parameter values
        **corner_kwargs : additional arguments for corner.corner
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Corner plot figure
        """
        try:
            import corner
            
            param_labels = [r'$\beta$', r'$\pi$', r'$\delta$', 
                           r'$\phi$', r'$\rho$', r'$V_0$']
            
            # Get prior bounds for plot limits
            prior_bounds = [
                (0.0, 20.0),     # β
                (200.0, 600.0),  # π  
                (1.0, 11.0),     # δ
                (0.0, 15.0),     # φ
                (0.0, 1.0),      # ρ
                (1.0, 148.0)     # V₀
            ]
            
            # Default corner plot settings
            default_kwargs = {
                'labels': param_labels,
                'truths': true_theta.numpy() if true_theta is not None else None,
                'truth_color': 'orange',
                'color': 'teal',
                'range': prior_bounds,
                'plot_datapoints': True,
                'plot_density': True,
                'plot_contours': True,
                'data_kwargs': {'alpha': 0.2, 'color': 'lightblue'},
                'hist_kwargs': {'alpha': 0.8, 'color': 'teal'},
                'contour_kwargs': {'colors': 'teal'},
                'smooth': 1.0,
                'smooth1d': 1.0,
                'quantiles': [0.16, 0.5, 0.84],
                'show_titles': True,
                'title_kwargs': {"fontsize": 12},
                'label_kwargs': {"fontsize": 14}
            }
            
            # Update with user-provided kwargs
            default_kwargs.update(corner_kwargs)
            
            fig = corner.corner(samples.numpy(), **default_kwargs)
            return fig
            
        except ImportError:
            print("Corner package not available - install with 'pip install corner'")
            return self.plot_pairwise(samples, true_theta)
        except Exception as e:
            print(f"Failed to create corner plot: {e}")
            return self.plot_pairwise(samples, true_theta)
    
    def posterior_predictive_check(self,
                                  posterior_samples: torch.Tensor,
                                  x_obs: torch.Tensor,
                                  true_theta: Optional[torch.Tensor] = None,
                                  n_pred_samples: int = 20) -> plt.Figure:
        """
        Perform posterior predictive check for TEIRV model.
        
        Parameters:
        -----------
        posterior_samples : torch.Tensor
            Samples from posterior distribution
        x_obs : torch.Tensor
            Observed data
        true_theta : torch.Tensor, optional
            True parameter values
        n_pred_samples : int
            Number of predictive samples to generate
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Predictive check plot
        """
        # Generate predictions from posterior samples
        generator = TEIRVDataGenerator(use_observations_only=True)
        
        predicted_obs = []
        for i in range(min(n_pred_samples, len(posterior_samples))):
            theta_sample = posterior_samples[i].numpy()
            try:
                times, pred_obs = generator.generate_test_patient_data(
                    true_theta=theta_sample,
                    noise_seed=i + 1000  # Different seed for each prediction
                )
                predicted_obs.append(pred_obs)
            except:
                continue
        
        if len(predicted_obs) == 0:
            print("No successful predictions generated")
            return plt.figure()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot predicted trajectories
        for i, pred in enumerate(predicted_obs):
            alpha = 0.1 if i > 0 else 0.3
            label = 'Posterior predictions' if i == 0 else None
            ax.plot(times, pred, color='teal', alpha=alpha, label=label)
        
        # Plot observed data
        ax.scatter(times, x_obs.numpy(), color='orange', s=60, 
                  label='Observed data', zorder=5)
        
        # Plot true prediction if available
        if true_theta is not None:
            try:
                times_true, true_pred = generator.generate_test_patient_data(
                    true_theta=true_theta.numpy(),
                    noise_seed=999
                )
                ax.plot(times_true, true_pred, color='red', linewidth=2, 
                       label='True parameters', zorder=4)
            except:
                pass
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('log₁₀ Viral Load')
        ax.set_title('TEIRV Posterior Predictive Check')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig