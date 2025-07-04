# Neural Posterior Estimation for Stochastic Lotka-Volterra Model

This project implements **Neural Posterior Estimation (NPE)** for parameter inference in a stochastic Lotka-Volterra predator-prey model using the Gillespie algorithm and the `sbi` library.

## Overview

The stochastic Lotka-Volterra model describes predator-prey dynamics as a continuous-time Markov chain (CTMC) with four reactions:

1. **Prey birth**: x� � x� + 1 at rate ��x�
2. **Predation**: x� � x� - 1, x� � x� + 1 at rate ��x��x�  
3. **Predator birth**: x� � x� + 1 at rate ��x��x�
4. **Predator death**: x� � x� - 1 at rate ��x�

Given noisy time-series observations, we use NPE to infer the posterior distribution over parameters � = (�, �, �, �).

## Installation

1. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate NPE_LV
   ```

2. **Install additional dependencies** (if conda creation failed):
   ```bash
   conda create -n NPE_LV python=3.10
   conda activate NPE_LV
   pip install torch sbi numpy scipy matplotlib seaborn jupyter pandas tqdm tensorboard
   ```

## Project Structure

```
NPE_LV/
   src/                     # Core modules
      simulator.py         # Gillespie algorithm implementation
      utils.py            # Utility functions and priors
      data_generation.py  # Training data generation
      inference.py        # NPE training and inference
   scripts/                # Executable scripts  
      generate_data.py    # Generate training data
      train_npe.py       # Train NPE model
      run_inference.py   # Run inference on observations
   notebooks/              # Analysis and visualization
      visualization.ipynb # Interactive analysis
   environment.yml         # Conda environment specification
```

## Quick Start

### 1. Generate Training Data

```bash
python scripts/generate_data.py \
    --n_samples 10000 \
    --output data/training_data.pkl \
    --t_max 10.0 \
    --dt 0.1
```

**Options**:
- `--summary_stats`: Use summary statistics instead of full trajectories
- `--x0_prey`, `--x0_pred`: Initial populations (default: 50, 100)
- `--seed`: Random seed for reproducibility

### 2. Train NPE Model

```bash
python scripts/train_npe.py \
    --data data/training_data.pkl \
    --output models/npe_model.pkl \
    --max_epochs 100 \
    --batch_size 512
```

**Options**:
- `--hidden_features`: Neural network width (default: 128)
- `--num_transforms`: Number of coupling layers (default: 5)
- `--device`: Training device (`cpu` or `cuda`)

### 3. Run Inference

```bash
python scripts/run_inference.py \
    --model models/npe_model.pkl \
    --theta_true 0.5 0.025 0.025 0.5 \
    --num_samples 5000 \
    --output_dir results/
```

This generates synthetic observations and samples from the posterior. For real data, use `--observation path/to/data.pt`.

## Usage Examples

### Python API

```python
from src.simulator import gillespie_lv
from src.utils import create_lv_prior, create_time_grid
from src.data_generation import LVDataGenerator
from src.inference import LVInference

# Generate training data
generator = LVDataGenerator(x0=(50, 100), t_max=10.0)
theta, x = generator.generate_batch(n_samples=1000)

# Train NPE
inference = LVInference()
training_info = inference.train(theta, x)

# Sample from posterior
x_obs = x[0]  # Use first observation as test
posterior_samples = inference.sample_posterior(x_obs, num_samples=1000)
```

### Jupyter Notebook

Use `notebooks/visualization.ipynb` for interactive analysis:
- Visualize simulator trajectories
- Analyze training data quality  
- Perform posterior predictive checks
- Evaluate coverage properties

## Model Configuration

### Prior Distribution
Default parameter bounds:
- � (prey birth): [0.01, 1.0]
- � (predation): [0.001, 0.1] 
- � (predator birth): [0.001, 0.1]
- � (predator death): [0.01, 1.0]

### Neural Network
- **Architecture**: Neural Spline Flow (NSF)
- **Hidden layers**: 128 units (configurable)
- **Coupling transforms**: 5 layers (configurable)
- **Training**: Adam optimizer with early stopping

## Performance Considerations

The Gillespie algorithm can be computationally expensive. Consider:

1. **Parallel simulation**: Use multiprocessing for large datasets
2. **JAX acceleration**: Implement JAX version for GPU acceleration  
3. **Summary statistics**: Use summary stats instead of full trajectories
4. **Early stopping**: Cap simulation time for unstable parameter regions

## Troubleshooting

**Common issues**:

1. **Simulation failures**: Reduce `t_max` or adjust parameter bounds
2. **Poor convergence**: Increase training data or network capacity
3. **Memory issues**: Reduce batch size or use summary statistics
4. **Conda environment**: Install packages manually if environment creation fails

## References

- [SBI Documentation](https://sbi-dev.github.io/sbi/)
- Papamakarios et al. (2019). Sequential Neural Likelihood
- Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions

## License

MIT License