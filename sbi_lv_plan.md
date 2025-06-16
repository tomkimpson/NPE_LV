
# Simulation-Based Inference Plan: Stochastic Predator-Prey System

## 🧭 Goal

We want to infer the unknown parameters of a **stochastic Lotka–Volterra (LV)** predator-prey model from discrete time-series data. The model is simulated via the **Gillespie algorithm** (CTMC), but our observations are noisy and discrete.

We’ll use **Neural Posterior Estimation (NPE)** to approximate the posterior distribution over parameters, conditioned on observed data.

---

## 📈 Problem Setup

- **Model**: CTMC with states \( (x_1(t), x_2(t)) \): prey and predator populations.
- **Events**:
  - Prey birth: \( x_1 \rightarrow x_1 + 1 \) at rate \( \alpha x_1 \)
  - Prey death (predation): \( x_1 \rightarrow x_1 - 1 \), at rate \( \beta x_1 x_2 \)
  - Predator birth: \( x_2 \rightarrow x_2 + 1 \) at rate \( \delta x_1 x_2 \)
  - Predator death: \( x_2 \rightarrow x_2 - 1 \) at rate \( \gamma x_2 \)
- **Observations**: Discrete in time, possibly noisy.

---

## 🛠️ Implementation Steps

### Step 1: Implement the Simulator (Gillespie algorithm)

- Inputs:
  - Parameters \( \theta = (\alpha, \beta, \delta, \gamma) \)
  - Initial conditions (e.g. \( x_1 = 50, x_2 = 100 \))
  - Final time \( T \), time grid for interpolation
- Output:
  - \( (t_k, x_1(t_k), x_2(t_k)) \)
  - Interpolated onto a regular time grid if needed

> **Pain point**: Gillespie produces irregular time steps. Interpolation is required.

---

### Step 2: Choose a Prior over Parameters

```python
prior = BoxUniform(low=torch.tensor([0.01, 0.001, 0.001, 0.01]),
                   high=torch.tensor([1.0, 0.1, 0.1, 1.0]))
```

> **Pain point**: Avoid extremely low rates that lead to slow simulation or extinction.

---

### Step 3: Define Summary Statistics (Optional)

- Use either:
  - Full time series (interpolated to fixed grid)
  - Summary statistics: mean, variance, autocorrelation, peak timing

---

### Step 4: Generate Training Data

1. Sample \( \theta \sim \text{prior} \)
2. Simulate the LV process
3. Interpolate or summarize
4. Store \( (\theta, x) \)

> **Pain point**: Simulation is expensive. Consider batching and checkpointing.

---

### Step 5: Train Neural Posterior Estimator (NPE)

```python
from sbi.inference import SNPE
inference = SNPE(prior)
inference.append_simulations(theta_tensor, x_tensor).train()
posterior = inference.build_posterior()
```

> **Pain point**: Long time series may require dimensionality reduction or regularization.

---

### Step 6: Inference on Observed Data

```python
posterior_samples = posterior.sample((1000,), x=x_obs_tensor)
```

- Condition on real data
- Visualize posterior

---

## 📦 Optional Enhancements

| Feature | Description |
|--------|-------------|
| ⚡ Parallel simulation | Use multiprocessing to speed up Gillespie |
| 🧠 Summary net | Train a neural network to learn summary statistics |
| 🎯 Prediction | Sample forward from posterior to forecast |
| 🔍 Diagnostics | Posterior predictive checks or SBC |

---

## ✅ Deliverables

1. Gillespie simulator
2. Data generation script
3. NPE training pipeline
4. Inference script
5. Optional: visualization notebook

---

## 🚩 Potential Pitfalls

| Issue | Mitigation |
|------|------------|
| Simulation time too long | Cap time / use early stopping |
| Irregular time steps | Interpolate before feeding to network |
| Poor convergence | Normalize, tune priors, use expressive models |
| Posterior collapse | Try NSF, more data, or deeper networks |
