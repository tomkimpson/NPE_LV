# Implementation Notes for TEIRV + Gillespie + NPE Pipeline

## 1. TEIRV Model Structure

### What do T, E, I, R, V compartments represent?

- **T**: Target cells — susceptible to infection by virions.
- **E**: Eclipsed cells — newly infected, not yet producing virus.
- **I**: Infectious cells — actively producing virions.
- **R**: Refractory cells — temporarily resistant to infection (e.g. due to interferon).
- **V**: Virions — free virus particles in the host.

### What are the reactions/transitions between compartments?

1. **Infection**:  
   \( T + V \xrightarrow{\beta} E \)  
   Target cells become infected by virions.

2. **Progression to infectious**:  
   \( E \xrightarrow{k} I \)  
   Eclipsed cells become infectious.

3. **Clearance of infected cells**:  
   \( I \xrightarrow{\delta} \emptyset \)  
   Infectious cells are cleared.

4. **Viral production**:  
   \( I \xrightarrow{\pi} I + V \)  
   Infectious cells produce virions.

5. **Viral clearance**:  
   \( V \xrightarrow{c} \emptyset \)  
   Virions are cleared from the system.

6. **Interferon-induced protection**:  
   \( T + I \xrightarrow{\Phi} R \)  
   Interferon causes target cells to become refractory.

7. **Loss of protection**:  
   \( R \xrightarrow{\rho} T \)  
   Refractory cells revert to susceptible state.

---

## 2. Gillespie Implementation

### How many reactions are there in total?

There are **7 reactions** corresponding to the transitions described above.

### What are the rate functions for each reaction?

These are the stochastic propensities used in the Gillespie algorithm:

1. \( \lambda_1 = \beta \cdot V \cdot T \)
2. \( \lambda_2 = k \cdot E \)
3. \( \lambda_3 = \delta \cdot I \)
4. \( \lambda_4 = \pi \cdot I \)
5. \( \lambda_5 = c \cdot V \)
6. \( \lambda_6 = \Phi \cdot I \cdot T \)
7. \( \lambda_7 = \rho \cdot R \)

---

## 3. Data/Observations

### What data do they fit to?

- Longitudinal **nasal swab data** from infected individuals.
- Specifically, **Cycle Number (CN)** values from RT-PCR tests.

### What's the observation model?

- CN values are transformed to log10 viral load using:
  \[
  \log_{10} V = 11.35 - 0.25 \cdot \text{CN}
  \]
- Observation model:
  \[
  y_t \sim \mathcal{N}(\log_{10} V(t), \sigma^2 = 1)
  \]
- Truncation is applied at the detection limit: \( \log_{10} V \geq -0.65 \).

### Any missing data or partial observations?

- Only the **virion count (V)** is observed.
- The selected dataset consists of 6 patients with **14 full observations** each.

---

## 4. Inference Approach (NPE)

### What are we trying to infer?

We infer **6 parameters**, while fixing 2 based on prior literature.

#### Inferred Parameters and Priors

| Parameter  | Description                                      | Prior Distribution                  |
|------------|--------------------------------------------------|--------------------------------------|
| \( \beta \)   | Infection rate (V infecting T)                   | \( \text{LogNormal}(\log(2.5 \times 10^{-9}), 1.0) \) |
| \( \pi \)     | Virion production rate (per infected cell)       | \( \text{LogNormal}(\log(10^2), 1.0) \)              |
| \( \delta \)  | Infected cell clearance rate                    | \( \text{LogNormal}(\log(0.5), 1.0) \)               |
| \( \Phi \)    | Interferon-induced protection rate (T → R)       | \( \text{LogNormal}(\log(10^{-9}), 2.0) \)           |
| \( \rho \)    | Reversion rate from R → T                        | \( \text{LogNormal}(\log(0.1), 1.0) \)               |
| \( V(0) \)    | Initial virion count                             | \( \text{LogNormal}(\log(10^3), 1.0) \)              |

#### Fixed Parameters

| Parameter  | Description                           | Value     |
|------------|---------------------------------------|-----------|
| \( k \)       | E → I transition rate                 | 4         |
| \( c \)       | Virion clearance rate                | 10        |

#### Initial Conditions

- \( T(0) = 8 \times 10^7 \)
- \( E(0) = 1 \)
- \( I(0) = 0 \)
- \( R(0) = 0 \)
- \( V(0) \) is inferred

---

### How do we handle stochastic dynamics?

- The TEIRV model is simulated using the **Gillespie algorithm** (exact stochastic simulation).
- Each simulation generates a full time series of states, which is passed through the observation model.

---

## 5. Neural Posterior Estimation (NPE) Pipeline

1. **Simulation**:
   - Sample parameters \( \theta_i \sim p(\theta) \)
   - Run Gillespie simulation to generate latent trajectories.
   - Apply observation model to produce synthetic observed data \( y_i \).

2. **Training**:
   - Train a conditional density estimator (e.g. MAF, NSF, Simformer) on dataset \( \{(\theta_i, y_i)\} \).

3. **Inference**:
   - Given real observed data \( y_{\text{obs}} \), evaluate or sample from posterior \( p(\theta \mid y_{\text{obs}}) \).

---

