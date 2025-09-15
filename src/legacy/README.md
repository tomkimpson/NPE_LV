# TEIRV Legacy Files

This directory contains older implementations and experimental versions of TEIRV model components that have been superseded by more efficient JAX-based implementations.

## Files Moved to Legacy

### Simulator Variants
- **`teirv_simulator.py`** - Original NumPy-based Gillespie implementation
- **`teirv_simulator_clean.py`** - Clean NumPy implementation without max_steps limitations
- **`teirv_simulator_optimized.py`** - Memory-optimized NumPy version with multiple storage strategies
- **`teirv_simulator_smart.py`** - NumPy version with adaptive computational budgets

### Data Generation Variants  
- **`teirv_data_generation.py`** - Original NumPy-based data generator
- **`teirv_batch_optimized.py`** - Parallel batch processing (NumPy-based)
- **`teirv_fast_batch.py`** - Fast sequential batch processing (NumPy-based)

## Why These Were Moved

These files were moved to legacy status because they have been superseded by JAX-based implementations that offer:

- **4-15x performance improvement** through JIT compilation
- **Vectorized batch processing** for better throughput
- **Memory efficiency** through sparse storage and optimized algorithms
- **GPU acceleration potential** for future scaling
- **No artificial computational limits** (proper simulation completion)

## Current Core Files

The active TEIRV implementation now consists of only 4 core files:

1. **`teirv_utils.py`** - Essential utilities, priors, and visualization functions
2. **`teirv_simulator_jax.py`** - JAX-accelerated Gillespie simulator with JIT compilation
3. **`teirv_data_generation_jax.py`** - JAX-accelerated data generation (4-15x faster)
4. **`teirv_inference.py`** - NPE training and inference using SBI

## Legacy File Preservation

These files are kept for reference purposes in case:
- Specific optimizations or algorithms need to be referenced
- Compatibility issues arise with JAX implementations
- Performance comparisons or benchmarking is needed
- Research into alternative implementation approaches is required

**Note:** These legacy files may have outdated import statements due to the recent codebase reorganization. If you need to use them, you may need to update the import paths accordingly.