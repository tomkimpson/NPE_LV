#!/usr/bin/env python3
"""
Quick script to run just the inference step using existing trained model.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from scripts.TEIRV_workflow import TEIRVWorkflow

def main():
    # Use the existing trained model from the production run
    model_path = "workflows/production_run_20250703_123040/npe_model.pkl"
    output_path = "workflows/production_run_20250703_123040/inference_results"
    
    print(f"🏥 Running inference with existing model: {model_path}")
    
    # Create inference arguments
    args = argparse.Namespace(
        model=model_path,
        output=output_path,
        data_dir="external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/data",
        n_samples=20000,  # Fixed: was inference_samples
        min_detections=5,
        min_peak_vl=2.0,
        patients=None,  # Use all patients
        device='cuda',
        seed=42
    )
    
    # Initialize workflow
    workflow = TEIRVWorkflow(device='cuda', seed=42)
    
    # Run inference only
    workflow.inference(args)
    
    print(f"✅ Inference completed! Results in: {output_path}")

if __name__ == "__main__":
    main()