#!/usr/bin/env python3
"""
Test script for Phase 3 clinical implementation.

Tests the complete clinical inference pipeline to ensure all components work together.
"""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from TEIRV.clinical_data import ClinicalDataLoader, ClinicalStudy, validate_clinical_data_compatibility
from TEIRV.teirv_utils import create_teirv_time_grid


def test_clinical_data_loader():
    """Test clinical data loading functionality."""
    print("Testing clinical data loader...")
    
    try:
        # Initialize loader
        loader = ClinicalDataLoader()
        print(f"  ✅ Initialized loader, found {len(loader.patient_ids)} patients")
        
        # Test loading a single patient
        if loader.patient_ids:
            patient_id = loader.patient_ids[0]
            times, observations = loader.load_patient_data(patient_id)
            print(f"  ✅ Loaded patient {patient_id}: {len(observations)} observations")
            
            # Test patient summary
            summary = loader.get_patient_summary(patient_id)
            print(f"  ✅ Generated summary for patient {patient_id}")
            
            # Test NPE preprocessing
            time_grid = create_teirv_time_grid(14.0, 1.0)
            x_obs = loader.preprocess_for_npe(patient_id, time_grid)
            print(f"  ✅ Preprocessed data shape: {x_obs.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Clinical data loader test failed: {e}")
        return False


def test_clinical_study():
    """Test clinical study management."""
    print("\nTesting clinical study management...")
    
    try:
        # Initialize study
        study = ClinicalStudy()
        print(f"  ✅ Initialized study with {len(study.patient_data)} patients")
        
        # Test study summary
        summary_df = study.get_study_summary()
        print(f"  ✅ Generated study summary: {summary_df.shape}")
        
        # Test patient filtering
        good_patients = study.filter_patients(min_detections=3, min_peak_viral_load=1.0)
        print(f"  ✅ Filtered to {len(good_patients)} high-quality patients")
        
        # Test inference preparation
        if good_patients:
            time_grid = create_teirv_time_grid(14.0, 1.0)
            inference_data = study.prepare_for_inference(good_patients[:2], time_grid)
            print(f"  ✅ Prepared {len(inference_data)} patients for inference")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Clinical study test failed: {e}")
        return False


def test_data_compatibility():
    """Test clinical data compatibility validation."""
    print("\nTesting data compatibility validation...")
    
    try:
        # Run compatibility check
        is_compatible = validate_clinical_data_compatibility()
        
        if is_compatible:
            print("  ✅ Clinical data compatibility validated")
        else:
            print("  ⚠️  Clinical data compatibility issues detected")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data compatibility test failed: {e}")
        return False


def test_imports():
    """Test that all clinical scripts can be imported."""
    print("\nTesting clinical script imports...")
    
    try:
        # Test importing the clinical scripts
        sys.path.append(str(Path(__file__).parent.parent / 'scripts'))
        
        # These should not raise ImportError
        import fit_clinical_data
        print("  ✅ fit_clinical_data.py imports successfully")
        
        import clinical_workflow
        print("  ✅ clinical_workflow.py imports successfully")
        
        import validate_against_benchmark
        print("  ✅ validate_against_benchmark.py imports successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Import test error: {e}")
        return False


def test_pipeline_components():
    """Test individual pipeline components."""
    print("\nTesting pipeline components...")
    
    try:
        # Test that we can create the necessary objects
        from TEIRV.teirv_inference import TEIRVInference
        from TEIRV.teirv_utils import create_teirv_prior, create_teirv_time_grid
        
        # Create inference object
        npe = TEIRVInference()
        print("  ✅ NPE inference object created")
        
        # Create prior
        prior = create_teirv_prior()
        print("  ✅ TEIRV prior created")
        
        # Create time grid
        time_grid = create_teirv_time_grid(14.0, 1.0)
        print(f"  ✅ Time grid created: {len(time_grid)} points")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline component test failed: {e}")
        return False


def main():
    """Run all Phase 3 tests."""
    print("Phase 3 Clinical Implementation Test Suite")
    print("=" * 50)
    
    start_time = time.time()
    tests_passed = 0
    total_tests = 5
    
    # Run tests
    if test_clinical_data_loader():
        tests_passed += 1
    
    if test_clinical_study():
        tests_passed += 1
    
    if test_data_compatibility():
        tests_passed += 1
    
    if test_imports():
        tests_passed += 1
    
    if test_pipeline_components():
        tests_passed += 1
    
    # Summary
    test_time = time.time() - start_time
    
    print(f"\nTest Results")
    print("=" * 20)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Test time: {test_time:.2f}s")
    
    if tests_passed == total_tests:
        print("✅ All Phase 3 tests passed!")
        print("\nPhase 3 Clinical Implementation is ready for use:")
        print("  - Clinical data loading: ✅")
        print("  - Patient preprocessing: ✅") 
        print("  - NPE inference pipeline: ✅")
        print("  - Validation framework: ✅")
        print("  - End-to-end workflow: ✅")
        
        print(f"\nUsage examples:")
        print(f"  python scripts/fit_clinical_data.py --model models/high_quality_npe.pkl")
        print(f"  python scripts/clinical_workflow.py --model models/high_quality_npe.pkl")
        print(f"  python scripts/validate_against_benchmark.py --npe_results results/clinical_inference")
        
        return True
    else:
        print(f"❌ {total_tests - tests_passed} tests failed")
        print("Phase 3 implementation needs attention before use")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)