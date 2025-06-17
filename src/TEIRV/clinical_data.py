"""
Clinical data loading and processing for TEIRV NPE.

Handles RT-PCR data from COVID patients in the JSFGermano2024 repository.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

from .teirv_utils import preprocess_clinical_data


class ClinicalDataLoader:
    """Loader for clinical RT-PCR data from JSFGermano2024 repository."""
    
    def __init__(self, data_directory: Optional[str] = None):
        """
        Initialize clinical data loader.
        
        Parameters:
        -----------
        data_directory : str, optional
            Path to clinical data directory. If None, uses default location.
        """
        if data_directory is None:
            # Default to submodule location
            self.data_dir = Path(__file__).parent.parent / "external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/data"
        else:
            self.data_dir = Path(data_directory)
            
        self.patient_ids = self._discover_patient_files()
        
    def _discover_patient_files(self) -> List[str]:
        """Discover available patient data files."""
        if not self.data_dir.exists():
            warnings.warn(f"Clinical data directory not found: {self.data_dir}")
            return []
            
        # Look for .ssv files (space-separated values)
        patient_files = list(self.data_dir.glob("*.ssv"))
        patient_ids = [f.stem for f in patient_files]
        
        print(f"Found {len(patient_ids)} patient datasets: {patient_ids}")
        return patient_ids
    
    def load_patient_data(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for a specific patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient ID (e.g., '432192')
            
        Returns:
        --------
        times : np.ndarray
            Time points (days)
        observations : np.ndarray
            RT-PCR observations (log₁₀ viral load)
        """
        filepath = self.data_dir / f"{patient_id}.ssv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Patient data not found: {filepath}")
            
        # Load space-separated values
        data = np.loadtxt(filepath, skiprows=1)
        times = data[:, 0]
        observations = data[:, 1]
        
        return times, observations
    
    def load_all_patients(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load data for all available patients.
        
        Returns:
        --------
        patient_data : dict
            Dictionary mapping patient_id -> (times, observations)
        """
        patient_data = {}
        
        for patient_id in self.patient_ids:
            try:
                times, observations = self.load_patient_data(patient_id)
                patient_data[patient_id] = (times, observations)
                print(f"Loaded patient {patient_id}: {len(observations)} timepoints")
            except Exception as e:
                warnings.warn(f"Failed to load patient {patient_id}: {e}")
                
        return patient_data
    
    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient ID
            
        Returns:
        --------
        summary : dict
            Summary statistics
        """
        times, observations = self.load_patient_data(patient_id)
        
        # Detection limit analysis
        detection_limit = -0.65
        above_detection = observations > detection_limit
        n_detected = np.sum(above_detection)
        
        # Valid observation statistics
        if n_detected > 0:
            valid_obs = observations[above_detection]
            peak_time = times[np.argmax(observations)]
            peak_value = np.max(observations)
        else:
            valid_obs = np.array([])
            peak_time = np.nan
            peak_value = np.nan
        
        summary = {
            'patient_id': patient_id,
            'n_timepoints': len(observations),
            'time_range': (times.min(), times.max()),
            'n_above_detection': n_detected,
            'detection_rate': n_detected / len(observations),
            'peak_time': peak_time,
            'peak_viral_load': peak_value,
            'mean_detected_vl': np.mean(valid_obs) if len(valid_obs) > 0 else np.nan,
            'std_detected_vl': np.std(valid_obs) if len(valid_obs) > 0 else np.nan,
            'viral_load_range': (valid_obs.min(), valid_obs.max()) if len(valid_obs) > 0 else (np.nan, np.nan)
        }
        
        return summary
    
    def preprocess_for_npe(self, 
                          patient_id: str,
                          target_times: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Preprocess patient data for NPE inference.
        
        Parameters:
        -----------
        patient_id : str
            Patient ID
        target_times : np.ndarray, optional
            Target time grid. If None, uses original times.
            
        Returns:
        --------
        x_obs : torch.Tensor
            Preprocessed observations ready for NPE
        """
        times, observations = self.load_patient_data(patient_id)
        
        # Interpolate to target time grid if provided
        if target_times is not None:
            if len(target_times) != len(times):
                # Interpolate observations to target times
                observations = np.interp(target_times, times, observations)
                times = target_times
                
        # Apply preprocessing (detection limits, etc.)
        processed_obs = preprocess_clinical_data(observations)
        
        return torch.tensor(processed_obs, dtype=torch.float32)


class ClinicalStudy:
    """Manages a complete clinical study with multiple patients."""
    
    def __init__(self, data_directory: Optional[str] = None):
        """Initialize clinical study manager."""
        self.loader = ClinicalDataLoader(data_directory)
        self.patient_data = self.loader.load_all_patients()
        
    def get_study_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all patients in the study.
        
        Returns:
        --------
        summary_df : pd.DataFrame
            Summary statistics for each patient
        """
        summaries = []
        
        for patient_id in self.loader.patient_ids:
            try:
                summary = self.loader.get_patient_summary(patient_id)
                summaries.append(summary)
            except Exception as e:
                warnings.warn(f"Failed to summarize patient {patient_id}: {e}")
                
        return pd.DataFrame(summaries)
    
    def filter_patients(self, 
                       min_detections: int = 5,
                       min_peak_viral_load: float = 2.0) -> List[str]:
        """
        Filter patients based on data quality criteria.
        
        Parameters:
        -----------
        min_detections : int
            Minimum number of observations above detection limit
        min_peak_viral_load : float
            Minimum peak viral load (log₁₀ scale)
            
        Returns:
        --------
        filtered_ids : list
            List of patient IDs meeting criteria
        """
        filtered_ids = []
        
        for patient_id in self.loader.patient_ids:
            try:
                summary = self.loader.get_patient_summary(patient_id)
                
                if (summary['n_above_detection'] >= min_detections and 
                    summary['peak_viral_load'] >= min_peak_viral_load):
                    filtered_ids.append(patient_id)
                    
            except Exception as e:
                warnings.warn(f"Error filtering patient {patient_id}: {e}")
                
        print(f"Filtered {len(filtered_ids)}/{len(self.loader.patient_ids)} patients meeting criteria")
        return filtered_ids
    
    def prepare_for_inference(self, 
                             patient_ids: Optional[List[str]] = None,
                             target_time_grid: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        """
        Prepare patient data for NPE inference.
        
        Parameters:
        -----------
        patient_ids : list, optional
            Specific patients to include. If None, uses all patients.
        target_time_grid : np.ndarray, optional
            Target time grid for interpolation
            
        Returns:
        --------
        inference_data : dict
            Dictionary mapping patient_id -> preprocessed observations
        """
        if patient_ids is None:
            patient_ids = self.loader.patient_ids
            
        inference_data = {}
        
        for patient_id in patient_ids:
            try:
                x_obs = self.loader.preprocess_for_npe(patient_id, target_time_grid)
                inference_data[patient_id] = x_obs
                print(f"Prepared patient {patient_id}: {x_obs.shape}")
            except Exception as e:
                warnings.warn(f"Failed to prepare patient {patient_id}: {e}")
                
        return inference_data


def load_clinical_validation_data() -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """
    Convenience function to load clinical data for validation.
    
    Returns:
    --------
    patient_observations : dict
        Patient data ready for inference
    study_summary : pd.DataFrame
        Summary statistics for all patients
    """
    study = ClinicalStudy()
    
    # Get study overview
    summary_df = study.get_study_summary()
    print("\nClinical Study Summary:")
    print("=" * 50)
    print(summary_df[['patient_id', 'n_above_detection', 'peak_viral_load', 'detection_rate']].to_string(index=False))
    
    # Filter high-quality patients
    good_patients = study.filter_patients(min_detections=5, min_peak_viral_load=2.0)
    
    # Prepare for inference  
    patient_observations = study.prepare_for_inference(good_patients)
    
    return patient_observations, summary_df


def validate_clinical_data_compatibility():
    """
    Validate that clinical data is compatible with our NPE pipeline.
    """
    print("Validating clinical data compatibility...")
    
    try:
        # Load data
        study = ClinicalStudy()
        summary_df = study.get_study_summary()
        
        # Check basic compatibility
        print(f"\n✅ Found {len(study.loader.patient_ids)} patient datasets")
        
        # Check time grid compatibility
        from .teirv_utils import create_teirv_time_grid
        target_grid = create_teirv_time_grid(14.0, 1.0)
        
        compatible_patients = 0
        for patient_id in study.loader.patient_ids:
            times, obs = study.loader.load_patient_data(patient_id)
            if len(obs) == len(target_grid):
                compatible_patients += 1
                
        print(f"✅ {compatible_patients}/{len(study.loader.patient_ids)} patients have compatible time grids")
        
        # Check detection rates
        detection_rates = summary_df['detection_rate'].dropna()
        if len(detection_rates) > 0:
            mean_detection = detection_rates.mean()
            print(f"✅ Mean detection rate: {mean_detection:.1%}")
        
        # Check viral load ranges
        peak_loads = summary_df['peak_viral_load'].dropna()
        if len(peak_loads) > 0:
            print(f"✅ Viral load range: [{peak_loads.min():.2f}, {peak_loads.max():.2f}] log₁₀")
            
        print("\n✅ Clinical data validation completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Clinical data validation failed: {e}")
        return False