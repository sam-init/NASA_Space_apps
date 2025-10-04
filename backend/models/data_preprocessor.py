"""
Data Preprocessor for ExoPlanet AI
NASA Space Apps Challenge 2025 - Team BrainRot

This module handles preprocessing of light curve data from various NASA missions
including Kepler, TESS, and K2 datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings

# Astronomical data handling
try:
    from astropy.io import fits
    from astropy.timeseries import TimeSeries
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    warnings.warn("Astropy not available. FITS file support will be limited.")

class DataPreprocessor:
    """
    Preprocessor for light curve data from NASA exoplanet missions.
    
    Handles:
    - CSV files with time/flux columns
    - FITS files from Kepler/TESS
    - Data cleaning and normalization
    - Outlier detection and removal
    - Detrending
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['csv', 'fits', 'dat', 'txt']
        
        # Default preprocessing parameters
        self.default_params = {
            'remove_outliers': True,
            'outlier_sigma': 3.0,
            'detrend': True,
            'normalize': True,
            'fill_gaps': True,
            'min_data_points': 100
        }
    
    def process_file(self, file_path: str, 
                    params: Optional[Dict] = None) -> Dict:
        """
        Process a light curve file and return cleaned data.
        
        Args:
            file_path: Path to the light curve file
            params: Processing parameters (optional)
            
        Returns:
            Dictionary containing processed data and metadata
        """
        try:
            # Use default params if none provided
            processing_params = {**self.default_params, **(params or {})}
            
            # Determine file type and load data
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.fits':
                time, flux, metadata = self._load_fits_file(file_path)
            elif file_extension in ['.csv', '.dat', '.txt']:
                time, flux, metadata = self._load_text_file(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Validate input data
            self._validate_data(time, flux)
            
            # Apply preprocessing steps
            processed_time, processed_flux = self._preprocess_lightcurve(
                time, flux, processing_params
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                time, flux, processed_time, processed_flux
            )
            
            return {
                'time': processed_time,
                'flux': processed_flux,
                'original_length': len(time),
                'processed_length': len(processed_time),
                'metadata': metadata,
                'quality_metrics': quality_metrics,
                'processing_params': processing_params,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"File processing failed: {str(e)}")
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def _load_fits_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load data from FITS file (Kepler/TESS format)."""
        if not ASTROPY_AVAILABLE:
            raise ImportError("Astropy required for FITS file support")
        
        try:
            with fits.open(file_path) as hdul:
                # Try to find the light curve data
                # Different missions have different HDU structures
                
                if len(hdul) > 1:
                    data = hdul[1].data  # Usually in first extension
                else:
                    data = hdul[0].data
                
                # Common column names for different missions
                time_cols = ['TIME', 'BTJD', 'TBJD', 'JD', 'MJD']
                flux_cols = ['FLUX', 'SAP_FLUX', 'PDCSAP_FLUX', 'LC_FLUX']
                
                time = None
                flux = None
                
                # Find time column
                for col in time_cols:
                    if col in data.columns.names:
                        time = data[col]
                        break
                
                # Find flux column
                for col in flux_cols:
                    if col in data.columns.names:
                        flux = data[col]
                        break
                
                if time is None or flux is None:
                    raise ValueError("Could not find time/flux columns in FITS file")
                
                # Extract metadata from header
                metadata = {
                    'mission': hdul[0].header.get('MISSION', 'Unknown'),
                    'object': hdul[0].header.get('OBJECT', 'Unknown'),
                    'ra': hdul[0].header.get('RA_OBJ', None),
                    'dec': hdul[0].header.get('DEC_OBJ', None),
                    'magnitude': hdul[0].header.get('KEPMAG', hdul[0].header.get('TESSMAG', None)),
                    'file_type': 'FITS'
                }
                
                return np.array(time), np.array(flux), metadata
                
        except Exception as e:
            raise ValueError(f"Failed to load FITS file: {str(e)}")
    
    def _load_text_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load data from text file (CSV, DAT, TXT)."""
        try:
            # Try different separators
            separators = [',', '\t', ' ', ';']
            
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, sep=sep, comment='#')
                    if len(df.columns) >= 2:
                        break
                except:
                    continue
            else:
                # If all separators fail, try automatic detection
                df = pd.read_csv(file_path, sep=None, engine='python', comment='#')
            
            # Identify time and flux columns
            time_col, flux_col = self._identify_columns(df)
            
            time = df[time_col].values
            flux = df[flux_col].values
            
            metadata = {
                'columns': list(df.columns),
                'time_column': time_col,
                'flux_column': flux_col,
                'file_type': 'TEXT'
            }
            
            return time, flux, metadata
            
        except Exception as e:
            raise ValueError(f"Failed to load text file: {str(e)}")
    
    def _identify_columns(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Identify time and flux columns in a DataFrame."""
        columns = df.columns.tolist()
        
        # Common time column names
        time_keywords = ['time', 't', 'bjd', 'jd', 'mjd', 'date']
        flux_keywords = ['flux', 'f', 'magnitude', 'mag', 'brightness']
        
        time_col = None
        flux_col = None
        
        # Find time column
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in time_keywords):
                time_col = col
                break
        
        # Find flux column
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in flux_keywords):
                flux_col = col
                break
        
        # If not found by name, use first two numeric columns
        if time_col is None or flux_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                time_col = numeric_cols[0]
                flux_col = numeric_cols[1]
            else:
                raise ValueError("Could not identify time and flux columns")
        
        return time_col, flux_col
    
    def _validate_data(self, time: np.ndarray, flux: np.ndarray):
        """Validate input time series data."""
        if len(time) != len(flux):
            raise ValueError("Time and flux arrays must have same length")
        
        if len(time) < self.default_params['min_data_points']:
            raise ValueError(f"Insufficient data points: {len(time)} < {self.default_params['min_data_points']}")
        
        if np.any(np.isnan(time)) or np.any(np.isnan(flux)):
            self.logger.warning("Data contains NaN values")
        
        if not np.all(np.diff(time) > 0):
            self.logger.warning("Time series is not monotonically increasing")
    
    def _preprocess_lightcurve(self, time: np.ndarray, flux: np.ndarray, 
                              params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Apply preprocessing steps to light curve data."""
        
        # Remove NaN values
        mask = ~(np.isnan(time) | np.isnan(flux))
        time_clean = time[mask]
        flux_clean = flux[mask]
        
        # Remove outliers
        if params['remove_outliers']:
            time_clean, flux_clean = self._remove_outliers(
                time_clean, flux_clean, params['outlier_sigma']
            )
        
        # Fill gaps if requested
        if params['fill_gaps']:
            time_clean, flux_clean = self._fill_gaps(time_clean, flux_clean)
        
        # Detrend the data
        if params['detrend']:
            flux_clean = self._detrend(time_clean, flux_clean)
        
        # Normalize the flux
        if params['normalize']:
            flux_clean = self._normalize_flux(flux_clean)
        
        return time_clean, flux_clean
    
    def _remove_outliers(self, time: np.ndarray, flux: np.ndarray, 
                        sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using sigma clipping."""
        median_flux = np.median(flux)
        mad = np.median(np.abs(flux - median_flux))
        
        # Use MAD-based outlier detection (more robust than std)
        threshold = sigma * 1.4826 * mad  # 1.4826 converts MAD to std equivalent
        
        mask = np.abs(flux - median_flux) < threshold
        
        self.logger.info(f"Removed {np.sum(~mask)} outliers ({np.sum(~mask)/len(flux)*100:.1f}%)")
        
        return time[mask], flux[mask]
    
    def _fill_gaps(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fill small gaps in time series data."""
        # Simple linear interpolation for small gaps
        # In production, you might want more sophisticated gap filling
        
        # Find gaps larger than median cadence
        dt = np.diff(time)
        median_dt = np.median(dt)
        large_gaps = dt > 3 * median_dt
        
        if np.any(large_gaps):
            self.logger.info(f"Found {np.sum(large_gaps)} large gaps in data")
        
        # For now, just return original data
        # TODO: Implement sophisticated gap filling
        return time, flux
    
    def _detrend(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """Remove long-term trends from flux data."""
        # Simple polynomial detrending
        # In production, you might use more sophisticated methods
        
        try:
            # Fit a low-order polynomial
            coeffs = np.polyfit(time, flux, deg=2)
            trend = np.polyval(coeffs, time)
            
            # Remove trend
            detrended = flux - trend + np.median(flux)
            
            self.logger.info("Applied polynomial detrending")
            return detrended
            
        except Exception as e:
            self.logger.warning(f"Detrending failed: {str(e)}, returning original flux")
            return flux
    
    def _normalize_flux(self, flux: np.ndarray) -> np.ndarray:
        """Normalize flux to have zero mean and unit variance."""
        normalized = (flux - np.mean(flux)) / np.std(flux)
        return normalized
    
    def _calculate_quality_metrics(self, orig_time: np.ndarray, orig_flux: np.ndarray,
                                  proc_time: np.ndarray, proc_flux: np.ndarray) -> Dict:
        """Calculate data quality metrics."""
        
        # Data completeness
        completeness = len(proc_time) / len(orig_time)
        
        # Noise level (standard deviation of processed flux)
        noise_level = np.std(proc_flux)
        
        # Time coverage
        time_span = np.max(proc_time) - np.min(proc_time)
        
        # Cadence statistics
        cadence = np.median(np.diff(proc_time))
        cadence_std = np.std(np.diff(proc_time))
        
        return {
            'completeness': float(completeness),
            'noise_level': float(noise_level),
            'time_span_days': float(time_span),
            'median_cadence': float(cadence),
            'cadence_stability': float(cadence_std / cadence) if cadence > 0 else 0,
            'snr_estimate': float(1.0 / noise_level) if noise_level > 0 else 0
        }
