"""
NASA Kepler KOI Dataset Loader
Loads and preprocesses the NASA Kepler Objects of Interest (KOI) dataset
Dataset: 9,564 samples with 49 features for exoplanet detection
"""

import pandas as pd
import numpy as np
import os
import requests
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import joblib

logger = logging.getLogger(__name__)

class KOIDatasetLoader:
    """
    Loads and manages the NASA Kepler KOI dataset for exoplanet detection.
    
    The KOI dataset contains 9,564 Kepler Objects of Interest with 49 features
    including stellar parameters, planetary parameters, and detection metrics.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.dataset_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+koi&format=csv"
        self.dataset_file = os.path.join(data_dir, "koi_dataset.csv")
        self.processed_file = os.path.join(data_dir, "koi_processed.pkl")
        
        # Dataset metadata
        self.expected_samples = 9564
        self.expected_features = 49
        self.target_column = 'koi_disposition'
        
        # Feature categories for better understanding
        self.stellar_features = [
            'koi_kepmag', 'koi_teff', 'koi_logg', 'koi_radius', 'koi_mass',
            'ra', 'dec', 'koi_gmag', 'koi_rmag', 'koi_imag', 'koi_zmag'
        ]
        
        self.planetary_features = [
            'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
            'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_dor'
        ]
        
        self.detection_features = [
            'koi_ror', 'koi_srho', 'koi_fittype', 'koi_disp_prov',
            'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
            'koi_fpflag_ec'
        ]
        
        os.makedirs(data_dir, exist_ok=True)
    
    def download_dataset(self, force_download: bool = False) -> bool:
        """
        Download the KOI dataset from NASA Exoplanet Archive.
        
        Args:
            force_download: Force re-download even if file exists
            
        Returns:
            bool: Success status
        """
        try:
            if os.path.exists(self.dataset_file) and not force_download:
                logger.info(f"Dataset already exists at {self.dataset_file}")
                return True
            
            logger.info("Downloading NASA Kepler KOI dataset...")
            
            response = requests.get(self.dataset_url, timeout=300)
            response.raise_for_status()
            
            with open(self.dataset_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logger.info(f"Dataset downloaded successfully to {self.dataset_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return False
    
    def load_raw_dataset(self) -> Optional[pd.DataFrame]:
        """
        Load the raw KOI dataset from CSV file.
        
        Returns:
            pd.DataFrame or None: Raw dataset
        """
        try:
            if not os.path.exists(self.dataset_file):
                logger.warning("Dataset file not found, attempting download...")
                if not self.download_dataset():
                    return None
            
            df = pd.read_csv(self.dataset_file, comment='#', low_memory=False)
            logger.info(f"Loaded raw dataset: {df.shape[0]} samples, {df.shape[1]} features")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load raw dataset: {e}")
            return None
    
    def preprocess_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the KOI dataset for machine learning.
        
        Args:
            df: Raw dataset DataFrame
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        try:
            logger.info("Preprocessing KOI dataset...")
            
            # Create a copy to avoid modifying original
            data = df.copy()
            
            # Handle target variable - convert disposition to binary
            # CONFIRMED = 1, FALSE POSITIVE = 0, CANDIDATE = 1 (conservative approach)
            target_mapping = {
                'CONFIRMED': 1,
                'FALSE POSITIVE': 0,
                'CANDIDATE': 1  # Treat candidates as potential exoplanets
            }
            
            if self.target_column not in data.columns:
                logger.error(f"Target column '{self.target_column}' not found")
                return None, None
            
            # Filter to only include rows with known dispositions
            data = data[data[self.target_column].isin(target_mapping.keys())].copy()
            target = data[self.target_column].map(target_mapping)
            
            # Select relevant features for exoplanet detection
            feature_columns = []
            
            # Add available stellar features
            for col in self.stellar_features:
                if col in data.columns:
                    feature_columns.append(col)
            
            # Add available planetary features
            for col in self.planetary_features:
                if col in data.columns:
                    feature_columns.append(col)
            
            # Add available detection features
            for col in self.detection_features:
                if col in data.columns:
                    feature_columns.append(col)
            
            # Select features
            features = data[feature_columns].copy()
            
            # Handle missing values
            # For numerical columns, use median imputation
            numerical_cols = features.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                median_val = features[col].median()
                features[col].fillna(median_val, inplace=True)
            
            # For categorical columns, use mode imputation
            categorical_cols = features.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                mode_val = features[col].mode().iloc[0] if not features[col].mode().empty else 'unknown'
                features[col].fillna(mode_val, inplace=True)
            
            # Convert categorical variables to numerical
            for col in categorical_cols:
                if features[col].dtype == 'object':
                    features[col] = pd.Categorical(features[col]).codes
            
            # Remove any remaining NaN values
            valid_indices = ~(features.isna().any(axis=1) | target.isna())
            features = features[valid_indices]
            target = target[valid_indices]
            
            logger.info(f"Preprocessed dataset: {features.shape[0]} samples, {features.shape[1]} features")
            logger.info(f"Target distribution: {target.value_counts().to_dict()}")
            
            return features, target
            
        except Exception as e:
            logger.error(f"Failed to preprocess dataset: {e}")
            return None, None
    
    def load_dataset(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load and preprocess the complete KOI dataset.
        
        Args:
            force_reload: Force reload from source
            
        Returns:
            Dict with dataset information and status
        """
        try:
            # Check if processed dataset exists
            if os.path.exists(self.processed_file) and not force_reload:
                logger.info("Loading preprocessed dataset from cache...")
                data = joblib.load(self.processed_file)
                return {
                    'success': True,
                    'source': 'cache',
                    'samples': data['features'].shape[0],
                    'features': data['features'].shape[1],
                    'target_distribution': data['target'].value_counts().to_dict(),
                    'timestamp': data.get('timestamp', 'unknown')
                }
            
            # Load and preprocess from raw data
            raw_df = self.load_raw_dataset()
            if raw_df is None:
                return {'success': False, 'error': 'Failed to load raw dataset'}
            
            features, target = self.preprocess_dataset(raw_df)
            if features is None or target is None:
                return {'success': False, 'error': 'Failed to preprocess dataset'}
            
            # Save processed dataset
            processed_data = {
                'features': features,
                'target': target,
                'feature_names': list(features.columns),
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(processed_data, self.processed_file)
            logger.info(f"Saved processed dataset to {self.processed_file}")
            
            return {
                'success': True,
                'source': 'processed',
                'samples': features.shape[0],
                'features': features.shape[1],
                'target_distribution': target.value_counts().to_dict(),
                'feature_names': list(features.columns),
                'timestamp': processed_data['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the KOI dataset.
        
        Returns:
            Dict with dataset metadata
        """
        return {
            'name': 'NASA Kepler Objects of Interest (KOI)',
            'description': 'Kepler mission exoplanet candidates and confirmed planets',
            'source': 'NASA Exoplanet Archive',
            'url': 'https://exoplanetarchive.ipac.caltech.edu/',
            'expected_samples': self.expected_samples,
            'expected_features': self.expected_features,
            'target_column': self.target_column,
            'feature_categories': {
                'stellar': self.stellar_features,
                'planetary': self.planetary_features,
                'detection': self.detection_features
            },
            'files': {
                'raw_dataset': self.dataset_file,
                'processed_dataset': self.processed_file,
                'raw_exists': os.path.exists(self.dataset_file),
                'processed_exists': os.path.exists(self.processed_file)
            }
        }
    
    def get_sample_data(self, n_samples: int = 5) -> Dict[str, Any]:
        """
        Get sample data for inspection.
        
        Args:
            n_samples: Number of samples to return
            
        Returns:
            Dict with sample data
        """
        try:
            data = joblib.load(self.processed_file)
            features = data['features']
            target = data['target']
            
            sample_indices = np.random.choice(len(features), min(n_samples, len(features)), replace=False)
            
            return {
                'success': True,
                'samples': {
                    'features': features.iloc[sample_indices].to_dict('records'),
                    'target': target.iloc[sample_indices].tolist(),
                    'feature_names': list(features.columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get sample data: {e}")
            return {'success': False, 'error': str(e)}
