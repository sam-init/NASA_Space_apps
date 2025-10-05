"""
Advanced Exoplanet Detector
NASA Space Apps Challenge 2025 - Team BrainRot

Ensemble machine learning model for exoplanet detection with 99%+ accuracy target.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os

# ML imports
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExoplanetDetector:
    """
    Advanced ensemble exoplanet detection model.
    
    Features:
    - Ensemble of Random Forest, Gradient Boosting, and XGBoost
    - SMOTE for class imbalance handling
    - Robust scaling for outlier resistance
    - Hyperparameter optimization
    - Target: 99%+ accuracy
    """
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = model_dir
        self.model_file = os.path.join(model_dir, "exoplanet_ensemble_model.pkl")
        self.scaler_file = os.path.join(model_dir, "feature_scaler.pkl")
        self.metadata_file = os.path.join(model_dir, "model_metadata.pkl")
        
        # Model components
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        # Model metadata
        self.model_info = {
            'name': 'ExoplanetEnsemble',
            'version': '2.0.0',
            'algorithm': 'Ensemble (RandomForest + GradientBoosting + XGBoost)',
            'target_accuracy': 0.99,
            'created': datetime.now().isoformat()
        }
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load existing trained model."""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
                self.model = joblib.load(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
                
                if os.path.exists(self.metadata_file):
                    self.model_info.update(joblib.load(self.metadata_file))
                
                self.is_trained = True
                logger.info("Pre-trained model loaded successfully")
                return True
        except Exception as e:
            logger.warning(f"Failed to load existing model: {e}")
        
        return False
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions on new data.
        
        Args:
            features: Input features DataFrame
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if not self.is_trained:
                return {
                    'error': 'Model not trained',
                    'prediction': False,
                    'confidence': 0.0
                }
            
            start_time = datetime.now()
            
            # Scale features
            X_scaled = self.scaler.transform(features)
            
            # Make predictions
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = float(probabilities[1] * 100)  # Confidence as percentage
            
            # Feature importance (if available)
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(
                    features.columns,
                    self.model.feature_importances_
                ))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'prediction': bool(prediction),
                'confidence': confidence,
                'probabilities': probabilities.tolist(),
                'processing_time': processing_time,
                'feature_importance': feature_importance,
                'model_info': {
                    'name': self.model_info['name'],
                    'version': self.model_info['version'],
                    'algorithm': self.model_info['algorithm']
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "error": str(e),
                "prediction": False,
                "confidence": 0.0
            }
    
    def train_model(self, features: pd.DataFrame, target: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42,
                   use_smote: bool = True, optimize_hyperparams: bool = False) -> Dict[str, Any]:
        """
        Train the ensemble exoplanet detection model.
        
        Args:
            features: Training features DataFrame
            target: Target labels Series
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            use_smote: Whether to use SMOTE for class balancing
            optimize_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Dict with training results and metrics
        """
        try:
            logger.info("Starting exoplanet detection model training...")
            start_time = datetime.now()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_size, random_state=random_state, stratify=target
            )
            
            logger.info(f"Training set: {X_train.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
            
            # Initialize preprocessing pipeline
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Handle class imbalance with SMOTE
            if use_smote:
                logger.info("Applying SMOTE for class balancing...")
                smote = SMOTE(random_state=random_state)
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                logger.info(f"After SMOTE: {X_train_scaled.shape[0]} samples")
            
            # Define base models
            base_models = [
                ('rf', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=random_state,
                    n_jobs=-1
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=random_state
                ))
            ]
            
            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                base_models.append(('xgb', xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=random_state,
                    eval_metric='logloss'
                )))
            
            # Create ensemble model
            self.model = VotingClassifier(
                estimators=base_models,
                voting='soft'
            )
            
            # Train the ensemble
            logger.info("Training ensemble model...")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
            test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, train_pred),
                'test_accuracy': accuracy_score(y_test, test_pred),
                'train_precision': precision_score(y_train, train_pred),
                'test_precision': precision_score(y_test, test_pred),
                'train_recall': recall_score(y_train, train_pred),
                'test_recall': recall_score(y_test, test_pred),
                'train_f1': f1_score(y_train, train_pred),
                'test_f1': f1_score(y_test, test_pred),
                'train_auc': roc_auc_score(y_train, train_proba),
                'test_auc': roc_auc_score(y_test, test_proba)
            }
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Update model info
            self.model_info.update({
                'training_samples': len(features),
                'features_used': list(features.columns),
                'metrics': metrics,
                'cv_scores': cv_scores.tolist(),
                'training_time': training_time,
                'last_trained': datetime.now().isoformat()
            })
            
            # Save model and components
            self._save_model()
            
            self.is_trained = True
            
            result = {
                'success': True,
                'metrics': metrics,
                'cv_scores': {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist()
                },
                'training_time': training_time,
                'model_info': self.model_info
            }
            
            logger.info(f"Training completed in {training_time:.2f}s")
            logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _save_model(self) -> None:
        """Save the trained model and associated components."""
        try:
            joblib.dump(self.model, self.model_file)
            joblib.dump(self.scaler, self.scaler_file)
            joblib.dump(self.model_info, self.metadata_file)
            logger.info(f"Model saved to {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model."""
        return {
            **self.model_info,
            "is_trained": self.is_trained,
            "model_files": {
                "model_exists": os.path.exists(self.model_file),
                "scaler_exists": os.path.exists(self.scaler_file),
                "metadata_exists": os.path.exists(self.metadata_file)
            },
            "timestamp": datetime.now().isoformat()
        }