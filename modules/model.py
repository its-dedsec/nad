"""
Module for anomaly detection modeling.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    A class for detecting network traffic anomalies using machine learning.
    """
    
    def __init__(self, model_type='isolation_forest', contamination=0.05):
        """
        Initialize the anomaly detector.
        
        Args:
            model_type (str): Type of model to use ('isolation_forest' or 'one_class_svm')
            contamination (float): Expected proportion of outliers in the data
        """
        self.model_type = model_type
        self.contamination = contamination
        self.model = None
        
        if model_type == 'isolation_forest':
            self.model = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'one_class_svm':
            self.model = OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma='scale'
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        logger.info(f"Initialized anomaly detector with {model_type}")

    def train(self, features):
        """
        Train the anomaly detection model.
        
        Args:
            features (pd.DataFrame): Feature DataFrame for training
            
        Returns:
            self: The trained model instance
        """
        try:
            logger.info(f"Training {self.model_type} model on {len(features)} samples")
            self.model.fit(features)
            logger.info("Model training completed")
            return self
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, features):
        """
        Predict anomalies in the data.
        
        Args:
            features (pd.DataFrame): Feature DataFrame for prediction
            
        Returns:
            np.ndarray: Anomaly scores (-1 for anomalies, 1 for normal points)
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            logger.info(f"Predicting anomalies on {len(features)} samples")
            predictions = self.model.predict(features)
            
            # Convert to binary labels
            return predictions
        
        except Exception as e:
            logger.error(f"Error predicting anomalies: {str(e)}")
            raise

    def decision_function(self, features):
        """
        Get anomaly scores for the data.
        
        Args:
            features (pd.DataFrame): Feature DataFrame for scoring
            
        Returns:
            np.ndarray: Anomaly scores (lower scores indicate more anomalous points)
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            logger.info(f"Computing anomaly scores for {len(features)} samples")
            scores = self.model.decision_function(features)
            
            # Normalize scores to [0, 1] range where 1 is most anomalous
            if self.model_type == 'isolation_forest':
                # For Isolation Forest, scores are already normalized, but in reverse
                # (higher values are normal, lower values are anomalous)
                scores = 1 - (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
            else:
                # For OneClassSVM, lower values are more anomalous
                scores = 1 - (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
                
            return scores
        
        except Exception as e:
            logger.error(f"Error computing anomaly scores: {str(e)}")
            raise

    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath}")
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            self.model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
