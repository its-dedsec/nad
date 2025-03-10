"""
Module for preprocessing network traffic data for anomaly detection.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Preprocess packet data for anomaly detection.
    
    Args:
        df (pd.DataFrame): DataFrame containing packet information
        
    Returns:
        tuple: (preprocessed_df, feature_df, scaler)
            - preprocessed_df: DataFrame with all features including engineered ones
            - feature_df: DataFrame with only numeric features for model training
            - scaler: StandardScaler fitted to the data
    """
    try:
        logger.info("Starting data preprocessing")
        
        # Make a copy to avoid modifying the original
        preprocessed_df = df.copy()
        
        # Handle missing values
        preprocessed_df = handle_missing_values(preprocessed_df)
        
        # Add engineered features
        preprocessed_df = add_engineered_features(preprocessed_df)
        
        # Encode categorical features
        preprocessed_df = encode_categorical_features(preprocessed_df)
        
        # Select and scale features for the model
        feature_df, scaler = scale_features(preprocessed_df)
        
        logger.info("Data preprocessing completed successfully")
        return preprocessed_df, feature_df, scaler
    
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    # Fill missing numeric values with 0
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(0)
    
    # Fill missing categorical values with 'Unknown'
    for col in df.select_dtypes(exclude=np.number).columns:
        df[col] = df[col].fillna('Unknown')
    
    return df

def add_engineered_features(df):
    """
    Add engineered features to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with additional engineered features
    """
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Extract time-based features
    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Create communication pairs for pattern analysis
    df['comm_pair'] = df['src_ip'] + ':' + df['src_port'].astype(str) + '-' + df['dst_ip'] + ':' + df['dst_port'].astype(str)
    
    # Flag for well-known ports (below 1024)
    df['is_well_known_port_src'] = (df['src_port'] < 1024).astype(int)
    df['is_well_known_port_dst'] = (df['dst_port'] < 1024).astype(int)
    
    # Flag common service ports
    common_ports = {80, 443, 22, 21, 25, 53, 3389, 3306, 5432}
    df['is_common_service_port'] = ((df['src_port'].isin(common_ports)) | (df['dst_port'].isin(common_ports))).astype(int)
    
    # Calculate packet size categories
    df['packet_size_category'] = pd.cut(
        df['length'], 
        bins=[0, 64, 256, 1024, 1500, float('inf')],
        labels=['tiny', 'small', 'medium', 'large', 'jumbo']
    )
    
    return df

def encode_categorical_features(df):
    """
    Encode categorical features for machine learning.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with encoded categorical features
    """
    # One-hot encode protocol
    df = pd.get_dummies(df, columns=['protocol'], prefix='protocol')
    
    # One-hot encode packet size category
    df = pd.get_dummies(df, columns=['packet_size_category'], prefix='size')
    
    # For TCP flags, we'll create boolean features for common flags
    if 'tcp_flags' in df.columns:
        # Check for common TCP flags (simplified version)
        df['has_syn'] = df['tcp_flags'].str.contains('S', na=False).astype(int)
        df['has_ack'] = df['tcp_flags'].str.contains('A', na=False).astype(int)
        df['has_fin'] = df['tcp_flags'].str.contains('F', na=False).astype(int)
        df['has_rst'] = df['tcp_flags'].str.contains('R', na=False).astype(int)
        df['has_psh'] = df['tcp_flags'].str.contains('P', na=False).astype(int)
    
    return df

def scale_features(df):
    """
    Select and scale numeric features for the model.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        tuple: (feature_df, scaler)
            - feature_df: DataFrame with selected and scaled features
            - scaler: StandardScaler fitted to the data
    """
    # Select numeric columns (excluding timestamp, packet_id, and datetime)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['timestamp', 'packet_id']]
    
    # Additional columns we don't want to use for modeling
    exclude_cols = ['datetime', 'src_ip', 'dst_ip', 'comm_pair', 'tcp_flags']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[feature_cols])
    
    # Create a new DataFrame with scaled features
    feature_df = pd.DataFrame(features_scaled, columns=feature_cols)
    
    return feature_df, scaler
