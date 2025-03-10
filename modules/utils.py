"""
Utility functions for the network anomaly detection system.
"""
import pandas as pd
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_to_csv(df, output_path):
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path to save the CSV file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving data to CSV: {str(e)}")
        raise

def load_from_csv(input_path):
    """
    Load DataFrame from CSV file.
    
    Args:
        input_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Data loaded from {input_path}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from CSV: {str(e)}")
        raise

def generate_output_filename(prefix, extension):
    """
    Generate a timestamped filename.
    
    Args:
        prefix (str): Prefix for the filename
        extension (str): File extension
        
    Returns:
        str: Generated filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def filter_columns_for_display(df, max_columns=20):
    """
    Filter DataFrame columns for display purposes.
    
    Args:
        df (pd.DataFrame): DataFrame to filter
        max_columns (int): Maximum number of columns to include
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    # Priority columns to always include
    priority_cols = [
        'packet_id', 'timestamp', 'protocol', 'src_ip', 'dst_ip', 
        'src_port', 'dst_port', 'length', 'is_anomaly', 'anomaly_score', 
        'interpretation'
    ]
    
    # Filter columns
    available_cols = [col for col in priority_cols if col in df.columns]
    
    # If we have room for more columns, add them
    other_cols = [col for col in df.columns if col not in priority_cols]
    remaining_slots = max_columns - len(available_cols)
    
    if remaining_slots > 0 and other_cols:
        available_cols.extend(other_cols[:remaining_slots])
    
    return df[available_cols]

def create_temp_directory():
    """
    Create a temporary directory for storing files.
    
    Returns:
        str: Path to the temporary directory
    """
    temp_dir = os.path.join(os.getcwd(), 'temp_files')
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir
