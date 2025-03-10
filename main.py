"""
Main module for the network anomaly detection system.
"""
import argparse
import os
import logging
from pathlib import Path

from modules.data_capture import load_pcap, extract_packet_info
from modules.data_preprocessing import preprocess_data
from modules.model import AnomalyDetector
from modules.analysis import analyze_anomalies, get_summary_stats
from modules.utils import save_to_csv, generate_output_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('network_anomaly_detection.log')
    ]
)
logger = logging.getLogger(__name__)

def process_pcap_file(pcap_file, output_dir=None, model_type='isolation_forest', contamination=0.05, threshold=0.8):
    """
    Process a pcap file for anomaly detection.
    
    Args:
        pcap_file (str): Path to the pcap file
        output_dir (str): Directory to save output files
        model_type (str): Type of anomaly detection model
        contamination (float): Expected proportion of anomalies
        threshold (float): Threshold for anomaly detection
        
    Returns:
        dict: Dictionary with results and file paths
    """
    try:
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Processing pcap file: {pcap_file}")
        
        # Load and extract packet information
        packets = load_pcap(pcap_file)
        raw_df = extract_packet_info(packets)
        
        # Save raw data
        raw_output_path = os.path.join(output_dir, generate_output_filename('raw_data', 'csv'))
        save_to_csv(raw_df, raw_output_path)
        
        # Preprocess data
        preprocessed_df, feature_df, scaler = preprocess_data(raw_df)
        
        # Save preprocessed data
        preprocessed_output_path = os.path.join(output_dir, generate_output_filename('preprocessed_data', 'csv'))
        save_to_csv(preprocessed_df, preprocessed_output_path)
        
        # Train anomaly detection model
        model = AnomalyDetector(model_type=model_type, contamination=contamination)
        model.train(feature_df)
        
        # Save the model
        model_path = os.path.join(output_dir, generate_output_filename('model', 'joblib'))
        model.save_model(model_path)
        
        # Predict anomalies
        anomaly_scores = model.decision_function(feature_df)
        
        # Analyze anomalies
        result_df = analyze_anomalies(preprocessed_df, anomaly_scores, threshold=threshold)
        
        # Save results
        result_output_path = os.path.join(output_dir, generate_output_filename('anomaly_results', 'csv'))
        save_to_csv(result_df, result_output_path)
        
        # Generate summary statistics
        summary = get_summary_stats(result_df)
        
        # Return results and file paths
        return {
            'raw_data_path': raw_output_path,
