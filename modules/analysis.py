"""
Module for analyzing anomalies in network traffic.
"""
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_anomalies(df, anomaly_scores, threshold=0.8):
    """
    Analyze detected anomalies and provide interpretations.
    
    Args:
        df (pd.DataFrame): Original DataFrame with packet information
        anomaly_scores (np.ndarray): Anomaly scores from the model
        threshold (float): Threshold for considering a point anomalous
        
    Returns:
        pd.DataFrame: DataFrame with anomaly information and interpretations
    """
    try:
        logger.info("Analyzing anomalies")
        
        # Add anomaly scores to the DataFrame
        result_df = df.copy()
        result_df['anomaly_score'] = anomaly_scores
        
        # Add binary label (1 for anomalies, 0 for normal)
        result_df['is_anomaly'] = (anomaly_scores > threshold).astype(int)
        
        # Add interpretations for anomalies
        result_df['interpretation'] = result_df.apply(
            lambda row: interpret_anomaly(row, df) if row['is_anomaly'] else "Normal traffic",
            axis=1
        )
        
        # Calculate percentage of anomalies
        anomaly_percentage = (result_df['is_anomaly'].sum() / len(result_df)) * 100
        logger.info(f"Detected {result_df['is_anomaly'].sum()} anomalies ({anomaly_percentage:.2f}%)")
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error analyzing anomalies: {str(e)}")
        raise

def interpret_anomaly(row, df):
    """
    Generate an interpretation for an anomalous packet.
    
    Args:
        row (pd.Series): Row containing packet information
        df (pd.DataFrame): Full DataFrame for context
        
    Returns:
        str: Interpretation of the anomaly
    """
    interpretations = []
    
    # Check for unusual ports
    if (row['is_well_known_port_src'] == 0 and row['src_port'] > 10000) and \
       (row['is_well_known_port_dst'] == 0 and row['dst_port'] > 10000):
        interpretations.append("Unusual port activity (high source and destination ports)")
    
    # Check for unusually large or small packets
    if 'length' in row and row['length'] > 1500:
        interpretations.append("Unusually large packet size")
    elif 'length' in row and row['length'] < 20:
        interpretations.append("Unusually small packet size")
    
    # Check for potential port scan
    if 'src_ip' in row and 'src_port' in row:
        # Count unique destination ports for this source IP
        src_ip = row['src_ip']
        src_df = df[df['src_ip'] == src_ip]
        if len(src_df) > 5:  # Only check if we have enough data
            unique_dst_ports = src_df['dst_port'].nunique()
            unique_dst_ips = src_df['dst_ip'].nunique()
            
            if unique_dst_ports > 5 and unique_dst_ports / len(src_df) > 0.8:
                interpretations.append("Potential port scanning behavior")
                
            if unique_dst_ips > 5 and unique_dst_ips / len(src_df) > 0.8:
                interpretations.append("Accessing multiple destinations rapidly")
    
    # Check for TCP flag anomalies
    if 'has_syn' in row and 'has_ack' in row:
        if row['has_syn'] == 1 and row['has_ack'] == 0 and row['has_fin'] == 0 and row['has_rst'] == 0:
            # SYN without ACK, potentially a SYN scan
            interpretations.append("SYN packet without ACK (possible SYN scan)")
        
        if row['has_rst'] == 1 and row['has_ack'] == 1:
            interpretations.append("Connection reset (RST+ACK flags)")
    
    # Check for unusual communication patterns
    if 'comm_pair' in row:
        pair_count = df[df['comm_pair'] == row['comm_pair']].shape[0]
        if pair_count == 1:
            interpretations.append("One-off communication between these endpoints")
    
    # If no specific interpretation found, provide a generic one based on score
    if not interpretations:
        if row['anomaly_score'] > 0.95:
            interpretations.append("Highly unusual traffic pattern")
        elif row['anomaly_score'] > 0.85:
            interpretations.append("Moderately unusual traffic pattern")
        else:
            interpretations.append("Slightly unusual traffic pattern")
    
    return " | ".join(interpretations)

def get_summary_stats(result_df):
    """
    Generate summary statistics for the analyzed traffic.
    
    Args:
        result_df (pd.DataFrame): DataFrame with analysis results
        
    Returns:
        dict: Dictionary with summary statistics
    """
    try:
        total_packets = len(result_df)
        anomalous_packets = result_df['is_anomaly'].sum()
        anomaly_percentage = (anomalous_packets / total_packets) * 100 if total_packets > 0 else 0
        
        # Protocol distribution
        protocol_cols = [col for col in result_df.columns if col.startswith('protocol_')]
        protocol_dist = {}
        for col in protocol_cols:
            protocol_name = col.replace('protocol_', '')
            protocol_dist[protocol_name] = result_df[col].sum()
        
        # Top source IPs with anomalies
        if 'src_ip' in result_df.columns and anomalous_packets > 0:
            top_anomalous_ips = result_df[result_df['is_anomaly'] == 1]['src_ip'].value_counts().head(5).to_dict()
        else:
            top_anomalous_ips = {}
        
        # Most common interpretations
        if anomalous_packets > 0:
            interpretation_counts = result_df[result_df['is_anomaly'] == 1]['interpretation'].value_counts().head(5).to_dict()
        else:
            interpretation_counts = {}
        
        # Packet size distribution
        size_cols = [col for col in result_df.columns if col.startswith('size_')]
        size_dist = {}
        for col in size_cols:
            size_name = col.replace('size_', '')
            size_dist[size_name] = result_df[col].sum()
        
        summary = {
            'total_packets': total_packets,
            'anomalous_packets': anomalous_packets,
            'anomaly_percentage': anomaly_percentage,
            'protocol_distribution': protocol_dist,
            'top_anomalous_source_ips': top_anomalous_ips,
            'common_anomaly_interpretations': interpretation_counts,
            'packet_size_distribution': size_dist
        }
        
        return summary
    
    except Exception as e:
        logger.error(f"Error generating summary statistics: {str(e)}")
        raise
