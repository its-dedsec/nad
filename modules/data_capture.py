"""
Module for capturing network traffic or loading existing pcap files.
"""
import os
import pyshark
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pcap(file_path):
    """
    Load packets from a pcap file.
    
    Args:
        file_path (str): Path to the pcap file
        
    Returns:
        list: List of packet objects
    """
    try:
        logger.info(f"Loading pcap file from {file_path}")
        cap = pyshark.FileCapture(file_path)
        packets = [packet for packet in cap]
        logger.info(f"Successfully loaded {len(packets)} packets")
        return packets
    except Exception as e:
        logger.error(f"Error loading pcap file: {str(e)}")
        raise

def extract_packet_info(packets):
    """
    Extract basic information from packets.
    
    Args:
        packets (list): List of packet objects
        
    Returns:
        pd.DataFrame: DataFrame containing packet information
    """
    packet_data = []
    
    try:
        for i, packet in enumerate(packets):
            try:
                # Basic packet info
                packet_info = {
                    'packet_id': i,
                    'timestamp': float(packet.sniff_timestamp),
                    'length': int(packet.length),
                    'protocol': packet.transport_layer if hasattr(packet, 'transport_layer') else 'Unknown'
                }
                
                # IP layer info
                if hasattr(packet, 'ip'):
                    packet_info.update({
                        'src_ip': packet.ip.src,
                        'dst_ip': packet.ip.dst,
                        'ttl': int(packet.ip.ttl)
                    })
                else:
                    packet_info.update({
                        'src_ip': 'Unknown',
                        'dst_ip': 'Unknown',
                        'ttl': 0
                    })
                
                # TCP/UDP layer info
                if hasattr(packet, 'tcp'):
                    packet_info.update({
                        'src_port': int(packet.tcp.srcport),
                        'dst_port': int(packet.tcp.dstport),
                        'tcp_flags': packet.tcp.flags if hasattr(packet.tcp, 'flags') else 'Unknown'
                    })
                elif hasattr(packet, 'udp'):
                    packet_info.update({
                        'src_port': int(packet.udp.srcport),
                        'dst_port': int(packet.udp.dstport),
                        'tcp_flags': 'N/A'  # Not applicable for UDP
                    })
                else:
                    packet_info.update({
                        'src_port': 0,
                        'dst_port': 0,
                        'tcp_flags': 'Unknown'
                    })
                
                packet_data.append(packet_info)
            except Exception as e:
                logger.warning(f"Error processing packet {i}: {str(e)}")
                continue
                
        df = pd.DataFrame(packet_data)
        logger.info(f"Successfully extracted information from {len(df)} packets")
        return df
    
    except Exception as e:
        logger.error(f"Error in extract_packet_info: {str(e)}")
        raise
