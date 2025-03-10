"""
Streamlit app for network anomaly detection.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from modules.data_capture import load_pcap, extract_packet_info
from modules.data_preprocessing import preprocess_data
from modules.model import AnomalyDetector
from modules.analysis import analyze_anomalies, get_summary_stats
from modules.utils import filter_columns_for_display, create_temp_directory

# Set page configuration
st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create temp directory
TEMP_DIR = create_temp_directory()

def main():
    # Sidebar
    st.sidebar.title("Network Anomaly Detection")
    st.sidebar.image("https://img.icons8.com/color/96/000000/cyber-security.png", width=100)
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Home", "Upload PCAP", "Analysis Results", "About"])
    
    if page == "Home":
        show_home_page()
    elif page == "Upload PCAP":
        show_upload_page()
    elif page == "Analysis Results":
        show_results_page()
    elif page == "About":
        show_about_page()

def show_home_page():
    st.title("Network Anomaly Detection System")
    st.write("## Welcome to the Network Anomaly Detection Tool")
    
    st.write("""
    This application helps you detect anomalies in network traffic using machine learning techniques.
    
    ### Features:
    - Upload and analyze PCAP files
    - Extract important features from network packets
    - Detect anomalies using machine learning algorithms
    - Visualize network traffic patterns and anomalies
    - Get interpretations of detected anomalies
    
    ### How to use:
    1. Navigate to the 'Upload PCAP' page
    2. Upload a PCAP file
    3. Configure detection parameters
    4. Run the analysis
    5. View the results and visualizations
    """)
    
    st.info("To get started, go to the 'Upload PCAP' page and upload a PCAP file.")

def show_upload_page():
    st.title("Upload PCAP File")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PCAP file", type=["pcap", "pcapng"])
    
    # Model parameters
    with st.expander("Model Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["isolation_forest", "one_class_svm"],
                help="Select the anomaly detection algorithm to use."
            )
            
            contamination = st.slider(
                "Contamination",
                min_value=0.01,
                max_value=0.2,
                value=0.05,
                step=0.01,
                help="Expected proportion of outliers in the data."
            )
        
        with col2:
            threshold = st.slider(
                "Anomaly Threshold",
                min_value=0.5,
                max_value=0.99,
                value=0.8,
                step=0.01,
                help="Threshold for considering a point anomalous."
            )
    
    # Process file when uploaded
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        
        # Save uploaded file to temp directory
        temp_file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state['pcap_path'] = temp_file_path
        st.session_state['file_name'] = uploaded_file.name
        
        # Process button
        if st.button("Process PCAP File"):
            with st.spinner("Processing PCAP file... This may take a while."):
                try:
                    # Load and extract packet information
                    packets = load_pcap(temp_file_path)
                    raw_df = extract_packet_info(packets)
                    st.session_state['raw_df'] = raw_df
                    
                    # Preprocess data
                    preprocessed_df, feature_df, scaler = preprocess_data(raw_df)
                    st.session_state['preprocessed_df'] = preprocessed_df
                    st.session_state['feature_df'] = feature_df
                    
                    # Train anomaly detection model
                    model = AnomalyDetector(model_type=model_type, contamination=contamination)
                    model.train(feature_df)
                    
                    # Predict anomalies
                    anomaly_scores = model.decision_function(feature_df)
                    
                    # Analyze anomalies
                    result_df = analyze_anomalies(preprocessed_df, anomaly_scores, threshold=threshold)
                    st.session_state['result_df'] = result_df
                    
                    # Generate summary statistics
                    summary = get_summary_stats(result_df)
                    st.session_state['summary'] = summary
                    
                    st.session_state['analysis_complete'] = True
                    
                    # Navigate to results page
                    st.success("Analysis complete! View the results on the 'Analysis Results' page.")
                    st.markdown("""<meta http-equiv="refresh" content="2; URL='/?page=Analysis+Results'" />""", unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.session_state['analysis_complete'] = False

def show_results_page():
    st.title("Analysis Results")
    
    if 'analysis_complete' not in st.session_state or not st.session_state['analysis_complete']:
        st.warning("No analysis results available. Please upload and process a PCAP file first.")
        return
    
    # Get data from session state
    result_df = st.session_state['result_df']
    summary = st.session_state['summary']
    file_name = st.session_state.get('file_name', 'Uploaded PCAP')
    
    # Summary statistics
    st.header(f"Summary for {file_name}")
    
    # Create metric cards for key statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Packets", f"{summary['total_packets']:,}")
    with col2:
        st.metric("Anomalous Packets", f"{summary['anomalous_packets']:,}")
    with col3:
        st.metric("Anomaly Percentage", f"{summary['anomaly_percentage']:.2f}%")
    with col4:
        # Count unique protocols
        protocol_count = len([k for k, v in summary['protocol_distribution'].items() if v > 0])
        st.metric("Unique Protocols", protocol_count)
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Overview", 
        "Anomaly Analysis", 
        "Protocol Analysis", 
        "Raw Data"
    ])
    
    with tab1:
        st.subheader("Data Overview")
        
        # Filter display columns
        display_df = filter_columns_for_display(result_df)
        
        # Distribution of packet sizes
        st.write("#### Packet Size Distribution")
        fig = px.histogram(
            display_df, 
            x="length", 
            nbins=50,
            color="is_anomaly",
            color_discrete_map={0: "blue", 1: "red"},
            labels={"length": "Packet Length (bytes)", "count": "Count", "is_anomaly": "Is Anomaly"},
            title="Distribution of Packet Sizes"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Timeline of packets
        st.write("#### Packet Timeline")
        if 'datetime' in result_df.columns:
            # Convert to datetime if it's not already
            timeline_df = result_df.copy()
            if not pd.api.types.is_datetime64_dtype(timeline_df['datetime']):
                timeline_df['datetime'] = pd.to_datetime(timeline_df['datetime'])
            
            # Create timeline chart
            fig = px.scatter(
                timeline_df, 
                x="datetime", 
                y="length",
                color="is_anomaly",
                color_discrete_map={0: "blue", 1: "red"},
                size="anomaly_score",
                hover_data=["src_ip", "dst_ip", "src_port", "dst_port", "protocol"],
                labels={
                    "datetime": "Time", 
                    "length": "Packet Length (bytes)", 
                    "is_anomaly": "Is Anomaly"
                },
                title="Timeline of Network Packets"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Anomaly Analysis")
        
        # Anomaly score distribution
        st.write("#### Anomaly Score Distribution")
        fig = px.histogram(
            result_df, 
            x="anomaly_score", 
            nbins=50,
            color="is_anomaly",
            color_discrete_map={0: "blue", 1: "red"},
            labels={"anomaly_score": "Anomaly Score", "count": "Count", "is_anomaly": "Is Anomaly"},
            title="Distribution of Anomaly Scores"
        )
        fig.add_vline(x=0.8, line_dash="dash", line_color="green", annotation_text="Threshold")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top anomaly interpretations
        if summary['common_anomaly_interpretations']:
            st.write("#### Common Anomaly Interpretations")
            interpretations = pd.DataFrame({
                'Interpretation': list(summary['common_anomaly_interpretations'].keys()),
                'Count': list(summary['common_anomaly_interpretations'].values())
            })
            fig = px.bar(
                interpretations,
                x="Count",
                y="Interpretation",
                orientation='h',
                labels={"Count": "Number of Packets", "Interpretation": "Interpretation"},
                title="Common Anomaly Interpretations"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top source IPs with anomalies
        if summary['top_anomalous_source_ips']:
            st.write("#### Top Source IPs with Anomalies")
            top_ips = pd.DataFrame({
                'Source IP': list(summary['top_anomalous_source_ips'].keys()),
                'Count': list(summary['top_anomalous_source_ips'].values())
            })
            fig = px.bar(
                top_ips,
                x="Source IP",
                y="Count",
                labels={"Count": "Number of Anomalous Packets", "Source IP": "Source IP Address"},
                title="Top Source IPs with Anomalies"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Protocol Analysis")
        
        # Protocol distribution
        st.write("#### Protocol Distribution")
        protocols = pd.DataFrame({
            'Protocol': list(summary['protocol_distribution'].keys()),
            'Count': list(summary['protocol_distribution'].values())
        })
        protocols = protocols[protocols['Count'] > 0]  # Filter out unused protocols
        
        if not protocols.empty:
            fig = px.pie(
                protocols,
                values="Count",
                names="Protocol",
                title="Distribution of Protocols"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No protocol information available.")
        
        # Anomalies by protocol
        if 'protocol' in result_df.columns:
            st.write("#### Anomalies by Protocol")
            protocol_cols = [col for col in result_df.columns if col.startswith('protocol_')]
            
            if protocol_cols:
                # Create a dataframe for protocol analysis
                protocol_data = []
                for col in protocol_cols:
                    protocol_name = col.replace('protocol_', '')
                    protocol_df = result_df[result_df[col] == 1]
                    
                    if not protocol_df.empty:
                        total = len(protocol_df)
                        anomalies = protocol_df['is_anomaly'].sum()
                        protocol_data.append({
                            'Protocol': protocol_name,
                            'Total Packets': total,
                            'Anomalous Packets': anomalies,
                            'Anomaly Percentage': (anomalies / total * 100) if total > 0 else 0
                        })
                
                if protocol_data:
                    protocol_df = pd.DataFrame(protocol_data)
                    fig = px.bar(
                        protocol_df,
                        x="Protocol",
                        y=["Total Packets", "Anomalous Packets"],
                        barmode="group",
                        labels={"value": "Number of Packets", "Protocol": "Protocol"},
                        title="Anomalies by Protocol"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Anomaly percentage by protocol
                    fig = px.bar(
                        protocol_df,
                        x="Protocol",
                        y="Anomaly Percentage",
                        labels={"Anomaly Percentage": "Percentage of Anomalies", "Protocol": "Protocol"},
                        title="Anomaly Percentage by Protocol"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Raw Data")
        
        # Filter to show anomalies only
        show_anomalies_only = st.checkbox("Show anomalies only")
        
        # Filter display columns and rows
        display_df = filter_columns_for_display(result_df)
        if show_anomalies_only:
            display_df = display_df[display_df['is_anomaly'] == 1]
        
        # Show DataFrame
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download Full Results as CSV",
            data=csv,
            file_name=f"anomaly_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_about_page():
    st.title("About the Network Anomaly Detection System")
    
    st.write("""
    ### Project Overview
    
    This application is designed to help cybersecurity professionals and network administrators identify unusual patterns in network traffic that might indicate security threats.
    
    ### Key Features
    
    - **Data Preprocessing**: Extracts relevant features from raw packet data and performs feature engineering.
    - **Machine Learning**: Uses unsupervised anomaly detection algorithms to identify unusual network traffic.
    - **Visualization**: Provides interactive visualizations to understand network traffic patterns and anomalies.
    - **Interpretation**: Offers insights into why certain packets were flagged as anomalous.
    
    ### Technologies Used
    
    - **Python**: For all data processing and analysis
    - **Streamlit**: For the web interface
    - **Scikit-learn**: For machine learning algorithms
    - **Pyshark**: For packet analysis
    - **Pandas**: For data manipulation
    - **Plotly & Matplotlib**: For data visualization
    
    ### How the Anomaly Detection Works
    
    1. **Feature Extraction**: The system extracts features from network packets (protocols, ports, packet sizes, etc.)
    2. **Feature Engineering**: Creates additional features that help identify patterns (time-based features, communication pairs, etc.)
    3. **Unsupervised Learning**: Uses algorithms like Isolation Forest or One-Class SVM to identify outliers
    4. **Scoring**: Assigns anomaly scores to each packet
    5. **Interpretation**: Analyzes the characteristics of anomalous packets to provide insights
    
    ### Privacy and Security
    
    All data processing is done locally. No data is sent to external servers.
    """)

if __name__ == "__main__":
    main()
