import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Cloud Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Cloud VM Anomaly Detection System")
st.markdown("Real-time anomaly detection using HistogramGradientBoosting model")
st.markdown("---")

# Load the model
@st.cache_resource
def load_model():
    try:
        # Try loading from joblib first (lighter)
        model_path = Path('best_pipeline_histgb.joblib')
        if model_path.exists():
            model = joblib.load(model_path)
            st.sidebar.success("✅ Model loaded from joblib")
            return model
        
        # Fallback to pickle
        model_path = Path('best_pipeline_histgb.pkl')
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            st.sidebar.success("✅ Model loaded from pickle")
            return model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.error("❌ Model file not found. Please ensure 'best_pipeline_histgb.joblib' or 'best_pipeline_histgb.pkl' is in the same directory.")
    st.stop()

# Define feature names based on your training data
NUMERIC_FEATURES = [
    'cpu_usage', 'memory_usage', 'network_traffic',
    'power_consumption', 'num_executed_instructions',
    'execution_time', 'energy_efficiency'
]

CATEGORICAL_FEATURES = [
    'task_type', 'task_priority', 'task_status'
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Sidebar for user input
st.sidebar.header("⚙️ Configuration")
input_method = st.sidebar.radio(
    "Select input method:",
    ["Manual Input", "CSV Upload", "Sample Data"]
)

input_data_df = None

if input_method == "Manual Input":
    st.sidebar.subheader("📊 Enter Feature Values")
    
    # Numeric inputs
    numeric_values = {}
    for feature in NUMERIC_FEATURES:
        numeric_values[feature] = st.sidebar.slider(
            f"{feature.replace('_', ' ').title()}",
            min_value=0.0,
            max_value=1000.0,
            value=50.0,
            step=1.0
        )
    
    # Categorical inputs
    categorical_values = {}
    categorical_values['task_type'] = st.sidebar.selectbox(
        "Task Type",
        options=['compute', 'io', 'network']
    )
    categorical_values['task_priority'] = st.sidebar.selectbox(
        "Task Priority",
        options=['low', 'medium', 'high']
    )
    categorical_values['task_status'] = st.sidebar.selectbox(
        "Task Status",
        options=['running', 'waiting', 'completed']
    )
    
    # Combine all values
    all_values = {**numeric_values, **categorical_values}
    input_data_df = pd.DataFrame([all_values])
    
    # Display the input data
    st.subheader("📋 Your Input Data")
    st.dataframe(input_data_df, use_container_width=True)
    
elif input_method == "CSV Upload":
    st.sidebar.subheader("📂 Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        input_data_df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview:")
        st.dataframe(input_data_df.head(10), use_container_width=True)
    else:
        st.sidebar.warning("⚠️ Please upload a CSV file")
        
else:  # Sample Data
    st.sidebar.subheader("🎲 Sample Data")
    num_samples = st.sidebar.slider("Number of samples", 1, 20, 5)
    
    # Generate realistic sample data
    np.random.seed(42)
    sample_data = {
        'cpu_usage': np.random.uniform(0, 100, num_samples),
        'memory_usage': np.random.uniform(0, 100, num_samples),
        'network_traffic': np.random.uniform(0, 1000, num_samples),
        'power_consumption': np.random.uniform(0, 500, num_samples),
        'num_executed_instructions': np.random.uniform(0, 10000, num_samples),
        'execution_time': np.random.uniform(0, 100, num_samples),
        'energy_efficiency': np.random.uniform(0, 1, num_samples),
        'task_type': np.random.choice(['compute', 'io', 'network'], num_samples),
        'task_priority': np.random.choice(['low', 'medium', 'high'], num_samples),
        'task_status': np.random.choice(['running', 'waiting', 'completed'], num_samples)
    }
    input_data_df = pd.DataFrame(sample_data)
    st.write("📊 Sample Data:")
    st.dataframe(input_data_df, use_container_width=True)

# Prediction section
if st.button("🚀 Run Anomaly Detection", key="predict"):
    if input_data_df is not None and len(input_data_df) > 0:
        try:
            st.info("⏳ Running predictions...")
            
            # Make predictions
            predictions = model.predict(input_data_df)
            
            # Get prediction probabilities
            try:
                anomaly_probs = model.predict_proba(input_data_df)[:, 1]
            except:
                anomaly_probs = predictions.astype(float)
            
            # Display results
            st.subheader("📊 Detection Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                normal_count = np.sum(predictions == 0)
                st.metric("✅ Normal", normal_count)
            
            with col2:
                anomaly_count = np.sum(predictions == 1)
                st.metric("⚠️ Anomalies", anomaly_count)
            
            with col3:
                anomaly_rate = (anomaly_count / len(predictions)) * 100 if len(predictions) > 0 else 0
                st.metric("🔴 Anomaly Rate", f"{anomaly_rate:.1f}%")
            
            with col4:
                avg_confidence = np.mean(anomaly_probs)
                st.metric("📈 Avg Confidence", f"{avg_confidence:.3f}")
            
            # Results table
            st.subheader("📋 Detailed Results")
            results_df = pd.DataFrame({
                'Sample': range(len(predictions)),
                'Prediction': ['⚠️ Anomaly' if p == 1 else '✅ Normal' for p in predictions],
                'Confidence': np.round(anomaly_probs * 100, 2),
                'Risk Level': ['🔴 High' if prob > 0.7 else '🟡 Medium' if prob > 0.4 else '🟢 Low' for prob in anomaly_probs]
            })
            st.dataframe(results_df, use_container_width=True)
            
            # Visualization
            st.subheader("📈 Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Classification distribution
                fig = go.Figure(data=[
                    go.Bar(
                        x=['✅ Normal', '⚠️ Anomaly'],
                        y=[normal_count, anomaly_count],
                        marker_color=['green', 'red']
                    )
                ])
                fig.update_layout(
                    title="Sample Classification",
                    xaxis_title="Classification",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig = go.Figure(data=[
                    go.Histogram(
                        x=anomaly_probs,
                        nbinsx=20,
                        marker_color='indianred'
                    )
                ])
                fig.update_layout(
                    title="Anomaly Probability Distribution",
                    xaxis_title="Anomaly Probability",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature values heatmap
            if len(input_data_df) > 0:
                st.subheader("🔥 Feature Heatmap")
                numeric_df = input_data_df[NUMERIC_FEATURES].head(10)
                fig = go.Figure(data=go.Heatmap(
                    z=numeric_df.values,
                    x=numeric_df.columns,
                    y=[f"Sample {i}" for i in range(len(numeric_df))],
                    colorscale='Viridis'
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Detection complete!")
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.error(f"Details: {str(e)}")
    else:
        st.warning("⚠️ Please provide input data")

# Model information
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Model Information")
st.sidebar.write(f"**Model Type:** HistogramGradientBoosting")
st.sidebar.write(f"**Test ROC-AUC:** 0.862")
st.sidebar.write(f"**Test PR-AUC:** 0.798")
st.sidebar.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.markdown("---")
st.sidebar.markdown("📖 [Streamlit Docs](https://docs.streamlit.io/)")
st.sidebar.markdown("🤖 [Model Source](https://github.com/Omnikam3010/cloud-anomaly-detection-streamlit)")
