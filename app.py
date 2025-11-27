import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Cloud Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('best_pipeline_histgb.joblib')
    return model

pipeline = load_model()

# Title and description
st.title("🔍 Cloud VM Anomaly Detection")
st.markdown("""
This application uses a **HistGradientBoostingClassifier** machine learning model 
to detect anomalies in cloud virtual machine operations. Simply input the cloud metrics 
below and the model will predict whether the VM behavior is **normal** or **anomalous**.
""")

# Display model performance metrics
with st.expander("📊 Model Performance Metrics", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ROC-AUC Score", "0.8034", "↑ High")
    with col2:
        st.metric("PR-AUC Score", "0.7149", "↑ Good")
    st.info("The model was trained on cloud VM operational metrics to identify anomalous behavior patterns.")

# Sidebar for user inputs
st.sidebar.header("📝 Input Cloud Metrics")
st.sidebar.markdown("Enter the cloud VM operational metrics below:")

# Numeric inputs
st.sidebar.subheader("Numeric Metrics")
cpu_usage = st.sidebar.slider(
    "CPU Usage (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=1.0,
    help="CPU utilization percentage"
)

memory_usage = st.sidebar.slider(
    "Memory Usage (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=1.0,
    help="Memory utilization percentage"
)

network_traffic = st.sidebar.slider(
    "Network Traffic (Mbps)",
    min_value=0.0,
    max_value=1000.0,
    value=100.0,
    step=10.0,
    help="Network traffic in megabits per second"
)

power_consumption = st.sidebar.slider(
    "Power Consumption (W)",
    min_value=0.0,
    max_value=500.0,
    value=100.0,
    step=5.0,
    help="Power consumption in watts"
)

num_executed_instructions = st.sidebar.slider(
    "Number of Executed Instructions (millions)",
    min_value=0.0,
    max_value=1000.0,
    value=100.0,
    step=10.0,
    help="Number of CPU instructions executed (in millions)"
)

execution_time = st.sidebar.slider(
    "Execution Time (ms)",
    min_value=0.0,
    max_value=10000.0,
    value=100.0,
    step=50.0,
    help="Task execution time in milliseconds"
)

energy_efficiency = st.sidebar.slider(
    "Energy Efficiency (MIPS/W)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=0.5,
    help="Million instructions per second per watt"
)

# Categorical inputs
st.sidebar.subheader("Task Information")
task_type = st.sidebar.selectbox(
    "Task Type",
    options=["Web Server", "Database", "Compute", "Storage"],
    help="Type of task running on the VM"
)

task_priority = st.sidebar.selectbox(
    "Task Priority",
    options=["Low", "Medium", "High"],
    help="Priority level of the task"
)

task_status = st.sidebar.selectbox(
    "Task Status",
    options=["Running", "Idle", "Waiting"],
    help="Current status of the task"
)

# Create a dataframe with user inputs
input_data = pd.DataFrame({
    'cpuusage': [cpu_usage],
    'memoryusage': [memory_usage],
    'networktraffic': [network_traffic],
    'powerconsumption': [power_consumption],
    'numexecutedinstructions': [num_executed_instructions],
    'executiontime': [execution_time],
    'energyefficiency': [energy_efficiency],
    'tasktype': [task_type],
    'taskpriority': [task_priority],
    'taskstatus': [task_status]
})

# Display input summary
with st.expander("📋 Input Summary", expanded=True):
    st.dataframe(input_data, use_container_width=True)

# Make prediction
if st.button("🚀 Predict Anomaly", type="primary", use_container_width=True):
    with st.spinner("Analyzing cloud metrics..."):
        try:
            # Make prediction
            prediction = pipeline.predict(input_data)[0]
            prediction_proba = pipeline.predict_proba(input_data)[0]
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            # Create three columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 0:
                    st.metric(
                        "Status",
                        "🟢 NORMAL",
                        "No anomaly detected"
                    )
                else:
                    st.metric(
                        "Status",
                        "🔴 ANOMALOUS",
                        "Anomaly detected!"
                    )
            
            with col2:
                confidence = max(prediction_proba) * 100
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%",
                    "Prediction confidence"
                )
            
            with col3:
                normal_prob = prediction_proba[0] * 100
                st.metric(
                    "Normal Probability",
                    f"{normal_prob:.2f}%",
                    "Likelihood of normal behavior"
                )
            
            # Display probability distribution
            st.subheader("📈 Prediction Probabilities")
            prob_data = pd.DataFrame({
                'Classification': ['Normal', 'Anomalous'],
                'Probability': [prediction_proba[0], prediction_proba[1]]
            })
            st.bar_chart(prob_data.set_index('Classification'), use_container_width=True)
            
            # Display detailed metrics
            st.subheader("📊 Input Metrics Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Numeric Metrics:**")
                numeric_summary = pd.DataFrame({
                    'Metric': ['CPU Usage', 'Memory Usage', 'Network Traffic', 'Power Consumption', 
                              'Executed Instructions', 'Execution Time', 'Energy Efficiency'],
                    'Value': [f"{cpu_usage}%", f"{memory_usage}%", f"{network_traffic} Mbps", 
                             f"{power_consumption} W", f"{num_executed_instructions}M", 
                             f"{execution_time} ms", f"{energy_efficiency} MIPS/W"]
                })
                st.dataframe(numeric_summary, use_container_width=True)
            
            with col2:
                st.write("**Task Information:**")
                task_summary = pd.DataFrame({
                    'Property': ['Task Type', 'Task Priority', 'Task Status'],
                    'Value': [task_type, task_priority, task_status]
                })
                st.dataframe(task_summary, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Please ensure the model file is uploaded correctly.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Cloud Anomaly Detection System | Powered by HistGradientBoostingClassifier | ML Model v1.0
    </div>
    """, unsafe_allow_html=True)
