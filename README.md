# Cloud Anomaly Detection - Streamlit App

🔍 **Real-time Cloud VM Anomaly Detection using Machine Learning**

A Streamlit web application that detects anomalies in cloud virtual machine operations using a trained HistGradientBoostingClassifier model.

## Overview

This application provides a user-friendly interface for detecting anomalies in cloud infrastructure metrics. Users can input operational metrics for cloud VMs, and the model will predict whether the behavior is normal or anomalous.

## Features

✅ **Real-time Predictions**: Instantly classify cloud VM behavior as normal or anomalous
✅ **Interactive UI**: User-friendly input sliders and dropdowns for all 10 features
✅ **Probability Distribution**: View prediction confidence and probability scores
✅ **Model Performance**: Display of ROC-AUC (0.8034) and PR-AUC (0.7149) metrics
✅ **Detailed Analysis**: Comprehensive breakdown of input metrics and predictions
✅ **Responsive Design**: Works seamlessly on desktop and mobile devices

## Input Features

The model accepts the following 10 cloud metrics:

### Numeric Metrics (7 features)
- **CPU Usage (%)**: CPU utilization percentage (0-100%)
- **Memory Usage (%)**: Memory utilization percentage (0-100%)
- **Network Traffic (Mbps)**: Network traffic in megabits per second (0-1000 Mbps)
- **Power Consumption (W)**: Power consumption in watts (0-500 W)
- **Number of Executed Instructions**: CPU instructions executed in millions (0-1000M)
- **Execution Time (ms)**: Task execution time in milliseconds (0-10,000 ms)
- **Energy Efficiency (MIPS/W)**: Million instructions per second per watt (0-100 MIPS/W)

### Categorical Metrics (3 features)
- **Task Type**: Web Server, Database, Compute, or Storage
- **Task Priority**: Low, Medium, or High
- **Task Status**: Running, Idle, or Waiting

## Model Performance

- **Algorithm**: HistGradientBoostingClassifier
- **ROC-AUC Score**: 0.8034 (High)
- **PR-AUC Score**: 0.7149 (Good)
- **Training Data**: Cloud VM operational metrics for anomaly detection

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Omnikam3010/cloud-anomaly-detection-streamlit.git
   cd cloud-anomaly-detection-streamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the model file**
   - Download `best_pipeline_histgb.joblib` from your Colab notebook
   - Place it in the root directory of the project

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   - Open your browser and navigate to `http://localhost:8501`

## Deployment on Streamlit Cloud

### Step 1: Prepare Your Repository
- Ensure all files are committed to GitHub:
  - `app.py` - Main Streamlit application
  - `requirements.txt` - Python dependencies
  - `best_pipeline_histgb.joblib` - Trained model (optional, can be downloaded from cloud storage)
  - `.gitignore` - Configured for Python

### Step 2: Deploy to Streamlit Community Cloud

1. **Go to Streamlit Cloud**
   - Visit https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Deploy new app**
   - Click "New app"
   - Select GitHub repository: `Omnikam3010/cloud-anomaly-detection-streamlit`
   - Select branch: `main`
   - Set main file path: `app.py`

3. **Configure (if needed)**
   - Advanced settings → Python version: 3.10
   - Save and deploy

4. **Access your deployed app**
   - Your app will be available at: `https://share.streamlit.io/Omnikam3010/cloud-anomaly-detection-streamlit/main/app.py`
   - Copy the URL and share with others

## Usage Guide

1. **Open the application** in your browser
2. **Adjust metrics** in the sidebar using sliders and dropdowns
3. **Review input summary** in the main area
4. **Click "Predict Anomaly"** button
5. **View results**:
   - Status badge (🟢 NORMAL or 🔴 ANOMALOUS)
   - Confidence percentage
   - Probability distribution chart
   - Detailed metrics summary

## File Structure

```
cloud-anomaly-detection-streamlit/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── best_pipeline_histgb.joblib     # Trained ML model
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## Technologies Used

- **Streamlit**: Web framework for building data apps
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **joblib**: Model persistence

## Troubleshooting

### Issue: "Model file not found"
- **Solution**: Ensure `best_pipeline_histgb.joblib` is in the same directory as `app.py`

### Issue: "ModuleNotFoundError"
- **Solution**: Install dependencies: `pip install -r requirements.txt`

### Issue: App runs slowly
- **Solution**: Streamlit caches the model automatically; refresh the page if needed

## API & Integration

The model can also be used programmatically:

```python
import joblib
import pandas as pd

# Load model
pipeline = joblib.load('best_pipeline_histgb.joblib')

# Create input data
input_data = pd.DataFrame({
    'cpuusage': [50.0],
    'memoryusage': [60.0],
    'networktraffic': [100.0],
    'powerconsumption': [150.0],
    'numexecutedinstructions': [200.0],
    'executiontime': [500.0],
    'energyefficiency': [10.0],
    'tasktype': ['Web Server'],
    'taskpriority': ['High'],
    'taskstatus': ['Running']
})

# Make prediction
prediction = pipeline.predict(input_data)
prob = pipeline.predict_proba(input_data)
print(f"Prediction: {prediction[0]}, Probability: {prob[0]}")
```

## Performance Metrics

The model achieved excellent performance on the test dataset:
- **ROC-AUC**: 0.8034 - Indicates strong discrimination between normal and anomalous instances
- **PR-AUC**: 0.7149 - Good precision-recall balance

## Future Enhancements

- [ ] Add historical data visualization
- [ ] Implement batch prediction for multiple VMs
- [ ] Add model retraining functionality
- [ ] Export predictions to CSV
- [ ] Add anomaly severity scoring
- [ ] Integration with cloud monitoring tools

## License

This project is open source and available under the MIT License.

## Contact & Support

- **GitHub**: https://github.com/Omnikam3010
- **Email**: For support, please open an issue on GitHub

---

**Built with ❤️ using Streamlit and scikit-learn**

Cloud Anomaly Detection System | Powered by HistGradientBoostingClassifier | ML Model v1.0
