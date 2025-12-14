import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import numpy as np
import tensorflow as tf

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SolarInverter AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS & THEME ---
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* General App Styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-weight: 700;
    }
    p {
            
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        # background-color: #FFFFFF;
        # border-right: 1px solid #E2E8F0;
    }
    
    /* Custom Button */
    .stButton>button {
        width: 100%;
        # background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        # color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        # box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.4);
        # transform: translateY(-1px);
    }

    /* Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        # background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #E2E8F0;
        flex: 1;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        # color: #0F172A;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Info Box */
    .stAlert {
        # background-color: #E0F2FE;
        # border: 1px solid #BAE6FD;
        # color: #0369A1;
        # border-radius: 8px;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/solar-panel.png", width=60)
    st.title("SolarInverter AI")
    st.markdown("Batch Prediction Tool")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Plant Data (CSV)", type=['csv'])
    
    run_prediction = False
    if uploaded_file is not None:
        st.success("File Uploaded Successfully")
        run_prediction = st.button("Generate Prediction", type="primary")
    
    st.markdown("---")
    st.markdown("### Requirements")
    st.code("DATE_TIME\nDC_POWER\nAMBIENT_TEMPERATURE\nMODULE_TEMPERATURE\nIRRADIATION\n[AC_POWER (Optional)]\n", language="csv")
    st.caption("Ensure your CSV matches the required schema.")

# --- MAIN AREA ---

# If no file
if uploaded_file is None:
    st.title("Welcome to SolarInverter AI ⚡")
    st.markdown("#### The professional tool for solar energy forecasting")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col2:
        st.markdown("##### Sample Data Structure:")
        st.dataframe(pd.DataFrame({
            'DATE_TIME': ['2023-01-01 12:00'],
            'DC_POWER': [5000],
            'IRRADIATION': [0.8],
            'AMBIENT_TEMPERATURE': [25],
            'MODULE_TEMPERATURE': [45.2],
            'AC_POWER': [4500] # optional AC_POWER 
        }), hide_index=True)
        
    
    with col1:
        st.info("👈 **Start by uploading your CSV file in the sidebar.**")
        st.markdown("""
        Unlock the potential of your solar plant with AI-driven insights. 
        Upload your sensor data to instantly generate accurate AC power output predictions.
        
        **Why use this tool?**
        - 🚀 **Fast**: Process thousands of records in seconds.
        - 🎯 **Accurate**: Powered by advanced machine learning models.
        - 📊 **Visual**: Interactive charts and actionable metrics.
        
        *Note: Include an `AC_POWER` column for Actual vs. Predicted comparison.*
        """)

# State 2 & 3: File Uploaded
else:
    # Load Data
    try:
        # Load the raw data to keep the DATE_TIME and potential AC_POWER column for plotting
        raw_df = pd.read_csv(uploaded_file)
        df = raw_df.copy()

        # Validation for required input features
        required_input_cols = ['DATE_TIME','DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE','IRRADIATION']
        missing_cols = [col for col in required_input_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns for prediction: {', '.join(missing_cols)}")
            st.stop()

        # PIPELINE    
        if 'DATE_TIME' in df.columns:
            raw_df['DATE_TIME'] = pd.to_datetime(raw_df['DATE_TIME'])
            df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
            
            # Feature Engineering for Model
            hour = df['DATE_TIME'].dt.hour
            minute = df['DATE_TIME'].dt.minute
            day = df['DATE_TIME'].dt.day

            df['day_sin'] = np.sin(2*np.pi*day/365)
            df['day_cos'] = np.cos(2*np.pi*day/365)

            df['hour_sin'] = np.sin(2*np.pi*hour/24)
            df['hour_cos'] = np.cos(2*np.pi*hour/24)

            df['mins_sin'] = np.sin(2*np.pi*minute/60)
            df['mins_cos'] = np.cos(2*np.pi*minute/60)
            
            # Drop the original DATE_TIME and any potential AC_POWER column from the feature set
            cols_to_drop = [col for col in ['DATE_TIME', 'AC_POWER'] if col in df.columns]
            df = df.drop(columns=cols_to_drop, axis=1, errors='ignore')
            
        #  Load model/scalers ONCE and cache them 
        @st.cache_resource
        def load_model_and_scalers():
            try:
                model_path = "solar_lstm_model_with_weather.keras"
                feat_scaler_path = "feature_scaler_with_weather.save"
                target_scaler_path = "target_scaler_with_weather.save"
                
                # Check if files exist
                if not os.path.exists(model_path):
                    st.error(f"Model file not found at: {model_path}")
                    return None, None, None
                    
                if not os.path.exists(feat_scaler_path) or not os.path.exists(target_scaler_path):
                    st.error("Scaler files not found.")
                    return None, None, None

                # Load resources
                with tf.keras.utils.custom_object_scope({'leaky_relu': tf.nn.leaky_relu}):
                    model = tf.keras.models.load_model(model_path, compile=False)
                feat_scaler = joblib.load(feat_scaler_path)
                target_scaler = joblib.load(target_scaler_path)
                
                return model, feat_scaler, target_scaler
            except Exception as e:
                st.error(f"Error loading resources: {str(e)}")
                return None, None, None

        # PREDICTION
        if run_prediction:
            st.subheader("⚡ Prediction Results")
            with st.spinner("Running AI Model..."):
                try:
                    # --- Load model and scalers ---
                    model, feat_scaler, target_scaler = load_model_and_scalers()
                    
                    if model is None:
                        st.stop()

                    # --- Scale features ---
                    X_scaled = feat_scaler.transform(df.values)
                    seq_len = 96
                    X = []
                    for i in range(len(X_scaled) - seq_len + 1):
                        X.append(X_scaled[i:i + seq_len])

                    X = np.array(X)  # (samples, 96, num_features)

                    # --- Predict ---
                    preds_scaled = model.predict(X, verbose=0)
                    preds = target_scaler.inverse_transform(preds_scaled).flatten()
                    preds = np.maximum(preds, 0)  # no negative power

                    # --- Align with timestamps ---
                    pred_df = raw_df.iloc[seq_len - 1:].copy() 
                    pred_df["PREDICTED_AC_POWER"] = preds
                    
                    total_power = pred_df["PREDICTED_AC_POWER"].sum()
                    avg_power = pred_df["PREDICTED_AC_POWER"].mean()
                    peak_power = pred_df["PREDICTED_AC_POWER"].max()
                    
                    # Check if actual AC power is available
                    has_actual_ac = 'AC_POWER' in pred_df.columns
                    
                    # --- Display Metrics ---
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-card">
                            <div class="metric-label">🔋Total Predicted Power</div>
                            <div class="metric-value">{total_power:,.2f} kW</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">📊Average Predicted Output</div>
                            <div class="metric-value">{avg_power:,.2f} kW</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">⚡Peak Predicted Power</div>
                            <div class="metric-value">{peak_power:,.2f} kW</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # PLOTTING 
                    
                    if has_actual_ac:
                        st.subheader("📈 Actual vs. Predicted AC Power")
                        title_text = "Actual vs. Predicted AC Power Output"
                    else:
                        st.subheader("📈 Predicted AC Power Output")
                        title_text = "Predicted AC Power Output"
                        
                    fig = go.Figure()

                    # Always plot Predicted AC Power
                    fig.add_trace(go.Scatter(
                        x=pred_df['DATE_TIME'], 
                        y=pred_df['PREDICTED_AC_POWER'], 
                        mode='lines', 
                        name='Predicted AC Power',
                        line=dict(color='#F59E0B', width=3) # Make predicted line more prominent
                    ))

                    if has_actual_ac:
                        # Plot Actual AC Power if the column exists
                        fig.add_trace(go.Scatter(
                            x=pred_df['DATE_TIME'], 
                            y=pred_df['AC_POWER'], 
                            mode='lines', 
                            name='Actual AC Power',
                            line=dict(color='#10B981', width=2, dash='solid')
                        ))
                    
                    # Update Layout
                    fig.update_layout(
                        title=title_text,
                        xaxis_title='Date and Time',
                        yaxis_title='AC Power (kW)',
                        legend_title="Series",
                        hovermode="x unified",
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("📋 Raw Data & Predictions")
                    st.dataframe(pred_df, use_container_width=True)
                    
                    #  Download Button 
                    @st.cache_data
                    def convert_df_to_csv(df):
                        # Cache the conversion to prevent computation on every rerun
                        return df.to_csv(index=False).encode('utf-8')

                    csv_data = convert_df_to_csv(pred_df)

                    st.download_button(
                        label="⬇️ Download Predictions CSV",
                        data=csv_data,
                        file_name='solar_predictions.csv',
                        mime='text/csv',
                    )
                    
                except Exception as e:
                    st.error(f"Prediction Error: {str(e)}")

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")