import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import xgboost as xgb

# --- P1: APP CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="Churn & Retention Decision System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- P2: PROFESSIONAL STYLING (FAANG-Style Internal Tool) ---
st.markdown("""
<style>
    /* Global clean font */
    * {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Styling */
    h1 {
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #1a202c;
    }
    h2, h3 {
        font-weight: 600;
        color: #2d3748;
    }
    
    /* Card Container */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 5px;
    }
    
    /* Recommendations Box */
    .recommendation-box {
        background-color: #ebf8ff;
        border-left: 4px solid #4299e1;
        padding: 16px;
        border-radius: 4px;
        margin-top: 20px;
    }
    
    /* Hide Default Streamlit Menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- P3: LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    if not os.path.exists("model"):
        return None, None, None, None, None
    
    artifacts = {}
    try:
        with open("model/scaler.pkl", "rb") as f:
            artifacts["scaler"] = pickle.load(f)
        with open("model/label_encoders.pkl", "rb") as f:
            artifacts["label_encoders"] = pickle.load(f)
        
        models = {}
        model_files = [f for f in os.listdir("model") if f.endswith(".pkl") and f not in ["scaler.pkl", "preprocessor.pkl", "label_encoders.pkl", "confusion_matrices.pkl"]]
        for name in model_files:
            clean_name = name.replace(".pkl", "").replace("_", " ").title()
            with open(f"model/{name}", "rb") as f:
                 models[clean_name] = pickle.load(f)
        
        # Load Confusion Matrices
        confusion_matrices = {}
        if os.path.exists("model/confusion_matrices.pkl"):
            with open("model/confusion_matrices.pkl", "rb") as f:
                confusion_matrices = pickle.load(f)
        
        # Load Performance Metrics
        metrics_df = None
        if os.path.exists("model_performance.csv"):
            metrics_df = pd.read_csv("model_performance.csv")
            
        return artifacts["scaler"], artifacts["label_encoders"], models, metrics_df, confusion_matrices
    except Exception as e:
        return None, None, None, None, None

scaler, label_encoders, models, metrics_df, confusion_matrices = load_artifacts()

def preprocess_data(df, scaler, label_encoders, expected_cols):
    """
    Applies the same preprocessing steps as the training pipeline.
    """
    df_proc = df.copy()
    
    # 1. Scale Numerical Columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # Ensure columns exist before scaling
    if all(c in df_proc.columns for c in num_cols):
        df_proc[num_cols] = scaler.transform(df_proc[num_cols])
    
    # 2. Encode Categorical Columns
    for col, le in label_encoders.items():
        if col in df_proc.columns:
            # Handle unknown categories safely
            df_proc[col] = df_proc[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
            df_proc[col] = le.transform(df_proc[col])
            
    # 3. Ensure all expected columns exist (padding with 0)
    for col in expected_cols:
        if col not in df_proc.columns:
            df_proc[col] = 0
            
    # 4. Reorder and set type
    df_proc = df_proc[expected_cols].astype(float)
    return df_proc

# Expected columns from training
EXPECTED_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
                 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

# --- P4: MAIN LAYOUT ---

# Header
st.title("Customer Churn Prediction & Retention Decision System")
st.markdown("""
<p style='font-size: 1.1rem; color: #4a5568;'>
    <b>Machine Learning–based churn risk classification</b> with actionable business recommendations. 
    This system supports valid decision-making for customer retention strategies.
</p>
<hr style='margin-bottom: 30px;'>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Model Selection
    model_names = list(models.keys()) if models else ["No Models Found"]
    selected_model_name = st.selectbox(
        "Select Classification Model", 
        model_names,
        index=0
    )
    
    st.info(f"Currently utilizing **{selected_model_name}** for inference.")
    
    st.divider()
    st.caption("v2.4.0 • Enterprise Release")

# --- MAIN SECTIONS ---

if not models:
    st.error("🚨 Critical Error: Models not found. Please ensure the training pipeline has been executed.")
    st.stop()

model = models[selected_model_name]

# TABS for workflow
tab1, tab2, tab3 = st.tabs(["🚀 Single Prediction", "📂 Batch Upload", "📊 Model Evaluation"])

# --- TAB 1: Single Prediction (The Decision Engine) ---
with tab1:
    st.subheader("Customer Profile Assessment")
    
    with st.expander("📝 Input Customer Details", expanded=True):
        # Organize inputs into logical columns
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown("##### 👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            
        with c2:
            st.markdown("##### 📞 Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", label_encoders['MultipleLines'].classes_ if 'MultipleLines' in label_encoders else ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", label_encoders['InternetService'].classes_)
            
        with c3:
            st.markdown("##### 🛡️ Tech Specs")
            online_security = st.selectbox("Online Security", label_encoders['OnlineSecurity'].classes_)
            online_backup = st.selectbox("Online Backup", label_encoders['OnlineBackup'].classes_ if 'OnlineBackup' in label_encoders else ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", label_encoders['DeviceProtection'].classes_ if 'DeviceProtection' in label_encoders else ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", label_encoders['TechSupport'].classes_)
            
        with c4:
            st.markdown("##### 📺 Streaming")
            streaming_tv = st.selectbox("Streaming TV", label_encoders['StreamingTV'].classes_ if 'StreamingTV' in label_encoders else ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", label_encoders['StreamingMovies'].classes_ if 'StreamingMovies' in label_encoders else ["No", "Yes", "No internet service"])

    with st.expander("💳 Account & Billing Details", expanded=True):
        a1, a2, a3 = st.columns(3)
        with a1:
            contract = st.selectbox("Contract Type", label_encoders['Contract'].classes_)
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", label_encoders['PaymentMethod'].classes_)
        with a2:
            tenure = st.slider("Tenure (Months)", 0, 80, 12)
        with a3:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 300.0, 70.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)


    if st.button("Evaluate Churn Risk", type="primary"):
        # Preprocessing
        input_dict = {
            "tenure": tenure, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
            "Contract": contract, "InternetService": internet_service, "TechSupport": tech_support,
            "OnlineSecurity": online_security, "PaymentMethod": payment_method,
            "gender": 1 if gender == "Male" else 0,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Dependents": 1 if dependents == "Yes" else 0,
            "PaperlessBilling": 1 if paperless == "Yes" else 0,
            "Partner": 1 if partner == "Yes" else 0,
            "PhoneService": 1 if phone_service == "Yes" else 0, 
            "MultipleLines": multiple_lines,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies
        }
        
        # Build DataFrame
        df_raw = pd.DataFrame([input_dict])
        
        # Process
        df_input = preprocess_data(df_raw, scaler, label_encoders, EXPECTED_COLS)

        # Predict
        prediction = model.predict(df_input)[0]
        try:
            prob = model.predict_proba(df_input)[0][1]
        except:
            prob = 0.0 # Fallback for models without probability

        # --- OUTPUT CARDS ---
        st.markdown("### Decision Support Output")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Churn Risk Class</div>
                    <div class="metric-value" style="color: {'#e53e3e' if prediction==1 else '#38a169'}">
                        {'HIGH' if prediction==1 else 'LOW'}
                    </div>
                </div>""", 
                unsafe_allow_html=True
            )
            
        with col2:
             st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Churn Probability</div>
                    <div class="metric-value">
                        {prob:.1%}
                    </div>
                </div>""", 
                unsafe_allow_html=True
            )
            
        with col3:
             confidence = prob if prediction == 1 else (1 - prob)
             st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Model Confidence</div>
                    <div class="metric-value">
                        {confidence:.1%}
                    </div>
                </div>""", 
                unsafe_allow_html=True
            )
            
        # Recommendation Engine
        st.markdown("#### 💡 Retention Recommendation")
        
        if prediction == 1:
            if prob > 0.8:
                rec_title = "🔴 Immediate Intervention Required"
                rec_text = "Customer is at **Critical Risk**. Suggest offering a **20% discount** on the next 3 months or a **free service upgrade** immediately. Assign to 'Priority Retention Team'."
            else:
                rec_title = "🟠 Proactive Retention"
                rec_text = "Customer is at **Risk**. Schedule a check-in call to discuss usage satisfaction. Consider waiving late fees or offering a better contract term."
            
            st.markdown(f"""
            <div class="recommendation-box" style="border-left-color: #e53e3e; background-color: #fff5f5;">
                <h4 style="margin:0; color: #c53030;">{rec_title}</h4>
                <p style="margin-top: 5px;">{rec_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div class="recommendation-box" style="border-left-color: #38a169; background-color: #f0fff4;">
                <h4 style="margin:0; color: #2f855a;">🟢 Maintain Relation</h4>
                <p style="margin-top: 5px;">Customer is <strong>Healthy</strong>. No immediate action required. Continue standard engagement emails.</p>
            </div>
            """, unsafe_allow_html=True)


# --- TAB 2: Batch Upload ---
with tab2:
    st.subheader("Bulk Assessment")
    uploaded_file = st.file_uploader("Upload Customer Data (CSV) for Batch Scoring", type="csv")
    
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        st.info(f"Loaded **{len(df_upload)}** customer records.")
        st.dataframe(df_upload.head(), width='stretch')
        
        if st.button("Score All Records"):
             with st.spinner("Processing batch predictions..."):
                 # Handle Binary Columns (Map Yes/No/Male/Female to 1/0)
                 binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'No phone service': 0, 'No internet service': 0}
                 
                 # Apply mapping to known binary columns if they exist as strings
                 for col in df_upload.columns:
                     if df_upload[col].dtype == 'object':
                         # Try mapping only if values allow
                         if set(df_upload[col].dropna().unique()).issubset(set(binary_map.keys())):
                              df_upload[col] = df_upload[col].map(binary_map)
                 
                 # Preprocess
                 try:
                     df_batch_processed = preprocess_data(df_upload, scaler, label_encoders, EXPECTED_COLS)
                     
                     # Predict
                     df_upload['Churn_Prediction'] = model.predict(df_batch_processed)
                     try:
                         df_upload['Churn_Probability'] = model.predict_proba(df_batch_processed)[:, 1]
                     except:
                         df_upload['Churn_Probability'] = 0.0
                         
                     df_upload['Churn_Label'] = df_upload['Churn_Prediction'].apply(lambda x: 'High Risk' if x == 1 else 'Low Risk')
                     
                     st.success("Batch processing complete.")
                     st.dataframe(df_upload[['customerID', 'Churn_Label', 'Churn_Probability']] if 'customerID' in df_upload.columns else df_upload.head())
                     
                     csv = df_upload.to_csv(index=False).encode('utf-8')
                     st.download_button("Download Results CSV", csv, "churn_predictions.csv", "text/csv")
                     
                 except Exception as e:
                     st.error(f"Error during batch processing: {str(e)}")


# --- TAB 3: Model Evaluation ---
with tab3:
    st.subheader("Model Performance Audit")
    
    if metrics_df is not None:
        # Custom styling: Highlight selected model, defaults for others
        def style_table(row):
            is_selected = row['Model'] == selected_model_name
            if is_selected:
                # High contrast highlight for selected model
                return ['background-color: #e6fffa; color: #2d3748; font-weight: bold'] * len(row)
            else:
                # Keep default background (transparent in Streamlit), enforce white text
                return ['color: white'] * len(row)

        st.dataframe(
            metrics_df.style.apply(style_table, axis=1).format(
                "{:.2%}", subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            ),
            width='stretch'
        )
        
        st.markdown("#### 🧩 Confusion Matrix")
        if confusion_matrices and selected_model_name in confusion_matrices:
            cm = np.array(confusion_matrices[selected_model_name])
            
            # Simple manual heatmap using styled dataframe to avoid heavy matplotlib import overhead in rendering if we can
            # But standard is matplotlib/seaborn. Let's use st.pyplot
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(3, 2))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                ax.set_title(f"Confusion Matrix: {selected_model_name}", fontsize=8)
                ax.set_ylabel('Actual', fontsize=8)
                ax.set_xlabel('Predicted', fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=8)
                st.pyplot(fig, use_container_width=False)
            except:
                st.write(cm) # Fallback
        else:
             st.info("Confusion matrix not available for this model.")
    else:
        st.warning("Performance metrics not found.")
        
    st.markdown("### Benchmarking Definitions")
    m1, m2, m3 = st.columns(3)
    m1.info("**Precision:** Accuracy of positive predictions (Churn).")
    m2.info("**Recall:** Ability to find all actual Churn cases.")
    m3.info("**AUC:** Overall capability to distinguish classes.")
    
    st.markdown("---")
    st.markdown("### Model Performance Observations")
    
    obs_data = {
        "ML Model": [
            "Logistic Regression", "Decision Tree", "kNN", 
            "Naive Bayes", "Random Forest (Ensemble)", "XGBoost (Ensemble)"
        ],
        "Observation": [
            "Achieved the highest overall accuracy and AUC, indicating strong baseline performance with good interpretability and balanced precision–recall trade-off.",
            "Showed lower performance across most metrics, indicating overfitting and limited generalization capability on unseen data.",
            "Delivered moderate performance with balanced accuracy and F1-score, but performance is sensitive to feature scaling and distance calculations.",
            "Achieved the highest recall, making it effective at identifying churned customers, but with lower precision leading to more false positives.",
            "Provided stable and balanced performance across metrics by reducing overfitting through ensemble learning.",
            "Demonstrated strong predictive performance with high AUC and F1-score by effectively capturing complex patterns in the data."
        ]
    }
    
    obs_df = pd.DataFrame(obs_data)
    st.table(obs_df)
    


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a0aec0; font-size: 0.8rem;">
    © 2025 Telco Analytics Group • Internal Use Only • v2.4.0
</div>
""", unsafe_allow_html=True)
