import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="Adult Income Classification", layout="wide")

st.title("💼 Adult Income Classification App")

# -----------------------------
# Model Selection
# -----------------------------
model_options = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]

selected_model = st.selectbox("Select Model", model_options)

# -----------------------------
# Load Model
# -----------------------------
model_filename = selected_model.replace(" ", "_") + ".pkl"
model_path = os.path.join("saved_models", model_filename)

model = joblib.load(model_path)

# -----------------------------
# Load Saved Metrics
# -----------------------------
st.subheader("📊 Original Model Performance")

metrics_path = os.path.join("saved_models", "model_metrics.csv")

if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path, index_col=0)
    st.dataframe(metrics_df.loc[selected_model])
else:
    st.warning("Model metrics file not found.")

# -----------------------------
# Upload Data
# -----------------------------
st.subheader("📁 Upload Processed Test CSV")

uploaded_file = st.file_uploader("Upload Processed CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data Preview")
    st.dataframe(data.head())

    # If income column exists → separate
    if "income" in data.columns:
        y_true = data["income"]
        X_input = data.drop("income", axis=1)
    else:
        y_true = None
        X_input = data

    # Predictions
    predictions = model.predict(X_input)

    st.write("### 🔮 Predictions")
    st.write(predictions)

    # If true labels exist → calculate metrics
    if y_true is not None:
        st.subheader("📈 Evaluation on Uploaded Data")

        acc = accuracy_score(y_true, predictions)
        prec = precision_score(y_true, predictions)
        rec = recall_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)
        mcc = matthews_corrcoef(y_true, predictions)

        metrics_dict = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "MCC": mcc
        }

        st.write(metrics_dict)

        st.subheader("📌 Confusion Matrix")
        cm = confusion_matrix(y_true, predictions)
        st.write(cm)