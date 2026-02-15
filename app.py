import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)

st.set_page_config(page_title="Adult Income Classification", layout="wide")

st.title("💼 Adult Income Prediction Dashboard")
st.markdown("Predict whether an individual earns more than $50K per year.")

# --------------------------------------------------
# Sidebar - Model Selection
# --------------------------------------------------

st.sidebar.header("Model Selection")

model_options = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]

selected_model = st.sidebar.selectbox("Choose Model", model_options)

model_filename = selected_model.replace(" ", "_") + ".pkl"
model_path = os.path.join("saved_models", model_filename)

model = joblib.load(model_path)

# --------------------------------------------------
# Display Original Metrics
# --------------------------------------------------

st.subheader("📊 Original Model Performance")

metrics_path = os.path.join("saved_models", "train_metrics.csv")

if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path, index_col=0)
    row = metrics_df.loc[selected_model]

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Accuracy", round(row["Accuracy"], 3))
    col2.metric("Precision", round(row["Precision"], 3))
    col3.metric("Recall", round(row["Recall"], 3))

    col4.metric("F1 Score", round(row["F1"], 3))
    col5.metric("MCC", round(row["MCC"], 3))
    col6.metric("AUC", round(row["AUC"], 3))

else:
    st.warning("Model metrics file not found.")

st.divider()

# --------------------------------------------------
# Upload Data Section
# --------------------------------------------------

st.subheader("📁 Upload Processed Test Dataset")

uploaded_file = st.file_uploader("Upload CSV file (Processed Data)", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if "income" in data.columns:
        y_true = data["income"]
        X_input = data.drop("income", axis=1)
    else:
        y_true = None
        X_input = data

    predictions = model.predict(X_input)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_input)[:, 1]
    else:
        probabilities = None

    # --------------------------------------------------
    # Show Predictions Table
    # --------------------------------------------------

    st.subheader("🔮 Predictions")

    result_df = pd.DataFrame({
        "Prediction": predictions
    })

    if probabilities is not None:
        result_df["Probability (>50K)"] = probabilities.round(4)

    if y_true is not None:
        result_df["Actual"] = y_true.values

    st.dataframe(result_df, use_container_width=True)

    # --------------------------------------------------
    # Evaluation Metrics (If Labels Exist)
    # --------------------------------------------------

    if y_true is not None:

        st.subheader("📈 Evaluation on Uploaded Data")

        acc = accuracy_score(y_true, predictions)
        prec = precision_score(y_true, predictions)
        rec = recall_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)
        mcc = matthews_corrcoef(y_true, predictions)

        if probabilities is not None:
            auc = roc_auc_score(y_true, probabilities)
        else:
            auc = None

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Accuracy", round(acc, 3))
        col2.metric("Precision", round(prec, 3))
        col3.metric("Recall", round(rec, 3))

        col4.metric("F1 Score", round(f1, 3))
        col5.metric("MCC", round(mcc, 3))

        if auc is not None:
            col6.metric("AUC", round(auc, 3))

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------

        st.subheader("📌 Confusion Matrix")

        cm = confusion_matrix(y_true, predictions)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["<=50K", ">50K"],
            yticklabels=["<=50K", ">50K"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)