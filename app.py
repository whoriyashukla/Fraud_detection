import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams["figure.autolayout"] = True
#st.set_option('deprecation.showPyplotGlobalUse', False)

# Load model and features
model_options = {
    "Random Forest": "rf_full_model.joblib",
    "Logistic Regression": "log_reg_full_model.joblib",
    "SVM": "svm_model.joblib"
}

selected_model_name = st.selectbox("🔀 Choose Model", list(model_options.keys()))
model_path = f"fraud_detection/{model_options[selected_model_name]}"

model = joblib.load(model_path)
features = joblib.load("fraud_detection/features_full.joblib")
threshold = joblib.load("fraud_detection/custom_threshold.joblib")


# UI Config
st.set_page_config(page_title="💳 Credit Card Fraud Detector", layout="wide")
st.title("🔍 Retail Credit Card Fraud Detection")
st.markdown("Upload transaction data and predict fraud using a trained Random Forest model.")

# Upload
st.markdown("### 📥 Need Sample Data?")
with open("sample_transactions.csv", "rb") as f:
    st.download_button("⬇️ Download Sample CSV", f, file_name="sample_transactions.csv", mime="text/csv")

uploaded_file = st.file_uploader("📁 Upload a CSV file with the required features", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("🧾 Uploaded Data Preview:", data.head())

    if all(f in data.columns for f in features):
        X_input = data[features]
        y_proba = model.predict_proba(X_input)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        data["Fraud Probability"] = y_proba
        data["Fraud Prediction"] = y_pred

        total = len(data)
        frauds = data["Fraud Prediction"].sum()
        fraud_rate = frauds / total * 100

        st.markdown("### 📊 Fraud Detection Summary")
        st.markdown(f"### 🤖 Model Used: `{selected_model_name}`")
        st.write(f"🔹 Total Transactions: `{total}`")
        st.write(f"🔸 Fraudulent Transactions: `{frauds}`")
        st.write(f"🔸 Fraud Rate: `{fraud_rate:.2f}%`")

        # Side-by-side plots
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔍 Fraud vs Non-Fraud Count")
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            sns.countplot(x="Fraud Prediction", data=data, palette="coolwarm", ax=ax1)
            ax1.set_xticklabels(["Non-Fraud", "Fraud"])
            st.pyplot(fig1)

        with col2:
            st.markdown("### 📉 Fraud Probability Distribution")
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            sns.histplot(data["Fraud Probability"], bins=20, kde=True, color="purple", ax=ax2)
            st.pyplot(fig2)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### 💰 Transaction Amounts by Prediction")
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            sns.boxplot(x="Fraud Prediction", y="Amount", data=data, palette="Set2", ax=ax3)
            ax3.set_xticklabels(["Non-Fraud", "Fraud"])
            st.pyplot(fig3)

        with col4:
            st.markdown("### 🕒 Time vs Fraud Prediction")
            fig4, ax4 = plt.subplots(figsize=(5.5, 3))
            sns.scatterplot(x="Time", y="Amount", hue="Fraud Prediction", data=data, palette="cool", ax=ax4)
            st.pyplot(fig4)

        # Display results
        st.success("✅ Predictions completed.")
        st.dataframe(data[["Fraud Probability", "Fraud Prediction"]].head())

        frauds = data[data["Fraud Prediction"] == 1]
        if not frauds.empty:
            st.subheader("🚨 Flagged Fraudulent Transactions")
            st.write(frauds)
        else:
            st.info("✅ No fraudulent transactions detected in this file.")

        # Download
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results CSV", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

    else:
        missing = set(features) - set(data.columns)
        st.error(f"❌ Missing required columns: {', '.join(missing)}")
