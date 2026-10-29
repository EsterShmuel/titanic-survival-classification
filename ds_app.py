"""
אפליקציית Streamlit ל-inference והערכה של מודל הטיטאניק.

המשתמש מזין נתיב לקובץ CSV, המודל נטען מהדיסק (נוצר ע"י train.py),
ומוצגות תחזיות. אם בקובץ יש עמודת Survived - מוצגים גם מדדים וגרפים.

הרצה:
    streamlit run ds_app.py
"""
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

from train import TitanicClassifier, preprocess_data

MODEL_PATH = 'models/model.pt'
PREPROCESSOR_PATH = 'models/preprocessor.joblib'
TEST_PATH = 'data/holdout_test.csv'

# העמודות הגולמיות ש-preprocess_data צריכה (Name - בשביל Title)
REQUIRED_COLUMNS = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


# ---------- 1. טעינת המודל וה-preprocessor ----------
@st.cache_resource
def load_model():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = TitanicClassifier(input_dim=len(preprocessor['features']))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model, preprocessor


# ---------- 2. חיזוי ----------
def predict(model, preprocessor, df):
    X = preprocess_data(df, preprocessor['fill_values'])
    # ערך לא מוכר (למשל Sex='M') הופך ל-NaN במיפוי - עוצרים עם הודעה ברורה
    bad_cols = X.columns[X.isnull().any()].tolist()
    if bad_cols:
        raise ValueError(f"unrecognized values in columns {bad_cols}")
    X = preprocessor['scaler'].transform(X)
    with torch.no_grad():
        probs = model(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()
    preds = (probs >= 0.5).astype(int)
    return probs, preds


st.title("Titanic Survival - Inference & Evaluation")

# ---------- 3. קלט מהמשתמש ----------
csv_path = st.text_input("Path to test CSV file", value=TEST_PATH)
if not st.button("Run inference"):
    st.stop()

# ---------- 4. טיפול בשגיאות ----------
# המודל לא קיים - לא הריצו train.py
if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
    st.error("Trained model not found. Run `python train.py` first.")
    st.stop()

# הקובץ לא קיים - טעות בנתיב
if not os.path.exists(csv_path):
    st.error(f"File not found: {csv_path}")
    st.stop()

# הקובץ לא CSV תקין
try:
    test_df = pd.read_csv(csv_path)
except Exception as e:
    st.error(f"Could not read the file as CSV: {e}")
    st.stop()

# חסרות עמודות שהמודל צריך
missing = [c for c in REQUIRED_COLUMNS if c not in test_df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ערכים לא תקינים (למשל Sex='M' במקום 'male') הופכים ל-NaN וה-scaler זורק ValueError
model, preprocessor = load_model()
try:
    probs, preds = predict(model, preprocessor, test_df)
except ValueError as e:
    st.error(f"Invalid values in the data: {e}")
    st.stop()

st.success(f"Inference completed on {len(test_df)} passengers")

# ---------- 5. מדדים וגרפים - רק אם יש תשובות אמיתיות ----------
if 'Survived' in test_df.columns:
    y_true = test_df['Survived'].values

    st.subheader("Evaluation metrics")
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{accuracy_score(y_true, preds):.3f}")
    cols[1].metric("Precision", f"{precision_score(y_true, preds):.3f}")
    cols[2].metric("Recall", f"{recall_score(y_true, preds):.3f}")
    cols[3].metric("F1", f"{f1_score(y_true, preds):.3f}")
    cols[4].metric("ROC-AUC", f"{roc_auc_score(y_true, probs):.3f}")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_true, preds), annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    with col2:
        fpr, tpr, _ = roc_curve(y_true, probs)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, probs):.3f}")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
else:
    st.info("No 'Survived' column in this file - showing predictions only "
            "(metrics require the true labels).")

# ---------- 6. טבלת תחזיות ----------
st.subheader("Predictions")
cols = [c for c in ['PassengerId', 'Name', 'Sex', 'Age', 'Pclass', 'Survived'] if c in test_df.columns]
result = test_df[cols].copy()
result['Predicted'] = preds
result['Probability'] = probs.round(3)
st.dataframe(result)
