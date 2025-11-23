# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
)

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    try:
        fake = pd.read_csv("Fake.csv")
        real = pd.read_csv("True.csv")
    except FileNotFoundError:
        st.error("❌ Dataset not found. Ensure 'Fake.csv' and 'True.csv' exist in the same folder.")
        return None

    if "text" not in fake.columns or "text" not in real.columns:
        st.error("❌ CSV files must contain a column named 'text'.")
        return None

    fake["label"] = 0
    real["label"] = 1
    data = (
        pd.concat([fake, real], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    return data


# ---------------- Preprocess Text ----------------
def preprocess_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower()


# ---------------- Train Model ----------------
@st.cache_resource
def train_model(data):
    data = data.copy()
    data["cleaned_text"] = data["text"].apply(preprocess_text)
    X = data["cleaned_text"]
    y = data["label"]

    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy


# ---------------- Custom Styling ----------------
st.markdown(
    """
    <style>
    .stTextArea textarea {font-size: 16px;}
    .stSuccess, .stInfo, .stWarning, .stError {font-size: 18px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- App Title ----------------
st.title("📰 Fake News Detection App")
st.caption("Detect whether a news article is **Real** or **Fake** using Machine Learning.")

# ---------------- Input Area ----------------
user_input = st.text_area(
    "✍️ Enter News Article",
    placeholder="Type or paste a news article here...",
    height=200,
)

# ---------------- Buttons ----------------
col1, col2 = st.columns([1, 3])
with col1:
    classify_btn = st.button("🚀 Classify Article", use_container_width=True)
with col2:
    st.empty()

# ---------------- Main Logic ----------------
if classify_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter a news article before classification.")
    else:
        data = load_data()
        if data is not None:
            with st.spinner("Training model and analyzing your input..."):
                model, vectorizer, accuracy = train_model(data)
                cleaned_input = preprocess_text(user_input)
                user_input_vec = vectorizer.transform([cleaned_input])
                prediction = model.predict(user_input_vec)
                result = "✅ Real News" if prediction[0] == 1 else "❌ Fake News"

            st.success(f"**Prediction:** {result}")
            st.info(f"**Model Accuracy:** {accuracy * 100:.2f}%")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Developed by Guruvel ")
