# =========================================================
# ENTERPRISE CREDIT CARD APPROVAL SYSTEM (CLEAN FINAL)
# =========================================================

import psycopg2
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():
    try:
        conn = psycopg2.connect(
            host=st.secrets["DB_HOST"],
            database=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            port=st.secrets["DB_PORT"],
            sslmode="require"
        )
        return conn

    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None
    
def test_connection():

    conn = get_connection()

    if conn:
            st.success("Supabase Connected Successfully")
            conn.close()
    else:
            st.error("Connection Failed")

if st.button("Test Database Connection"):
        test_connection()
# =========================================================
# USER ACCOUNT FUNCTIONS
# =========================================================

def create_user(username, password):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, password)
        )
        conn.commit()
        st.success("Account Created Successfully 🎉")

    except Exception:
        st.error("Username already exists")

    finally:
        cursor.close()
        conn.close()


def verify_user(username, password):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username=%s AND password=%s",
        (username, password)
    )

    user = cursor.fetchone()

    cursor.close()
    conn.close()

    return user
# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Credit Risk System",
    page_icon="💳",
    layout="wide"
)

# ---------------------------------------------------------
# BACKGROUND STYLE
# ---------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1707075891545-41b982930351?q=80&w=1170&auto=format&fit=crop");
    background-size: cover;
    background-attachment: fixed;
}
html, body, div, p, span, label, h1, h2, h3 {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOGIN SYSTEM
# ---------------------------------------------------------

# ---------------------------------------------------------
# LOGIN / SIGNUP SYSTEM
# ---------------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# SHOW LOGIN ONLY IF USER NOT LOGGED IN
if not st.session_state.logged_in:

    menu = st.sidebar.radio(
        "Account",
        ["Login", "Create Account"]
    )

    if menu == "Login":

        st.title("🔐 Enterprise Login Portal")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            user = verify_user(username, password)

            if user:
                st.session_state.logged_in = True
                st.success("Login Successful ✅")
                st.rerun()

            else:
                st.error("Invalid Username or Password ❌")

    elif menu == "Create Account":

        st.title("📝 Create New Account")

        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            create_user(new_user, new_pass)

    st.stop()
# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("clean_dataset.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

target_column = df.columns[-1]

if df[target_column].dtype == "object":
    df[target_column] = df[target_column].replace({"+": 1, "-": 0})

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ---------------------------------------------------------
# MODEL BUILDING
# ---------------------------------------------------------
log_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=300, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

log_acc = accuracy_score(y_test, log_model.predict(X_test))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("💳 Navigation Panel")

# Model Selection
# ---------------------------------------------------------
# SIDEBAR PANEL (ENTERPRISE VERSION)
# ---------------------------------------------------------

# 👤 User Profile
st.sidebar.markdown("## 👤 User Profile")
st.sidebar.info("""
**Name:** Thirisha  
**Role:** Credit Risk Analyst  
**Access Level:** Admin  
""")

st.sidebar.markdown("---")

# ⚙ System Status
st.sidebar.markdown("## ⚙ System Status")

if rf_acc > log_acc:
    system_status = "Optimal ✅"
else:
    system_status = "Stable ✅"

st.sidebar.success(f"Model Status: {system_status}")
st.sidebar.write(f"Best Accuracy: {max(log_acc, rf_acc)*100:.2f}%")

st.sidebar.markdown("---")

# 🤖 Model Selection
st.sidebar.markdown("## 🤖 Model Selection")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Auto (Best)", "Logistic Regression", "Random Forest"]
)

if model_choice == "Logistic Regression":
    best_model = log_model
    best_accuracy = log_acc
elif model_choice == "Random Forest":
    best_model = rf_model
    best_accuracy = rf_acc
else:
    if rf_acc > log_acc:
        best_model = rf_model
        best_accuracy = rf_acc
    else:
        best_model = log_model
        best_accuracy = log_acc

st.sidebar.markdown("---")

# 📊 Quick Stats
st.sidebar.markdown("## 📊 Quick Stats")
st.sidebar.metric("Total Applicants", len(df))
st.sidebar.metric(
    "Approval Rate",
    f"{(df[target_column].mean()*100):.2f}%"
)

st.sidebar.markdown("---")

# 🕒 System Time
import datetime

st.sidebar.markdown("## 🕒 System Time")
current_time = datetime.datetime.now().strftime("%d-%m-%Y  %H:%M:%S")
st.sidebar.success(current_time)

# Page Navigation
page = st.sidebar.radio(
    "Go to",
    ["📊 Dashboard", "🤖 Prediction"],
    key="main_navigation"
)

# Logout
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# =========================================================
# DASHBOARD PAGE
# =========================================================
if page == "📊 Dashboard":

    st.title("📊 Credit Risk Analytics Dashboard")
    st.success(f"🏆 Selected Model Accuracy: {best_accuracy*100:.2f}%")

    col1, col2 = st.columns(2)

    if "Income" in df.columns:
        with col1:
            fig1 = px.histogram(df, x="Income", color=target_column,
                                title="Income Distribution")
            st.plotly_chart(fig1, use_container_width=True)

    if "Debt" in df.columns:
        with col2:
            fig2 = px.box(df, x=target_column, y="Debt",
                          title="Debt vs Approval")
            st.plotly_chart(fig2, use_container_width=True)

    if "CreditScore" in df.columns:
        fig3 = px.box(df, x=target_column, y="CreditScore",
                      title="Credit Score vs Approval")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

# =========================================================
# PREDICTION PAGE
# =========================================================
elif page == "🤖 Prediction":

    st.title("🤖 Real-Time Credit Card Assessment")

    user_input = {}

    for col in X.columns:
        if X[col].dtype == "object":
            user_input[col] = st.selectbox(col, X[col].unique())
        else:
            user_input[col] = st.slider(
                col,
                float(X[col].min()),
                float(X[col].max()),
                float(X[col].mean())
            )

    input_df = pd.DataFrame([user_input])

    if st.button("Run Assessment"):

        prediction = best_model.predict(input_df)[0]
        probability = best_model.predict_proba(input_df)[0][1]

        if probability >= 0.75:
            risk = "Low Risk 🟢"
        elif probability >= 0.50:
            risk = "Moderate Risk 🟡"
        else:
            risk = "High Risk 🔴"

        if prediction == 1:
            st.success("✅ APPROVED")
        else:
            st.error("❌ REJECTED")

        st.write(f"Approval Probability: {probability:.2f}")
        st.write(f"Risk Category: {risk}")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown("Enterprise Credit Risk Engine | Production ML Pipeline")