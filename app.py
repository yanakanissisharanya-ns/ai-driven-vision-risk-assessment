import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ---------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Vision Risk Assessment",
    layout="wide"
)

st.title("AI-Driven Personalized Vision Risk Assessment")
st.write(
"""
This application predicts **Vision Risk Level** based on
screen time and lifestyle behaviors using Machine Learning.
"""
)

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dataset Overview",
        "Exploratory Data Analysis",
        "Model Training",
        "Vision Risk Prediction"
    ]
)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("vision_screen_time_dataset.csv")
    return data


data = load_data()

# ---------------------------------------------------
# Dataset Overview
# ---------------------------------------------------

if page == "Dataset Overview":

    st.header("Dataset Overview")

    st.subheader("First 5 Rows")

    st.dataframe(data.head())

    st.subheader("Dataset Shape")

    st.write(data.shape)

    st.subheader("Statistical Summary")

    st.write(data.describe())


# ---------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------

df = data.copy()

# Handle Missing Values
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode Categorical Data
label_encoders = {}

categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Target
X = df.drop("Vision_Risk", axis=1)
y = df["Vision_Risk"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------------

if page == "Exploratory Data Analysis":

    st.header("Exploratory Data Analysis Dashboard")

    col1, col2 = st.columns(2)

    # Correlation Heatmap
    with col1:
        st.subheader("Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Screen Time Distribution
    with col2:
        st.subheader("Daily Screen Time Distribution")

        fig = px.histogram(
            data,
            x="Daily_Screen_Time",
            nbins=20,
            color_discrete_sequence=["blue"]
        )

        st.plotly_chart(fig)

    # Sleep Hours vs Vision Risk
    st.subheader("Sleep Hours vs Vision Risk")

    fig = px.box(
        data,
        x="Vision_Risk",
        y="Sleep_Hours",
        color="Vision_Risk"
    )

    st.plotly_chart(fig)

    # Outdoor Activity vs Vision Risk
    st.subheader("Outdoor Activity vs Vision Risk")

    fig = px.box(
        data,
        x="Vision_Risk",
        y="Outdoor_Activity_Hours",
        color="Vision_Risk"
    )

    st.plotly_chart(fig)

# ---------------------------------------------------
# Machine Learning Model Training
# ---------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(),

    "Random Forest": RandomForestClassifier(),

    "KNN": KNeighborsClassifier(),

    "SVM": SVC(probability=True)

}

results = {}

best_model = None
best_accuracy = 0

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model


# ---------------------------------------------------
# Model Results
# ---------------------------------------------------

if page == "Model Training":

    st.header("Model Performance Comparison")

    results_df = pd.DataFrame(
        list(results.items()),
        columns=["Model", "Accuracy"]
    )

    st.dataframe(results_df)

    fig = px.bar(
        results_df,
        x="Model",
        y="Accuracy",
        color="Accuracy",
        title="Model Accuracy Comparison"
    )

    st.plotly_chart(fig)

    st.success(f"Best Model Selected: {type(best_model).__name__}")

    # Feature Importance using Random Forest

    st.subheader("Feature Importance")

    rf = RandomForestClassifier()

    rf.fit(X_train, y_train)

    importance = rf.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig = px.bar(
        feature_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Factors Affecting Vision Risk"
    )

    st.plotly_chart(fig)

# ---------------------------------------------------
# Vision Risk Prediction
# ---------------------------------------------------

if page == "Vision Risk Prediction":

    st.header("Enter Your Details")

    with st.form("prediction_form"):

        age = st.number_input("Age", 10, 80)

        gender = st.selectbox("Gender", ["Male", "Female"])

        screen = st.slider("Daily Screen Time", 0, 16)

        mobile = st.slider("Mobile Usage Hours", 0, 12)

        laptop = st.slider("Laptop Usage Hours", 0, 12)

        night = st.selectbox("Night Screen Usage", ["Yes", "No"])

        break_freq = st.number_input("Break Frequency (minutes)", 5, 120)

        blue = st.selectbox("Blue Light Filter", ["Yes", "No"])

        distance = st.slider("Screen Distance (cm)", 20, 100)

        sleep = st.slider("Sleep Hours", 3, 12)

        outdoor = st.slider("Outdoor Activity Hours", 0, 6)

        strain = st.selectbox("Eye Strain", ["Yes", "No"])

        headache = st.slider("Headache Frequency", 0, 10)

        power = st.number_input("Existing Eye Power")

        submit = st.form_submit_button("Predict Vision Risk")

    if submit:

        input_df = pd.DataFrame({

            "Age":[age],
            "Gender":[gender],
            "Daily_Screen_Time":[screen],
            "Mobile_Usage_Hours":[mobile],
            "Laptop_Usage_Hours":[laptop],
            "Night_Screen_Usage":[night],
            "Break_Frequency":[break_freq],
            "Blue_Light_Filter":[blue],
            "Screen_Distance":[distance],
            "Sleep_Hours":[sleep],
            "Outdoor_Activity_Hours":[outdoor],
            "Eye_Strain":[strain],
            "Headache_Frequency":[headache],
            "Existing_Eye_Power":[power]

        })

        # Encode categorical values
        for col in label_encoders:
            if col in input_df.columns:
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Scale
        input_scaled = scaler.transform(input_df)

        pred = best_model.predict(input_scaled)[0]

        prob = best_model.predict_proba(input_scaled)[0]

        risk_score = int(np.max(prob) * 100)

        risk_label = label_encoders["Vision_Risk"].inverse_transform([pred])[0]

        st.subheader("Prediction Result")

        st.success(f"Vision Risk Level: {risk_label}")

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Vision Risk Score"},
            gauge={'axis': {'range': [0,100]}}
        ))

        st.plotly_chart(fig)

        # Recommendations

        st.subheader("Recommendations")

        if risk_label == "High":

            st.error("High Risk Detected")

            st.write("""
            • Follow the **20-20-20 rule**  
            • Reduce screen time  
            • Maintain **40-70 cm screen distance**  
            • Increase outdoor activity  
            • Avoid night screen exposure  
            • Improve sleep habits
            """)

        elif risk_label == "Medium":

            st.warning("Moderate Risk")

            st.write("""
            • Take frequent screen breaks  
            • Use blue light filter  
            • Increase outdoor activity
            """)

        else:

            st.success("Low Risk")

            st.write("""
            • Maintain healthy screen habits  
            • Continue taking regular breaks
            """)