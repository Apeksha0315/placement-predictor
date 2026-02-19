import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="AI Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load model
model = joblib.load("models/placement_model.pkl")

# Custom CSS styling
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-title {
        font-size: 18px;
        color: #555;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">🎓 AI Student Placement Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict placement chances using Machine Learning</p>', unsafe_allow_html=True)

st.markdown("---")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📘 Academic Details")
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
    attendance = st.slider("Attendance %", 0, 100, 80)
    backlogs = st.number_input("Backlogs", 0, 10, 0)

with col2:
    st.subheader("💻 Skill & Performance")
    internships = st.number_input("Internships", 0, 10, 1)
    projects = st.number_input("Projects", 0, 10, 2)
    technical = st.slider("Technical Score", 0, 10, 7)
    aptitude = st.slider("Aptitude Score", 0, 100, 70)
    communication = st.slider("Communication Score", 0, 10, 7)

st.markdown("---")

if st.button("🔍 Predict Placement", use_container_width=True):

    features = np.array([[cgpa, internships, projects, technical,
                          aptitude, communication, backlogs, attendance]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    if prediction == 1:
        st.success("🎉 Student is Likely to be Placed!")
        st.progress(int(probability * 100))
        st.write(f"**Placement Probability:** {probability*100:.2f}%")
    else:
        st.error("❌ Student is Not Likely to be Placed")
        st.progress(int((1 - probability) * 100))
        st.write(f"**Placement Probability:** {(1 - probability)*100:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)
