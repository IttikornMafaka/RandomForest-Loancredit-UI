import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# ⚙️ CONFIG
# ======================
st.set_page_config(page_title="Loan AI", layout="wide")

# ======================
# 🎨 CSS (Fintech Style)
# ======================
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #eef2f3, #ffffff);
}

.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.header {
    font-size: 36px;
    font-weight: bold;
}

.sub {
    color: gray;
    margin-bottom: 20px;
}

.result-card {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 24px;
    font-weight: bold;
}

.green {
    background: linear-gradient(135deg, #00b09b, #96c93d);
}

.orange {
    background: linear-gradient(135deg, #f7971e, #ffd200);
}

.red {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
}
</style>
""", unsafe_allow_html=True)

# ======================
# ⚡ LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    return pickle.load(open(r'C:\Users\Ittikorn\OneDrive\Desktop\Ai\Machinelearning\ssswork\Notebook\model.pkl', "rb"))

model = load_model()

# ======================
# 📊 FEATURE IMPORTANCE
# ======================
@st.cache_data
def get_feature_importance(model):
    feature_names = ["Dependents", "Self_Employed", "LoanAmount",
                     "Credit_History", "Income", "CoIncome"]

    importance = model.feature_importances_

    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=True)

# ======================
# 🏦 HEADER
# ======================
st.markdown("<div class='header'>🏦 Loan Approval System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>AI-powered credit decision platform</div>", unsafe_allow_html=True)

# ======================
# 📥 INPUT + RESULT
# ======================
col1, col2 = st.columns([1, 1])

# ===== INPUT =====
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📥 Applicant Info")

    with st.form("form"):
        income = st.number_input("รายได้ผู้กู้", min_value=0)
        co_income = st.number_input("รายได้คนค้ำ", min_value=0)
        loan_amount = st.number_input("จำนวนเงินกู้", min_value=0)

        credit_map = {"มี": 1, "ไม่มี": 0}
        emp_map = {"มีงาน": 1, "ไม่มีงาน": 0}

        credit_text = st.selectbox("ประวัติเครดิต", list(credit_map.keys()))
        self_emp_text = st.selectbox("สถานะการทำงาน", list(emp_map.keys()))

        dep_text = st.selectbox("ผู้อยู่ในอุปการะ", ["0", "1", "2", "3+"])

        submit = st.form_submit_button("🚀 Predict")

    st.markdown("</div>", unsafe_allow_html=True)

# ===== RESULT =====
with col2:
    if submit:
        dependents = 3 if dep_text == "3+" else int(dep_text)
        credit_history = credit_map[credit_text]
        self_employed = emp_map[self_emp_text]

        input_data = np.array([[dependents, self_employed, loan_amount,
                                credit_history, income, co_income]])

        prob = model.predict_proba(input_data)[0][1]

        # เลือกสี
        if prob > 0.7:
            color_class = "green"
            label = "APPROVED ✅"
        elif prob > 0.4:
            color_class = "orange"
            label = "REVIEW ⚠️"
        else:
            color_class = "red"
            label = "REJECTED ❌"

        st.markdown(f"""
        <div class='result-card {color_class}'>
            {label} <br><br>
            {prob*100:.2f}% Approval Chance
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(prob * 100))

# ======================
# 📊 FEATURE IMPORTANCE
# ======================
st.markdown("## 📊 Model Insight")

try:
    df = get_feature_importance(model)

    fig, ax = plt.subplots()
    ax.barh(df["Feature"], df["Importance"])

    st.pyplot(fig)

except:
    st.warning("No feature importance available")