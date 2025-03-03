import streamlit as st
import pickle
import numpy as np



dataset = pickle.load(open("loan_approval_dataset.pkl", "rb"))
pipe = pickle.load(open("loan_approval_gb_model1.pkl", "rb"))



st.title("Loan Approval Prediction")

col1, col2 = st.columns(2)

with col1:
    dep = st.number_input("No of Dependents")

with col2:
    edu = st.selectbox("Education" , ["Graduate","Not Graduate"])

    if edu == "Graduate":
        edu = 1
    else:
        edu = 0





col1, col2 = st.columns(2)

with col1:
    self_e = st.selectbox("Self-Employed" , ["Yes" , "No"])

    if self_e == "Yes":
        self_e = 1
    else:
        self_e = 0

with col2:
    income = st.number_input("Annual Income")



col1 , col2  = st.columns(2)

with col1:
    l_amount = st.number_input("Loan Amount")

with col2:
    l_term = st.number_input("Loan Term")




col1 , col2  = st.columns(2)

with col1:
    c_score = st.number_input("Cibil Score")

with col2:
    r_asset = st.number_input("Residential Assets Value")




col1 , col2  = st.columns(2)

with col1:
    c_asset = st.number_input("Commercial Assets Value")

with col2:
    l_asset = st.number_input("Luxury Assets Value")


b_asset = st.number_input("Bank Assets Value")


if st.button("Predict"):

    query = np.array([dep, edu, self_e, income, l_amount, l_term,c_score, r_asset,c_asset, l_asset, b_asset])


    query = query.reshape(1, 11)

    z = pipe.predict(query)[0]

    if z == 1:
        st.header("LOAN IS APPROVED")

    else:
        st.header("LOAN IS REJECTED")
