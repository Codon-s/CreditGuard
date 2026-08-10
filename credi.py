import streamlit as st

st.set_page_config(
    page_title="Credit Card Limit Predictor",
    page_icon="💳"
)

st.title("💳 Credit Card Limit Predictor")
st.write("Enter your details to estimate your credit card limit.")

# Inputs
name = st.text_input("Enter your name")

age = st.number_input(
    "Enter your age",
    min_value=18,
    max_value=100,
    value=25
)

credit_score = st.number_input(
    "Enter your credit score",
    min_value=300,
    max_value=900,
    value=700
)

# Prediction
if st.button("Predict Credit Limit"):

    if name == "":
        st.warning("Please enter your name.")

    else:
        # Simple rule-based logic
        if credit_score >= 800:
            limit = 500000

        elif credit_score >= 750:
            limit = 300000

        elif credit_score >= 700:
            limit = 200000

        elif credit_score >= 650:
            limit = 100000

        elif credit_score >= 600:
            limit = 50000

        else:
            limit = 20000

        st.success(f"Hello {name}! 👋")

        st.metric(
            label="Estimated Credit Card Limit",
            value=f"₹{limit:,}"
        )

        st.info(
            f"Based on your age ({age}) and credit score ({credit_score}), "
            f"your estimated limit is ₹{limit:,}."
        )
