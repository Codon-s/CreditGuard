import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Card Analytics",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("💳 Credit Card Customer Analytics")
st.caption("Interactive Customer Spending & Financial Behaviour Dashboard")


# =========================================================
# FILE UPLOAD
# =========================================================

st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Credit Card Excel File",
    type=["xlsx", "xls"]
)


# =========================================================
# STOP IF NO FILE
# =========================================================

if uploaded_file is None:

    st.info("👈 Please upload your Excel dataset from the sidebar.")

    st.markdown("""
    ### Expected Dataset

    Your Excel file should contain columns such as:

    - Customer_ID
    - Age
    - Gender
    - Occupation
    - Employment_Type
    - Residential_Status
    - PAN_Verified
    - KYC_Status
    - Fraud_Flag
    - Loan_Count
    - Annual_Income
    - Avg_Monthly_Spending
    - EMI_Group
    - DTI_Group
    - Savings_Group
    - Investment_Group
    - Transaction_Group
    - Utilization_Group
    """)

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

try:

    df = pd.read_excel(uploaded_file)

except Exception as e:

    st.error("❌ Unable to read the Excel file.")

    st.exception(e)

    st.stop()


# =========================================================
# AGE GROUP
# =========================================================

def age_group(age):

    if age < 20:
        return "Teen"

    elif age < 30:
        return "Young Adult"

    elif age < 50:
        return "Adult"

    elif age < 60:
        return "Middle Aged"

    else:
        return "Senior Citizen"


df["Age_Group"] = df["Age"].apply(age_group)


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Filters")


# Employment

employment_list = sorted(
    df["Employment_Type"].dropna().unique().tolist()
)

selected_employment = st.sidebar.selectbox(
    "Employment Type",
    ["All"] + employment_list
)


if selected_employment != "All":

    temp_df = df[
        df["Employment_Type"] == selected_employment
    ]

else:

    temp_df = df.copy()


# Occupation

occupation_list = sorted(
    temp_df["Occupation"].dropna().unique().tolist()
)

selected_occupation = st.sidebar.selectbox(
    "Occupation",
    ["All"] + occupation_list
)


if selected_occupation != "All":

    temp_df = temp_df[
        temp_df["Occupation"] == selected_occupation
    ]


# Age Group

age_list = [
    "Teen",
    "Young Adult",
    "Adult",
    "Middle Aged",
    "Senior Citizen"
]

selected_age = st.sidebar.selectbox(
    "Age Group",
    ["All"] + age_list
)


if selected_age != "All":

    temp_df = temp_df[
        temp_df["Age_Group"] == selected_age
    ]


final_df = temp_df.copy()


# =========================================================
# KPI
# =========================================================

st.subheader("📌 Customer Overview")


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "👥 Customers",
        f"{len(final_df):,}"
    )


with c2:

    avg_spending = final_df[
        "Avg_Monthly_Spending"
    ].mean()

    st.metric(
        "💳 Avg Monthly Spending",
        f"₹{avg_spending:,.0f}"
    )


with c3:

    avg_income = final_df[
        "Annual_Income"
    ].mean()

    st.metric(
        "💰 Avg Annual Income",
        f"₹{avg_income:,.0f}"
    )


with c4:

    avg_age = final_df["Age"].mean()

    st.metric(
        "Average Age",
        f"{avg_age:.1f}"
    )


st.markdown("---")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Spending",
        "👥 Customer Behaviour",
        "💰 Financial Behaviour",
        "📋 Data"
    ]
)


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.header("📊 Spending Analysis")


    # -----------------------------------------
    # Spending Distribution
    # -----------------------------------------

    fig = px.histogram(
        final_df,
        x="Avg_Monthly_Spending",
        nbins=20,
        title="Monthly Spending Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------
    # Age Group
    # -----------------------------------------

    age_data = (
        final_df
        .groupby("Age_Group")
        ["Avg_Monthly_Spending"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        age_data,
        x="Age_Group",
        y="Avg_Monthly_Spending",
        title="Average Spending by Age Group",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------
    # Occupation
    # -----------------------------------------

    occupation_data = (
        final_df
        .groupby("Occupation")
        ["Avg_Monthly_Spending"]
        .mean()
        .reset_index()
        .sort_values("Avg_Monthly_Spending")
    )

    fig = px.bar(
        occupation_data,
        x="Avg_Monthly_Spending",
        y="Occupation",
        orientation="h",
        title="Average Spending by Occupation",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------
    # Employment Type
    # -----------------------------------------

    employment_data = (
        final_df
        .groupby("Employment_Type")
        ["Avg_Monthly_Spending"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        employment_data,
        x="Employment_Type",
        y="Avg_Monthly_Spending",
        title="Average Spending by Employment Type",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------
    # Top 10 Customers
    # -----------------------------------------

    top10 = final_df.nlargest(
        10,
        "Avg_Monthly_Spending"
    )

    fig = px.bar(
        top10,
        x="Customer_ID",
        y="Avg_Monthly_Spending",
        title="Top 10 High Spending Customers",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------
    # Income vs Spending
    # -----------------------------------------

    fig = px.scatter(
        final_df,
        x="Annual_Income",
        y="Avg_Monthly_Spending",
        hover_data=[
            "Customer_ID",
            "Age",
            "Gender",
            "Occupation"
        ],
        title="Annual Income vs Monthly Spending"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.header("👥 Customer Behaviour")


    col1, col2 = st.columns(2)


    # Gender

    with col1:

        data = (
            final_df
            .groupby("Gender")
            ["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            data,
            x="Gender",
            y="Avg_Monthly_Spending",
            title="Spending by Gender",
            text_auto=".2s"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Residential

    with col2:

        data = (
            final_df
            .groupby("Residential_Status")
            ["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            data,
            x="Residential_Status",
            y="Avg_Monthly_Spending",
            title="Spending by Residential Status",
            text_auto=".2s"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    col1, col2, col3 = st.columns(3)


    # PAN

    with col1:

        data = (
            final_df
            .groupby("PAN_Verified")
            ["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            data,
            x="PAN_Verified",
            y="Avg_Monthly_Spending",
            title="PAN Verification",
            text_auto=".2s"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # KYC

    with col2:

        data = (
            final_df
            .groupby("KYC_Status")
            ["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            data,
            x="KYC_Status",
            y="Avg_Monthly_Spending",
            title="KYC Status",
            text_auto=".2s"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Fraud

    with col3:

        data = (
            final_df
            .groupby("Fraud_Flag")
            ["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            data,
            x="Fraud_Flag",
            y="Avg_Monthly_Spending",
            title="Fraud Flag",
            text_auto=".2s"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Loan Count

    data = (
        final_df
        .groupby("Loan_Count")
        ["Avg_Monthly_Spending"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        data,
        x="Loan_Count",
        y="Avg_Monthly_Spending",
        title="Spending by Loan Count",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.header("💰 Financial Behaviour")


    col1, col2 = st.columns(2)


    # EMI

    with col1:

        data = (
            final_df
            .groupby(
                "EMI_Group",
                observed=True
            )["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            data,
            x="EMI_Group",
            y="Avg_Monthly_Spending",
            title="Spending by EMI Group",
            text_auto=".2s"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # DTI

    with col2:

        data = (
            final_df
            .groupby(
                "DTI_Group",
                observed=True
            )["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            data,
            x="DTI_Group",
            y="Avg_Monthly_Spending",
            markers=True,
            title="Debt-to-Income Ratio"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    col1, col2 = st.columns(2)


    # Savings

    with col1:

        data = (
            final_df
            .groupby(
                "Savings_Group",
                observed=True
            )["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            data,
            x="Savings_Group",
            y="Avg_Monthly_Spending",
            title="Savings Balance",
            text_auto=".2s"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Investment

    with col2:

        data = (
            final_df
            .groupby(
                "Investment_Group",
                observed=True
            )["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            data,
            x="Investment_Group",
            y="Avg_Monthly_Spending",
            title="Investment Value",
            text_auto=".2s"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    col1, col2 = st.columns(2)


    # Transactions

    with col1:

        data = (
            final_df
            .groupby(
                "Transaction_Group",
                observed=True
            )["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            data,
            x="Transaction_Group",
            y="Avg_Monthly_Spending",
            markers=True,
            title="Monthly Transactions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Utilization

    with col2:

        data = (
            final_df
            .groupby(
                "Utilization_Group",
                observed=True
            )["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            data,
            x="Utilization_Group",
            y="Avg_Monthly_Spending",
            title="Credit Utilization",
            text_auto=".2s"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.header("📋 Customer Data")

    st.write(
        f"Showing **{len(final_df):,} customers**"
    )

    st.dataframe(
        final_df,
        use_container_width=True,
        height=500
    )


    # Download

    csv = final_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        "⬇️ Download Filtered Data",
        csv,
        "filtered_credit_card_data.csv",
        "text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "💳 Credit Card Customer Spending Analytics Dashboard"
)
