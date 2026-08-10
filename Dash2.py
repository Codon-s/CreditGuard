import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Card Spending Analytics",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.block-container {
    padding-top: 2rem;
}

.dashboard-title {
    font-size: 40px;
    font-weight: 700;
    color: #17365D;
    text-align: center;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 30px;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_excel("../Dataset/Credir_Card_Bank.xlsx")

    return df


df = load_data()


# =========================================================
# AGE GROUP FUNCTION
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
# TITLE
# =========================================================

st.markdown(
    '<div class="dashboard-title">💳 Credit Card Spending Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Interactive Customer Spending & Financial Behaviour Dashboard'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.title("🎛️ Dashboard Filters")

st.sidebar.markdown("### Customer Segmentation")


# Employment Type

employment_options = ["All"] + sorted(
    df["Employment_Type"].dropna().unique().tolist()
)

selected_emp = st.sidebar.selectbox(
    "Employment Type",
    employment_options
)


if selected_emp != "All":

    filtered_emp = df[
        df["Employment_Type"] == selected_emp
    ]

else:

    filtered_emp = df.copy()


# Occupation

occupation_options = ["All"] + sorted(
    filtered_emp["Occupation"].dropna().unique().tolist()
)

selected_occ = st.sidebar.selectbox(
    "Occupation",
    occupation_options
)


if selected_occ != "All":

    filtered_occ = filtered_emp[
        filtered_emp["Occupation"] == selected_occ
    ]

else:

    filtered_occ = filtered_emp.copy()


# Age Group

age_options = ["All"] + [
    "Teen",
    "Young Adult",
    "Adult",
    "Middle Aged",
    "Senior Citizen"
]

selected_age = st.sidebar.selectbox(
    "Age Group",
    age_options
)


if selected_age != "All":

    final_df = filtered_occ[
        filtered_occ["Age_Group"] == selected_age
    ]

else:

    final_df = filtered_occ.copy()


# =========================================================
# RESET / INFO
# =========================================================

st.sidebar.markdown("---")

st.sidebar.info(
    f"Showing **{len(final_df):,} customers**"
)


# =========================================================
# KPI SECTION
# =========================================================

st.markdown("## 📌 Customer Overview")

if len(final_df) > 0:

    total_customers = len(final_df)

    avg_spending = final_df[
        "Avg_Monthly_Spending"
    ].mean()

    avg_income = final_df[
        "Annual_Income"
    ].mean()

    total_spending = final_df[
        "Avg_Monthly_Spending"
    ].sum()

    avg_age = final_df["Age"].mean()

else:

    total_customers = 0
    avg_spending = 0
    avg_income = 0
    total_spending = 0
    avg_age = 0


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "👥 Customers",
        f"{total_customers:,}"
    )


with col2:

    st.metric(
        "💳 Avg Monthly Spending",
        f"₹{avg_spending:,.0f}"
    )


with col3:

    st.metric(
        "💰 Avg Annual Income",
        f"₹{avg_income:,.0f}"
    )


with col4:

    st.metric(
        "📊 Total Spending",
        f"₹{total_spending:,.0f}"
    )


with col5:

    st.metric(
        "🎂 Average Age",
        f"{avg_age:.1f}"
    )


st.markdown("---")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Spending Overview",
    "👥 Customer Behaviour",
    "💰 Financial Behaviour",
    "📋 Customer Data"
])


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader("📊 Spending Overview")


    # -----------------------------------------
    # 1. Monthly Spending Distribution
    # -----------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        fig = px.histogram(
            final_df,
            x="Avg_Monthly_Spending",
            nbins=20,
            title="Monthly Spending Distribution",
            labels={
                "Avg_Monthly_Spending":
                "Average Monthly Spending"
            }
        )

        fig.update_layout(
            height=450,
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------
    # 2. Spending by Age Group
    # -----------------------------------------

    with col2:

        age = (
            final_df
            .groupby(
                "Age_Group",
                observed=True
            )["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            age,
            x="Age_Group",
            y="Avg_Monthly_Spending",
            title="Average Spending by Age Group",
            text_auto=".2s"
        )

        fig.update_layout(
            height=450,
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------
    # 3. Occupation
    # -----------------------------------------

    occupation = (
        final_df
        .groupby("Occupation")
        ["Avg_Monthly_Spending"]
        .mean()
        .reset_index()
        .sort_values(
            "Avg_Monthly_Spending"
        )
    )

    fig = px.bar(
        occupation,
        x="Avg_Monthly_Spending",
        y="Occupation",
        orientation="h",
        title="Average Spending by Occupation",
        text_auto=".2s"
    )

    fig.update_layout(
        height=500,
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------
    # 4. Employment Type
    # -----------------------------------------

    employment = (
        final_df
        .groupby("Employment_Type")
        ["Avg_Monthly_Spending"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        employment,
        x="Employment_Type",
        y="Avg_Monthly_Spending",
        title="Average Spending by Employment Type",
        text_auto=".2s"
    )

    fig.update_layout(
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------
    # 5. Top 10 Customers
    # -----------------------------------------

    top10 = (
        final_df
        .nlargest(
            10,
            "Avg_Monthly_Spending"
        )
    )

    fig = px.bar(
        top10,
        x="Customer_ID",
        y="Avg_Monthly_Spending",
        title="Top 10 High Spending Customers",
        text_auto=".2s"
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------
    # 6. Income vs Spending
    # -----------------------------------------

    fig = px.scatter(
        final_df,
        x="Annual_Income",
        y="Avg_Monthly_Spending",
        title="Annual Income vs Monthly Spending",
        hover_data=[
            "Customer_ID",
            "Age",
            "Occupation",
            "Employment_Type"
        ]
    )

    fig.update_layout(
        height=500,
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader("👥 Customer Behaviour Analysis")


    col1, col2 = st.columns(2)


    # Gender

    with col1:

        gender = (
            final_df
            .groupby("Gender")
            ["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            gender,
            x="Gender",
            y="Avg_Monthly_Spending",
            title="Spending by Gender",
            text_auto=".2s"
        )

        fig.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Residential Status

    with col2:

        residential = (
            final_df
            .groupby("Residential_Status")
            ["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            residential,
            x="Residential_Status",
            y="Avg_Monthly_Spending",
            title="Spending by Residential Status",
            text_auto=".2s"
        )

        fig.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    col1, col2, col3 = st.columns(3)


    # PAN

    with col1:

        pan = (
            final_df
            .groupby("PAN_Verified")
            ["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            pan,
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

        kyc = (
            final_df
            .groupby("KYC_Status")
            ["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            kyc,
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

        fraud = (
            final_df
            .groupby("Fraud_Flag")
            ["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            fraud,
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

    loan = (
        final_df
        .groupby("Loan_Count")
        ["Avg_Monthly_Spending"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        loan,
        x="Loan_Count",
        y="Avg_Monthly_Spending",
        title="Spending by Loan Count",
        text_auto=".2s"
    )

    fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.subheader("💰 Financial Behaviour Analysis")


    col1, col2 = st.columns(2)


    # EMI

    with col1:

        emi = (
            final_df
            .groupby(
                "EMI_Group",
                observed=True
            )["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            emi,
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

        dti = (
            final_df
            .groupby(
                "DTI_Group",
                observed=True
            )["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            dti,
            x="DTI_Group",
            y="Avg_Monthly_Spending",
            markers=True,
            title="Debt-to-Income Ratio"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    col1, col2, col3 = st.columns(3)


    # Savings

    with col1:

        saving = (
            final_df
            .groupby(
                "Savings_Group",
                observed=True
            )["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            saving,
            x="Avg_Monthly_Spending",
            y="Savings_Group",
            orientation="h",
            title="Savings Balance",
            text_auto=".2s"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Investment

    with col2:

        invest = (
            final_df
            .groupby(
                "Investment_Group",
                observed=True
            )["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            invest,
            x="Investment_Group",
            y="Avg_Monthly_Spending",
            title="Investment Value",
            text_auto=".2s"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Transactions

    with col3:

        tran = (
            final_df
            .groupby(
                "Transaction_Group",
                observed=True
            )["Avg_Monthly_Spending"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            tran,
            x="Transaction_Group",
            y="Avg_Monthly_Spending",
            markers=True,
            title="Monthly Transactions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Credit Utilization

    util = (
        final_df
        .groupby(
            "Utilization_Group",
            observed=True
        )["Avg_Monthly_Spending"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        util,
        x="Utilization_Group",
        y="Avg_Monthly_Spending",
        title="Credit Utilization",
        text_auto=".2s"
    )

    fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# TAB 4 - CUSTOMER DATA
# =========================================================

with tab4:

    st.subheader("📋 Filtered Customer Dataset")

    st.write(
        f"Displaying **{len(final_df):,} customers**"
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
        label="⬇️ Download Filtered Data",
        data=csv,
        file_name="filtered_credit_card_customers.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "<center>💳 Credit Card Customer Analytics Dashboard</center>",
    unsafe_allow_html=True
)
