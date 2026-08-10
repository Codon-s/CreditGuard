import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CreditGuard - Credit Card Risk Analysis",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("💳 CreditGuard")
st.subheader("Credit Card Risk Analysis Dashboard")

st.markdown(
    """
    This interactive dashboard analyzes customer credit risk based on
    demographic, financial, employment and credit-status information.
    Use the filters in the sidebar to explore the data dynamically.
    """
)

# =========================================================
# LOAD DATA FROM GITHUB ZIP FILES
# =========================================================

@st.cache_data
def load_data():

    # GitHub raw URLs
    applications_url = (
        "https://github.com/Abhishek131004/CreditGuard/"
        "raw/refs/heads/main/Applications_cc.csv.zip"
    )

    credit_record_url = (
        "https://github.com/Abhishek131004/CreditGuard/"
        "raw/refs/heads/main/credit_record%20_cc.csv.zip"
    )

    # Read CSV directly from ZIP
    a = pd.read_csv(
        applications_url,
        compression="zip"
    )

    b = pd.read_csv(
        credit_record_url,
        compression="zip"
    )

    # =====================================================
    # FIND COMMON CUSTOMERS
    # =====================================================

    common = a[a["ID"].isin(b["ID"])]

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    a_cleaned = a.drop_duplicates(
        subset=["ID"],
        keep="first"
    )

    b_cleaned = b.drop_duplicates(
        subset=["ID"],
        keep="first"
    )

    # =====================================================
    # MERGE DATASETS
    # =====================================================

    df = pd.merge(
        a_cleaned,
        b_cleaned,
        on="ID",
        how="inner"
    )

    # =====================================================
    # DATA CLEANING
    # =====================================================

    # Education
    df["NAME_EDUCATION_TYPE"] = df[
        "NAME_EDUCATION_TYPE"
    ].replace(
        "Secondary / secondary special",
        "Secondary special"
    )

    # Family status
    df["NAME_FAMILY_STATUS"] = df[
        "NAME_FAMILY_STATUS"
    ].replace(
        "Single / not married",
        "Single"
    )

    # Housing
    df["NAME_HOUSING_TYPE"] = df[
        "NAME_HOUSING_TYPE"
    ].replace(
        "House / apartment",
        "House"
    )

    # Occupation
    df["OCCUPATION_TYPE"] = df[
        "OCCUPATION_TYPE"
    ].fillna("Unknown")

    df["OCCUPATION_TYPE"] = df[
        "OCCUPATION_TYPE"
    ].replace(
        "Waiters/barmen staff",
        "Waiters"
    )

    # =====================================================
    # CREDIT RISK
    # =====================================================

    df["STATUS"] = df["STATUS"].astype(str)

    status_map = {
        "0": 0,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
        "5": 1,
        "C": 0,
        "X": 0
    }

    df["CREDIT_RISK"] = df[
        "STATUS"
    ].map(status_map)

    # Remove records with unmapped status
    df = df.dropna(
        subset=["CREDIT_RISK"]
    )

    df["CREDIT_RISK"] = df[
        "CREDIT_RISK"
    ].astype(int)

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    df["AGE"] = (
        -df["DAYS_BIRTH"] // 365
    ).astype(int)

    df["EMPLOYMENT_YEARS"] = (
        -df["DAYS_EMPLOYED"] / 365
    )

    # Remove unrealistic negative employment years
    df.loc[
        df["EMPLOYMENT_YEARS"] < 0,
        "EMPLOYMENT_YEARS"
    ] = np.nan

    return df


# =========================================================
# LOAD DATA
# =========================================================

with st.spinner("Loading CreditGuard data..."):

    try:
        df = load_data()

    except Exception as e:

        st.error(
            "Unable to load the datasets from GitHub."
        )

        st.code(str(e))

        st.stop()


# =========================================================
# DATASET INFORMATION
# =========================================================

with st.expander("📁 Dataset Information"):

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Total Columns",
        len(df.columns)
    )

    col3.metric(
        "Unique Customers",
        f"{df['ID'].nunique():,}"
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🔍 Dashboard Filters")

st.sidebar.markdown(
    "Use the filters below to explore specific customer segments."
)

# =========================================================
# GENDER
# =========================================================

gender_options = ["All"] + sorted(
    df["CODE_GENDER"]
    .dropna()
    .unique()
    .tolist()
)

selected_gender = st.sidebar.selectbox(
    "Gender",
    gender_options
)


# =========================================================
# CAR OWNERSHIP
# =========================================================

car_options = ["All"] + sorted(
    df["FLAG_OWN_CAR"]
    .dropna()
    .unique()
    .tolist()
)

selected_car = st.sidebar.selectbox(
    "Owns Car",
    car_options
)


# =========================================================
# REAL ESTATE
# =========================================================

realty_options = ["All"] + sorted(
    df["FLAG_OWN_REALTY"]
    .dropna()
    .unique()
    .tolist()
)

selected_realty = st.sidebar.selectbox(
    "Owns Property",
    realty_options
)


# =========================================================
# INCOME TYPE
# =========================================================

income_options = ["All"] + sorted(
    df["NAME_INCOME_TYPE"]
    .dropna()
    .unique()
    .tolist()
)

selected_income_type = st.sidebar.selectbox(
    "Income Type",
    income_options
)


# =========================================================
# EDUCATION
# =========================================================

education_options = ["All"] + sorted(
    df["NAME_EDUCATION_TYPE"]
    .dropna()
    .unique()
    .tolist()
)

selected_education = st.sidebar.selectbox(
    "Education",
    education_options
)


# =========================================================
# FAMILY STATUS
# =========================================================

family_options = ["All"] + sorted(
    df["NAME_FAMILY_STATUS"]
    .dropna()
    .unique()
    .tolist()
)

selected_family = st.sidebar.selectbox(
    "Family Status",
    family_options
)


# =========================================================
# HOUSING
# =========================================================

housing_options = ["All"] + sorted(
    df["NAME_HOUSING_TYPE"]
    .dropna()
    .unique()
    .tolist()
)

selected_housing = st.sidebar.selectbox(
    "Housing Type",
    housing_options
)


# =========================================================
# OCCUPATION
# =========================================================

occupation_options = ["All"] + sorted(
    df["OCCUPATION_TYPE"]
    .dropna()
    .unique()
    .tolist()
)

selected_occupation = st.sidebar.selectbox(
    "Occupation",
    occupation_options
)


# =========================================================
# CREDIT RISK
# =========================================================

risk_options = [
    "All",
    "Good Risk",
    "Bad Risk"
]

selected_risk = st.sidebar.selectbox(
    "Credit Risk",
    risk_options
)


# =========================================================
# AGE FILTER
# =========================================================

st.sidebar.subheader("📊 Numerical Filters")

age_min = int(df["AGE"].min())
age_max = int(df["AGE"].max())

age_range = st.sidebar.slider(
    "Age Range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max)
)


# =========================================================
# INCOME FILTER
# =========================================================

income_min = float(
    df["AMT_INCOME_TOTAL"].min()
)

income_max = float(
    df["AMT_INCOME_TOTAL"].max()
)

income_range = st.sidebar.slider(
    "Annual Income",
    min_value=income_min,
    max_value=income_max,
    value=(income_min, income_max)
)


# =========================================================
# FAMILY MEMBER FILTER
# =========================================================

family_min = int(
    df["CNT_FAM_MEMBERS"].min()
)

family_max = int(
    df["CNT_FAM_MEMBERS"].max()
)

family_range = st.sidebar.slider(
    "Family Members",
    min_value=family_min,
    max_value=family_max,
    value=(family_min, family_max)
)


# =========================================================
# RESET FILTER INFORMATION
# =========================================================

st.sidebar.markdown("---")

st.sidebar.info(
    "💡 Change any filter to dynamically update "
    "all KPIs, charts and customer records."
)


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_df = df.copy()


# Gender
if selected_gender != "All":

    filtered_df = filtered_df[
        filtered_df["CODE_GENDER"]
        == selected_gender
    ]


# Car
if selected_car != "All":

    filtered_df = filtered_df[
        filtered_df["FLAG_OWN_CAR"]
        == selected_car
    ]


# Property
if selected_realty != "All":

    filtered_df = filtered_df[
        filtered_df["FLAG_OWN_REALTY"]
        == selected_realty
    ]


# Income type
if selected_income_type != "All":

    filtered_df = filtered_df[
        filtered_df["NAME_INCOME_TYPE"]
        == selected_income_type
    ]


# Education
if selected_education != "All":

    filtered_df = filtered_df[
        filtered_df["NAME_EDUCATION_TYPE"]
        == selected_education
    ]


# Family status
if selected_family != "All":

    filtered_df = filtered_df[
        filtered_df["NAME_FAMILY_STATUS"]
        == selected_family
    ]


# Housing
if selected_housing != "All":

    filtered_df = filtered_df[
        filtered_df["NAME_HOUSING_TYPE"]
        == selected_housing
    ]


# Occupation
if selected_occupation != "All":

    filtered_df = filtered_df[
        filtered_df["OCCUPATION_TYPE"]
        == selected_occupation
    ]


# Age
filtered_df = filtered_df[
    (filtered_df["AGE"] >= age_range[0])
    &
    (filtered_df["AGE"] <= age_range[1])
]


# Income
filtered_df = filtered_df[
    (filtered_df["AMT_INCOME_TOTAL"] >= income_range[0])
    &
    (filtered_df["AMT_INCOME_TOTAL"] <= income_range[1])
]


# Family members
filtered_df = filtered_df[
    (filtered_df["CNT_FAM_MEMBERS"] >= family_range[0])
    &
    (filtered_df["CNT_FAM_MEMBERS"] <= family_range[1])
]


# Credit risk
if selected_risk == "Good Risk":

    filtered_df = filtered_df[
        filtered_df["CREDIT_RISK"] == 0
    ]

elif selected_risk == "Bad Risk":

    filtered_df = filtered_df[
        filtered_df["CREDIT_RISK"] == 1
    ]


# =========================================================
# MAIN DASHBOARD
# =========================================================

st.markdown("---")

st.header("📊 Credit Risk Overview")


# =========================================================
# KPI CALCULATIONS
# =========================================================

total_customers = len(filtered_df)

good_risk = (
    filtered_df["CREDIT_RISK"] == 0
).sum()

bad_risk = (
    filtered_df["CREDIT_RISK"] == 1
).sum()

if total_customers > 0:

    bad_risk_percentage = (
        bad_risk / total_customers
    ) * 100

    good_risk_percentage = (
        good_risk / total_customers
    ) * 100

    average_income = (
        filtered_df["AMT_INCOME_TOTAL"].mean()
    )

    average_age = (
        filtered_df["AGE"].mean()
    )

else:

    bad_risk_percentage = 0
    good_risk_percentage = 0
    average_income = 0
    average_age = 0


# =========================================================
# KPI CARDS
# =========================================================

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "👥 Customers",
    f"{total_customers:,}"
)

col2.metric(
    "🟢 Good Risk",
    f"{good_risk:,}"
)

col3.metric(
    "🔴 Bad Risk",
    f"{bad_risk:,}"
)

col4.metric(
    "⚠️ Bad Risk %",
    f"{bad_risk_percentage:.2f}%"
)

col5.metric(
    "💰 Avg Income",
    f"{average_income:,.0f}"
)


# =========================================================
# NO DATA
# =========================================================

if filtered_df.empty:

    st.warning(
        "⚠️ No customers match the selected filters."
    )

    st.info(
        "Try changing the filters in the sidebar."
    )

    st.stop()


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Risk Overview",
        "👥 Customer Insights",
        "💼 Occupation & Income",
        "🔗 Relationships",
        "📋 Customer Data"
    ]
)


# =========================================================
# TAB 1
# RISK OVERVIEW
# =========================================================

with tab1:

    st.subheader("Credit Risk Distribution")

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # CREDIT RISK
    # -----------------------------------------------------

    with col1:

        risk_counts = pd.DataFrame({
            "Risk": [
                "Good Risk",
                "Bad Risk"
            ],
            "Customers": [
                good_risk,
                bad_risk
            ]
        })

        st.bar_chart(
            risk_counts.set_index("Risk")
        )

    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    with col2:

        status_counts = (
            filtered_df["STATUS"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            status_counts
        )

    # -----------------------------------------------------
    # INCOME VS RISK
    # -----------------------------------------------------

    st.subheader(
        "Annual Income vs Credit Risk"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sb.boxplot(
        data=filtered_df,
        x="CREDIT_RISK",
        y="AMT_INCOME_TOTAL",
        ax=ax
    )

    ax.set_title(
        "Annual Income vs Credit Risk"
    )

    ax.set_xlabel(
        "Credit Risk (0 = Good, 1 = Bad)"
    )

    ax.set_ylabel(
        "Annual Income"
    )

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# TAB 2
# CUSTOMER INSIGHTS
# =========================================================

with tab2:

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # AGE
    # -----------------------------------------------------

    with col1:

        st.subheader(
            "Age Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sb.histplot(
            data=filtered_df,
            x="AGE",
            bins=25,
            kde=True,
            ax=ax
        )

        ax.set_title(
            "Customer Age Distribution"
        )

        ax.set_xlabel(
            "Age"
        )

        st.pyplot(fig)

        plt.close(fig)

    # -----------------------------------------------------
    # FAMILY MEMBERS
    # -----------------------------------------------------

    with col2:

        st.subheader(
            "Family Member Distribution"
        )

        family_counts = (
            filtered_df[
                "CNT_FAM_MEMBERS"
            ]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            family_counts
        )

    # -----------------------------------------------------
    # EMPLOYMENT
    # -----------------------------------------------------

    st.subheader(
        "Employment Length Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sb.histplot(
        data=filtered_df,
        x="EMPLOYMENT_YEARS",
        bins=30,
        kde=True,
        ax=ax
    )

    ax.set_title(
        "Employment Length Distribution"
    )

    ax.set_xlabel(
        "Years Employed"
    )

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# TAB 3
# OCCUPATION & INCOME
# =========================================================

with tab3:

    # -----------------------------------------------------
    # OCCUPATION RISK
    # -----------------------------------------------------

    st.subheader(
        "Credit Risk Percentage by Occupation"
    )

    occupation_risk = pd.crosstab(
        filtered_df["OCCUPATION_TYPE"],
        filtered_df["CREDIT_RISK"],
        normalize="index"
    ) * 100

    occupation_risk = occupation_risk.rename(
        columns={
            0: "Good Risk %",
            1: "Bad Risk %"
        }
    )

    st.bar_chart(
        occupation_risk
    )

    # -----------------------------------------------------
    # INCOME TYPE RISK
    # -----------------------------------------------------

    st.subheader(
        "Credit Risk Percentage by Income Type"
    )

    income_risk = pd.crosstab(
        filtered_df["NAME_INCOME_TYPE"],
        filtered_df["CREDIT_RISK"],
        normalize="index"
    ) * 100

    income_risk = income_risk.rename(
        columns={
            0: "Good Risk %",
            1: "Bad Risk %"
        }
    )

    st.bar_chart(
        income_risk
    )

    # -----------------------------------------------------
    # AVERAGE INCOME
    # -----------------------------------------------------

    st.subheader(
        "Average Income by Credit Risk"
    )

    avg_income_risk = (
        filtered_df
        .groupby("CREDIT_RISK")[
            "AMT_INCOME_TOTAL"
        ]
        .mean()
    )

    avg_income_risk.index = [
        "Good Risk"
        if x == 0
        else "Bad Risk"
        for x in avg_income_risk.index
    ]

    st.bar_chart(
        avg_income_risk
    )


# =========================================================
# TAB 4
# RELATIONSHIPS
# =========================================================

with tab4:

    st.subheader(
        "Numerical Feature Correlation"
    )

    numerical_cols = filtered_df.select_dtypes(
        include=["int64", "float64"]
    )

    if len(numerical_cols.columns) > 1:

        correlation = numerical_cols.corr()

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        sb.heatmap(
            correlation,
            annot=True,
            cmap="RdBu_r",
            fmt=".2f",
            ax=ax
        )

        ax.set_title(
            "Correlation Matrix"
        )

        st.pyplot(fig)

        plt.close(fig)

    # -----------------------------------------------------
    # INCOME DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "Annual Income Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sb.histplot(
        data=filtered_df,
        x="AMT_INCOME_TOTAL",
        bins=30,
        kde=True,
        ax=ax
    )

    ax.set_title(
        "Distribution of Annual Income"
    )

    ax.set_xlabel(
        "Annual Income"
    )

    st.pyplot(fig)

    plt.close(fig)

    # -----------------------------------------------------
    # INCOME BOX PLOT
    # -----------------------------------------------------

    st.subheader(
        "Income Outlier Analysis"
    )

    fig, ax = plt.subplots(
        figsize=(10, 3)
    )

    sb.boxplot(
        x=filtered_df["AMT_INCOME_TOTAL"],
        ax=ax
    )

    ax.set_title(
        "Boxplot of Annual Income"
    )

    ax.set_xlabel(
        "Annual Income"
    )

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# TAB 5
# CUSTOMER DATA
# =========================================================

with tab5:

    st.subheader(
        "Filtered Customer Records"
    )

    display_columns = [
        "ID",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "CNT_CHILDREN",
        "AMT_INCOME_TOTAL",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "CNT_FAM_MEMBERS",
        "AGE",
        "EMPLOYMENT_YEARS",
        "STATUS",
        "CREDIT_RISK"
    ]

    available_columns = [
        col
        for col in display_columns
        if col in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[available_columns],
        use_container_width=True,
        height=500
    )

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    csv = filtered_df[
        available_columns
    ].to_csv(index=False)

    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv,
        file_name="CreditGuard_filtered_data.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "CreditGuard | Credit Card Risk Analysis | "
    "Python • Pandas • Seaborn • Matplotlib • Streamlit"
)
