import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Credit Card Risk Analysis",
    page_icon="💳",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM COLORS
# ---------------------------------------------------------

colors = sb.color_palette("crest", 2)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("💳 Credit Card Risk Analysis Dashboard")
st.markdown(
    "Interactive analysis of customer demographics, income, employment "
    "and credit risk."
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    a = pd.read_csv("Applications_cc.csv")
    b = pd.read_csv("credit_record _cc.csv")

    # Keep common customers
    common = a[a["ID"].isin(b["ID"])]

    # Remove duplicate IDs
    a_cleaned = a.drop_duplicates(
        subset=["ID"],
        keep="first"
    )

    b_cleaned = b.drop_duplicates(
        subset=["ID"],
        keep="first"
    )

    # Merge datasets
    df = pd.merge(
        a_cleaned,
        b_cleaned,
        on="ID",
        how="inner"
    )

    # -----------------------------------------------------
    # DATA CLEANING
    # -----------------------------------------------------

    df["NAME_EDUCATION_TYPE"] = df["NAME_EDUCATION_TYPE"].replace(
        "Secondary / secondary special",
        "Secondary special"
    )

    df["NAME_FAMILY_STATUS"] = df["NAME_FAMILY_STATUS"].replace(
        "Single / not married",
        "Single"
    )

    df["NAME_HOUSING_TYPE"] = df["NAME_HOUSING_TYPE"].replace(
        "House / apartment",
        "House"
    )

    df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna(
        "Unknown"
    )

    df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].replace(
        "Waiters/barmen staff",
        "Waiters"
    )

    # -----------------------------------------------------
    # CREDIT RISK
    # -----------------------------------------------------

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

    df["CREDIT_RISK"] = df["STATUS"].map(status_map)

    # Remove unknown status records
    df = df.dropna(subset=["CREDIT_RISK"])

    df["CREDIT_RISK"] = df["CREDIT_RISK"].astype(int)

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    df["AGE"] = (-df["DAYS_BIRTH"] // 365).astype(int)

    df["EMPLOYMENT_YEARS"] = (
        -df["DAYS_EMPLOYED"] / 365
    )

    # Fix unrealistic employment values
    df.loc[
        df["EMPLOYMENT_YEARS"] < 0,
        "EMPLOYMENT_YEARS"
    ] = np.nan

    return df


df = load_data()

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------

st.sidebar.header("🔍 Filters")

# Gender
gender_options = ["All"] + sorted(
    df["CODE_GENDER"].dropna().unique().tolist()
)

selected_gender = st.sidebar.selectbox(
    "Gender",
    gender_options
)

# Car ownership
car_options = ["All"] + sorted(
    df["FLAG_OWN_CAR"].dropna().unique().tolist()
)

selected_car = st.sidebar.selectbox(
    "Owns Car",
    car_options
)

# Property ownership
realty_options = ["All"] + sorted(
    df["FLAG_OWN_REALTY"].dropna().unique().tolist()
)

selected_realty = st.sidebar.selectbox(
    "Owns Property",
    realty_options
)

# Income type
income_options = ["All"] + sorted(
    df["NAME_INCOME_TYPE"].dropna().unique().tolist()
)

selected_income_type = st.sidebar.selectbox(
    "Income Type",
    income_options
)

# Education
education_options = ["All"] + sorted(
    df["NAME_EDUCATION_TYPE"].dropna().unique().tolist()
)

selected_education = st.sidebar.selectbox(
    "Education",
    education_options
)

# Family status
family_options = ["All"] + sorted(
    df["NAME_FAMILY_STATUS"].dropna().unique().tolist()
)

selected_family = st.sidebar.selectbox(
    "Family Status",
    family_options
)

# Housing
housing_options = ["All"] + sorted(
    df["NAME_HOUSING_TYPE"].dropna().unique().tolist()
)

selected_housing = st.sidebar.selectbox(
    "Housing Type",
    housing_options
)

# Occupation
occupation_options = ["All"] + sorted(
    df["OCCUPATION_TYPE"].dropna().unique().tolist()
)

selected_occupation = st.sidebar.selectbox(
    "Occupation",
    occupation_options
)

# Credit risk
risk_options = ["All", "Good Risk", "Bad Risk"]

selected_risk = st.sidebar.selectbox(
    "Credit Risk",
    risk_options
)

# ---------------------------------------------------------
# NUMERICAL FILTERS
# ---------------------------------------------------------

st.sidebar.subheader("Numerical Filters")

min_age = int(df["AGE"].min())
max_age = int(df["AGE"].max())

age_range = st.sidebar.slider(
    "Age",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

min_income = float(df["AMT_INCOME_TOTAL"].min())
max_income = float(df["AMT_INCOME_TOTAL"].max())

income_range = st.sidebar.slider(
    "Annual Income",
    min_value=min_income,
    max_value=max_income,
    value=(min_income, max_income)
)

# ---------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------

filtered_df = df.copy()

if selected_gender != "All":
    filtered_df = filtered_df[
        filtered_df["CODE_GENDER"] == selected_gender
    ]

if selected_car != "All":
    filtered_df = filtered_df[
        filtered_df["FLAG_OWN_CAR"] == selected_car
    ]

if selected_realty != "All":
    filtered_df = filtered_df[
        filtered_df["FLAG_OWN_REALTY"] == selected_realty
    ]

if selected_income_type != "All":
    filtered_df = filtered_df[
        filtered_df["NAME_INCOME_TYPE"] == selected_income_type
    ]

if selected_education != "All":
    filtered_df = filtered_df[
        filtered_df["NAME_EDUCATION_TYPE"] == selected_education
    ]

if selected_family != "All":
    filtered_df = filtered_df[
        filtered_df["NAME_FAMILY_STATUS"] == selected_family
    ]

if selected_housing != "All":
    filtered_df = filtered_df[
        filtered_df["NAME_HOUSING_TYPE"] == selected_housing
    ]

if selected_occupation != "All":
    filtered_df = filtered_df[
        filtered_df["OCCUPATION_TYPE"] == selected_occupation
    ]

filtered_df = filtered_df[
    (filtered_df["AGE"] >= age_range[0]) &
    (filtered_df["AGE"] <= age_range[1])
]

filtered_df = filtered_df[
    (filtered_df["AMT_INCOME_TOTAL"] >= income_range[0]) &
    (filtered_df["AMT_INCOME_TOTAL"] <= income_range[1])
]

if selected_risk == "Good Risk":
    filtered_df = filtered_df[
        filtered_df["CREDIT_RISK"] == 0
    ]

elif selected_risk == "Bad Risk":
    filtered_df = filtered_df[
        filtered_df["CREDIT_RISK"] == 1
    ]

# ---------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------

st.subheader("📊 Key Metrics")

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
else:
    bad_risk_percentage = 0

avg_income = (
    filtered_df["AMT_INCOME_TOTAL"].mean()
    if total_customers > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "👥 Customers",
    f"{total_customers:,}"
)

col2.metric(
    "✅ Good Risk",
    f"{good_risk:,}"
)

col3.metric(
    "⚠️ Bad Risk",
    f"{bad_risk:,}"
)

col4.metric(
    "📈 Bad Risk %",
    f"{bad_risk_percentage:.2f}%"
)

st.divider()

# ---------------------------------------------------------
# NO DATA MESSAGE
# ---------------------------------------------------------

if filtered_df.empty:

    st.warning(
        "No customers match the selected filters. "
        "Please modify the filters."
    )

else:

    # -----------------------------------------------------
    # TABS
    # -----------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Risk Overview",
            "👥 Customer Insights",
            "💼 Occupation & Income",
            "📋 Customer Data"
        ]
    )

    # =====================================================
    # TAB 1 - RISK OVERVIEW
    # =====================================================

    with tab1:

        col1, col2 = st.columns(2)

        # Risk distribution
        with col1:

            st.subheader("Credit Risk Distribution")

            risk_counts = (
                filtered_df["CREDIT_RISK"]
                .value_counts()
                .sort_index()
            )

            risk_df = pd.DataFrame({
                "Risk": [
                    "Good Risk",
                    "Bad Risk"
                ],
                "Customers": [
                    risk_counts.get(0, 0),
                    risk_counts.get(1, 0)
                ]
            })

            st.bar_chart(
                risk_df.set_index("Risk")
            )

        # Status distribution
        with col2:

            st.subheader("Credit Status Distribution")

            status_counts = (
                filtered_df["STATUS"]
                .value_counts()
                .sort_index()
            )

            st.bar_chart(status_counts)

        # Income vs Risk

        st.subheader("Annual Income vs Credit Risk")

        fig, ax = plt.subplots(figsize=(10, 5))

        sb.boxplot(
            data=filtered_df,
            x="CREDIT_RISK",
            y="AMT_INCOME_TOTAL",
            ax=ax
        )

        ax.set_xlabel(
            "Credit Risk (0 = Good, 1 = Bad)"
        )

        ax.set_ylabel("Annual Income")

        ax.set_title(
            "Annual Income vs Credit Risk"
        )

        st.pyplot(fig)

        plt.close(fig)

    # =====================================================
    # TAB 2 - CUSTOMER INSIGHTS
    # =====================================================

    with tab2:

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Age Distribution")

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

            st.pyplot(fig)

            plt.close(fig)

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

            st.bar_chart(family_counts)

        # Employment

        st.subheader(
            "Employment Years Distribution"
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

        ax.set_xlabel(
            "Years Employed"
        )

        ax.set_title(
            "Employment Length Distribution"
        )

        st.pyplot(fig)

        plt.close(fig)

    # =====================================================
    # TAB 3 - OCCUPATION & INCOME
    # =====================================================

    with tab3:

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

        # Average income by risk

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
            "Good Risk" if x == 0
            else "Bad Risk"
            for x in avg_income_risk.index
        ]

        st.bar_chart(
            avg_income_risk
        )

    # =====================================================
    # TAB 4 - CUSTOMER DATA
    # =====================================================

    with tab4:

        st.subheader(
            "Filtered Customer Data"
        )

        display_columns = [
            "ID",
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "AMT_INCOME_TOTAL",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "OCCUPATION_TYPE",
            "AGE",
            "EMPLOYMENT_YEARS",
            "STATUS",
            "CREDIT_RISK"
        ]

        available_columns = [
            col for col in display_columns
            if col in filtered_df.columns
        ]

        st.dataframe(
            filtered_df[available_columns],
            use_container_width=True,
            height=500
        )

        # Download

        csv = filtered_df[
            available_columns
        ].to_csv(index=False)

        st.download_button(
            label="⬇️ Download Filtered Data",
            data=csv,
            file_name="filtered_credit_risk_data.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Credit Card Risk Analysis | Built with Python, Pandas, "
    "Seaborn, Matplotlib and Streamlit"
)
