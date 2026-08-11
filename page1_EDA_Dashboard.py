import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CreditGuard | EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

APP_URL = (
    "https://github.com/Abhishek131004/CreditGuard/"
    "raw/refs/heads/main/Applications_cc.csv.zip"
)

CREDIT_URL = (
    "https://github.com/Abhishek131004/CreditGuard/"
    "raw/refs/heads/main/credit_record%20_cc.csv.zip"
)


# ============================================================
# ENTERPRISE CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color:#f5f7fa;
        color:#172033;
    }

    [data-testid="stSidebar"] {
        background-color:#0b172a;
        border-right:1px solid #1d3048;
    }

    [data-testid="stSidebar"] * {
        color:#e8eef7 !important;
    }

    .brand-container {
        padding:8px 5px 22px 5px;
    }

    .brand-logo {
        display:flex;
        align-items:center;
        gap:12px;
    }

    .shield {
        width:42px;
        height:42px;
        background:#1e5eff;
        border-radius:11px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:22px;
    }

    .brand-name {
        font-size:23px;
        font-weight:750;
        color:white;
    }

    .brand-subtitle {
        font-size:11px;
        color:#8ea2bd;
    }

    .topbar {
        background:white;
        border:1px solid #e2e7ef;
        border-radius:14px;
        padding:12px 18px;
        margin-bottom:22px;
    }

    .page-title {
        font-size:29px;
        font-weight:750;
        color:#152238;
        margin-bottom:2px;
    }

    .page-subtitle {
        color:#68768a;
        font-size:14px;
        margin-bottom:20px;
    }

    .kpi-card {
        background:white;
        border:1px solid #e1e7ef;
        border-radius:14px;
        padding:18px;
        min-height:125px;
        box-shadow:0 3px 12px rgba(16,32,55,0.045);
    }

    .kpi-label {
        color:#68768a;
        font-size:13px;
        font-weight:550;
    }

    .kpi-value {
        color:#142033;
        font-size:27px;
        font-weight:750;
        margin-top:7px;
    }

    .kpi-trend {
        margin-top:8px;
        font-size:12px;
        color:#6d7b8f;
    }

    .section-card {
        background:white;
        border:1px solid #e2e7ef;
        border-radius:14px;
        padding:18px;
        box-shadow:0 3px 12px rgba(16,32,55,0.04);
    }

    .badge {
        display:inline-block;
        padding:4px 10px;
        border-radius:999px;
        font-size:11px;
        font-weight:700;
    }

    .badge-low {
        background:#e7f7ef;
        color:#087849;
    }

    .badge-high {
        background:#fdeaea;
        color:#b32929;
    }

    .alert-card {
        background:white;
        border:1px solid #e1e6ed;
        border-left:4px solid #d85a5a;
        border-radius:10px;
        padding:15px;
        margin-bottom:10px;
    }

    .alert-warning {
        border-left-color:#d99020;
    }

    .alert-info {
        border-left-color:#3778d8;
    }

    .footer {
        color:#8490a1;
        font-size:11px;
        text-align:center;
        padding:25px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():

    applications = pd.read_csv(
        APP_URL,
        compression="zip"
    )

    credit_record = pd.read_csv(
        CREDIT_URL,
        compression="zip"
    )

    # Remove duplicate application IDs
    applications = applications.drop_duplicates(
        subset=["ID"],
        keep="first"
    )

    # Remove duplicate credit IDs
    credit_record = credit_record.drop_duplicates(
        subset=["ID"],
        keep="first"
    )

    # Merge
    df = pd.merge(
        applications,
        credit_record,
        on="ID",
        how="inner"
    )

    # --------------------------------------------------------
    # CLEAN LABELS
    # --------------------------------------------------------

    df["NAME_EDUCATION_TYPE"] = (
        df["NAME_EDUCATION_TYPE"]
        .replace(
            "Secondary / secondary special",
            "Secondary special"
        )
    )

    df["NAME_FAMILY_STATUS"] = (
        df["NAME_FAMILY_STATUS"]
        .replace(
            "Single / not married",
            "Single"
        )
    )

    df["NAME_HOUSING_TYPE"] = (
        df["NAME_HOUSING_TYPE"]
        .replace(
            "House / apartment",
            "House"
        )
    )

    df["OCCUPATION_TYPE"] = (
        df["OCCUPATION_TYPE"]
        .fillna("Unknown")
    )

    df["OCCUPATION_TYPE"] = (
        df["OCCUPATION_TYPE"]
        .replace(
            "Waiters/barmen staff",
            "Waiters"
        )
    )

    # --------------------------------------------------------
    # CREDIT RISK
    # --------------------------------------------------------

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

    df["CREDIT_RISK"] = (
        df["STATUS"]
        .map(status_map)
    )

    df = df.dropna(
        subset=["CREDIT_RISK"]
    )

    df["CREDIT_RISK"] = (
        df["CREDIT_RISK"]
        .astype(int)
    )

    # --------------------------------------------------------
    # DERIVED FIELDS
    # --------------------------------------------------------

    df["AGE"] = (
        -df["DAYS_BIRTH"] // 365
    ).astype(int)

    df["EMPLOYMENT_YEARS"] = (
        -df["DAYS_EMPLOYED"] / 365
    )

    df.loc[
        df["EMPLOYMENT_YEARS"] < 0,
        "EMPLOYMENT_YEARS"
    ] = np.nan

    df["RISK_CATEGORY"] = np.where(
        df["CREDIT_RISK"] == 0,
        "Low Risk",
        "High Risk"
    )

    df["INCOME_BAND"] = pd.cut(
        df["AMT_INCOME_TOTAL"],
        bins=[
            -np.inf,
            200000,
            500000,
            1000000,
            2000000,
            np.inf
        ],
        labels=[
            "< ₹2L",
            "₹2–5L",
            "₹5–10L",
            "₹10–20L",
            "₹20L+"
        ]
    )

    df["AGE_GROUP"] = pd.cut(
        df["AGE"],
        bins=[
            17,
            25,
            35,
            45,
            55,
            np.inf
        ],
        labels=[
            "18–25",
            "26–35",
            "36–45",
            "46–55",
            "56+"
        ]
    )

    return df


# ============================================================
# HELPERS
# ============================================================

def money(value):

    if pd.isna(value):
        return "N/A"

    if value >= 10000000:
        return f"₹{value / 10000000:.1f}Cr"

    if value >= 100000:
        return f"₹{value / 100000:.1f}L"

    return f"₹{value:,.0f}"


def pct(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:.1f}%"


def risk_badge(category):

    if category == "Low Risk":
        return (
            '<span class="badge badge-low">'
            'LOW RISK'
            '</span>'
        )

    return (
        '<span class="badge badge-high">'
        'HIGH RISK'
        '</span>'
    )


def create_kpi(
    label,
    value,
    trend="N/A"
):

    return f"""
    <div class="kpi-card">

        <div class="kpi-label">
            {label}
        </div>

        <div class="kpi-value">
            {value}
        </div>

        <div class="kpi-trend">
            {trend}
        </div>

    </div>
    """


def chart_layout(
    fig,
    height=380
):

    fig.update_layout(
        height=height,
        margin=dict(
            l=10,
            r=10,
            t=45,
            b=10
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Inter, Arial",
            color="#243247"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0
        )
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#e4e8ee"
    )

    fig.update_yaxes(
        gridcolor="#edf0f4",
        zeroline=False
    )

    return fig


# ============================================================
# LOAD DATA
# ============================================================

try:

    with st.spinner(
        "Loading CreditGuard portfolio..."
    ):

        df = load_data()

except Exception as e:

    st.error(
        "Unable to load portfolio data."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="brand-container">

        <div class="brand-logo">

            <div class="shield">
                🛡️
            </div>

            <div>

                <div class="brand-name">
                    CreditGuard
                </div>

                <div class="brand-subtitle">
                    EDA & Portfolio Intelligence
                </div>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown(
    "### EDA Navigation"
)


pages = [
    "Overview",
    "Applications",
    "Customer 360",
    "Risk Analysis",
    "Portfolio Analytics",
    "Risk Alerts",
    "Reports",
    "Audit & Data Quality"
]


page = st.sidebar.radio(
    "Navigation",
    pages,
    label_visibility="collapsed"
)


# ============================================================
# FILTERS
# ============================================================

with st.sidebar.expander(
    "Global Filters",
    expanded=False
):

    filter_gender = st.multiselect(
        "Gender",
        sorted(
            df["CODE_GENDER"]
            .dropna()
            .unique()
        )
    )

    filter_education = st.multiselect(
        "Education",
        sorted(
            df["NAME_EDUCATION_TYPE"]
            .dropna()
            .unique()
        )
    )

    filter_occupation = st.multiselect(
        "Occupation",
        sorted(
            df["OCCUPATION_TYPE"]
            .dropna()
            .unique()
        )
    )

    filter_housing = st.multiselect(
        "Housing",
        sorted(
            df["NAME_HOUSING_TYPE"]
            .dropna()
            .unique()
        )
    )

    filter_family = st.multiselect(
        "Family Status",
        sorted(
            df["NAME_FAMILY_STATUS"]
            .dropna()
            .unique()
        )
    )

    filter_risk = st.multiselect(
        "Risk Category",
        [
            "Low Risk",
            "High Risk"
        ]
    )

    age_range = st.slider(
        "Age",
        int(df["AGE"].min()),
        int(df["AGE"].max()),
        (
            int(df["AGE"].min()),
            int(df["AGE"].max())
        )
    )

    income_range = st.slider(
        "Annual Income",
        float(
            df["AMT_INCOME_TOTAL"].min()
        ),
        float(
            df["AMT_INCOME_TOTAL"].max()
        ),
        (
            float(
                df["AMT_INCOME_TOTAL"].min()
            ),
            float(
                df["AMT_INCOME_TOTAL"].max()
            )
        )
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = df.copy()


if filter_gender:

    filtered = filtered[
        filtered["CODE_GENDER"]
        .isin(filter_gender)
    ]


if filter_education:

    filtered = filtered[
        filtered["NAME_EDUCATION_TYPE"]
        .isin(filter_education)
    ]


if filter_occupation:

    filtered = filtered[
        filtered["OCCUPATION_TYPE"]
        .isin(filter_occupation)
    ]


if filter_housing:

    filtered = filtered[
        filtered["NAME_HOUSING_TYPE"]
        .isin(filter_housing)
    ]


if filter_family:

    filtered = filtered[
        filtered["NAME_FAMILY_STATUS"]
        .isin(filter_family)
    ]


if filter_risk:

    filtered = filtered[
        filtered["RISK_CATEGORY"]
        .isin(filter_risk)
    ]


filtered = filtered[
    (filtered["AGE"] >= age_range[0])
    &
    (filtered["AGE"] <= age_range[1])
]


filtered = filtered[
    (filtered["AMT_INCOME_TOTAL"] >= income_range[0])
    &
    (filtered["AMT_INCOME_TOTAL"] <= income_range[1])
]


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
    """
    <div class="topbar">

        <b>CreditGuard</b>
        &nbsp; / &nbsp;
        Exploratory Data Analysis

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="page-title">'
        'Credit Portfolio Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Explore customer demographics, portfolio composition '
        'and historical credit-risk patterns.'
        '</div>',
        unsafe_allow_html=True
    )

    total = len(filtered)

    low_risk = (
        filtered["CREDIT_RISK"] == 0
    ).sum()

    high_risk = (
        filtered["CREDIT_RISK"] == 1
    ).sum()

    risk_rate = (
        high_risk / total * 100
        if total
        else 0
    )

    avg_income = (
        filtered["AMT_INCOME_TOTAL"].mean()
        if total
        else np.nan
    )

    avg_age = (
        filtered["AGE"].mean()
        if total
        else np.nan
    )


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:

        st.markdown(
            create_kpi(
                "Customer Records",
                f"{total:,}",
                "Current filtered portfolio"
            ),
            unsafe_allow_html=True
        )

    with k2:

        st.markdown(
            create_kpi(
                "Low-Risk Customers",
                f"{low_risk:,}",
                pct(
                    low_risk / total * 100
                ) if total else "N/A"
            ),
            unsafe_allow_html=True
        )

    with k3:

        st.markdown(
            create_kpi(
                "High-Risk Customers",
                f"{high_risk:,}",
                pct(risk_rate)
            ),
            unsafe_allow_html=True
        )

    with k4:

        st.markdown(
            create_kpi(
                "Average Income",
                money(avg_income),
                "Annual income"
            ),
            unsafe_allow_html=True
        )

    with k5:

        st.markdown(
            create_kpi(
                "Average Age",
                f"{avg_age:.1f} years"
                if total else "N/A",
                "Customer profile"
            ),
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    c1, c2 = st.columns(
        [1, 1.7]
    )

    with c1:

        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Customer Risk Distribution"
        )

        risk_counts = (
            filtered["RISK_CATEGORY"]
            .value_counts()
        )

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=[
                        "Low Risk",
                        "High Risk"
                    ],
                    values=[
                        risk_counts.get(
                            "Low Risk",
                            0
                        ),
                        risk_counts.get(
                            "High Risk",
                            0
                        )
                    ],
                    hole=0.68,
                    marker=dict(
                        colors=[
                            "#16a36a",
                            "#d64d4d"
                        ]
                    ),
                    textinfo="percent"
                )
            ]
        )

        fig.update_layout(
            height=330,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            ),
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Portfolio Snapshot"
        )

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Customer Records",
                    "Low-Risk Customers",
                    "High-Risk Customers",
                    "Average Income",
                    "Average Age"
                ],
                "Value": [
                    f"{total:,}",
                    f"{low_risk:,}",
                    f"{high_risk:,}",
                    money(avg_income),
                    (
                        f"{avg_age:.1f} years"
                        if total
                        else "N/A"
                    )
                ]
            }
        )

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # AGE RISK
    # --------------------------------------------------------

    st.subheader(
        "Risk Trend by Age Group"
    )

    age_risk = (
        filtered
        .groupby(
            [
                "AGE_GROUP",
                "RISK_CATEGORY"
            ],
            observed=True
        )
        .size()
        .reset_index(
            name="Customers"
        )
    )

    fig = px.bar(
        age_risk,
        x="AGE_GROUP",
        y="Customers",
        color="RISK_CATEGORY",
        barmode="group",
        color_discrete_map={
            "Low Risk": "#16a36a",
            "High Risk": "#d64d4d"
        }
    )

    st.plotly_chart(
        chart_layout(fig),
        use_container_width=True
    )


    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    st.subheader(
        "Customer Risk Concentration"
    )

    heatmap = pd.crosstab(
        filtered["AGE_GROUP"],
        filtered["INCOME_BAND"]
    )

    heatmap = heatmap.reindex(
        index=[
            "18–25",
            "26–35",
            "36–45",
            "46–55",
            "56+"
        ],
        columns=[
            "< ₹2L",
            "₹2–5L",
            "₹5–10L",
            "₹10–20L",
            "₹20L+"
        ]
    )

    fig = px.imshow(
        heatmap,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[
            "#eef5ff",
            "#8bb5ef",
            "#174ea6"
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# APPLICATIONS
# ============================================================

elif page == "Applications":

    st.markdown(
        '<div class="page-title">'
        'Credit Card Applications'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Explore customer application records and '
        'risk classification.'
        '</div>',
        unsafe_allow_html=True
    )

    search_id = st.text_input(
        "Search Customer / Application ID",
        placeholder="Enter customer ID"
    )

    app_data = filtered.copy()


    if search_id:

        app_data = app_data[
            app_data["ID"]
            .astype(str)
            .str.contains(
                search_id,
                case=False,
                na=False
            )
        ]


    risk_filter = st.multiselect(
        "Risk Category",
        [
            "Low Risk",
            "High Risk"
        ]
    )


    if risk_filter:

        app_data = app_data[
            app_data["RISK_CATEGORY"]
            .isin(risk_filter)
        ]


    st.metric(
        "Matching Records",
        f"{len(app_data):,}"
    )


    display = app_data[
        [
            "ID",
            "AGE",
            "AMT_INCOME_TOTAL",
            "OCCUPATION_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_HOUSING_TYPE",
            "RISK_CATEGORY",
            "STATUS"
        ]
    ].copy()


    display.columns = [
        "Customer ID",
        "Age",
        "Annual Income",
        "Occupation",
        "Education",
        "Housing",
        "Risk Category",
        "Credit Status"
    ]


    st.dataframe(
        display,
        use_container_width=True,
        height=520
    )


    csv_data = display.to_csv(
        index=False
    )


    st.download_button(
        "Export CSV",
        csv_data,
        "creditguard_applications.csv",
        "text/csv"
    )


# ============================================================
# CUSTOMER 360
# ============================================================

elif page == "Customer 360":

    st.markdown(
        '<div class="page-title">'
        'Customer 360'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Complete customer profile and historical '
        'risk context.'
        '</div>',
        unsafe_allow_html=True
    )


    customer_id = st.text_input(
        "Search Customer ID",
        placeholder="Example: 5008804"
    )


    if customer_id:

        result = df[
            df["ID"]
            .astype(str)
            == customer_id.strip()
        ]


        if result.empty:

            st.warning(
                "No customer found."
            )

        else:

            customer = result.iloc[0]

            st.success(
                "Customer profile loaded."
            )


            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                "Customer ID",
                str(customer["ID"])
            )

            c2.metric(
                "Age",
                f"{customer['AGE']} years"
            )

            c3.metric(
                "Annual Income",
                money(
                    customer[
                        "AMT_INCOME_TOTAL"
                    ]
                )
            )

            c4.metric(
                "Family Size",
                str(
                    customer[
                        "CNT_FAM_MEMBERS"
                    ]
                )
            )

            c5.markdown(
                f"""
                <div style="padding-top:5px;">

                    <div style="
                        color:#68768a;
                        font-size:13px;
                    ">
                        Risk Category
                    </div>

                    <div style="margin-top:9px;">
                        {risk_badge(
                            customer[
                                "RISK_CATEGORY"
                            ]
                        )}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown("---")


            left, right = st.columns(2)


            with left:

                st.subheader(
                    "Applicant Information"
                )

                applicant_info = pd.DataFrame(
                    {
                        "Field": [
                            "Customer ID",
                            "Gender",
                            "Age",
                            "Family Status",
                            "Number of Children",
                            "Family Members"
                        ],
                        "Value": [
                            customer["ID"],
                            customer["CODE_GENDER"],
                            customer["AGE"],
                            customer[
                                "NAME_FAMILY_STATUS"
                            ],
                            customer[
                                "CNT_CHILDREN"
                            ],
                            customer[
                                "CNT_FAM_MEMBERS"
                            ]
                        ]
                    }
                )

                st.dataframe(
                    applicant_info,
                    hide_index=True,
                    use_container_width=True
                )


            with right:

                st.subheader(
                    "Financial Information"
                )

                financial_info = pd.DataFrame(
                    {
                        "Field": [
                            "Annual Income",
                            "Income Type",
                            "Occupation",
                            "Employment Duration",
                            "Education",
                            "Housing Type"
                        ],
                        "Value": [
                            money(
                                customer[
                                    "AMT_INCOME_TOTAL"
                                ]
                            ),
                            customer[
                                "NAME_INCOME_TYPE"
                            ],
                            customer[
                                "OCCUPATION_TYPE"
                            ],
                            (
                                f"{customer['EMPLOYMENT_YEARS']:.1f} years"
                                if pd.notna(
                                    customer[
                                        "EMPLOYMENT_YEARS"
                                    ]
                                )
                                else "N/A"
                            ),
                            customer[
                                "NAME_EDUCATION_TYPE"
                            ],
                            customer[
                                "NAME_HOUSING_TYPE"
                            ]
                        ]
                    }
                )

                st.dataframe(
                    financial_info,
                    hide_index=True,
                    use_container_width=True
                )


            st.subheader(
                "Asset & Contact Availability"
            )


            c1, c2, c3, c4 = st.columns(4)


            c1.metric(
                "Car Ownership",
                customer["FLAG_OWN_CAR"]
            )

            c2.metric(
                "Property Ownership",
                customer["FLAG_OWN_REALTY"]
            )

            c3.metric(
                "Phone Availability",
                customer["FLAG_PHONE"]
            )

            c4.metric(
                "Email Availability",
                customer["FLAG_EMAIL"]
            )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.markdown(
        '<div class="page-title">'
        'Risk Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Analyze credit-risk concentration across '
        'key customer segments.'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # INCOME
    # --------------------------------------------------------

    st.subheader(
        "Risk by Income Band"
    )

    income_risk = pd.crosstab(
        filtered["INCOME_BAND"],
        filtered["RISK_CATEGORY"],
        normalize="index"
    ) * 100

    income_risk = (
        income_risk
        .reset_index()
    )

    fig = px.bar(
        income_risk,
        x="INCOME_BAND",
        y=[
            c for c in [
                "Low Risk",
                "High Risk"
            ]
            if c in income_risk.columns
        ],
        barmode="group",
        color_discrete_map={
            "Low Risk": "#16a36a",
            "High Risk": "#d64d4d"
        }
    )

    st.plotly_chart(
        chart_layout(fig),
        use_container_width=True
    )


    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    st.subheader(
        "Risk by Age Group"
    )

    age_risk = pd.crosstab(
        filtered["AGE_GROUP"],
        filtered["RISK_CATEGORY"],
        normalize="index"
    ) * 100

    age_risk = age_risk.reset_index()


    fig = px.bar(
        age_risk,
        x="AGE_GROUP",
        y=[
            c for c in [
                "Low Risk",
                "High Risk"
            ]
            if c in age_risk.columns
        ],
        barmode="group",
        color_discrete_map={
            "Low Risk": "#16a36a",
            "High Risk": "#d64d4d"
        }
    )

    st.plotly_chart(
        chart_layout(fig),
        use_container_width=True
    )


    # --------------------------------------------------------
    # OCCUPATION
    # --------------------------------------------------------

    st.subheader(
        "Risk by Occupation"
    )

    occupation_risk = pd.crosstab(
        filtered["OCCUPATION_TYPE"],
        filtered["RISK_CATEGORY"],
        normalize="index"
    ) * 100


    if "High Risk" in occupation_risk.columns:

        occupation_risk = (
            occupation_risk
            .sort_values(
                "High Risk",
                ascending=True
            )
            .tail(15)
            .reset_index()
        )


        fig = px.bar(
            occupation_risk,
            x="High Risk",
            y="OCCUPATION_TYPE",
            orientation="h",
            labels={
                "High Risk":
                    "High-Risk Customers (%)"
            },
            color_discrete_sequence=[
                "#d64d4d"
            ]
        )

        st.plotly_chart(
            chart_layout(fig, 450),
            use_container_width=True
        )


    # --------------------------------------------------------
    # EDUCATION / HOUSING
    # --------------------------------------------------------

    c1, c2 = st.columns(2)


    with c1:

        st.subheader(
            "Risk by Education"
        )

        edu_risk = pd.crosstab(
            filtered["NAME_EDUCATION_TYPE"],
            filtered["RISK_CATEGORY"],
            normalize="index"
        ) * 100

        edu_risk = (
            edu_risk
            .reset_index()
        )


        fig = px.bar(
            edu_risk,
            x="NAME_EDUCATION_TYPE",
            y=[
                c for c in [
                    "Low Risk",
                    "High Risk"
                ]
                if c in edu_risk.columns
            ],
            barmode="group",
            color_discrete_map={
                "Low Risk": "#16a36a",
                "High Risk": "#d64d4d"
            }
        )

        fig.update_xaxes(
            tickangle=-35
        )

        st.plotly_chart(
            chart_layout(fig, 420),
            use_container_width=True
        )


    with c2:

        st.subheader(
            "Risk by Housing"
        )

        housing_risk = pd.crosstab(
            filtered["NAME_HOUSING_TYPE"],
            filtered["RISK_CATEGORY"],
            normalize="index"
        ) * 100

        housing_risk = (
            housing_risk
            .reset_index()
        )


        fig = px.bar(
            housing_risk,
            x="NAME_HOUSING_TYPE",
            y=[
                c for c in [
                    "Low Risk",
                    "High Risk"
                ]
                if c in housing_risk.columns
            ],
            barmode="group",
            color_discrete_map={
                "Low Risk": "#16a36a",
                "High Risk": "#d64d4d"
            }
        )

        fig.update_xaxes(
            tickangle=-35
        )

        st.plotly_chart(
            chart_layout(fig, 420),
            use_container_width=True
        )


# ============================================================
# PORTFOLIO ANALYTICS
# ============================================================

elif page == "Portfolio Analytics":

    st.markdown(
        '<div class="page-title">'
        'Portfolio Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Executive-level customer segmentation and '
        'risk concentration analysis.'
        '</div>',
        unsafe_allow_html=True
    )


    total = len(filtered)

    high_risk = (
        filtered["CREDIT_RISK"] == 1
    ).sum()

    avg_income = (
        filtered["AMT_INCOME_TOTAL"].mean()
        if total
        else np.nan
    )

    risk_rate = (
        high_risk / total * 100
        if total
        else 0
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Customer Records",
        f"{total:,}"
    )

    c2.metric(
        "High-Risk Customers",
        f"{high_risk:,}"
    )

    c3.metric(
        "High-Risk Rate",
        pct(risk_rate)
    )

    c4.metric(
        "Average Income",
        money(avg_income)
    )


    st.markdown("---")


    segment_dimension = st.selectbox(
        "Segment Portfolio By",
        [
            "AGE_GROUP",
            "INCOME_BAND",
            "OCCUPATION_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_HOUSING_TYPE",
            "NAME_FAMILY_STATUS",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY"
        ]
    )


    segment = (
        filtered
        .groupby(segment_dimension)
        .agg(
            Customers=("ID", "count"),
            Average_Income=(
                "AMT_INCOME_TOTAL",
                "mean"
            ),
            High_Risk=(
                "CREDIT_RISK",
                "sum"
            )
        )
        .reset_index()
    )


    segment["High_Risk_Percentage"] = (
        segment["High_Risk"]
        / segment["Customers"]
        * 100
    )


    segment["Risk Level"] = np.select(
        [
            segment[
                "High_Risk_Percentage"
            ] <= 5,

            segment[
                "High_Risk_Percentage"
            ] <= 15
        ],
        [
            "Low",
            "Medium"
        ],
        default="High"
    )


    segment["Average Income"] = (
        segment["Average_Income"]
        .apply(money)
    )


    segment["High-Risk %"] = (
        segment[
            "High_Risk_Percentage"
        ]
        .round(1)
        .astype(str)
        + "%"
    )


    segment_display = segment[
        [
            segment_dimension,
            "Customers",
            "Average Income",
            "High-Risk %",
            "Risk Level"
        ]
    ].copy()


    segment_display.columns = [
        "Segment",
        "Customers",
        "Average Income",
        "High-Risk %",
        "Risk Level"
    ]


    st.dataframe(
        segment_display,
        hide_index=True,
        use_container_width=True
    )


    st.subheader(
        "High-Risk Concentration"
    )


    chart_data = (
        segment
        .sort_values(
            "High_Risk_Percentage",
            ascending=False
        )
        .head(15)
    )


    fig = px.bar(
        chart_data,
        x="High_Risk_Percentage",
        y=segment_dimension,
        orientation="h",
        labels={
            "High_Risk_Percentage":
                "High-Risk Customers (%)"
        },
        color_discrete_sequence=[
            "#d64d4d"
        ]
    )


    st.plotly_chart(
        chart_layout(fig, 450),
        use_container_width=True
    )


# ============================================================
# RISK ALERTS
# ============================================================

elif page == "Risk Alerts":

    st.markdown(
        '<div class="page-title">'
        'Risk Alerts'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Operational risk monitoring based on '
        'current portfolio data.'
        '</div>',
        unsafe_allow_html=True
    )


    total = len(filtered)

    high_risk = (
        filtered["CREDIT_RISK"] == 1
    ).sum()

    high_risk_rate = (
        high_risk / total * 100
        if total
        else 0
    )


    c1, c2, c3 = st.columns(3)

    c1.metric(
        "High Risk",
        f"{high_risk:,}"
    )

    c2.metric(
        "High-Risk Rate",
        pct(high_risk_rate)
    )

    c3.metric(
        "Unknown Occupation",
        f"{(
            df['OCCUPATION_TYPE'] == 'Unknown'
        ).sum():,}"
    )


    if high_risk > 0:

        st.markdown(
            f"""
            <div class="alert-card">

                <b>
                    🔴 High-Risk Customer Concentration
                </b>

                <br>

                <span style="color:#68768a;">

                {high_risk:,} customers are currently
                classified as high risk, representing
                {high_risk_rate:.1f}% of the filtered portfolio.

                </span>

            </div>
            """,
            unsafe_allow_html=True
        )


    income_alert = (
        filtered
        .groupby("INCOME_BAND")
        .agg(
            Customers=("ID", "count"),
            High_Risk=("CREDIT_RISK", "sum")
        )
    )


    if not income_alert.empty:

        income_alert["Rate"] = (
            income_alert["High_Risk"]
            / income_alert["Customers"]
            * 100
        )


        highest_band = (
            income_alert["Rate"]
            .idxmax()
        )


        highest_rate = (
            income_alert.loc[
                highest_band,
                "Rate"
            ]
        )


        st.markdown(
            f"""
            <div class="alert-card alert-warning">

                <b>
                    🟠 Risk Concentration
                </b>

                <br>

                <span style="color:#68768a;">

                The {highest_band} income segment
                has the highest observed high-risk
                rate at {highest_rate:.1f}%.

                </span>

            </div>
            """,
            unsafe_allow_html=True
        )


    missing_occupation = (
        df["OCCUPATION_TYPE"]
        .eq("Unknown")
        .sum()
    )


    if missing_occupation > 0:

        st.markdown(
            f"""
            <div class="alert-card alert-info">

                <b>
                    🟡 Incomplete Customer Profile
                </b>

                <br>

                <span style="color:#68768a;">

                {missing_occupation:,} customer records
                have an unknown occupation value.

                </span>

            </div>
            """,
            unsafe_allow_html=True
        )


    alert_table = pd.DataFrame(
        {
            "Alert Type": [
                "High-Risk Customer Concentration",
                "Risk Concentration by Income",
                "Incomplete Customer Profile"
            ],
            "Severity": [
                "Critical",
                "High",
                "Medium"
            ],
            "Status": [
                "Open",
                "Open",
                "Open"
            ]
        }
    )


    st.dataframe(
        alert_table,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":

    st.markdown(
        '<div class="page-title">'
        'Reporting Center'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Export current portfolio and customer analytics.'
        '</div>',
        unsafe_allow_html=True
    )


    reports = [
        (
            "Daily Risk Summary",
            "Current customer risk distribution."
        ),
        (
            "Customer Risk Report",
            "Customer-level risk classification."
        ),
        (
            "Application Analysis Report",
            "Application-related customer information."
        ),
        (
            "High-Risk Exposure Report",
            "High-risk customer segmentation."
        ),
        (
            "Risk Trend Report",
            "Current risk distribution by segment."
        )
    ]


    for name, description in reports:

        st.markdown(
            f"""
            <div class="section-card"
                 style="margin-bottom:12px;">

                <b>{name}</b>

                <div style="
                    color:#68768a;
                    font-size:13px;
                    margin-top:5px;
                ">

                    {description}

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.subheader(
        "Export Current Customer Data"
    )


    export_df = filtered.copy()

    export_df.columns = [
        c.replace(
            "_",
            " "
        ).title()
        for c in export_df.columns
    ]


    csv = export_df.to_csv(
        index=False
    )


    st.download_button(
        "Download CSV",
        csv,
        "CreditGuard_Report.csv",
        "text/csv"
    )


    excel_buffer = io.BytesIO()


    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        export_df.to_excel(
            writer,
            index=False,
            sheet_name="CreditGuard Report"
        )


    st.download_button(
        "Download Excel",
        excel_buffer.getvalue(),
        "CreditGuard_Report.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ============================================================
# AUDIT & DATA QUALITY
# ============================================================

elif page == "Audit & Data Quality":

    st.markdown(
        '<div class="page-title">'
        'Audit & Data Quality'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Data-quality visibility and operational transparency.'
        '</div>',
        unsafe_allow_html=True
    )


    total_records = len(df)

    complete_records = (
        df.notna()
        .all(axis=1)
        .sum()
    )

    incomplete_records = (
        total_records
        - complete_records
    )

    duplicate_records = (
        df.duplicated()
        .sum()
    )

    missing_values = (
        df.isna()
        .sum()
        .sum()
    )


    c1, c2, c3, c4, c5 = st.columns(5)


    c1.metric(
        "Total Records",
        f"{total_records:,}"
    )

    c2.metric(
        "Complete Records",
        f"{complete_records:,}"
    )

    c3.metric(
        "Incomplete Records",
        f"{incomplete_records:,}"
    )

    c4.metric(
        "Duplicate Records",
        f"{duplicate_records:,}"
    )

    c5.metric(
        "Missing Values",
        f"{missing_values:,}"
    )


    completeness = (
        complete_records
        / total_records
        * 100
        if total_records
        else 0
    )


    st.subheader(
        "Data Completeness"
    )


    st.progress(
        int(completeness)
    )


    st.write(
        f"{completeness:.1f}% complete records"
    )


    st.subheader(
        "Field-Level Data Quality"
    )


    quality = pd.DataFrame(
        {
            "Field": df.columns,

            "Missing Values": [
                int(
                    df[c]
                    .isna()
                    .sum()
                )
                for c in df.columns
            ],

            "Missing %": [
                round(
                    df[c]
                    .isna()
                    .mean()
                    * 100,
                    2
                )
                for c in df.columns
            ]
        }
    )


    st.dataframe(
        quality,
        hide_index=True,
        use_container_width=True
    )


    st.subheader(
        "Dataset Preview"
    )


    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=450
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        CreditGuard • EDA & Portfolio Intelligence

        <br>

        Enterprise analytics workspace

    </div>
    """,
    unsafe_allow_html=True
)
