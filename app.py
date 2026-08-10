import io
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CreditGuard | Credit Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ENTERPRISE CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background-color: #f5f7fa;
        color: #172033;
    }

    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background-color: #0b172a;
        border-right: 1px solid #1d3048;
    }

    [data-testid="stSidebar"] * {
        color: #e8eef7 !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        padding: 7px 8px;
        border-radius: 8px;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #162942;
    }

    /* ---------- SIDEBAR BRAND ---------- */

    .brand-container {
        padding: 8px 5px 22px 5px;
    }

    .brand-logo {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .shield {
        width: 42px;
        height: 42px;
        background: #1e5eff;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
    }

    .brand-name {
        font-size: 23px;
        font-weight: 750;
        color: white;
    }

    .brand-subtitle {
        font-size: 11px;
        color: #8ea2bd;
        margin-top: 2px;
    }

    /* ---------- TOP BAR ---------- */

    .topbar {
        background: white;
        border: 1px solid #e2e7ef;
        border-radius: 14px;
        padding: 12px 18px;
        margin-bottom: 22px;
        box-shadow: 0 2px 8px rgba(17, 31, 54, 0.04);
    }

    .online-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #16a36a;
        border-radius: 50%;
        margin-right: 5px;
    }

    /* ---------- PAGE HEADER ---------- */

    .page-title {
        font-size: 29px;
        font-weight: 750;
        color: #152238;
        margin-bottom: 2px;
    }

    .page-subtitle {
        color: #68768a;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* ---------- KPI CARDS ---------- */

    .kpi-card {
        background: white;
        border: 1px solid #e1e7ef;
        border-radius: 14px;
        padding: 18px;
        min-height: 132px;
        box-shadow: 0 3px 12px rgba(16, 32, 55, 0.045);
    }

    .kpi-label {
        color: #68768a;
        font-size: 13px;
        font-weight: 550;
        margin-bottom: 9px;
    }

    .kpi-value {
        color: #142033;
        font-size: 27px;
        font-weight: 750;
    }

    .kpi-trend {
        margin-top: 10px;
        font-size: 12px;
        color: #6d7b8f;
    }

    .trend-positive {
        color: #16865b;
        font-weight: 650;
    }

    .trend-negative {
        color: #c54242;
        font-weight: 650;
    }

    /* ---------- SECTION CARDS ---------- */

    .section-card {
        background: white;
        border: 1px solid #e2e7ef;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 3px 12px rgba(16, 32, 55, 0.04);
    }

    /* ---------- BADGES ---------- */

    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
    }

    .badge-low {
        background: #e7f7ef;
        color: #087849;
    }

    .badge-medium {
        background: #fff3dc;
        color: #a66700;
    }

    .badge-high {
        background: #fdeaea;
        color: #b32929;
    }

    .badge-neutral {
        background: #edf1f6;
        color: #536276;
    }

    /* ---------- ALERTS ---------- */

    .alert-card {
        background: white;
        border: 1px solid #e1e6ed;
        border-left: 4px solid #d85a5a;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }

    .alert-warning {
        border-left-color: #d99020;
    }

    .alert-info {
        border-left-color: #3778d8;
    }

    /* ---------- INSIGHTS ---------- */

    .insight-card {
        background: white;
        border: 1px solid #e1e7ef;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }

    .insight-title {
        font-size: 13px;
        font-weight: 700;
        color: #26364b;
    }

    .insight-text {
        font-size: 13px;
        color: #66758a;
        margin-top: 5px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        color: #8490a1;
        font-size: 11px;
        text-align: center;
        padding: 25px 0 10px 0;
    }

    /* ---------- TABLE ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 768px) {

        .page-title {
            font-size: 22px;
        }

        .kpi-value {
            font-size: 22px;
        }

        .kpi-card {
            min-height: 110px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
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

TODAY = "10 Aug 2026"


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

    # --------------------------------------------------------
    # APPLICATION DATA
    # --------------------------------------------------------

    applications = applications.drop_duplicates(
        subset=["ID"],
        keep="first"
    )

    # --------------------------------------------------------
    # CREDIT RECORD
    # --------------------------------------------------------
    #
    # Preserve the same logic as the original analysis:
    # one record per ID.
    #
    # --------------------------------------------------------

    credit_record = credit_record.drop_duplicates(
        subset=["ID"],
        keep="first"
    )

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

    df["CREDIT_RISK"] = df["STATUS"].map(status_map)

    df = df.dropna(
        subset=["CREDIT_RISK"]
    )

    df["CREDIT_RISK"] = (
        df["CREDIT_RISK"]
        .astype(int)
    )

    # --------------------------------------------------------
    # DERIVED BUSINESS FIELDS
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

    # Risk label
    df["RISK_CATEGORY"] = np.where(
        df["CREDIT_RISK"] == 0,
        "Low Risk",
        "High Risk"
    )

    # Income band
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

    # Age group
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
# HELPER FUNCTIONS
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

    if category == "Medium Risk":

        return (
            '<span class="badge badge-medium">'
            'MEDIUM'
            '</span>'
        )

    if category == "High Risk":

        return (
            '<span class="badge badge-high">'
            'HIGH RISK'
            '</span>'
        )

    return (
        '<span class="badge badge-neutral">'
        'N/A'
        '</span>'
    )


def create_kpi(
    label,
    value,
    trend="N/A",
    trend_type="neutral"
):

    trend_class = ""

    if trend_type == "positive":
        trend_class = "trend-positive"

    elif trend_type == "negative":
        trend_class = "trend-negative"

    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-trend">
            <span class="{trend_class}">
                {trend}
            </span>
        </div>
    </div>
    """


def chart_layout(fig, height=360):

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
        ),
        hoverlabel=dict(
            bgcolor="white"
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

    with st.spinner("Loading portfolio data..."):

        df = load_data()

except Exception as e:

    st.error("Unable to load portfolio data.")

    st.info(
        "Please verify that the GitHub dataset files are available."
    )

    with st.expander("Technical details"):

        st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR BRANDING
# ============================================================

st.sidebar.markdown(
    """
    <div class="brand-container">
        <div class="brand-logo">
            <div class="shield">🛡️</div>
            <div>
                <div class="brand-name">CreditGuard</div>
                <div class="brand-subtitle">
                    Credit Risk Intelligence
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.markdown("### Navigation")

pages = [
    "Overview",
    "Applications",
    "Customer 360",
    "Risk Analysis",
    "Portfolio Analytics",
    "Risk Alerts",
    "Reports",
    "Audit & Compliance",
    "Settings"
]

page = st.sidebar.radio(
    "Navigation",
    pages,
    label_visibility="collapsed"
)


# ============================================================
# SIDEBAR STATUS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="
        font-size:12px;
        color:#8ea2bd;
        margin-bottom:5px;
    ">
        SYSTEM STATUS
    </div>

    <div style="
        font-size:13px;
        font-weight:600;
        color:#e8eef7;
    ">
        <span style="color:#28c47c;">●</span>
        All systems operational
    </div>

    <div style="
        font-size:12px;
        color:#8ea2bd;
        margin-top:15px;
    ">
        DATA UPDATED
    </div>

    <div style="
        font-size:13px;
        color:#e8eef7;
    ">
        10 Aug 2026
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# GLOBAL FILTERS
# ============================================================

with st.sidebar.expander("Global Filters", expanded=False):

    filter_gender = st.multiselect(
        "Gender",
        sorted(df["CODE_GENDER"].dropna().unique()),
        default=[]
    )

    filter_education = st.multiselect(
        "Education",
        sorted(
            df["NAME_EDUCATION_TYPE"]
            .dropna()
            .unique()
        ),
        default=[]
    )

    filter_occupation = st.multiselect(
        "Occupation",
        sorted(
            df["OCCUPATION_TYPE"]
            .dropna()
            .unique()
        ),
        default=[]
    )

    filter_housing = st.multiselect(
        "Housing",
        sorted(
            df["NAME_HOUSING_TYPE"]
            .dropna()
            .unique()
        ),
        default=[]
    )

    filter_family = st.multiselect(
        "Family Status",
        sorted(
            df["NAME_FAMILY_STATUS"]
            .dropna()
            .unique()
        ),
        default=[]
    )

    filter_risk = st.multiselect(
        "Risk Category",
        ["Low Risk", "High Risk"],
        default=[]
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
        float(df["AMT_INCOME_TOTAL"].min()),
        float(df["AMT_INCOME_TOTAL"].max()),
        (
            float(df["AMT_INCOME_TOTAL"].min()),
            float(df["AMT_INCOME_TOTAL"].max())
        )
    )


# ============================================================
# APPLY GLOBAL FILTERS
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
# TOP NAVIGATION BAR
# ============================================================

top1, top2, top3, top4 = st.columns(
    [5, 1.3, 1.2, 1.5]
)

with top1:

    st.markdown(
        """
        <div class="topbar">
            <b>CreditGuard</b>
            &nbsp; / &nbsp;
            Enterprise Risk Workspace
        </div>
        """,
        unsafe_allow_html=True
    )

with top2:

    st.caption("Data")

with top3:

    st.caption("🔔 3")

with top4:

    st.caption("Risk Analyst")


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="page-title">Credit Risk Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Monitor customer exposure, application activity '
        'and portfolio risk.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PERIOD SELECTOR
    # --------------------------------------------------------

    period = st.segmented_control(
        "Analysis Period",
        ["Today", "7 Days", "30 Days", "90 Days", "Custom"],
        default="30 Days"
    )

    st.markdown("")

    # --------------------------------------------------------
    # KPI VALUES FROM REAL DATA
    # --------------------------------------------------------

    total_customers = len(filtered)

    good_customers = (
        filtered["CREDIT_RISK"] == 0
    ).sum()

    high_risk_customers = (
        filtered["CREDIT_RISK"] == 1
    ).sum()

    risk_rate = (
        high_risk_customers / total_customers * 100
        if total_customers
        else np.nan
    )

    average_income = (
        filtered["AMT_INCOME_TOTAL"].mean()
        if total_customers
        else np.nan
    )

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.markdown(
            create_kpi(
                "Customer Records",
                f"{total_customers:,}",
                "Current filtered portfolio"
            ),
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            create_kpi(
                "Low-Risk Customers",
                f"{good_customers:,}",
                pct(
                    good_customers / total_customers * 100
                )
                if total_customers else "N/A"
            ),
            unsafe_allow_html=True
        )

    with k3:
        st.markdown(
            create_kpi(
                "High-Risk Customers",
                f"{high_risk_customers:,}",
                pct(risk_rate)
            ),
            unsafe_allow_html=True
        )

    with k4:
        st.markdown(
            create_kpi(
                "Portfolio Risk Rate",
                pct(risk_rate),
                "Based on available credit-status data"
            ),
            unsafe_allow_html=True
        )

    with k5:
        st.markdown(
            create_kpi(
                "Average Customer Income",
                money(average_income),
                "Current filtered portfolio"
            ),
            unsafe_allow_html=True
        )

    with k6:
        st.markdown(
            create_kpi(
                "Application Status",
                "N/A",
                "Not available in source data"
            ),
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # RISK DISTRIBUTION + APPLICATION INFORMATION
    # --------------------------------------------------------

    c1, c2 = st.columns([1, 1.7])

    with c1:

        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        st.subheader("Customer Risk Distribution")

        risk_counts = (
            filtered["RISK_CATEGORY"]
            .value_counts()
        )

        labels = ["Low Risk", "High Risk"]

        values = [
            risk_counts.get("Low Risk", 0),
            risk_counts.get("High Risk", 0)
        ]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.68,
                    marker=dict(
                        colors=[
                            "#16a36a",
                            "#d64d4d"
                        ]
                    ),
                    textinfo="percent",
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Customers: %{value:,}<br>"
                        "%{percent}<extra></extra>"
                    )
                )
            ]
        )

        fig.update_layout(
            height=340,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            ),
            showlegend=True,
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Medium-risk classification is not available "
            "in the current source data."
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

        st.subheader("Portfolio Snapshot")

        summary = pd.DataFrame({
            "Metric": [
                "Customer Records",
                "Low-Risk Customers",
                "High-Risk Customers",
                "Average Income",
                "Average Age",
                "Application Status"
            ],
            "Value": [
                f"{len(filtered):,}",
                f"{good_customers:,}",
                f"{high_risk_customers:,}",
                money(
                    filtered[
                        "AMT_INCOME_TOTAL"
                    ].mean()
                ),
                (
                    f"{filtered['AGE'].mean():.1f} years"
                    if len(filtered)
                    else "N/A"
                ),
                "N/A"
            ]
        })

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )

        st.info(
            "Application dates and application-status "
            "fields are not present in the current source data."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # RISK BY AGE
    # --------------------------------------------------------

    st.subheader("Risk Trend by Age Group")

    age_risk = (
        filtered
        .groupby(
            ["AGE_GROUP", "RISK_CATEGORY"],
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
        },
        title=""
    )

    fig = chart_layout(fig, 390)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # RISK HEATMAP
    # --------------------------------------------------------

    st.subheader("Customer Risk Concentration")

    heatmap_data = pd.crosstab(
        filtered["AGE_GROUP"],
        filtered["INCOME_BAND"]
    )

    heatmap_data = heatmap_data.reindex(
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
        heatmap_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[
            "#eef5ff",
            "#8bb5ef",
            "#174ea6"
        ]
    )

    fig.update_layout(
        height=400,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# APPLICATIONS PAGE
# ============================================================

elif page == "Applications":

    st.markdown(
        '<div class="page-title">Credit Card Applications</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Explore customer application records and risk classification.'
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
        ["Low Risk", "High Risk"],
        default=[]
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
        '<div class="page-title">Customer 360</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Complete customer profile and risk context.'
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

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.subheader("Customer Summary")

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
                    customer["AMT_INCOME_TOTAL"]
                )
            )

            c4.metric(
                "Family Size",
                str(
                    customer["CNT_FAM_MEMBERS"]
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

                    <div style="
                        margin-top:9px;
                    ">
                        {risk_badge(
                            customer["RISK_CATEGORY"]
                        )}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("---")

            # ------------------------------------------------
            # CUSTOMER INFORMATION
            # ------------------------------------------------

            left, right = st.columns(2)

            with left:

                st.subheader(
                    "Applicant Information"
                )

                applicant_info = pd.DataFrame({
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
                        customer["NAME_FAMILY_STATUS"],
                        customer["CNT_CHILDREN"],
                        customer["CNT_FAM_MEMBERS"]
                    ]
                })

                st.dataframe(
                    applicant_info,
                    hide_index=True,
                    use_container_width=True
                )

            with right:

                st.subheader(
                    "Financial Information"
                )

                financial_info = pd.DataFrame({
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
                })

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

            st.subheader("Risk Profile")

            risk_text = (
                "LOW RISK"
                if customer["CREDIT_RISK"] == 0
                else "HIGH RISK"
            )

            st.markdown(
                f"""
                <div class="section-card">

                    <div style="
                        font-size:13px;
                        color:#68768a;
                    ">
                        Risk Category
                    </div>

                    <div style="
                        font-size:25px;
                        font-weight:750;
                        margin-top:7px;
                    ">
                        {risk_text}
                    </div>

                    <div style="
                        margin-top:10px;
                        color:#68768a;
                        font-size:12px;
                    ">
                        A numerical risk score and portfolio
                        exposure amount are not available
                        in the source dataset.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.markdown(
        '<div class="page-title">Risk Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Analyze customer risk concentration across key business segments.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # RISK BY INCOME
    # --------------------------------------------------------

    st.subheader("Risk by Income Band")

    income_risk = pd.crosstab(
        filtered["INCOME_BAND"],
        filtered["RISK_CATEGORY"],
        normalize="index"
    ) * 100

    income_risk = income_risk.reset_index()

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
    # RISK BY AGE
    # --------------------------------------------------------

    st.subheader("Risk by Age Group")

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

    st.subheader("Risk by Occupation")

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

        st.subheader("Risk by Education")

        edu_risk = pd.crosstab(
            filtered["NAME_EDUCATION_TYPE"],
            filtered["RISK_CATEGORY"],
            normalize="index"
        ) * 100

        edu_risk = edu_risk.reset_index()

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

        st.subheader("Risk by Housing")

        housing_risk = pd.crosstab(
            filtered["NAME_HOUSING_TYPE"],
            filtered["RISK_CATEGORY"],
            normalize="index"
        ) * 100

        housing_risk = housing_risk.reset_index()

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
        '<div class="page-title">Portfolio Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Executive-level customer portfolio segmentation and exposure analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    total = len(filtered)

    high_risk = (
        filtered["CREDIT_RISK"] == 1
    ).sum()

    avg_income = (
        filtered["AMT_INCOME_TOTAL"].mean()
        if total else np.nan
    )

    risk_rate = (
        high_risk / total * 100
        if total else np.nan
    )

    c1, c2, c3, c4, c5 = st.columns(5)

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

    c5.metric(
        "Portfolio Exposure",
        "N/A"
    )

    st.markdown("---")

    st.subheader("Portfolio Segmentation")

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
            segment["High_Risk_Percentage"] <= 5,
            segment["High_Risk_Percentage"] <= 15
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
        segment["High_Risk_Percentage"]
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

    st.subheader("High-Risk Concentration")

    fig = px.bar(
        segment.sort_values(
            "High_Risk_Percentage",
            ascending=False
        ).head(15),
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
        chart_layout(fig, 480),
        use_container_width=True
    )


# ============================================================
# RISK ALERTS
# ============================================================

elif page == "Risk Alerts":

    st.markdown(
        '<div class="page-title">Risk Alerts</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Operational risk monitoring based on current portfolio data.'
        '</div>',
        unsafe_allow_html=True
    )

    total = len(filtered)

    high_risk = (
        filtered["CREDIT_RISK"] == 1
    ).sum()

    high_risk_rate = (
        high_risk / total * 100
        if total else 0
    )

    critical = high_risk

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Critical",
        f"{critical:,}"
    )

    c2.metric(
        "High",
        f"{high_risk:,}"
    )

    c3.metric(
        "Medium",
        "N/A"
    )

    c4.metric(
        "Resolved",
        "N/A"
    )

    st.markdown("---")

    if high_risk > 0:

        st.markdown(
            f"""
            <div class="alert-card">
                <b>🔴 High-Risk Customer Concentration</b>
                <br>
                <span style="color:#68768a;">
                {high_risk:,} customers are currently classified
                as high risk, representing
                {high_risk_rate:.1f}% of the filtered portfolio.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Income segment alert

    income_alert = (
        filtered
        .groupby("INCOME_BAND")
        .agg(
            Customers=("ID", "count"),
            High_Risk=("CREDIT_RISK", "sum")
        )
    )

    income_alert["Rate"] = (
        income_alert["High_Risk"]
        / income_alert["Customers"]
        * 100
    )

    if not income_alert.empty:

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
                <b>🟠 Risk Concentration</b>
                <br>
                <span style="color:#68768a;">
                The {highest_band} income segment has the
                highest observed high-risk rate at
                {highest_rate:.1f}%.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Missing occupation

    missing_occupation = (
        df["OCCUPATION_TYPE"]
        .eq("Unknown")
        .sum()
    )

    if missing_occupation > 0:

        st.markdown(
            f"""
            <div class="alert-card alert-info">
                <b>🟡 Incomplete Customer Profile</b>
                <br>
                <span style="color:#68768a;">
                {missing_occupation:,} customer records have
                an unknown occupation value.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Alert Data")

    alert_table = pd.DataFrame({
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
        ],
        "Date": [
            TODAY,
            TODAY,
            TODAY
        ]
    })

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
        '<div class="page-title">Reporting Center</div>',
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

    st.subheader("Export Current Customer Data")

    export_df = filtered.copy()

    export_df.columns = [
        c.replace("_", " ").title()
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

    # Excel export

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

    st.info(
        "PDF generation is not enabled because the current "
        "dashboard does not contain a dedicated report-layout "
        "engine. CSV and Excel exports contain the actual data."
    )


# ============================================================
# AUDIT & COMPLIANCE
# ============================================================

elif page == "Audit & Compliance":

    st.markdown(
        '<div class="page-title">Audit & Compliance</div>',
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

    st.markdown("---")

    completeness = (
        complete_records
        / total_records
        * 100
        if total_records
        else 0
    )

    st.subheader("Data Completeness")

    st.progress(
        int(completeness)
    )

    st.write(
        f"{completeness:.1f}% complete records"
    )

    st.subheader("Field-Level Data Quality")

    quality = pd.DataFrame({
        "Field": df.columns,
        "Missing Values": [
            int(df[c].isna().sum())
            for c in df.columns
        ],
        "Missing %": [
            round(
                df[c].isna().mean() * 100,
                2
            )
            for c in df.columns
        ]
    })

    st.dataframe(
        quality,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Audit Activity")

    audit = pd.DataFrame({
        "Timestamp": [TODAY],
        "User": ["Risk Analyst"],
        "Action": ["Portfolio Data Review"],
        "Application ID": ["N/A"],
        "Customer ID": ["N/A"],
        "Previous Status": ["N/A"],
        "New Status": ["N/A"],
        "Reason": [
            "Dashboard portfolio analysis"
        ]
    })

    st.dataframe(
        audit,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# SETTINGS
# ============================================================

elif page == "Settings":

    st.markdown(
        '<div class="page-title">Settings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Dashboard and data configuration.'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader("Dashboard Preferences")

    theme = st.selectbox(
        "Interface Theme",
        [
            "Enterprise Light",
            "Enterprise Dark"
        ]
    )

    st.selectbox(
        "Default Analysis Period",
        [
            "Today",
            "7 Days",
            "30 Days",
            "90 Days"
        ],
        index=2
    )

    st.subheader("Data Source")

    st.code(
        "GitHub Repository: Abhishek131004/CreditGuard"
    )

    st.success(
        "Data source connection configured."
    )

    st.subheader("Available Source Fields")

    st.dataframe(
        pd.DataFrame({
            "Field": df.columns
        }),
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        CreditGuard &nbsp;•&nbsp;
        Credit Risk Intelligence & Portfolio Management
        <br>
        Enterprise analytics workspace
    </div>
    """,
    unsafe_allow_html=True
)
