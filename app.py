import streamlit as st


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

    /* ========================================================
       GLOBAL
    ======================================================== */

    .stApp {
        background-color: #f5f7fa;
        color: #172033;
    }

    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }

    /* ========================================================
       SIDEBAR
    ======================================================== */

    [data-testid="stSidebar"] {
        background-color: #0b172a;
        border-right: 1px solid #1d3048;
    }

    [data-testid="stSidebar"] * {
        color: #e8eef7 !important;
    }

    /* ========================================================
       BRAND
    ======================================================== */

    .brand-container {
        padding: 8px 5px 25px 5px;
    }

    .brand-logo {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .shield {
        width: 44px;
        height: 44px;
        background: #1e5eff;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 23px;
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

    /* ========================================================
       HERO
    ======================================================== */

    .hero {
        background: white;
        border: 1px solid #e1e7ef;
        border-radius: 20px;
        padding: 55px 50px;
        box-shadow: 0 6px 25px rgba(16, 32, 55, 0.05);
        margin-bottom: 25px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        background: #eaf1ff;
        color: #1e5eff;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        color: #142033;
        line-height: 1.1;
        margin-bottom: 15px;
    }

    .hero-title span {
        color: #1e5eff;
    }

    .hero-text {
        font-size: 16px;
        color: #68768a;
        max-width: 850px;
        line-height: 1.7;
    }

    /* ========================================================
       CARDS
    ======================================================== */

    .dashboard-card {
        background: white;
        border: 1px solid #e1e7ef;
        border-radius: 16px;
        padding: 25px;
        min-height: 260px;
        box-shadow: 0 4px 16px rgba(16, 32, 55, 0.04);
    }

    .dashboard-icon {
        font-size: 35px;
        margin-bottom: 12px;
    }

    .dashboard-title {
        font-size: 22px;
        font-weight: 750;
        color: #152238;
    }

    .dashboard-text {
        color: #68768a;
        font-size: 14px;
        line-height: 1.7;
        margin-top: 10px;
    }

    .feature {
        margin-top: 8px;
        color: #405069;
        font-size: 13px;
    }

    /* ========================================================
       KPI
    ======================================================== */

    .kpi-card {
        background: white;
        border: 1px solid #e1e7ef;
        border-radius: 14px;
        padding: 20px;
        min-height: 115px;
    }

    .kpi-label {
        color: #68768a;
        font-size: 12px;
        font-weight: 600;
    }

    .kpi-value {
        color: #142033;
        font-size: 25px;
        font-weight: 800;
        margin-top: 7px;
    }

    /* ========================================================
       FOOTER
    ======================================================== */

    .footer {
        color: #8490a1;
        font-size: 11px;
        text-align: center;
        padding: 35px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR BRANDING
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
                    Credit Risk Intelligence
                </div>
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="
        font-size:12px;
        color:#8ea2bd;
        margin-bottom:6px;
    ">
        WORKSPACE
    </div>

    <div style="
        font-size:13px;
        color:#e8eef7;
        line-height:1.6;
    ">
        Credit Risk Analytics<br>
        Portfolio Intelligence<br>
        Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)


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
    ">
        <span style="color:#28c47c;">●</span>
        All systems operational
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            CREDIT RISK INTELLIGENCE PLATFORM
        </div>

        <div class="hero-title">
            Welcome to <span>CreditGuard</span>
        </div>

        <div class="hero-text">
            An enterprise-style credit risk analytics platform
            designed to explore customer portfolios, identify
            risk patterns and evaluate applicant credit risk
            using machine-learning models.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KPI ROW
# ============================================================

k1, k2, k3, k4 = st.columns(4)


with k1:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">ANALYTICS MODULE</div>
            <div class="kpi-value">EDA</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with k2:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">ML MODELS</div>
            <div class="kpi-value">3</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with k3:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">RISK ENGINE</div>
            <div class="kpi-value">LIVE</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with k4:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">PLATFORM</div>
            <div class="kpi-value">STREAMLIT</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# DASHBOARD CARDS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        """
        <div class="dashboard-card">

            <div class="dashboard-icon">
                📊
            </div>

            <div class="dashboard-title">
                EDA & Portfolio Intelligence
            </div>

            <div class="dashboard-text">

                Understand the credit portfolio through
                interactive exploratory analysis and
                customer segmentation.

            </div>

            <div class="feature">
                ✓ Customer demographics
            </div>

            <div class="feature">
                ✓ Income & employment analysis
            </div>

            <div class="feature">
                ✓ Education & housing analysis
            </div>

            <div class="feature">
                ✓ Risk concentration
            </div>

            <div class="feature">
                ✓ Portfolio segmentation
            </div>

            <div class="feature">
                ✓ Interactive filters & reports
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="dashboard-card">

            <div class="dashboard-icon">
                🤖
            </div>

            <div class="dashboard-title">
                ML Credit Risk Intelligence
            </div>

            <div class="dashboard-text">

                Assess an individual applicant using
                trained classification models and
                compare model performance.

            </div>

            <div class="feature">
                ✓ Applicant risk assessment
            </div>

            <div class="feature">
                ✓ Risk probability
            </div>

            <div class="feature">
                ✓ Logistic Regression
            </div>

            <div class="feature">
                ✓ Random Forest
            </div>

            <div class="feature">
                ✓ XGBoost
            </div>

            <div class="feature">
                ✓ Confusion matrix & metrics
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    """
    <div style="
        font-size:22px;
        font-weight:750;
        color:#152238;
        margin-bottom:15px;
    ">
        How CreditGuard Works
    </div>
    """,
    unsafe_allow_html=True
)


steps = st.columns(3)


with steps[0]:

    st.markdown(
        """
        <div class="dashboard-card"
             style="min-height:170px;">

            <b>01 · Explore</b>

            <div class="dashboard-text">
                Analyze customer characteristics,
                portfolio composition and historical
                credit-risk patterns.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with steps[1]:

    st.markdown(
        """
        <div class="dashboard-card"
             style="min-height:170px;">

            <b>02 · Assess</b>

            <div class="dashboard-text">
                Enter applicant information and
                evaluate credit risk using the
                selected machine-learning model.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with steps[2]:

    st.markdown(
        """
        <div class="dashboard-card"
             style="min-height:170px;">

            <b>03 · Decide</b>

            <div class="dashboard-text">
                Review the predicted risk,
                probability and model performance
                before making an analytical decision.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        CreditGuard • Credit Risk Intelligence & Portfolio Analytics

        <br>

        Enterprise analytics workspace

    </div>
    """,
    unsafe_allow_html=True
)
