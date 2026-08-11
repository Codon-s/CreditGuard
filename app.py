import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from eda_dashboard import render_eda
from ml_dashboard import render_ml

st.set_page_config(
    page_title="CreditGuard | Credit Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- THEME ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --navy:#081426; --navy2:#0d1f36; --blue:#2f6fed; --blue2:#eaf1ff;
  --text:#122033; --muted:#6b788b; --border:#e2e8f0; --surface:#ffffff;
  --bg:#f4f7fb; --green:#15956b; --red:#d94a4a; --amber:#c98618;
}
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background:var(--bg); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,var(--navy),#0a192d); }
[data-testid="stSidebar"] * { color:#edf4ff !important; }
.block-container { max-width:1450px; padding-top:1.5rem; padding-bottom:3rem; }
div[data-testid="stMetric"] {
  background:var(--surface); border:1px solid var(--border); border-radius:16px;
  padding:16px 18px; box-shadow:0 6px 22px rgba(12,31,55,.05);
}
.hero {
  background:linear-gradient(135deg,#0a1729 0%,#102846 65%,#173d72 100%);
  color:white; border-radius:24px; padding:34px 38px; margin:0 0 22px 0;
  box-shadow:0 14px 40px rgba(10,28,52,.18);
}
.hero .eyebrow { color:#9ec1ff; font-size:12px; font-weight:800; letter-spacing:1.8px; }
.hero h1 { font-size:42px; margin:8px 0 10px; letter-spacing:-1.4px; }
.hero p { color:#c9d6e8; max-width:780px; line-height:1.65; margin:0; }
.brand { padding:14px 8px 20px; }
.brand-row { display:flex; gap:12px; align-items:center; }
.shield { width:44px; height:44px; display:flex; align-items:center; justify-content:center;
          background:#2f6fed; border-radius:13px; font-size:22px; }
.brand-name { font-size:21px; font-weight:800; color:white; }
.brand-sub { font-size:11px; color:#9eb0c7; margin-top:2px; }
.nav-note { color:#8ea2bd; font-size:11px; letter-spacing:1px; font-weight:700; margin:12px 0 8px; }
.card { background:white; border:1px solid var(--border); border-radius:18px; padding:22px;
        box-shadow:0 7px 25px rgba(12,31,55,.045); }
.card h3 { margin:0 0 7px; color:var(--text); }
.card p { color:var(--muted); line-height:1.55; }
.kicker { font-size:11px; font-weight:800; letter-spacing:1.2px; color:#5f7aa4; }
.big { font-size:28px; font-weight:800; color:var(--text); margin:5px 0; }
.pill { display:inline-block; padding:5px 10px; border-radius:999px; font-size:11px; font-weight:800; }
.pill-blue { background:#eaf1ff; color:#255cc3; }
.pill-green { background:#e6f7f0; color:#0b7d56; }
.pill-red { background:#fdeceb; color:#b82e2e; }
.small { color:var(--muted); font-size:12px; }
div.stButton > button { border-radius:11px; font-weight:700; min-height:42px; }
[data-testid="stSidebar"] .stRadio > div { gap:5px; }
[data-testid="stSidebar"] .stRadio label { padding:8px 10px; border-radius:10px; }
[data-testid="stSidebar"] .stRadio label:hover { background:#152b49; }
hr { border-color:#e8edf4; }
@media (max-width:900px) {
  .hero h1 { font-size:32px; }
  .block-container { padding-left:1rem; padding-right:1rem; }
}
</style>
""", unsafe_allow_html=True)

APP_URL = "https://github.com/Abhishek131004/CreditGuard/raw/refs/heads/main/Applications_cc.csv.zip"
CREDIT_URL = "https://github.com/Abhishek131004/CreditGuard/raw/refs/heads/main/credit_record%20_cc.csv.zip"

@st.cache_data(show_spinner=False)
def load_data():
    def read_source(url, names):
        for p in names:
            if Path(p).exists():
                return pd.read_csv(p, compression="zip" if str(p).endswith(".zip") else None)
        return pd.read_csv(url, compression="zip")

    applications = read_source(APP_URL, [
        "Applications_cc.csv.zip", "Applications_cc.csv",
        "data/Applications_cc.csv.zip", "data/Applications_cc.csv"
    ])
    credit_record = read_source(CREDIT_URL, [
        "credit_record _cc.csv.zip", "credit_record_cc.csv.zip",
        "credit_record _cc.csv", "credit_record_cc.csv",
        "data/credit_record _cc.csv.zip", "data/credit_record_cc.csv.zip"
    ])

    applications = applications.drop_duplicates(subset=["ID"], keep="first")
    credit_record = credit_record.drop_duplicates(subset=["ID"], keep="first")
    df = applications.merge(credit_record, on="ID", how="inner")

    # Preserve the user's existing cleaning / risk definition.
    df["NAME_EDUCATION_TYPE"] = df["NAME_EDUCATION_TYPE"].replace(
        "Secondary / secondary special", "Secondary special"
    )
    df["NAME_FAMILY_STATUS"] = df["NAME_FAMILY_STATUS"].replace(
        "Single / not married", "Single"
    )
    df["NAME_HOUSING_TYPE"] = df["NAME_HOUSING_TYPE"].replace(
        "House / apartment", "House"
    )
    df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown").replace(
        "Waiters/barmen staff", "Waiters"
    )

    df["STATUS"] = df["STATUS"].astype(str)
    status_map = {"0":0, "1":1, "2":1, "3":1, "4":1, "5":1, "C":0, "X":0}
    df["CREDIT_RISK"] = df["STATUS"].map(status_map)
    df = df.dropna(subset=["CREDIT_RISK"]).copy()
    df["CREDIT_RISK"] = df["CREDIT_RISK"].astype(int)

    df["AGE"] = (-df["DAYS_BIRTH"] // 365).astype(int)
    df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365
    df.loc[df["EMPLOYMENT_YEARS"] < 0, "EMPLOYMENT_YEARS"] = np.nan
    df["RISK_CATEGORY"] = np.where(df["CREDIT_RISK"].eq(0), "Low Risk", "High Risk")
    df["INCOME_BAND"] = pd.cut(
        df["AMT_INCOME_TOTAL"],
        [-np.inf,200000,500000,1000000,2000000,np.inf],
        labels=["< ₹2L","₹2–5L","₹5–10L","₹10–20L","₹20L+"]
    )
    df["AGE_GROUP"] = pd.cut(
        df["AGE"], [17,25,35,45,55,np.inf],
        labels=["18–25","26–35","36–45","46–55","56+"]
    )
    return df

def money(v):
    if pd.isna(v): return "N/A"
    if v >= 1e7: return f"₹{v/1e7:.1f}Cr"
    if v >= 1e5: return f"₹{v/1e5:.1f}L"
    return f"₹{v:,.0f}"

def styled_chart(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(l=10,r=10,t=45,b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, Arial", color="#243247"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e4e8ee")
    fig.update_yaxes(gridcolor="#edf0f4", zeroline=False)
    return fig

try:
    df = load_data()
except Exception as e:
    st.error("Unable to load CreditGuard data.")
    st.info("Place the two ZIP/CSV files in the project root or verify the GitHub source files.")
    with st.expander("Technical details"):
        st.code(str(e))
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.markdown("""
<div class="brand">
  <div class="brand-row">
    <div class="shield">🛡️</div>
    <div><div class="brand-name">CreditGuard</div>
    <div class="brand-sub">Credit Risk Intelligence</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="nav-note">WORKSPACE</div>', unsafe_allow_html=True)
PAGES = ["Executive Overview", "EDA Dashboard", "ML Model Dashboard"]
if "page" not in st.session_state:
    st.session_state["page"] = PAGES[0]
page = st.sidebar.radio(
    "Workspace",
    PAGES,
    label_visibility="collapsed",
    key="page",
)

with st.sidebar.expander("Global Filters", expanded=False):
    gender = st.multiselect("Gender", sorted(df["CODE_GENDER"].dropna().unique()))
    education = st.multiselect("Education", sorted(df["NAME_EDUCATION_TYPE"].dropna().unique()))
    risk = st.multiselect("Risk Category", ["Low Risk","High Risk"])
    age = st.slider("Age", int(df["AGE"].min()), int(df["AGE"].max()),
                    (int(df["AGE"].min()), int(df["AGE"].max())))
    income = st.slider("Annual Income", float(df["AMT_INCOME_TOTAL"].min()),
                       float(df["AMT_INCOME_TOTAL"].max()),
                       (float(df["AMT_INCOME_TOTAL"].min()), float(df["AMT_INCOME_TOTAL"].max())))

filtered = df.copy()
if gender: filtered = filtered[filtered["CODE_GENDER"].isin(gender)]
if education: filtered = filtered[filtered["NAME_EDUCATION_TYPE"].isin(education)]
if risk: filtered = filtered[filtered["RISK_CATEGORY"].isin(risk)]
filtered = filtered[filtered["AGE"].between(age[0], age[1])]
filtered = filtered[filtered["AMT_INCOME_TOTAL"].between(income[0], income[1])]

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="small">● <b style="color:#4dd99b">All systems operational</b><br><br>'
    '<span style="color:#8ea2bd">DATASET</span><br>CreditGuard portfolio<br><br>'
    '<span style="color:#8ea2bd">RECORDS</span><br>'
    f'{len(df):,} source records</div>', unsafe_allow_html=True
)

# ---------- EXECUTIVE HOME ----------
if page == "Executive Overview":
    total = len(filtered)
    high = int(filtered["CREDIT_RISK"].eq(1).sum())
    low = int(filtered["CREDIT_RISK"].eq(0).sum())
    risk_rate = high / total * 100 if total else 0
    avg_income = filtered["AMT_INCOME_TOTAL"].mean() if total else np.nan

    st.markdown("""
    <div class="hero">
      <div class="eyebrow">CREDIT RISK INTELLIGENCE PLATFORM</div>
      <h1>Welcome to CreditGuard</h1>
      <p>Enterprise-style portfolio intelligence for exploring customer risk,
      financial segments and credit-status patterns from the underlying portfolio data.</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Customer Records", f"{total:,}")
    c2.metric("Low-Risk Customers", f"{low:,}")
    c3.metric("High-Risk Customers", f"{high:,}")
    c4.metric("Portfolio Risk Rate", f"{risk_rate:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    a,b = st.columns([1.1, 1.9])
    with a:
        st.markdown('<div class="card"><div class="kicker">RISK MIX</div><h3>Portfolio Risk Distribution</h3>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["Low Risk","High Risk"], values=[low,high], hole=.68,
            marker=dict(colors=["#15956b","#d94a4a"]),
            textinfo="percent", hovertemplate="<b>%{label}</b><br>%{value:,} customers<extra></extra>"
        ))
        fig.update_layout(height=330, margin=dict(l=5,r=5,t=10,b=10), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="kicker">PORTFOLIO SNAPSHOT</div><h3>Current Data Profile</h3>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Metric":["Source records","Filtered records","Average annual income","Average age","Unknown occupation"],
            "Value":[f"{len(df):,}",f"{total:,}",money(avg_income),
                     f"{filtered['AGE'].mean():.1f} years" if total else "N/A",
                     f"{int(filtered['OCCUPATION_TYPE'].eq('Unknown').sum()):,}"]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True, height=260)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    x,y,z = st.columns(3)
    with x:
        st.markdown('<div class="card"><span class="pill pill-blue">EDA</span><h3>Explore the portfolio</h3><p>Distributions, relationships, segmentation and data-quality views.</p></div>', unsafe_allow_html=True)
        if st.button("Open EDA Dashboard →", key="open_eda", use_container_width=True):
            st.session_state["page"] = "EDA Dashboard"
            st.rerun()
    with y:
        st.markdown('<div class="card"><span class="pill pill-green">RISK ENGINE</span><h3>Assess customer risk</h3><p>Run the trained classifiers and compare their business-relevant performance.</p></div>', unsafe_allow_html=True)
        if st.button("Open ML Dashboard →", key="open_ml", use_container_width=True):
            st.session_state["page"] = "ML Model Dashboard"
            st.rerun()
    with z:
        st.markdown('<div class="card"><span class="pill pill-red">DATA</span><h3>Inspect source records</h3><p>Use the filters in the sidebar to narrow the portfolio before opening either workspace.</p></div>', unsafe_allow_html=True)

elif page == "EDA Dashboard":
    render_eda(filtered, df)

else:
    render_ml(df)

st.markdown('<div style="text-align:center;color:#8a96a8;font-size:11px;padding:30px 0 0;">CreditGuard • Credit Risk Intelligence & Portfolio Analytics</div>', unsafe_allow_html=True)
