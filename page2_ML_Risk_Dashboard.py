import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CreditGuard | ML Risk",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DATA URLS
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
# CSS
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
    }

    .page-subtitle {
        color:#68768a;
        font-size:14px;
        margin-bottom:22px;
    }

    .section-card {
        background:white;
        border:1px solid #e2e7ef;
        border-radius:14px;
        padding:22px;
        box-shadow:0 3px 12px rgba(16,32,55,0.04);
    }

    .section-title {
        font-size:19px;
        font-weight:750;
        color:#152238;
        margin-bottom:15px;
    }

    .metric-card {
        background:white;
        border:1px solid #e1e7ef;
        border-radius:14px;
        padding:18px;
        min-height:120px;
    }

    .metric-label {
        color:#68768a;
        font-size:12px;
        font-weight:600;
    }

    .metric-value {
        color:#142033;
        font-size:26px;
        font-weight:800;
        margin-top:7px;
    }

    .risk-low {
        background:#ecfdf5;
        border:1px solid #a7e8c8;
        border-left:6px solid #16a36a;
        border-radius:15px;
        padding:25px;
    }

    .risk-high {
        background:#fff1f1;
        border:1px solid #f2b4b4;
        border-left:6px solid #d64d4d;
        border-radius:15px;
        padding:25px;
    }

    .risk-title {
        font-size:31px;
        font-weight:800;
        margin-top:5px;
    }

    .risk-description {
        color:#68768a;
        font-size:13px;
        margin-top:8px;
    }

    .model-card {
        background:white;
        border:1px solid #e1e7ef;
        border-radius:14px;
        padding:18px;
        min-height:150px;
    }

    .model-name {
        color:#68768a;
        font-size:12px;
        font-weight:700;
    }

    .model-score {
        color:#142033;
        font-size:26px;
        font-weight:800;
        margin-top:7px;
    }

    .footer {
        color:#8490a1;
        font-size:11px;
        text-align:center;
        padding:30px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA PREPARATION
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


    applications = (
        applications
        .drop_duplicates(
            subset=["ID"],
            keep="first"
        )
    )


    credit_record = (
        credit_record
        .drop_duplicates(
            subset=["ID"],
            keep="first"
        )
    )


    df = pd.merge(
        applications,
        credit_record,
        on="ID",
        how="inner"
    )


    # --------------------------------------------------------
    # SAME CLEANING AS EDA
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

    df["STATUS"] = (
        df["STATUS"]
        .astype(str)
    )


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


    return df


# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource(show_spinner=False)
def train_models(df):

    data = df.copy()


    # --------------------------------------------------------
    # EXACT FEATURE REMOVAL FROM YOUR ML CODE
    # --------------------------------------------------------

    data = data.drop(
        [
            "ID",
            "FLAG_MOBIL",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED"
        ],
        axis=1,
        errors="ignore"
    )


    # --------------------------------------------------------
    # LABEL ENCODING
    # --------------------------------------------------------

    categorical_columns = [
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE"
    ]


    encoders = {}


    for column in categorical_columns:

        encoder = LabelEncoder()

        data[column] = (
            encoder
            .fit_transform(
                data[column]
                .astype(str)
            )
        )

        encoders[column] = encoder


    # --------------------------------------------------------
    # STATUS ENCODER
    # --------------------------------------------------------

    status_encoder = LabelEncoder()

    data["STATUS"] = (
        status_encoder
        .fit_transform(
            data["STATUS"]
            .astype(str)
        )
    )


    encoders["STATUS"] = (
        status_encoder
    )


    # --------------------------------------------------------
    # X / Y
    # --------------------------------------------------------

    X = data.drop(
        [
            "CREDIT_RISK",
            "STATUS"
        ],
        axis=1
    )


    y = data["CREDIT_RISK"]


    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )


    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    logistic = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )


    logistic.fit(
        X_train,
        y_train
    )


    logistic_pred = (
        logistic.predict(
            X_test
        )
    )


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf_base = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )


    rf_params = {
        "n_estimators": [
            50,
            100,
            200
        ]
    }


    rf_grid = GridSearchCV(
        estimator=rf_base,
        param_grid=rf_params,
        scoring="accuracy",
        cv=5,
        n_jobs=-1
    )


    rf_grid.fit(
        X_train,
        y_train
    )


    random_forest = (
        rf_grid.best_estimator_
    )


    rf_pred = (
        random_forest.predict(
            X_test
        )
    )


    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    xgb = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )


    xgb.fit(
        X_train,
        y_train
    )


    xgb_pred = (
        xgb.predict(
            X_test
        )
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    def calculate_metrics(
        actual,
        predicted
    ):

        return {

            "accuracy":
                accuracy_score(
                    actual,
                    predicted
                ),

            "precision":
                precision_score(
                    actual,
                    predicted,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    actual,
                    predicted,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    actual,
                    predicted,
                    zero_division=0
                ),

            "confusion_matrix":
                confusion_matrix(
                    actual,
                    predicted
                )

        }


    metrics = {

        "Logistic Regression":
            calculate_metrics(
                y_test,
                logistic_pred
            ),

        "Random Forest":
            calculate_metrics(
                y_test,
                rf_pred
            ),

        "XGBoost":
            calculate_metrics(
                y_test,
                xgb_pred
            )
    }


    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
        "XGBoost": xgb,
        "encoders": encoders,
        "feature_columns": list(X.columns),
        "metrics": metrics,
        "best_rf_params": rf_grid.best_params_
    }


# ============================================================
# LOAD DATA / TRAIN
# ============================================================

with st.spinner(
    "Preparing credit-risk intelligence engine..."
):

    df = load_data()

    models = train_models(
        df
    )


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
                    ML Risk Intelligence
                </div>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown(
    "### ML Navigation"
)


ml_page = st.sidebar.radio(
    "Navigation",
    [
        "Risk Prediction",
        "Model Performance",
        "Model Details"
    ],
    label_visibility="collapsed"
)


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
    """
    <div class="topbar">

        <b>CreditGuard</b>
        &nbsp; / &nbsp;
        Machine Learning Risk Intelligence

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RISK PREDICTION
# ============================================================

if ml_page == "Risk Prediction":

    st.markdown(
        '<div class="page-title">'
        'AI Credit Risk Assessment'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Enter applicant information to generate a '
        'machine-learning-based credit-risk assessment.'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    selected_model = st.selectbox(
        "Prediction Model",
        [
            "Random Forest",
            "XGBoost",
            "Logistic Regression"
        ]
    )


    model = models[
        selected_model
    ]


    st.markdown(
        '<div class="section-title">'
        '👤 Applicant Profile'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # APPLICANT PROFILE
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)


    with c1:

        gender = st.selectbox(
            "Gender",
            sorted(
                df[
                    "CODE_GENDER"
                ]
                .dropna()
                .unique()
            )
        )


    with c2:

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=20,
            value=0
        )


    with c3:

        family_members = st.number_input(
            "Family Members",
            min_value=1,
            max_value=20,
            value=2
        )


    # --------------------------------------------------------
    # FINANCIAL PROFILE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '💰 Financial Profile'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        income = st.number_input(
            "Annual Income (₹)",
            min_value=0.0,
            value=300000.0,
            step=10000.0
        )


    with c2:

        car = st.selectbox(
            "Owns Car",
            ["Yes", "No"]
        )


    with c3:

        property_owner = st.selectbox(
            "Owns Property",
            ["Yes", "No"]
        )


    # --------------------------------------------------------
    # EMPLOYMENT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '💼 Employment Profile'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2 = st.columns(2)


    income_types = sorted(
        df[
            "NAME_INCOME_TYPE"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    occupations = sorted(
        df[
            "OCCUPATION_TYPE"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    with c1:

        income_type = st.selectbox(
            "Income Type",
            income_types
        )


    with c2:

        occupation = st.selectbox(
            "Occupation",
            occupations
        )


    # --------------------------------------------------------
    # EDUCATION / FAMILY / HOUSING
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🎓 Education & Housing'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    education_types = sorted(
        df[
            "NAME_EDUCATION_TYPE"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    family_statuses = sorted(
        df[
            "NAME_FAMILY_STATUS"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    housing_types = sorted(
        df[
            "NAME_HOUSING_TYPE"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    with c1:

        education = st.selectbox(
            "Education",
            education_types
        )


    with c2:

        family_status = st.selectbox(
            "Family Status",
            family_statuses
        )


    with c3:

        housing = st.selectbox(
            "Housing Type",
            housing_types
        )


    # --------------------------------------------------------
    # CONTACT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📱 Contact Information'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        work_phone = st.selectbox(
            "Work Phone",
            ["Yes", "No"]
        )


    with c2:

        phone = st.selectbox(
            "Phone",
            ["Yes", "No"]
        )


    with c3:

        email = st.selectbox(
            "Email",
            ["Yes", "No"]
        )


    with c4:

        months_balance = st.number_input(
            "Months Balance",
            min_value=0,
            max_value=60,
            value=12
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # PREDICT
    # ========================================================

    predict = st.button(
        "🔍 ASSESS CREDIT RISK",
        type="primary",
        use_container_width=True
    )


    if predict:

        # ----------------------------------------------------
        # RAW INPUT
        # ----------------------------------------------------

        input_data = {

            "CODE_GENDER":
                gender,

            "FLAG_OWN_CAR":
                car == "Yes",

            "FLAG_OWN_REALTY":
                property_owner == "Yes",

            "CNT_CHILDREN":
                children,

            "AMT_INCOME_TOTAL":
                income,

            "NAME_INCOME_TYPE":
                income_type,

            "NAME_EDUCATION_TYPE":
                education,

            "NAME_FAMILY_STATUS":
                family_status,

            "NAME_HOUSING_TYPE":
                housing,

            "FLAG_WORK_PHONE":
                int(
                    work_phone == "Yes"
                ),

            "FLAG_PHONE":
                int(
                    phone == "Yes"
                ),

            "FLAG_EMAIL":
                int(
                    email == "Yes"
                ),

            "OCCUPATION_TYPE":
                occupation,

            "CNT_FAM_MEMBERS":
                family_members,

            "MONTHS_BALANCE":
                months_balance
        }


        input_df = pd.DataFrame(
            [input_data]
        )


        # ----------------------------------------------------
        # ENCODE
        # ----------------------------------------------------

        for column, encoder in (
            models[
                "encoders"
            ].items()
        ):

            if column == "STATUS":
                continue


            if column in input_df.columns:

                value = str(
                    input_df.loc[
                        0,
                        column
                    ]
                )


                classes = list(
                    encoder.classes_
                )


                if value in classes:

                    input_df[column] = (
                        encoder.transform(
                            [value]
                        )[0]
                    )

                else:

                    st.error(
                        f"Unknown category "
                        f"for {column}: {value}"
                    )

                    st.stop()


        # ----------------------------------------------------
        # BOOLEAN COLUMNS
        # ----------------------------------------------------

        input_df[
            "FLAG_OWN_CAR"
        ] = (
            input_df[
                "FLAG_OWN_CAR"
            ]
            .astype(int)
        )


        input_df[
            "FLAG_OWN_REALTY"
        ] = (
            input_df[
                "FLAG_OWN_REALTY"
            ]
            .astype(int)
        )


        # ----------------------------------------------------
        # FEATURE ORDER
        # ----------------------------------------------------

        input_df = input_df[
            models[
                "feature_columns"
            ]
        ]


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        )[0]


        probability = (
            model.predict_proba(
                input_df
            )[0][1]
        )


        risk_probability = (
            probability * 100
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:

            risk_title = (
                "🔴 HIGH CREDIT RISK"
            )

            risk_class = "risk-high"

            description = (
                "The model identifies an elevated "
                "probability of credit-risk behavior."
            )

        else:

            risk_title = (
                "🟢 LOW CREDIT RISK"
            )

            risk_class = "risk-low"

            description = (
                "The model identifies a relatively "
                "lower probability of credit-risk behavior."
            )


        st.markdown("<br>", unsafe_allow_html=True)


        st.markdown(
            f"""
            <div class="{risk_class}">

                <div style="
                    font-size:12px;
                    font-weight:700;
                    color:#68768a;
                ">
                    CREDIT RISK ASSESSMENT
                </div>

                <div class="risk-title">
                    {risk_title}
                </div>

                <div class="risk-description">
                    {description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown("<br>", unsafe_allow_html=True)


        # ----------------------------------------------------
        # RESULT METRICS
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)


        with c1:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        RISK PROBABILITY
                    </div>

                    <div class="metric-value">
                        {risk_probability:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with c2:

            prediction_text = (
                "Credit Risk"
                if prediction == 1
                else "No Credit Risk"
            )


            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        MODEL DECISION
                    </div>

                    <div class="metric-value"
                         style="font-size:21px;">

                        {prediction_text}

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with c3:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        MODEL USED
                    </div>

                    <div class="metric-value"
                         style="font-size:20px;">

                        {selected_model}

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # PROBABILITY CHART
        # ----------------------------------------------------

        st.markdown("<br>", unsafe_allow_html=True)


        probability_df = pd.DataFrame(
            {
                "Risk Type": [
                    "No Credit Risk",
                    "Credit Risk"
                ],

                "Probability": [
                    100 - risk_probability,
                    risk_probability
                ]
            }
        )


        fig = px.bar(
            probability_df,
            x="Risk Type",
            y="Probability",
            text="Probability",
            title="Prediction Probability"
        )


        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )


        fig.update_layout(
            height=380,
            paper_bgcolor="white",
            plot_bgcolor="white",
            yaxis=dict(
                range=[
                    0,
                    105
                ],
                title="Probability (%)"
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif ml_page == "Model Performance":

    st.markdown(
        '<div class="page-title">'
        'Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Compare classification performance across '
        'the three trained models.'
        '</div>',
        unsafe_allow_html=True
    )


    metrics = models[
        "metrics"
    ]


    # --------------------------------------------------------
    # MODEL CARDS
    # --------------------------------------------------------

    model_names = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ]


    columns = st.columns(3)


    for i, name in enumerate(
        model_names
    ):

        m = metrics[name]


        with columns[i]:

            st.markdown(
                f"""
                <div class="model-card">

                    <div class="model-name">
                        {name.upper()}
                    </div>

                    <div class="model-score">
                        {m["accuracy"] * 100:.2f}%
                    </div>

                    <div style="
                        color:#68768a;
                        font-size:12px;
                        margin-top:8px;
                    ">

                        Accuracy

                        <br>

                        Precision:
                        {m["precision"]:.3f}

                        <br>

                        Recall:
                        {m["recall"]:.3f}

                        <br>

                        F1:
                        {m["f1"]:.3f}

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        {
            "Model": model_names,

            "Accuracy": [
                metrics[m]["accuracy"]
                for m in model_names
            ],

            "Precision": [
                metrics[m]["precision"]
                for m in model_names
            ],

            "Recall": [
                metrics[m]["recall"]
                for m in model_names
            ],

            "F1 Score": [
                metrics[m]["f1"]
                for m in model_names
            ]
        }
    )


    st.subheader(
        "Model Comparison"
    )


    display_comparison = comparison.copy()


    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        display_comparison[
            column
        ] = (
            display_comparison[
                column
            ] * 100
        ).round(2).astype(str) + "%"


    st.dataframe(
        display_comparison,
        hide_index=True,
        use_container_width=True
    )


    # --------------------------------------------------------
    # PERFORMANCE CHART
    # --------------------------------------------------------

    chart_data = comparison.melt(
        id_vars="Model",
        value_vars=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        var_name="Metric",
        value_name="Score"
    )


    chart_data["Score"] = (
        chart_data["Score"]
        * 100
    )


    fig = px.bar(
        chart_data,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model Performance Comparison"
    )


    fig.update_layout(
        height=450,
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis_title="Score (%)"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )


    matrix_model = st.selectbox(
        "Select Model",
        model_names
    )


    cm = metrics[
        matrix_model
    ]["confusion_matrix"]


    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual: No Risk",
            "Actual: Risk"
        ],
        columns=[
            "Predicted: No Risk",
            "Predicted: Risk"
        ]
    )


    st.dataframe(
        cm_df,
        use_container_width=True
    )


# ============================================================
# MODEL DETAILS
# ============================================================

elif ml_page == "Model Details":

    st.markdown(
        '<div class="page-title">'
        'ML Model Details'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Technical information about the credit-risk '
        'classification pipeline.'
        '</div>',
        unsafe_allow_html=True
    )


    st.subheader(
        "Models Used"
    )


    model_info = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "XGBoost"
            ],

            "Type": [
                "Linear Classification",
                "Bagging Ensemble",
                "Gradient Boosting"
            ],

            "Purpose": [
                "Baseline classification",
                "Non-linear ensemble classification",
                "Advanced boosting classification"
            ]
        }
    )


    st.dataframe(
        model_info,
        hide_index=True,
        use_container_width=True
    )


    st.subheader(
        "Feature Pipeline"
    )


    st.write(
        """
        The prediction pipeline follows the same feature
        engineering logic used in the original ML analysis.
        """
    )


    feature_df = pd.DataFrame(
        {
            "Feature": models[
                "feature_columns"
            ]
        }
    )


    st.dataframe(
        feature_df,
        hide_index=True,
        use_container_width=True
    )


    st.subheader(
        "Random Forest Grid Search"
    )


    st.write(
        "Best parameters:"
    )


    st.code(
        str(
            models[
                "best_rf_params"
            ]
        )
    )


    st.warning(
        """
        This application is intended for analytical,
        educational and portfolio demonstration purposes.
        Machine-learning output should not be used as the
        sole basis for real-world lending decisions.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        CreditGuard • Machine Learning Risk Intelligence

        <br>

        Logistic Regression • Random Forest • XGBoost

    </div>
    """,
    unsafe_allow_html=True
)
