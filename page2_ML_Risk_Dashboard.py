import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

CATEGORICAL = [
    "CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY","NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE"
]
DROP = ["ID","FLAG_MOBIL","DAYS_BIRTH","DAYS_EMPLOYED","STATUS","CREDIT_RISK"]

@st.cache_data(show_spinner=False)
def prepare_training_data(df):
    work=df.copy()
    encoders={}
    for col in CATEGORICAL:
        le=LabelEncoder()
        work[col]=le.fit_transform(work[col].astype(str))
        encoders[col]=dict(zip(le.classes_,le.transform(le.classes_)))
    X=work.drop(columns=DROP,errors="ignore")
    # Remove derived display-only columns that were not in the original ML code.
    X=X.drop(columns=["AGE","EMPLOYMENT_YEARS","RISK_CATEGORY","INCOME_BAND","AGE_GROUP"],errors="ignore")
    X=X.select_dtypes(include=[np.number]).copy()
    y=work["CREDIT_RISK"].astype(int)
    return X,y,encoders

@st.cache_resource(show_spinner=False)
def train_models(df):
    X,y,_=prepare_training_data(df)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)

    models={
        "Logistic Regression":LogisticRegression(max_iter=1000,random_state=42),
        "Random Forest":RandomForestClassifier(random_state=42,n_estimators=100),
    }
    if XGB_AVAILABLE:
        models["XGBoost"]=XGBClassifier(random_state=42,eval_metric="logloss")

    fitted={}
    rows=[]
    matrices={}
    for name,model in models.items():
        model.fit(X_train,y_train)
        pred=model.predict(X_test)
        fitted[name]=model
        rows.append({
            "Model":name,
            "Accuracy":accuracy_score(y_test,pred)*100,
            "Precision":precision_score(y_test,pred,zero_division=0)*100,
            "Recall":recall_score(y_test,pred,zero_division=0)*100,
            "F1 Score":f1_score(y_test,pred,zero_division=0),
        })
        matrices[name]=confusion_matrix(y_test,pred)
    results=pd.DataFrame(rows).sort_values("F1 Score",ascending=False).reset_index(drop=True)
    return fitted,results,matrices,X_test,y_test

def metric_card(label,value,sub):
    st.markdown(f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:18px;'
                f'box-shadow:0 6px 20px rgba(12,31,55,.04);"><div style="font-size:12px;color:#6b788b;">'
                f'{label}</div><div style="font-size:28px;font-weight:800;color:#122033;margin:4px 0;">{value}</div>'
                f'<div style="font-size:11px;color:#8a96a8;">{sub}</div></div>',unsafe_allow_html=True)

def render_ml(df):
    st.markdown('<div class="hero"><div class="eyebrow">MACHINE LEARNING MODEL LAB</div>'
                '<h1>Credit Risk Prediction</h1><p>Train, compare and inspect the three classifiers from your original analysis, '
                'then use the preferred classifier for an individual risk assessment.</p></div>',unsafe_allow_html=True)

    fitted,results,matrices,X_test,y_test=train_models(df)

    c1,c2,c3,c4=st.columns(4)
    best=results.iloc[0]
    c1.metric("Best F1",f'{best["F1 Score"]:.3f}',best["Model"])
    c2.metric("Best Accuracy",f'{results["Accuracy"].max():.2f}%',"holdout test set")
    c3.metric("Test Records",f"{len(y_test):,}","20% holdout")
    c4.metric("Risk Cases",f"{int(y_test.sum()):,}","positive class in test set")

    tabs=st.tabs(["Model Comparison","Confusion Matrices","Feature Importance","Live Assessment","Cross-Validation"])

    with tabs[0]:
        st.subheader("Classifier Performance")
        display=results.copy()
        for c in ["Accuracy","Precision","Recall"]:
            display[c]=display[c].map(lambda x:f"{x:.2f}%")
        display["F1 Score"]=display["F1 Score"].map(lambda x:f"{x:.3f}")
        st.dataframe(display,hide_index=True,use_container_width=True)
        st.info("Because the target is highly imbalanced, F1 Score and minority-class recall are more informative than accuracy alone.")

    with tabs[1]:
        cols=st.columns(len(matrices))
        for col,(name,cm) in zip(cols,matrices.items()):
            with col:
                st.markdown(f"**{name}**")
                import plotly.figure_factory as ff
                fig=ff.create_annotated_heatmap(
                    z=cm.tolist(),x=["Predicted Low","Predicted High"],
                    y=["Actual Low","Actual High"],colorscale=[[0,"#eef3fa"],[1,"#2f6fed"]],
                    showscale=False
                )
                fig.update_layout(height=300,margin=dict(l=20,r=10,t=25,b=20))
                st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    with tabs[2]:
        model_name=st.selectbox("Inspect feature importance",list(fitted.keys()),index=1 if "Random Forest" in fitted else 0)
        model=fitted[model_name]
        feature_names=X_test.columns
        if hasattr(model,"feature_importances_"):
            imp=pd.DataFrame({"Feature":feature_names,"Importance":model.feature_importances_}).sort_values("Importance",ascending=False).head(15)
        else:
            perm=permutation_importance(model,X_test,y_test,n_repeats=5,random_state=42,scoring="f1")
            imp=pd.DataFrame({"Feature":feature_names,"Importance":perm.importances_mean}).sort_values("Importance",ascending=False).head(15)
        fig=__import__("plotly.express",fromlist=["bar"]).bar(imp.sort_values("Importance"),x="Importance",y="Feature",orientation="h")
        fig.update_layout(height=500)
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(imp,hide_index=True,use_container_width=True)

    with tabs[3]:
        st.subheader("Individual Customer Risk Assessment")
        st.caption("The input controls use the same feature set as the original training pipeline.")
        _,_,encoders=prepare_training_data(df)
        feature_order=X_test.columns.tolist()

        c1,c2,c3=st.columns(3)
        with c1:
            gender=st.selectbox("Gender",list(encoders["CODE_GENDER"].keys()))
            car=st.selectbox("Car Ownership",list(encoders["FLAG_OWN_CAR"].keys()))
            realty=st.selectbox("Property Ownership",list(encoders["FLAG_OWN_REALTY"].keys()))
            children=st.number_input("Number of Children",0,20,0)
            income=st.number_input("Annual Income",0.0,50000000.0,300000.0,step=10000.0)
        with c2:
            income_type=st.selectbox("Income Type",list(encoders["NAME_INCOME_TYPE"].keys()))
            education=st.selectbox("Education",list(encoders["NAME_EDUCATION_TYPE"].keys()))
            family=st.selectbox("Family Status",list(encoders["NAME_FAMILY_STATUS"].keys()))
            housing=st.selectbox("Housing Type",list(encoders["NAME_HOUSING_TYPE"].keys()))
            family_members=st.number_input("Family Members",1.0,20.0,2.0,step=1.0)
        with c3:
            occupation=st.selectbox("Occupation",list(encoders["OCCUPATION_TYPE"].keys()))
            work_phone=st.selectbox("Work Phone",["0","1"])
            phone=st.selectbox("Phone Available",["0","1"])
            email=st.selectbox("Email Available",["0","1"])
            gender_dummy=gender

        if st.button("Assess Credit Risk",type="primary",use_container_width=True):
            row={
                "CODE_GENDER":encoders["CODE_GENDER"][gender],
                "FLAG_OWN_CAR":encoders["FLAG_OWN_CAR"][car],
                "FLAG_OWN_REALTY":encoders["FLAG_OWN_REALTY"][realty],
                "CNT_CHILDREN":children,"AMT_INCOME_TOTAL":income,
                "NAME_INCOME_TYPE":encoders["NAME_INCOME_TYPE"][income_type],
                "NAME_EDUCATION_TYPE":encoders["NAME_EDUCATION_TYPE"][education],
                "NAME_FAMILY_STATUS":encoders["NAME_FAMILY_STATUS"][family],
                "NAME_HOUSING_TYPE":encoders["NAME_HOUSING_TYPE"][housing],
                "FLAG_WORK_PHONE":int(work_phone),"FLAG_PHONE":int(phone),
                "FLAG_EMAIL":int(email),"OCCUPATION_TYPE":encoders["OCCUPATION_TYPE"][occupation],
                "CNT_FAM_MEMBERS":family_members,
            }
            sample=pd.DataFrame([row])
            sample=sample.reindex(columns=feature_order,fill_value=0)
            preferred="Random Forest" if "Random Forest" in fitted else results.iloc[0]["Model"]
            model=fitted[preferred]
            pred=int(model.predict(sample)[0])
            proba=float(model.predict_proba(sample)[0][1]) if hasattr(model,"predict_proba") else np.nan
            if pred==1:
                st.error(f"⚠️ HIGH RISK — {preferred}")
            else:
                st.success(f"✓ LOW RISK — {preferred}")
            if not np.isnan(proba):
                st.progress(min(max(proba,0),1))
                st.write(f"Estimated positive-class probability: **{proba*100:.2f}%**")
            st.caption("This assessment follows the trained feature pipeline and is intended for project analysis, not a lending decision.")

    with tabs[4]:
        st.subheader("15-Fold Cross-Validation")
        st.caption("This reproduces the cross-validation approach used in your original code. It may take some time.")
        if st.button("Run 15-Fold Accuracy CV",type="secondary"):
            X,y,_=prepare_training_data(df)
            cv_rows=[]
            cv_models={
                "Logistic Regression":LogisticRegression(max_iter=1000,random_state=42),
                "Random Forest":RandomForestClassifier(random_state=42,n_estimators=100),
            }
            if XGB_AVAILABLE: cv_models["XGBoost"]=XGBClassifier(random_state=42,eval_metric="logloss")
            progress=st.progress(0)
            for i,(name,m) in enumerate(cv_models.items(),1):
                scores=cross_val_score(m,X,y,cv=15,scoring="accuracy")
                cv_rows.append({"Model":name,"Mean CV Accuracy":scores.mean()*100,"Std":scores.std()*100})
                progress.progress(i/len(cv_models))
            cv_df=pd.DataFrame(cv_rows)
            cv_df["Mean CV Accuracy"]=cv_df["Mean CV Accuracy"].map(lambda x:f"{x:.2f}%")
            cv_df["Std"]=cv_df["Std"].map(lambda x:f"{x:.2f}%")
            st.dataframe(cv_df,hide_index=True,use_container_width=True)
