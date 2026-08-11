import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def chart_style(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(l=10,r=10,t=40,b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, Arial", color="#243247"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#edf0f4", zeroline=False)
    return fig

def card(title, subtitle=""):
    st.markdown(
        f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:20px;'
        f'box-shadow:0 7px 25px rgba(12,31,55,.045);"><div style="font-size:11px;font-weight:800;'
        f'letter-spacing:1.2px;color:#5f7aa4;">EDA WORKSPACE</div><h2 style="margin:5px 0 4px;color:#122033;">'
        f'{title}</h2><div style="color:#6b788b;font-size:13px;">{subtitle}</div></div>',
        unsafe_allow_html=True
    )

def render_eda(filtered, full_df):
    st.markdown('<div class="hero"><div class="eyebrow">EXPLORATORY DATA ANALYSIS</div>'
                '<h1>Portfolio Intelligence</h1><p>Understand customer composition, income patterns, '
                'risk concentration and data quality using the current filtered portfolio.</p></div>',
                unsafe_allow_html=True)

    tabs = st.tabs(["Overview", "Risk Segmentation", "Customer Profile", "Data Quality", "Raw Data"])

    with tabs[0]:
        total=len(filtered); high=int(filtered["CREDIT_RISK"].eq(1).sum())
        avg_income=filtered["AMT_INCOME_TOTAL"].mean() if total else np.nan
        avg_age=filtered["AGE"].mean() if total else np.nan
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Records",f"{total:,}"); c2.metric("High Risk",f"{high:,}")
        c3.metric("High-Risk Rate",f"{high/total*100:.2f}%" if total else "N/A")
        c4.metric("Average Income",f"₹{avg_income:,.0f}" if pd.notna(avg_income) else "N/A")
        st.markdown("<br>",unsafe_allow_html=True)
        a,b=st.columns(2)
        with a:
            fig=px.histogram(filtered,x="AGE",color="RISK_CATEGORY",nbins=30,
                             color_discrete_map={"Low Risk":"#15956b","High Risk":"#d94a4a"},
                             labels={"AGE":"Age","count":"Customers"})
            st.plotly_chart(chart_style(fig,370),use_container_width=True)
        with b:
            fig=px.histogram(filtered,x="AMT_INCOME_TOTAL",color="RISK_CATEGORY",nbins=35,
                             color_discrete_map={"Low Risk":"#15956b","High Risk":"#d94a4a"},
                             labels={"AMT_INCOME_TOTAL":"Annual Income","count":"Customers"})
            fig.update_xaxes(tickformat=",.0f")
            st.plotly_chart(chart_style(fig,370),use_container_width=True)

        a,b=st.columns(2)
        with a:
            counts=filtered["NAME_EDUCATION_TYPE"].value_counts().reset_index()
            counts.columns=["Education","Customers"]
            fig=px.bar(counts,x="Customers",y="Education",orientation="h")
            st.plotly_chart(chart_style(fig,360),use_container_width=True)
        with b:
            counts=filtered["NAME_INCOME_TYPE"].value_counts().reset_index()
            counts.columns=["Income Type","Customers"]
            fig=px.bar(counts,x="Customers",y="Income Type",orientation="h")
            st.plotly_chart(chart_style(fig,360),use_container_width=True)

    with tabs[1]:
        st.subheader("Risk by Age Group")
        age_risk=pd.crosstab(filtered["AGE_GROUP"],filtered["RISK_CATEGORY"],normalize="index").mul(100).reset_index()
        fig=px.bar(age_risk,x="AGE_GROUP",y=[c for c in ["Low Risk","High Risk"] if c in age_risk.columns],
                   barmode="group",color_discrete_map={"Low Risk":"#15956b","High Risk":"#d94a4a"})
        st.plotly_chart(chart_style(fig,390),use_container_width=True)

        st.subheader("Risk by Income Band")
        inc=pd.crosstab(filtered["INCOME_BAND"],filtered["RISK_CATEGORY"],normalize="index").mul(100).reset_index()
        fig=px.bar(inc,x="INCOME_BAND",y=[c for c in ["Low Risk","High Risk"] if c in inc.columns],
                   barmode="group",color_discrete_map={"Low Risk":"#15956b","High Risk":"#d94a4a"})
        st.plotly_chart(chart_style(fig,390),use_container_width=True)

        a,b=st.columns(2)
        with a:
            occ=pd.crosstab(filtered["OCCUPATION_TYPE"],filtered["RISK_CATEGORY"],normalize="index").mul(100)
            if "High Risk" in occ:
                occ=occ.sort_values("High Risk").tail(15).reset_index()
                fig=px.bar(occ,x="High Risk",y="OCCUPATION_TYPE",orientation="h",
                           labels={"High Risk":"High-Risk Rate (%)"},
                           color_discrete_sequence=["#d94a4a"])
                st.plotly_chart(chart_style(fig,460),use_container_width=True)
        with b:
            house=pd.crosstab(filtered["NAME_HOUSING_TYPE"],filtered["RISK_CATEGORY"],normalize="index").mul(100).reset_index()
            fig=px.bar(house,x="NAME_HOUSING_TYPE",y=[c for c in ["Low Risk","High Risk"] if c in house.columns],
                       barmode="group",color_discrete_map={"Low Risk":"#15956b","High Risk":"#d94a4a"})
            fig.update_xaxes(tickangle=-25)
            st.plotly_chart(chart_style(fig,460),use_container_width=True)

        st.subheader("Age × Income Customer Concentration")
        heat=pd.crosstab(filtered["AGE_GROUP"],filtered["INCOME_BAND"]).reindex(
            index=["18–25","26–35","36–45","46–55","56+"],
            columns=["< ₹2L","₹2–5L","₹5–10L","₹10–20L","₹20L+"],fill_value=0)
        fig=px.imshow(heat,text_auto=True,aspect="auto",color_continuous_scale=["#eef5ff","#8bb5ef","#174ea6"])
        st.plotly_chart(chart_style(fig,400),use_container_width=True)

    with tabs[2]:
        st.subheader("Customer Profile Explorer")
        idq=st.text_input("Search Customer ID",placeholder="Enter an ID from the portfolio")
        if idq:
            r=full_df[full_df["ID"].astype(str).eq(idq.strip())]
            if r.empty: st.warning("No customer found for that ID.")
            else:
                row=r.iloc[0]
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Customer ID",str(row["ID"])); c2.metric("Age",f'{row["AGE"]} years')
                c3.metric("Annual Income",f'₹{row["AMT_INCOME_TOTAL"]:,.0f}')
                c4.metric("Risk",row["RISK_CATEGORY"])
                left,right=st.columns(2)
                with left:
                    st.dataframe(pd.DataFrame({"Field":["Gender","Family Status","Children","Family Members","Education"],
                                               "Value":[row["CODE_GENDER"],row["NAME_FAMILY_STATUS"],row["CNT_CHILDREN"],
                                                        row["CNT_FAM_MEMBERS"],row["NAME_EDUCATION_TYPE"]]}),
                                 hide_index=True,use_container_width=True)
                with right:
                    st.dataframe(pd.DataFrame({"Field":["Income Type","Occupation","Housing","Car Ownership","Property Ownership"],
                                               "Value":[row["NAME_INCOME_TYPE"],row["OCCUPATION_TYPE"],row["NAME_HOUSING_TYPE"],
                                                        row["FLAG_OWN_CAR"],row["FLAG_OWN_REALTY"]]}),
                                 hide_index=True,use_container_width=True)
        else:
            st.info("Enter a Customer ID to open a focused profile.")

    with tabs[3]:
        st.subheader("Data Quality Monitor")
        missing=full_df.isna().sum()
        c1,c2,c3=st.columns(3)
        c1.metric("Rows",f"{len(full_df):,}"); c2.metric("Columns",f"{full_df.shape[1]:,}")
        c3.metric("Missing Cells",f"{int(missing.sum()):,}")
        quality=pd.DataFrame({"Field":full_df.columns,
                              "Missing Values":[int(full_df[c].isna().sum()) for c in full_df.columns],
                              "Missing %":[round(full_df[c].isna().mean()*100,2) for c in full_df.columns]})
        quality=quality.sort_values(["Missing Values","Field"],ascending=[False,True])
        st.dataframe(quality,hide_index=True,use_container_width=True,height=430)
        st.download_button("Download data-quality report",quality.to_csv(index=False),
                           "creditguard_data_quality.csv","text/csv")

    with tabs[4]:
        st.subheader("Portfolio Records")
        display=filtered.copy()
        labels={"ID":"Customer ID","AMT_INCOME_TOTAL":"Annual Income","AGE":"Age",
                "NAME_INCOME_TYPE":"Income Type","NAME_EDUCATION_TYPE":"Education",
                "NAME_FAMILY_STATUS":"Family Status","NAME_HOUSING_TYPE":"Housing",
                "OCCUPATION_TYPE":"Occupation","RISK_CATEGORY":"Risk Category"}
        keep=[c for c in labels if c in display.columns]
        display=display[keep].rename(columns=labels)
        st.dataframe(display,hide_index=True,use_container_width=True,height=520)
        st.download_button("Export filtered portfolio",display.to_csv(index=False),
                           "creditguard_filtered_portfolio.csv","text/csv")
