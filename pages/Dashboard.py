import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os

st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/Telco-Customer-Churn.csv')
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        return df
    except:
        st.error("Could not load data file. Please ensure Telco-Customer-Churn.csv is in the data folder.")
        return None

st.title("📊 Customer Analytics Dashboard")
st.markdown("### Comprehensive Data Analysis & Insights")
st.markdown("---")

df = load_data()

if df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Executive Summary", "📈 Customer Analytics", "💰 Revenue Analysis", "🔍 Advanced Insights"])
    
    with tab1:
        st.header("🎯 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_customers = len(df)
        churned_customers = df['Churn'].value_counts().get('Yes', 0)
        churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0
        avg_monthly_charges = df['MonthlyCharges'].mean() if 'MonthlyCharges' in df.columns else 0
        
        with col1:
            st.metric(
                label="📊 Total Customers",
                value=f"{total_customers:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="❌ Churned Customers",
                value=f"{churned_customers:,}",
                delta=f"{churn_rate:.1f}% rate",
                delta_color="inverse"
            )
        
        with col3:
            avg_revenue = df['MonthlyCharges'].sum() if 'MonthlyCharges' in df.columns else 0
            st.metric(
                label="💰 Monthly Revenue",
                value=f"${avg_revenue:,.0f}",
                delta=f"${avg_monthly_charges:.2f} avg"
            )
        
        with col4:
            retained_customers = total_customers - churned_customers
            retention_rate = (retained_customers / total_customers * 100) if total_customers > 0 else 0
            st.metric(
                label="✅ Retention Rate",
                value=f"{retention_rate:.1f}%",
                delta=f"{retained_customers:,} customers"
            )
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Churn Overview")
            churn_data = df['Churn'].value_counts()
            
            fig = px.pie(
                values=churn_data.values, 
                names=['Retained', 'Churned'],
                title="Customer Retention vs Churn",
                color_discrete_map={'Retained': '#00cc96', 'Churned': '#ef553b'},
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📋 Key Insights")
            st.markdown(f"""
            **📈 Business Health:**
            - Customer base: {total_customers:,}
            - Monthly revenue: ${avg_revenue:,.0f}
            - Average spend: ${avg_monthly_charges:.2f}
            
            **⚠️ Risk Indicators:**
            - Churn rate: {churn_rate:.1f}%
            - Lost customers: {churned_customers:,}
            - Revenue at risk: ${churned_customers * avg_monthly_charges:.0f}/month
            
            **🎯 Opportunities:**
            - Focus on month-to-month contracts
            - Improve fiber optic satisfaction
            - Target high-value customers
            """)
    
    with tab2:
        st.header("📈 Customer Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'tenure' in df.columns:
                st.subheader("📅 Customer Tenure Analysis")
                
                fig = px.histogram(
                    df, x='tenure', color='Churn',
                    title="Customer Tenure Distribution",
                    nbins=30,
                    color_discrete_map={'Yes': '#ef553b', 'No': '#00cc96'},
                    barmode='overlay'
                )
                fig.update_layout(height=400)
                fig.update_traces(opacity=0.7)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Contract' in df.columns:
                st.subheader("📋 Contract Type Analysis")
                
                contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
                
                fig = px.bar(
                    contract_churn, 
                    x=contract_churn.index, 
                    y=['No', 'Yes'],
                    title="Churn Rate by Contract Type (%)",
                    color_discrete_map={'Yes': '#ef553b', 'No': '#00cc96'},
                    barmode='stack'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'InternetService' in df.columns:
                st.subheader("🌐 Internet Service Impact")
                
                internet_stats = df.groupby(['InternetService', 'Churn']).size().unstack(fill_value=0)
                internet_stats_pct = internet_stats.div(internet_stats.sum(axis=1), axis=0) * 100
                
                fig = px.bar(
                    internet_stats_pct,
                    x=internet_stats_pct.index,
                    y=['No', 'Yes'],
                    title="Churn Rate by Internet Service (%)",
                    color_discrete_map={'Yes': '#ef553b', 'No': '#00cc96'},
                    barmode='stack'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'PaymentMethod' in df.columns:
                st.subheader("💳 Payment Method Analysis")
                
                payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
                
                fig = px.bar(
                    payment_churn,
                    x=payment_churn.index,
                    y='Yes',
                    title="Churn Rate by Payment Method (%)",
                    color_discrete_sequence=['#ef553b']
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("👥 Demographics Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'gender' in df.columns:
                gender_churn = pd.crosstab(df['gender'], df['Churn'], normalize='index') * 100
                
                fig = px.bar(
                    gender_churn,
                    x=gender_churn.index,
                    y='Yes',
                    title="Churn Rate by Gender (%)",
                    color_discrete_sequence=['#ff7f0e']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'SeniorCitizen' in df.columns:
                df['SeniorCitizenLabel'] = df['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior'})
                senior_churn = pd.crosstab(df['SeniorCitizenLabel'], df['Churn'], normalize='index') * 100
                
                fig = px.bar(
                    senior_churn,
                    x=senior_churn.index,
                    y='Yes',
                    title="Churn Rate by Age Group (%)",
                    color_discrete_sequence=['#9467bd']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if 'Partner' in df.columns:
                partner_churn = pd.crosstab(df['Partner'], df['Churn'], normalize='index') * 100
                
                fig = px.bar(
                    partner_churn,
                    x=partner_churn.index,
                    y='Yes',
                    title="Churn Rate by Partner Status (%)",
                    color_discrete_sequence=['#2ca02c']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("💰 Revenue Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'MonthlyCharges' in df.columns:
                st.subheader("💵 Monthly Charges Distribution")
                
                fig = px.box(
                    df, x='Churn', y='MonthlyCharges',
                    title="Monthly Charges by Churn Status",
                    color='Churn',
                    color_discrete_map={'Yes': '#ef553b', 'No': '#00cc96'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'TotalCharges' in df.columns:
                st.subheader("💰 Total Charges Analysis")
                
                df_clean = df.dropna(subset=['TotalCharges'])
                
                fig = px.histogram(
                    df_clean, x='TotalCharges', color='Churn',
                    title="Total Charges Distribution",
                    nbins=30,
                    color_discrete_map={'Yes': '#ef553b', 'No': '#00cc96'},
                    barmode='overlay'
                )
                fig.update_layout(height=400)
                fig.update_traces(opacity=0.7)
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Revenue Metrics by Segment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            revenue_by_contract = df.groupby(['Contract', 'Churn'])['MonthlyCharges'].agg(['sum', 'mean', 'count']).round(2)
            st.write("**Revenue by Contract Type:**")
            st.dataframe(revenue_by_contract, use_container_width=True)
        
        with col2:
            if 'InternetService' in df.columns:
                revenue_by_internet = df.groupby(['InternetService', 'Churn'])['MonthlyCharges'].agg(['sum', 'mean', 'count']).round(2)
                st.write("**Revenue by Internet Service:**")
                st.dataframe(revenue_by_internet, use_container_width=True)
        
        st.subheader("💡 Revenue Impact Analysis")
        
        lost_revenue_monthly = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
        avg_customer_value = df['MonthlyCharges'].mean()
        potential_recovery = lost_revenue_monthly * 0.5
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="💸 Lost Monthly Revenue",
                value=f"${lost_revenue_monthly:,.0f}",
                delta="From churned customers",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                label="🎯 Recovery Potential (50%)",
                value=f"${potential_recovery:,.0f}",
                delta="Monthly opportunity"
            )
        
        with col3:
            annual_impact = lost_revenue_monthly * 12
            st.metric(
                label="📅 Annual Impact",
                value=f"${annual_impact:,.0f}",
                delta="Total yearly loss"
            )
    
    with tab4:
        st.header("🔍 Advanced Insights")
        
        st.subheader("🎯 Customer Segmentation Analysis")
        
        if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
            df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 72], labels=['0-12 months', '1-2 years', '2-3 years', '3+ years'])
            df['charges_group'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 65, 95, 120], labels=['Low', 'Medium', 'High', 'Premium'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                segment_analysis = pd.crosstab([df['tenure_group'], df['charges_group']], df['Churn'], normalize='index') * 100
                
                fig = px.imshow(
                    segment_analysis['Yes'].unstack().fillna(0),
                    title="Churn Rate Heatmap: Tenure vs Charges (%)",
                    color_continuous_scale="Reds",
                    aspect="auto"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🚨 High-Risk Segments")
                
                high_risk_segments = segment_analysis[segment_analysis['Yes'] > 40].sort_values('Yes', ascending=False)
                
                if not high_risk_segments.empty:
                    for idx, row in high_risk_segments.head(5).iterrows():
                        tenure_grp, charges_grp = idx
                        st.error(f"**{tenure_grp} + {charges_grp} Charges**: {row['Yes']:.1f}% churn rate")
                else:
                    st.success("No segments with >40% churn rate found!")
                
                st.subheader("✅ Safe Segments")
                low_risk_segments = segment_analysis[segment_analysis['Yes'] < 20].sort_values('Yes')
                
                if not low_risk_segments.empty:
                    for idx, row in low_risk_segments.head(3).iterrows():
                        tenure_grp, charges_grp = idx
                        st.success(f"**{tenure_grp} + {charges_grp} Charges**: {row['Yes']:.1f}% churn rate")
        
        st.subheader("📊 Service Adoption Impact")
        
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        available_services = [col for col in service_cols if col in df.columns]
        
        if available_services:
            col1, col2 = st.columns(2)
            
            with col1:
                service_impact = {}
                for service in available_services:
                    service_churn = pd.crosstab(df[service], df['Churn'], normalize='index') * 100
                    if 'Yes' in service_churn.columns and 'Yes' in service_churn.index:
                        service_impact[service] = service_churn.loc['Yes', 'Yes']
                
                if service_impact:
                    fig = px.bar(
                        x=list(service_impact.keys()),
                        y=list(service_impact.values()),
                        title="Churn Rate by Service Adoption (%)",
                        color=list(service_impact.values()),
                        color_continuous_scale="RdYlGn_r"
                    )
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Service Recommendations")
                
                if service_impact:
                    sorted_services = sorted(service_impact.items(), key=lambda x: x[1])
                    
                    st.markdown("**🏆 Best Retention Services:**")
                    for service, churn_rate in sorted_services[:3]:
                        st.success(f"{service}: {churn_rate:.1f}% churn rate")
                    
                    st.markdown("**⚠️ High-Risk Services:**")
                    for service, churn_rate in sorted_services[-2:]:
                        st.error(f"{service}: {churn_rate:.1f}% churn rate")
        
        st.subheader("📈 Predictive Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🎯 Retention Opportunity",
                value=f"{(100 - churn_rate):.1f}%",
                delta="Current baseline"
            )
        
        with col2:
            improvement_potential = min(10, churn_rate * 0.3)
            st.metric(
                label="📈 Improvement Potential",
                value=f"+{improvement_potential:.1f}%",
                delta="With targeted interventions"
            )
        
        with col3:
            roi_potential = improvement_potential * total_customers * avg_monthly_charges * 12 / 100
            st.metric(
                label="💰 Annual ROI Potential",
                value=f"${roi_potential:,.0f}",
                delta="Revenue recovery"
            )

else:
    st.error("⚠️ Unable to load data. Please ensure the Telco-Customer-Churn.csv file is available in the data folder.")
    
    st.markdown("""
    ### 📁 Expected Data Structure
    
    The dashboard expects a CSV file with the following columns:
    - `Churn`: Customer churn status (Yes/No)
    - `tenure`: Customer tenure in months
    - `MonthlyCharges`: Monthly charges amount
    - `Contract`: Contract type
    - `InternetService`: Internet service type
    - `PaymentMethod`: Payment method
    - And other customer attributes...
    """)

st.markdown("---")
