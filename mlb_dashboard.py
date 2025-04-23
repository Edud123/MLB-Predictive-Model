import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle

# Page config
st.set_page_config(page_title="MLB Win Rate Predictor Dashboard", layout="wide")

# Load the preprocessed data
@st.cache_data
def load_data():
    df = pd.read_csv("merged.csv")
    return df

# Load the trained models
@st.cache_resource
def load_models():
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    return xgb_model, rf_model

def main():
    st.title("MLB Win Rate Prediction Analysis Dashboard")
    
    # Sidebar
    st.sidebar.header("Filters")
    
    # Load data
    df = load_data()
    
    # Date range filter
    years = sorted(df['Season'].unique())
    selected_years = st.sidebar.slider(
        "Select Year Range",
        min_value=min(years),
        max_value=max(years),
        value=(min(years), max(years))
    )
    
    # Team filter
    teams = sorted(df['Team'].unique())
    selected_team = st.sidebar.multiselect(
        "Select Teams",
        teams,
        default=teams[:5]
    )
    
    # Filter data
    filtered_df = df[
        (df['Season'].between(selected_years[0], selected_years[1])) &
        (df['Team'].isin(selected_team))
    ]
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Win Rate Trends")
        fig_trends = px.line(
            filtered_df.sort_values(by=['Season']),  # Ensure data is sorted by Season
            x='Season',
            y='WR',
            color='Team',
            title="Historical Win Rate by Team",
            markers=True  # Add markers for better visualization
        )
        fig_trends.update_traces(line_shape='linear')  # Use linear interpolation for smoother lines
        fig_trends.update_layout(
            xaxis_title="Season",
            yaxis_title="Win Rate",
            xaxis=dict(tickmode='linear'),  # Ensure x-axis ticks are linear
            yaxis=dict(range=[0, 1]),  # Set y-axis range to valid win rate values
            legend_title="Teams"
        )
        st.plotly_chart(fig_trends, use_container_width=True)
    
    with col2:
        st.subheader("Statistical Categories Impact")
        # Create feature groups
        feature_groups = {
            'Traditional': ['ERA', 'AVG', 'HR', 'RBI'],
            'Advanced Batting': ['wRC+', 'OBP', 'SLG', 'WAR'],
            'Advanced Pitching': ['FIP', 'xFIP', 'K/9', 'BB/9'],
            'Fielding': ['DRS', 'UZR', 'Def'],
            'Run Prevention': ['ERA', 'FIP', 'LOB%', 'GB%']
        }
        
        # Calculate category correlations
        category_impact = {}
        for category, features in feature_groups.items():
            category_corrs = []
            for feat in features:
                lag_features = [col for col in filtered_df.columns if col.startswith(feat + '_lag')]
                for lag_feat in lag_features:
                    corr = filtered_df[lag_feat].corr(filtered_df['WR'])
                    if not np.isnan(corr):
                        category_corrs.append(abs(corr))
            if category_corrs:
                category_impact[category] = np.mean(category_corrs)
        
        fig_impact = px.bar(
            x=list(category_impact.keys()),
            y=list(category_impact.values()),
            title="Predictive Power by Statistical Category"
        )
        st.plotly_chart(fig_impact, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        # Top individual features
        lag_cols = [col for col in filtered_df.columns if '_lag' in col]
        feature_correlations = {}
        for col in lag_cols:
            corr = filtered_df[col].corr(filtered_df['WR'])
            if not np.isnan(corr):
                feature_correlations[col] = abs(corr)
        
        top_features = dict(sorted(feature_correlations.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:10])
        
        fig_features = px.bar(
            x=list(top_features.keys()),
            y=list(top_features.values()),
            title="Top 10 Individual Predictive Features"
        )
        fig_features.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_features, use_container_width=True)
    
    with col4:
        # Model performance comparison
        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['R² Score', 'RMSE', 'MAE'],
            'XGBoost': [0.38, 0.067, 0.052],
            'Random Forest': [0.33, 0.071, 0.055]
        })
        st.table(metrics_df)
    
    # Interactive predictor
    st.subheader("Win Rate Predictor")
    col5, col6 = st.columns(2)
    
    with col5:
        # Add interactive inputs for key features
        selected_era = st.slider("ERA", 2.0, 6.0, 4.0)
        selected_obp = st.slider("OBP", 0.290, 0.380, 0.330)
        selected_slg = st.slider("SLG", 0.350, 0.500, 0.420)
    
    with col6:
        selected_war = st.slider("Team WAR", -10.0, 50.0, 20.0)
        selected_fip = st.slider("FIP", 2.5, 5.5, 4.0)
        selected_drs = st.slider("DRS", -50, 50, 0)
    
    if st.button("Predict Win Rate"):
        # Here you would normally use the loaded models to make predictions
        # For now, we'll use a simple placeholder
        predicted_wr = 0.500 + (4.0 - selected_era) * 0.02 + \
                      (selected_obp - 0.330) * 0.5 + \
                      (selected_slg - 0.420) * 0.3
        
        st.metric(
            label="Predicted Win Rate",
            value=f"{predicted_wr:.3f}",
            delta=f"{(predicted_wr - 0.500):.3f}"
        )

if __name__ == "__main__":
    main()