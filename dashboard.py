"""
AI Model Health Monitoring Dashboard

Interactive Streamlit dashboard for visualizing model health,
drift metrics, and failure probability in real-time.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"
REFRESH_INTERVAL = 30  # seconds
LOG_FILE = "logs/prediction_logs.csv"
ALERT_LOG_FILE = "logs/alert_history.csv"

# Page configuration
st.set_page_config(
    page_title="AI Model Health Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    .status-healthy {
        color: #00C851;
        font-size: 2rem;
        font-weight: bold;
    }
    .status-risky {
        color: #ffbb33;
        font-size: 2rem;
        font-weight: bold;
    }
    .status-critical {
        color: #ff4444;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def get_failure_risk():
    """Fetch failure risk from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/failure-risk", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return {"error": str(e)}

def get_drift_status():
    """Fetch drift status from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/drift-status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return {"error": str(e)}

def load_prediction_logs():
    """Load prediction logs"""
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame()

def load_alert_history():
    """Load alert history"""
    if os.path.exists(ALERT_LOG_FILE):
        return pd.read_csv(ALERT_LOG_FILE)
    return pd.DataFrame()

def get_status_badge(failure_prob):
    """Get status badge based on failure probability"""
    if failure_prob < 0.3:
        return "🟢 HEALTHY", "status-healthy"
    elif failure_prob < 0.7:
        return "🟡 RISKY", "status-risky"
    else:
        return "🔴 CRITICAL", "status-critical"

def create_gauge_chart(value, title, max_value=1.0):
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        number={'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.3], 'color': '#00C851'},
                {'range': [0.3, 0.7], 'color': '#ffbb33'},
                {'range': [0.7, max_value], 'color': '#ff4444'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Main Dashboard
def main():
    # Header
    st.title("🤖 AI Model Health Monitoring Dashboard")
    st.markdown("---")
    
    # Fetch data
    failure_risk = get_failure_risk()
    drift_status = get_drift_status()
    
    if failure_risk and "error" not in failure_risk:
        failure_prob = failure_risk.get("failure_probability", 0)
        risk_level = failure_risk.get("risk", "UNKNOWN")
        metrics = failure_risk.get("metrics", {})
        
        # Status Badge
        status_text, status_class = get_status_badge(failure_prob)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f'<p class="{status_class}">{status_text}</p>', unsafe_allow_html=True)
            st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("---")
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Failure Probability",
                value=f"{failure_prob:.2%}",
                delta=f"{risk_level}",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                label="Avg Confidence",
                value=f"{metrics.get('avg_confidence', 0):.2%}",
                delta="Good" if metrics.get('avg_confidence', 0) > 0.75 else "Low"
            )
        
        with col3:
            st.metric(
                label="PSI Score",
                value=f"{metrics.get('psi_score', 0):.4f}",
                delta="Drift" if metrics.get('psi_score', 0) > 0.2 else "OK"
            )
        
        with col4:
            st.metric(
                label="Concept Drift",
                value=f"{metrics.get('concept_drift_count', 0)}",
                delta="Detected" if metrics.get('concept_drift_count', 0) > 0 else "None"
            )
        
        st.markdown("---")
        
        # Gauge Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                create_gauge_chart(failure_prob, "Failure Probability"),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                create_gauge_chart(metrics.get('avg_confidence', 0), "Average Confidence"),
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Load prediction logs for visualizations
        df_logs = load_prediction_logs()
        
        if not df_logs.empty:
            # Time series charts
            st.subheader("📊 Drift Metrics Over Time")
            
            df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
            df_recent = df_logs.tail(100)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence over time
                fig_conf = px.line(
                    df_recent,
                    x='timestamp',
                    y='confidence',
                    title='Confidence Over Time',
                    labels={'confidence': 'Confidence', 'timestamp': 'Time'}
                )
                fig_conf.add_hline(y=0.75, line_dash="dash", line_color="green", annotation_text="Good Threshold")
                fig_conf.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="Risk Threshold")
                st.plotly_chart(fig_conf, use_container_width=True)
            
            with col2:
                # Latency over time
                fig_lat = px.line(
                    df_recent,
                    x='timestamp',
                    y='latency',
                    title='Latency Over Time',
                    labels={'latency': 'Latency (s)', 'timestamp': 'Time'}
                )
                st.plotly_chart(fig_lat, use_container_width=True)
            
            # Concept drift occurrences
            if 'concept_drift' in df_recent.columns:
                st.subheader("🔄 Concept Drift Detections")
                drift_counts = df_recent.groupby(df_recent['timestamp'].dt.date)['concept_drift'].sum()
                fig_drift = px.bar(
                    x=drift_counts.index,
                    y=drift_counts.values,
                    title='Concept Drift Occurrences by Day',
                    labels={'x': 'Date', 'y': 'Drift Count'}
                )
                st.plotly_chart(fig_drift, use_container_width=True)
        
        st.markdown("---")
        
        # Alert History
        st.subheader("🚨 Alert History")
        df_alerts = load_alert_history()
        
        if not df_alerts.empty:
            df_alerts['timestamp'] = pd.to_datetime(df_alerts['timestamp'])
            df_alerts_recent = df_alerts.tail(10).sort_values('timestamp', ascending=False)
            
            st.dataframe(
                df_alerts_recent[[
                    'timestamp', 'failure_probability', 'psi_score',
                    'avg_confidence', 'concept_drift_count'
                ]],
                use_container_width=True
            )
        else:
            st.info("No alerts triggered yet")
        
        # Sidebar - Detailed Metrics
        with st.sidebar:
            st.header("📈 Detailed Metrics")
            
            st.subheader("Failure Risk")
            st.json({
                "probability": f"{failure_prob:.4f}",
                "risk_level": risk_level,
                "predictions_analyzed": failure_risk.get("predictions_analyzed", 0)
            })
            
            st.subheader("Drift Detection")
            if drift_status and "error" not in drift_status:
                st.json({
                    "overall_drift": drift_status.get("overall_drift_detected", False),
                    "total_predictions": drift_status.get("total_predictions_analyzed", 0)
                })
            
            st.subheader("Alert System")
            alert_info = failure_risk.get("alert_system", {})
            if alert_info.get("alert_triggered"):
                st.warning("🚨 Alert Active!")
                st.json(alert_info)
            else:
                st.success("✅ No Active Alerts")
            
            # Refresh button
            if st.button("🔄 Refresh Now"):
                st.rerun()
    
    else:
        st.error("❌ Failed to fetch data from API. Make sure the API is running at http://127.0.0.1:8000")
        st.info("Start the API with: `uvicorn app:app --reload`")
    
    # Auto-refresh
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

if __name__ == "__main__":
    main()
