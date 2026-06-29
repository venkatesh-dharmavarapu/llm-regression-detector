import streamlit as st
import json
import os
import pandas as pd

RESULTS_FILE = "data/latest_run.json"

st.set_page_config(
    page_title="LLM Regression Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Model Regression Detection Pipeline")
st.markdown("### Continuous Evaluation & Drift Monitoring Scorecard")
st.write("---")

if not os.path.exists(RESULTS_FILE):
    st.warning("⚠️ No evaluation metrics found. Run `python src/evaluator.py` first to generate data.")
else:
    # Load metrics from file
    with open(RESULTS_FILE, 'r') as file:
        data = json.load(file)
        
    df = pd.DataFrame(data)
    
    # Calculate top-line metrics
    total_cases = len(df)
    passed_cases = df['category_match'].sum()
    failed_cases = total_cases - passed_cases
    accuracy = (passed_cases / total_cases) * 100
    avg_judge_score = df['summary_judge_score'].mean()
    avg_latency = df['latency_seconds'].mean()

    # Layout Key Performance Indicators (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Categorical Accuracy", f"{accuracy:.1f}%", f"{passed_cases}/{total_cases} Passed")
    col2.metric("Avg Judge Score", f"{avg_judge_score:.2f} / 5.0", None)
    col3.metric("Avg Latency", f"{avg_latency:.2f}s", None)
    
    if failed_cases > 0:
        col4.metric("Regressions / Failures", f"{failed_cases}", "- Critical Alert", delta_color="inverse")
    else:
        col4.metric("Regressions / Failures", "0", "System Healthy", delta_color="normal")

    st.write("---")
    
    # Detailed Data breakdown
    st.subheader("📋 Detailed Test Case Breakdown")
    
    # Create scannable UI indicators for the table
    display_df = df.copy()
    display_df['Status'] = display_df['category_match'].apply(lambda x: "✅ PASS" if x else "❌ FAIL")
    display_df['Judge Evaluation'] = display_df['summary_judge_score'].apply(lambda x: "⭐" * x)
    
    # Select columns for readable viewing
    display_df = display_df[[
        'test_case_id', 'Status', 'expected_category', 
        'actual_category', 'Judge Evaluation', 'latency_seconds'
    ]]
    
    st.dataframe(display_df, use_container_width=True)
    
    st.write("---")
    
    # Deep Dive Diff View
    st.subheader("🔍 Deep-Dive Output Inspector")
    selected_id = st.selectbox("Select a Test Case ID to inspect raw prompt completions:", df['test_case_id'])
    
    case_row = df[df['test_case_id'] == selected_id].iloc[0]
    
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**Input Document Context:**\n\n{case_row['input_text']}")
        st.success(f"**Expected Target Summary:**\n\n{case_row['expected_summary']}")
        
    with c2:
        st.metric("Categorical Output", f"Actual: {case_row['actual_category']}", f"Expected: {case_row['expected_category']}")
        st.warning(f"**Model Generated Summary:**\n\n{case_row['actual_summary']}")