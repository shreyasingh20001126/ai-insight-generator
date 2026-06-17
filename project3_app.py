# ============================================================
# PROJECT 3 — AI-Powered Business Insight Generator
# Uses Claude AI to analyse any CSV and generate executive reports
# Run: streamlit run project3_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
import os
import json
import requests

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Business Insight Generator",
    page_icon="🤖",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0F1F3D; }
    .main { background-color: #0F1F3D; }
    section[data-testid="stSidebar"] {
        background-color: #131E30;
        border-right: 1px solid #243554;
    }
    .insight-card {
        background: #1A2840;
        border: 1px solid #243554;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .metric-row {
        background: #1A2840;
        border: 1px solid #243554;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        color: #00C2FF;
        font-size: 1.8rem;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #8BA3BE;
    }
    .stButton button {
        background: linear-gradient(135deg, #00C2FF, #0080AA);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
    }
    .report-box {
        background: #1A2840;
        border: 1px solid #243554;
        border-left: 4px solid #00C2FF;
        border-radius: 8px;
        padding: 24px;
        margin-top: 20px;
    }
    h1, h2, h3 { color: #E2EAF4 !important; }
    p, li { color: #8BA3BE; }
    .stMarkdown { color: #8BA3BE; }
</style>
""", unsafe_allow_html=True)

# ── CHART STYLE ───────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor' : '#0F1F3D',
    'axes.facecolor'   : '#1A2840',
    'axes.edgecolor'   : '#243554',
    'axes.labelcolor'  : '#8BA3BE',
    'xtick.color'      : '#8BA3BE',
    'ytick.color'      : '#8BA3BE',
    'text.color'       : '#E2EAF4',
    'grid.color'       : '#243554',
    'grid.linestyle'   : '--',
    'grid.alpha'       : 0.5,
    'font.family'      : 'sans-serif',
})
ACCENT = '#00C2FF'; WARN = '#F59E0B'; DANGER = '#EF4444'
SAFE = '#10B981'; PURPLE = '#A78BFA'

# ── CLAUDE API CALL ───────────────────────────────────────────
def call_claude(prompt, api_key):
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        else:
            return f"API Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Connection error: {str(e)}"

# ── DATA ANALYSIS ─────────────────────────────────────────────
def analyse_dataframe(df):
    analysis = {}

    # Basic info
    analysis['rows']    = len(df)
    analysis['cols']    = len(df.columns)
    analysis['columns'] = list(df.columns)

    # Column types
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    analysis['numeric_cols']     = num_cols
    analysis['categorical_cols'] = cat_cols

    # Missing values
    missing = df.isnull().sum()
    analysis['missing'] = missing[missing > 0].to_dict()
    analysis['missing_pct'] = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)

    # Numeric summary
    if num_cols:
        desc = df[num_cols].describe()
        analysis['numeric_summary'] = desc.to_string()

    # Categorical summary
    cat_summary = {}
    for col in cat_cols[:5]:
        cat_summary[col] = df[col].value_counts().head(5).to_dict()
    analysis['categorical_summary'] = cat_summary

    # Correlations
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        # Find top correlations
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append({
                    'col1': corr.columns[i],
                    'col2': corr.columns[j],
                    'corr': corr.iloc[i, j]
                })
        corr_pairs = sorted(corr_pairs, key=lambda x: abs(x['corr']), reverse=True)
        analysis['top_correlations'] = corr_pairs[:3]

    return analysis

# ── AUTO CHARTS ───────────────────────────────────────────────
def generate_charts(df, analysis):
    charts = []
    num_cols = analysis['numeric_cols']
    cat_cols = analysis['categorical_cols']

    # Chart 1: Distribution of first numeric column
    if num_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        col = num_cols[0]
        ax.hist(df[col].dropna(), bins=30, color=ACCENT,
                edgecolor='none', alpha=0.85)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        plt.tight_layout()
        charts.append(('distribution', fig, f'Distribution of {col}'))

    # Chart 2: Top category counts
    if cat_cols:
        col = cat_cols[0]
        top_cats = df[col].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [ACCENT, WARN, SAFE, DANGER, PURPLE,
                  '#F472B6', '#34D399', '#60A5FA']
        ax.barh(top_cats.index.astype(str), top_cats.values,
                color=colors[:len(top_cats)], edgecolor='none', alpha=0.88)
        ax.set_title(f'Top Values — {col}')
        ax.set_xlabel('Count')
        ax.grid(axis='x')
        ax.set_axisbelow(True)
        plt.tight_layout()
        charts.append(('categories', fig, f'Top Values in {col}'))

    # Chart 3: Correlation heatmap
    if len(num_cols) >= 3:
        cols_to_use = num_cols[:6]
        corr = df[cols_to_use].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols_to_use)))
        ax.set_yticks(range(len(cols_to_use)))
        ax.set_xticklabels(cols_to_use, rotation=45,
                           ha='right', fontsize=9)
        ax.set_yticklabels(cols_to_use, fontsize=9)
        for i in range(len(cols_to_use)):
            for j in range(len(cols_to_use)):
                ax.text(j, i, f'{corr.values[i,j]:.2f}',
                        ha='center', va='center',
                        fontsize=8, color='white')
        plt.colorbar(im, ax=ax)
        ax.set_title('Correlation Heatmap')
        plt.tight_layout()
        charts.append(('correlation', fig, 'Correlation Heatmap'))

    # Chart 4: Numeric columns bar (means)
    if len(num_cols) >= 2:
        means = df[num_cols[:8]].mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(range(len(means)), means.values,
                      color=ACCENT, edgecolor='none', alpha=0.88)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=45,
                           ha='right', fontsize=9)
        ax.set_title('Mean Values — Numeric Columns')
        ax.set_ylabel('Mean Value')
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        plt.tight_layout()
        charts.append(('means', fig, 'Column Means Overview'))

    return charts

# ── BUILD PROMPT FOR CLAUDE ───────────────────────────────────
def build_prompt(df, analysis, business_context):
    prompt = f"""You are a senior business analyst and data scientist. 
Analyse this dataset and write a professional executive report.

DATASET OVERVIEW:
- Rows: {analysis['rows']:,}
- Columns: {analysis['cols']}
- Column names: {analysis['columns']}
- Numeric columns: {analysis['numeric_cols']}
- Categorical columns: {analysis['categorical_cols']}
- Missing data: {analysis['missing_pct']:.1f}% of values are missing

NUMERIC SUMMARY:
{analysis.get('numeric_summary', 'No numeric columns')}

CATEGORICAL SUMMARY:
{json.dumps(analysis.get('categorical_summary', {}), indent=2)}

TOP CORRELATIONS:
{json.dumps([{'columns': f"{c['col1']} vs {c['col2']}", 'correlation': round(c['corr'], 3)} for c in analysis.get('top_correlations', [])], indent=2)}

BUSINESS CONTEXT PROVIDED BY USER:
{business_context if business_context else 'No specific context provided - infer from column names'}

Write a professional executive report with these exact sections:

## Executive Summary
2-3 sentences summarising what this data shows at a high level.

## Key Findings
5 specific, numbered findings from the actual data. Include real numbers.

## Risk Areas
3 specific risks or concerns identified in the data.

## Business Recommendations
5 specific, actionable recommendations based on the data findings.

## Next Steps
3 concrete next steps for the business to take.

Be specific, use the actual column names and numbers from the data.
Write in professional business language suitable for a C-suite audience.
Do not use generic statements — every point must reference the actual data."""

    return prompt

# ════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════

# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div style='padding: 10px 0 20px'>
    <div style='color: #00C2FF; font-size: 12px; font-weight: 600;
                letter-spacing: 0.1em; text-transform: uppercase;
                margin-bottom: 8px'>
        MSc Data Science Portfolio — Project 3
    </div>
    <h1 style='margin:0; font-size: 2rem; font-weight: 800;
               color: #E2EAF4; letter-spacing: -0.02em'>
        🤖 AI Business Insight Generator
    </h1>
    <p style='color: #8BA3BE; margin-top: 8px; font-size: 15px'>
        Upload any CSV → Get an AI-generated executive report with charts and recommendations
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "Anthropic API Key",
    type="password",
    help="Get your free key at console.anthropic.com"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Business Context")
business_context = st.sidebar.text_area(
    "Describe your business/dataset (optional)",
    placeholder="e.g. This is sales data from a UK retail company. We want to understand revenue trends and customer behaviour.",
    height=120
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How it works:**
1. Upload your CSV file
2. App analyses the data automatically
3. Claude AI writes an executive report
4. Download charts and insights

**Supported:** Any CSV file up to 200MB
""")

# ── MAIN CONTENT ──────────────────────────────────────────────
col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload any CSV dataset to analyse"
    )

with col_info:
    st.markdown("""
    <div style='background:#1A2840; border:1px solid #243554;
                border-radius:10px; padding:16px; margin-top:28px'>
        <div style='color:#00C2FF; font-weight:700; margin-bottom:8px'>
            💡 Try these datasets
        </div>
        <div style='color:#8BA3BE; font-size:13px; line-height:1.8'>
            • UK Road Safety CSV<br>
            • Telco Churn CSV<br>
            • Any sales/HR/finance CSV<br>
            • ONS UK economic data
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── PROCESS FILE ──────────────────────────────────────────────
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded: **{len(df):,} rows × {df.shape[1]} columns**")

        # ── DATA OVERVIEW ─────────────────────────────────────
        st.subheader("📊 Dataset Overview")

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Rows", f"{len(df):,}")
        with m2:
            st.metric("Total Columns", df.shape[1])
        with m3:
            st.metric("Numeric Columns", len(num_cols))
        with m4:
            st.metric("Missing Data", f"{missing_pct:.1f}%")

        # ── DATA PREVIEW ──────────────────────────────────────
        with st.expander("👀 Preview Data (first 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)

        with st.expander("📈 Statistical Summary"):
            if num_cols:
                st.dataframe(df[num_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns found")

        # ── AUTO CHARTS ───────────────────────────────────────
        st.subheader("📉 Automatic Data Visualisations")

        analysis = analyse_dataframe(df)
        charts   = generate_charts(df, analysis)

        if charts:
            cols = st.columns(2)
            for i, (chart_type, fig, title) in enumerate(charts):
                with cols[i % 2]:
                    st.pyplot(fig)
                    plt.close(fig)
        else:
            st.info("No charts generated — dataset may need more numeric columns")

        # ── AI REPORT ─────────────────────────────────────────
        st.subheader("🤖 AI Executive Report")

        if not api_key:
            st.warning("""
            ⚠️ **API Key Required**

            To generate the AI report:
            1. Go to **console.anthropic.com**
            2. Sign up for free
            3. Go to API Keys → Create Key
            4. Paste it in the sidebar

            You get free credits to start!
            """)
        else:
            if st.button("🚀 Generate AI Executive Report"):
                with st.spinner("Claude AI is analysing your data and writing the report..."):
                    prompt = build_prompt(df, analysis, business_context)
                    report = call_claude(prompt, api_key)

                if report.startswith("API Error") or report.startswith("Connection"):
                    st.error(f"❌ {report}")
                    st.info("Check your API key is correct and has credits available.")
                else:
                    st.markdown("""
                    <div style='background:#1A2840; border:1px solid #243554;
                                border-left:4px solid #00C2FF; border-radius:8px;
                                padding:24px; margin-top:16px'>
                    """, unsafe_allow_html=True)
                    st.markdown(report)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Download button
                    report_with_header = f"""# AI Business Insight Report
Generated by: AI Business Insight Generator
Dataset: {uploaded_file.name}
Rows: {len(df):,} | Columns: {df.shape[1]}

---

{report}

---
Generated using Claude AI | MSc Data Science Portfolio | Shreya Singh
"""
                    st.download_button(
                        label="📥 Download Report as Text",
                        data=report_with_header,
                        file_name="business_insight_report.txt",
                        mime="text/plain"
                    )

        # ── KEY STATS TABLE ───────────────────────────────────
        if num_cols:
            st.subheader("📋 Column Statistics")
            stats_df = df[num_cols].agg(['mean','median','std','min','max']).round(2)
            st.dataframe(stats_df, use_container_width=True)

        # ── MISSING DATA ──────────────────────────────────────
        if analysis['missing']:
            st.subheader("⚠️ Missing Data Report")
            missing_df = pd.DataFrame({
                'Column': list(analysis['missing'].keys()),
                'Missing Count': list(analysis['missing'].values()),
                'Missing %': [round(v/len(df)*100, 1)
                              for v in analysis['missing'].values()]
            })
            st.dataframe(missing_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.info("Make sure your file is a valid CSV with headers in the first row.")

else:
    # ── EMPTY STATE ───────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:60px 20px;
                background:#1A2840; border:2px dashed #243554;
                border-radius:16px; margin-top:20px'>
        <div style='font-size:48px; margin-bottom:16px'>📂</div>
        <div style='color:#E2EAF4; font-size:20px;
                    font-weight:700; margin-bottom:8px'>
            Upload a CSV to get started
        </div>
        <div style='color:#8BA3BE; font-size:14px; max-width:400px; margin:0 auto'>
            The app will automatically analyse your data, generate charts,
            and use Claude AI to write a professional executive report
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("How It Works")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1️⃣", "Upload CSV", "Any dataset — sales, HR, finance, operations"),
        ("2️⃣", "Auto Analysis", "App calculates stats, finds patterns, builds charts"),
        ("3️⃣", "AI Report", "Claude writes a full executive report from your data"),
        ("4️⃣", "Download", "Save the report and share with stakeholders"),
    ]
    for col, (num, title, desc) in zip([c1,c2,c3,c4], steps):
        with col:
            st.markdown(f"""
            <div style='background:#1A2840; border:1px solid #243554;
                        border-radius:10px; padding:16px; text-align:center'>
                <div style='font-size:28px'>{num}</div>
                <div style='color:#E2EAF4; font-weight:700;
                            margin:8px 0 4px'>{title}</div>
                <div style='color:#8BA3BE; font-size:12px'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8BA3BE; font-size:12px'>
    MSc Data Science Portfolio — Project 3 | Shreya Singh |
    Built with Python · Streamlit · Claude AI API
</div>
""", unsafe_allow_html=True)
