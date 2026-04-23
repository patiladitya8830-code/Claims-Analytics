import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Life Insurance Claims Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AI-powered Life Insurance Claims Analytics Dashboard"
    }
)

# ==========================================
# CUSTOM CSS — DARK GLASSMORPHISM THEME
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* ── Metric Cards ── */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        backdrop-filter: blur(12px);
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 48px rgba(99,102,241,0.25);
        border-color: rgba(99,102,241,0.4);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* ── Headers ── */
    h1 { 
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.3rem !important;
    }
    h2, h3 { color: #c4b5fd !important; }

    /* ── Section Cards ── */
    .section-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8 !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(99,102,241,0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.5);
    }

    /* ── Inputs ── */
    .stNumberInput input, .stSelectbox, .stMultiSelect {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px;
        color: #e2e8f0 !important;
    }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* ── Radio buttons ── */
    .stRadio label { color: #e2e8f0 !important; }

    /* ── KPI badge ── */
    .kpi-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }

    /* ── Prediction result boxes ── */
    .pred-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1));
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 14px;
        padding: 1.2rem 1.8rem;
        margin-top: 1rem;
    }
    .pred-low {
        background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(22,163,74,0.1));
        border: 1px solid rgba(34,197,94,0.4);
        border-radius: 14px;
        padding: 1.2rem 1.8rem;
        margin-top: 1rem;
    }

    /* ── Info boxes ── */
    .stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# PLOTLY DEFAULT TEMPLATE
# ==========================================
PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Inter"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
    colorway=["#6366f1", "#8b5cf6", "#a78bfa", "#34d399", "#f59e0b", "#f43f5e",
              "#06b6d4", "#10b981", "#ec4899", "#84cc16"]
)


# ==========================================
# DATA LOADING & CACHING
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_group_death_claims.csv")
        df.fillna(df.median(numeric_only=True), inplace=True)
        df.drop_duplicates(inplace=True)
        df.drop(['Policy_ID'], axis=1, inplace=True, errors='ignore')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


df_original = load_data()


# ==========================================
# MODEL TRAINING & CACHING
# ==========================================
@st.cache_resource
def train_model(df):
    if df.empty:
        return None, None, None, None, None, None

    df_model = df.copy()
    median_claim = df_model['claims_paid_amt'].median()
    df_model['High_Claim'] = (df_model['claims_paid_amt'] > median_claim).astype(int)

    df_encoded = pd.get_dummies(df_model, drop_first=True)

    # Keep only numeric
    df_encoded = df_encoded.select_dtypes(include=np.number)

    # Separate target
    if 'High_Claim' not in df_encoded.columns:
        return None, None, None, None, None, None

    X = df_encoded.drop('High_Claim', axis=1)
    y = df_encoded['High_Claim']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                 max_depth=12, min_samples_leaf=2)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall':    recall_score(y_test, y_pred, zero_division=0),
        'F1 Score':  f1_score(y_test, y_pred, zero_division=0),
        'AUC-ROC':   roc_auc_score(y_test, y_prob)
    }

    return rf, X.columns, metrics, X_test, y_test, y_prob


model, features, metrics, X_test, y_test, y_prob = train_model(df_original)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);'>
        <div style='font-size:2.8rem;'>🏦</div>
        <div style='font-size:1.1rem; font-weight:700; color:#a78bfa; margin-top:6px;'>Claims Analytics</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:2px;'>AI-Powered Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "",
        ["📊 Dashboard & KPIs",
         "📈 Trends Analysis",
         "🗂️ Data Exploration",
         "⚖️ Claims Ratio Analysis",
         "🤖 Predictive Analytics"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # ── Sidebar Filters ──
    st.markdown("### 🔧 Filters")

    all_years = sorted(df_original['year'].unique().tolist()) if not df_original.empty else []
    selected_years = st.multiselect("📅 Year(s)", all_years, default=all_years,
                                     help="Filter data by financial year")

    all_insurers = sorted(df_original['life_insurer'].unique().tolist()) if not df_original.empty else []
    selected_insurers = st.multiselect("🏢 Insurer(s)", all_insurers, default=all_insurers,
                                        help="Filter by life insurer")

    all_categories = sorted(df_original['category'].unique().tolist()) if 'category' in df_original.columns and not df_original.empty else []
    selected_categories = st.multiselect("📁 Category", all_categories, default=all_categories,
                                          help="Filter by claim category")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748b; text-align:center;'>
        <b>Data:</b> Group Death Claims<br/>
        <b>Model:</b> Random Forest<br/>
        <b>v2.0.0</b> · 2025
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# APPLY FILTERS
# ==========================================
if df_original.empty:
    st.warning("⚠️ Data could not be loaded. Please ensure 'cleaned_group_death_claims.csv' exists.")
    st.stop()

df = df_original.copy()
if selected_years:
    df = df[df['year'].isin(selected_years)]
if selected_insurers:
    df = df[df['life_insurer'].isin(selected_insurers)]
if selected_categories and 'category' in df.columns:
    df = df[df['category'].isin(selected_categories)]

if df.empty:
    st.warning("No data matches the current filter selection. Please adjust the sidebar filters.")
    st.stop()


# ==========================================
# HELPER
# ==========================================
def apply_layout(fig, title="", height=400, legend=True):
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(size=15, color="#c4b5fd"), x=0.01),
        height=height,
        showlegend=legend,
        margin=dict(l=30, r=20, t=50, b=30)
    )
    return fig


PAGE = page.split(" ", 1)[1].strip()


# ══════════════════════════════════════════
# PAGE 1 — DASHBOARD & KPIs
# ══════════════════════════════════════════
if PAGE == "Dashboard & KPIs":
    st.title("📊 Executive Dashboard")
    st.markdown(f"<span style='color:#64748b;font-size:.9rem;'>Showing data for {len(df):,} records · {df['life_insurer'].nunique()} insurers · {df['year'].nunique()} year(s)</span>", unsafe_allow_html=True)

    # ── KPI Row ──
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Life Insurers", f"{df['life_insurer'].nunique()}")
    with col3:
        total_paid = df['claims_paid_amt'].sum()
        st.metric("Total Claims Paid (₹ Cr)", f"₹ {total_paid/1e7:,.1f} Cr")
    with col4:
        avg_paid_ratio = df['claims_paid_ratio_no'].mean() * 100
        st.metric("Avg Paid Ratio (No.)", f"{avg_paid_ratio:.1f}%")
    with col5:
        avg_repud_ratio = df['claims_repudiated_rejected_ratio_no'].mean() * 100
        st.metric("Avg Repudiation Rate", f"{avg_repud_ratio:.1f}%")

    st.markdown("---")

    # ── Row 1 ──
    colA, colB = st.columns(2)
    with colA:
        fig1 = px.histogram(df, x="claims_paid_amt", nbins=40,
                            color_discrete_sequence=["#6366f1"],
                            marginal="box")
        apply_layout(fig1, "Claims Paid Amount — Distribution", legend=False)
        fig1.update_xaxes(title_text="Amount (₹)")
        fig1.update_yaxes(title_text="Count")
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        insurer_data = (df.groupby('life_insurer')['claims_paid_amt']
                        .sum().reset_index()
                        .sort_values('claims_paid_amt', ascending=False).head(15))
        fig2 = px.bar(insurer_data, x='life_insurer', y='claims_paid_amt',
                      color='claims_paid_amt', color_continuous_scale='Purples')
        apply_layout(fig2, "Top 15 Insurers — Total Claims Paid (₹)", legend=False)
        fig2.update_xaxes(tickangle=-45, title_text="Insurer")
        fig2.update_yaxes(title_text="Total Paid (₹)")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2 ──
    colC, colD = st.columns(2)
    with colC:
        fig3 = px.scatter(df, x="claims_intimated_amt", y="claims_paid_amt",
                          color="life_insurer",
                          trendline="ols",
                          hover_data=["year", "category"])
        apply_layout(fig3, "Intimated vs Paid Amounts — Scatter")
        fig3.update_xaxes(title_text="Intimated Amount (₹)")
        fig3.update_yaxes(title_text="Paid Amount (₹)")
        st.plotly_chart(fig3, use_container_width=True)

    with colD:
        if 'category' in df.columns:
            cat_data = df['category'].value_counts().reset_index()
            cat_data.columns = ['category', 'count']
            fig4 = px.pie(cat_data, values='count', names='category', hole=0.45,
                          color_discrete_sequence=["#6366f1", "#8b5cf6", "#a78bfa", "#34d399"])
            apply_layout(fig4, "Volume by Claim Category")
            st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3 — Claims Lifecycle: Intimated vs Paid vs Pending ──
    st.markdown("### 🔄 Claims Lifecycle Overview")
    life_df = df.groupby('life_insurer')[
        ['claims_intimated_amt', 'claims_paid_amt', 'claims_repudiated_amt',
         'claims_rejected_amt', 'claims_pending_end_amt']
    ].sum().reset_index().sort_values('claims_intimated_amt', ascending=False).head(12)

    fig5 = go.Figure()
    cols_lifecycle = {
        'claims_intimated_amt': ('#6366f1', 'Intimated'),
        'claims_paid_amt': ('#34d399', 'Paid'),
        'claims_repudiated_amt': ('#f59e0b', 'Repudiated'),
        'claims_rejected_amt': ('#f43f5e', 'Rejected'),
        'claims_pending_end_amt': ('#06b6d4', 'Pending End'),
    }
    for col, (color, label) in cols_lifecycle.items():
        fig5.add_trace(go.Bar(name=label, x=life_df['life_insurer'],
                              y=life_df[col], marker_color=color))
    fig5.update_layout(barmode='group', **PLOTLY_TEMPLATE,
                       title=dict(text="Claims Lifecycle by Insurer (Top 12)", font=dict(size=15, color="#c4b5fd"), x=0.01),
                       height=420, margin=dict(l=30, r=20, t=50, b=80))
    fig5.update_xaxes(tickangle=-40)
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════
# PAGE 2 — TRENDS ANALYSIS
# ══════════════════════════════════════════
elif PAGE == "Trends Analysis":
    st.title("📈 Trends Analysis")
    st.markdown("<span style='color:#64748b;font-size:.9rem;'>Year-over-year movement across key claims metrics</span>", unsafe_allow_html=True)

    year_df = df.groupby('year').agg(
        total_paid=('claims_paid_amt', 'sum'),
        total_intimated=('claims_intimated_amt', 'sum'),
        total_repudiated=('claims_repudiated_amt', 'sum'),
        total_rejected=('claims_rejected_amt', 'sum'),
        total_pending_end=('claims_pending_end_amt', 'sum'),
        avg_paid_ratio_no=('claims_paid_ratio_no', 'mean'),
        avg_paid_ratio_amt=('claims_paid_ratio_amt', 'mean'),
        avg_repud_ratio=('claims_repudiated_rejected_ratio_no', 'mean'),
        total_paid_no=('claims_paid_no', 'sum'),
        total_intimated_no=('claims_intimated_no', 'sum'),
    ).reset_index()

    # ── Line Chart: Amount trends ──
    fig_trend1 = go.Figure()
    trend_metrics = {
        'total_intimated': ('#6366f1', 'Intimated'),
        'total_paid': ('#34d399', 'Paid'),
        'total_repudiated': ('#f59e0b', 'Repudiated'),
        'total_rejected': ('#f43f5e', 'Rejected'),
        'total_pending_end': ('#06b6d4', 'Pending End'),
    }
    for col, (color, label) in trend_metrics.items():
        fig_trend1.add_trace(go.Scatter(
            x=year_df['year'], y=year_df[col],
            mode='lines+markers', name=label,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
        ))
    fig_trend1.update_layout(**PLOTLY_TEMPLATE,
                              title=dict(text="Claims Amount Trends (₹) — Year over Year", font=dict(size=15, color="#c4b5fd"), x=0.01),
                              height=420, margin=dict(l=30, r=20, t=50, b=30),
                              hovermode="x unified")
    st.plotly_chart(fig_trend1, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        # Claims Paid Ratio (No.) trend
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(
            x=year_df['year'],
            y=year_df['avg_paid_ratio_no'] * 100,
            mode='lines+markers+text',
            name='Paid Ratio (No.)',
            line=dict(color='#34d399', width=3),
            marker=dict(size=10),
            text=[f"{v:.1f}%" for v in year_df['avg_paid_ratio_no'] * 100],
            textposition='top center'
        ))
        fig_ratio.add_trace(go.Scatter(
            x=year_df['year'],
            y=year_df['avg_repud_ratio'] * 100,
            mode='lines+markers+text',
            name='Repud+Rejected Ratio (No.)',
            line=dict(color='#f43f5e', width=3),
            marker=dict(size=10),
            text=[f"{v:.1f}%" for v in year_df['avg_repud_ratio'] * 100],
            textposition='bottom center'
        ))
        fig_ratio.update_layout(**PLOTLY_TEMPLATE,
                                 title=dict(text="Claims Paid vs Repudiation Rate (%) — YoY", font=dict(size=14, color="#c4b5fd"), x=0.01),
                                 height=380, margin=dict(l=30, r=20, t=50, b=30),
                                 yaxis_ticksuffix="%")
        st.plotly_chart(fig_ratio, use_container_width=True)

    with colB:
        # Volume trend (count)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=year_df['year'], y=year_df['total_intimated_no'],
                                  name='Intimated (No.)', marker_color='#6366f1'))
        fig_vol.add_trace(go.Bar(x=year_df['year'], y=year_df['total_paid_no'],
                                  name='Paid (No.)', marker_color='#34d399'))
        fig_vol.update_layout(**PLOTLY_TEMPLATE, barmode='group',
                               title=dict(text="Claims Volume (Count) — Intimated vs Paid", font=dict(size=14, color="#c4b5fd"), x=0.01),
                               height=380, margin=dict(l=30, r=20, t=50, b=30))
        st.plotly_chart(fig_vol, use_container_width=True)

    # ── Per-Insurer Heatmap: claims_paid_ratio_no by Year ──
    st.markdown("### 🌡️ Insurer Paid Ratio Heatmap (by Year)")
    heat_df = df.pivot_table(index='life_insurer', columns='year',
                              values='claims_paid_ratio_no', aggfunc='mean')
    fig_heat = px.imshow(heat_df * 100, text_auto=".1f",
                          color_continuous_scale='Viridis',
                          labels=dict(x="Year", y="Insurer", color="Paid Ratio (%)"))
    fig_heat.update_layout(**PLOTLY_TEMPLATE, height=max(400, len(heat_df) * 28 + 80),
                            title=dict(text="Claims Paid Ratio (%) by Insurer & Year", font=dict(size=14, color="#c4b5fd"), x=0.01),
                            margin=dict(l=10, r=20, t=50, b=30))
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════
# PAGE 3 — DATA EXPLORATION
# ══════════════════════════════════════════
elif PAGE == "Data Exploration":
    st.title("🗂️ Data Exploration")
    st.markdown("<span style='color:#64748b;font-size:.9rem;'>Raw data views, statistics, and correlation analysis</span>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📐 Statistics", "🔗 Correlations"])

    with tab1:
        st.dataframe(df, use_container_width=True, height=500)
        colX, colY = st.columns(2)
        with colX:
            st.markdown(f"**Rows:** {len(df):,}  |  **Columns:** {df.shape[1]}")
        with colY:
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Filtered CSV", csv, "filtered_claims.csv", "text/csv")

    with tab2:
        desc = df.describe().T
        desc.index.name = "Column"
        st.dataframe(desc, use_container_width=True)

        # Box plots for key numerics
        st.markdown("#### Distribution Comparison (Box Plots)")
        num_cols_select = [c for c in df.select_dtypes(include=np.number).columns
                           if df[c].std() > 0]
        chosen = st.multiselect("Select columns to compare:", num_cols_select,
                                 default=num_cols_select[:4])
        if chosen:
            fig_box = go.Figure()
            for c in chosen:
                fig_box.add_trace(go.Box(y=df[c], name=c, boxmean='sd'))
            fig_box.update_layout(**PLOTLY_TEMPLATE,
                                   title=dict(text="Box Plot Comparison", font=dict(size=14, color="#c4b5fd"), x=0.01),
                                   height=420, margin=dict(l=20, r=20, t=50, b=30))
            st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        st.subheader("Feature Correlation Matrix")
        corr = df.select_dtypes(include=np.number).corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                              color_continuous_scale="RdBu_r")
        fig_corr.update_layout(**PLOTLY_TEMPLATE, height=650,
                                title=dict(text="Pearson Correlation Heatmap", font=dict(size=14, color="#c4b5fd"), x=0.01),
                                margin=dict(l=10, r=20, t=50, b=30))
        st.plotly_chart(fig_corr, use_container_width=True)

        # Top correlations with claims_paid_amt
        st.markdown("#### Top Correlations with `claims_paid_amt`")
        top_corr = (corr['claims_paid_amt']
                    .drop('claims_paid_amt')
                    .abs()
                    .sort_values(ascending=False)
                    .reset_index())
        top_corr.columns = ['Feature', '|Correlation|']
        fig_corr2 = px.bar(top_corr, x='|Correlation|', y='Feature', orientation='h',
                            color='|Correlation|', color_continuous_scale='Purples')
        fig_corr2.update_layout(**PLOTLY_TEMPLATE,
                                 title=dict(text="Features Correlated with Claims Paid Amount", font=dict(size=14, color="#c4b5fd"), x=0.01),
                                 height=500, margin=dict(l=20, r=20, t=50, b=30))
        st.plotly_chart(fig_corr2, use_container_width=True)


# ══════════════════════════════════════════
# PAGE 4 — CLAIMS RATIO ANALYSIS
# ══════════════════════════════════════════
elif PAGE == "Claims Ratio Analysis":
    st.title("⚖️ Claims Ratio Analysis")
    st.markdown("<span style='color:#64748b;font-size:.9rem;'>Deep-dive into settlement, repudiation, and pending ratios</span>", unsafe_allow_html=True)

    # ── KPIs ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Settlement Ratio (No.)",
                  f"{df['claims_paid_ratio_no'].mean()*100:.1f}%",
                  delta=f"{(df['claims_paid_ratio_no'].mean() - df_original['claims_paid_ratio_no'].mean())*100:+.1f}% vs all")
    with col2:
        st.metric("Avg Settlement Ratio (Amt.)",
                  f"{df['claims_paid_ratio_amt'].mean()*100:.1f}%")
    with col3:
        st.metric("Avg Repud+Rej Ratio (No.)",
                  f"{df['claims_repudiated_rejected_ratio_no'].mean()*100:.1f}%")
    with col4:
        st.metric("Avg Pending End Ratio (No.)",
                  f"{df['claims_pending_ratio_no'].mean()*100:.1f}%")

    st.markdown("---")

    colA, colB = st.columns(2)
    with colA:
        # Claim Approval Rate by Category (from notebook)
        claim_rate = df.groupby('category')[
            ['claims_paid_ratio_no', 'claims_paid_ratio_amt']
        ].mean().reset_index()
        claim_rate['claims_paid_ratio_no_pct'] = claim_rate['claims_paid_ratio_no'] * 100
        claim_rate['claims_paid_ratio_amt_pct'] = claim_rate['claims_paid_ratio_amt'] * 100

        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(name='By Count (%)', x=claim_rate['category'],
                                  y=claim_rate['claims_paid_ratio_no_pct'],
                                  marker_color='#6366f1'))
        fig_cat.add_trace(go.Bar(name='By Amount (%)', x=claim_rate['category'],
                                  y=claim_rate['claims_paid_ratio_amt_pct'],
                                  marker_color='#34d399'))
        fig_cat.update_layout(**PLOTLY_TEMPLATE, barmode='group',
                               title=dict(text="Claim Settlement Rate by Category", font=dict(size=14, color="#c4b5fd"), x=0.01),
                               height=380, yaxis_ticksuffix="%",
                               margin=dict(l=20, r=20, t=50, b=30))
        st.plotly_chart(fig_cat, use_container_width=True)

    with colB:
        # Insurer-level settlement efficiency: scatter plot
        ins_ratio = df.groupby('life_insurer').agg(
            paid_ratio_no=('claims_paid_ratio_no', 'mean'),
            paid_ratio_amt=('claims_paid_ratio_amt', 'mean'),
            repud_ratio=('claims_repudiated_rejected_ratio_no', 'mean'),
            total_paid=('claims_paid_amt', 'sum')
        ).reset_index()

        fig_scatter = px.scatter(ins_ratio, x='paid_ratio_no', y='paid_ratio_amt',
                                  size='total_paid', color='repud_ratio',
                                  text='life_insurer',
                                  color_continuous_scale='RdYlGn_r',
                                  size_max=50,
                                  labels={
                                      'paid_ratio_no': 'Settlement Ratio (Count)',
                                      'paid_ratio_amt': 'Settlement Ratio (Amount)',
                                      'repud_ratio': 'Repud. Ratio'
                                  })
        fig_scatter.update_traces(textposition='top center', textfont_size=9)
        fig_scatter.update_layout(**PLOTLY_TEMPLATE,
                                   title=dict(text="Insurer Efficiency Map (Size = Total Paid)", font=dict(size=14, color="#c4b5fd"), x=0.01),
                                   height=380,
                                   margin=dict(l=20, r=20, t=50, b=30))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Full Insurer Leaderboard ──
    st.markdown("### 🏆 Insurer Leaderboard — Settlement Efficiency")
    leader = df.groupby('life_insurer').agg(
        Settlement_Ratio_No=('claims_paid_ratio_no', 'mean'),
        Settlement_Ratio_Amt=('claims_paid_ratio_amt', 'mean'),
        Repud_Rej_Ratio=('claims_repudiated_rejected_ratio_no', 'mean'),
        Pending_Ratio=('claims_pending_ratio_no', 'mean'),
        Total_Paid_Cr=('claims_paid_amt', lambda x: x.sum() / 1e7),
        Records=('claims_paid_amt', 'count')
    ).reset_index().sort_values('Settlement_Ratio_No', ascending=False)

    leader['Settlement_Ratio_No'] = (leader['Settlement_Ratio_No'] * 100).round(2)
    leader['Settlement_Ratio_Amt'] = (leader['Settlement_Ratio_Amt'] * 100).round(2)
    leader['Repud_Rej_Ratio'] = (leader['Repud_Rej_Ratio'] * 100).round(2)
    leader['Pending_Ratio'] = (leader['Pending_Ratio'] * 100).round(2)
    leader['Total_Paid_Cr'] = leader['Total_Paid_Cr'].round(2)

    st.dataframe(leader, use_container_width=True)

    # ── Box plots: Paid Ratio by Insurer ──
    st.markdown("### 📦 Settlement Ratio Distribution by Insurer")
    fig_box_ratio = px.box(df, x='life_insurer', y='claims_paid_ratio_no',
                            color='life_insurer',
                            labels={'claims_paid_ratio_no': 'Paid Ratio (No.)', 'life_insurer': 'Insurer'})
    fig_box_ratio.update_layout(**PLOTLY_TEMPLATE,
                                 title=dict(text="Settlement Ratio Spread per Insurer", font=dict(size=14, color="#c4b5fd"), x=0.01),
                                 height=440, showlegend=False,
                                 margin=dict(l=20, r=20, t=50, b=80))
    fig_box_ratio.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_box_ratio, use_container_width=True)


# ══════════════════════════════════════════
# PAGE 5 — PREDICTIVE ANALYTICS
# ══════════════════════════════════════════
elif PAGE == "Predictive Analytics":
    st.title("🤖 Predictive Analytics (AI Model)")
    st.markdown("""
    <span style='color:#64748b;font-size:.9rem;'>
    Random Forest Classifier predicts whether a claim record is a <b>High Claim</b> scenario
    (paid amount &gt; dataset median). Trained on the full unfiltered dataset.
    </span>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("❌ Model could not be trained. Please check the data.")
    else:
        # ── Model Metrics ──
        st.markdown("### 📐 Model Performance Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Accuracy",  f"{metrics['Accuracy']:.2%}")
        with m2:
            st.metric("Precision", f"{metrics['Precision']:.2%}")
        with m3:
            st.metric("Recall",    f"{metrics['Recall']:.2%}")
        with m4:
            st.metric("F1 Score",  f"{metrics['F1 Score']:.2%}")
        with m5:
            st.metric("AUC-ROC",   f"{metrics['AUC-ROC']:.4f}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        # Confusion Matrix
        with col1:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Purples',
                               labels=dict(x="Predicted", y="Actual"),
                               x=['Low Claim', 'High Claim'],
                               y=['Low Claim', 'High Claim'])
            fig_cm.update_layout(**PLOTLY_TEMPLATE,
                                  title=dict(text="Confusion Matrix", font=dict(size=14, color="#c4b5fd"), x=0.01),
                                  height=380, margin=dict(l=20, r=20, t=50, b=30))
            st.plotly_chart(fig_cm, use_container_width=True)

        # ROC Curve
        with col2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                          line=dict(color='#6366f1', width=2.5),
                                          name=f"ROC (AUC={metrics['AUC-ROC']:.4f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                          line=dict(color='#64748b', dash='dash'),
                                          name='Random'))
            fig_roc.update_layout(**PLOTLY_TEMPLATE,
                                   title=dict(text="ROC Curve", font=dict(size=14, color="#c4b5fd"), x=0.01),
                                   height=380, margin=dict(l=20, r=20, t=50, b=30),
                                   xaxis_title="False Positive Rate",
                                   yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)

        # Feature Importance
        st.markdown("### 🌟 Feature Importance")
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]
        imp_df = pd.DataFrame({
            'Feature': [features[i] for i in indices],
            'Importance': importances[indices]
        }).sort_values('Importance', ascending=True)

        fig_feat = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                           color='Importance', color_continuous_scale='Purples')
        fig_feat.update_layout(**PLOTLY_TEMPLATE,
                                title=dict(text="Top 20 Feature Importances", font=dict(size=14, color="#c4b5fd"), x=0.01),
                                height=520, margin=dict(l=20, r=20, t=50, b=30),
                                showlegend=False)
        st.plotly_chart(fig_feat, use_container_width=True)

        # ── Prediction Form ──
        st.markdown("---")
        st.markdown("### 🔮 Make a Prediction")
        st.markdown("<span style='color:#64748b;font-size:.88rem;'>Enter values below. Categorical fields default to the most common category.</span>", unsafe_allow_html=True)

        num_cols = df_original.select_dtypes(include=np.number).columns.tolist()
        num_cols = [c for c in num_cols if c not in ['claims_paid_amt']]

        with st.form("predict_form"):
            st.markdown("#### Numerical Inputs")
            form_cols = st.columns(3)
            user_inputs = {}
            for i, col in enumerate(num_cols):
                with form_cols[i % 3]:
                    val = st.number_input(
                        col.replace("_", " ").title(),
                        value=float(df_original[col].median()),
                        format="%.4f",
                        key=f"input_{col}"
                    )
                    user_inputs[col] = val

            submitted = st.form_submit_button("⚡ Run Prediction", use_container_width=True)

        if submitted:
            # Build input row aligned to training features
            # Use most-common value for categoricals
            input_row = {}
            for col in df_original.columns:
                if col == 'claims_paid_amt':
                    input_row[col] = df_original[col].median()
                elif col in user_inputs:
                    input_row[col] = user_inputs[col]
                elif df_original[col].dtype == object:
                    input_row[col] = df_original[col].mode()[0]
                else:
                    input_row[col] = df_original[col].median()

            input_df = pd.DataFrame([input_row])

            # Add dummy High_Claim col (needed for get_dummies alignment)
            input_df['High_Claim'] = 0
            df_model_tmp = df_original.copy()
            df_model_tmp['High_Claim'] = 0

            # Encode together to align categories, then take only the new row
            combined = pd.concat([df_model_tmp, input_df], ignore_index=True)
            combined_enc = pd.get_dummies(combined, drop_first=True).select_dtypes(include=np.number)
            input_enc = combined_enc.iloc[[-1]]

            # Align to model features
            for c in features:
                if c not in input_enc.columns:
                    input_enc[c] = 0
            input_enc = input_enc[features]

            prediction = model.predict(input_enc)[0]
            prob = model.predict_proba(input_enc)[0]

            if prediction == 1:
                st.markdown(f"""
                <div class='pred-high'>
                    <h3 style='color:#f87171; margin:0;'>🚨 HIGH CLAIM PREDICTED</h3>
                    <p style='margin:6px 0 0; color:#fca5a5;'>
                        Confidence: <b>{prob[1]:.2%}</b> · This record is likely above the median claim amount.
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='pred-low'>
                    <h3 style='color:#4ade80; margin:0;'>✅ LOW CLAIM PREDICTED</h3>
                    <p style='margin:6px 0 0; color:#86efac;'>
                        Confidence: <b>{prob[0]:.2%}</b> · This record is likely below the median claim amount.
                    </p>
                </div>""", unsafe_allow_html=True)

            # Show probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob[1] * 100,
                title={'text': "High Claim Probability", 'font': {'color': '#c4b5fd', 'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#64748b'},
                    'bar': {'color': '#6366f1'},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'bordercolor': 'rgba(255,255,255,0.1)',
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(34,197,94,0.2)'},
                        {'range': [40, 65], 'color': 'rgba(245,158,11,0.2)'},
                        {'range': [65, 100], 'color': 'rgba(239,68,68,0.2)'},
                    ],
                    'threshold': {'line': {'color': '#f43f5e', 'width': 3}, 'value': 50}
                },
                number={'suffix': '%', 'font': {'color': '#e2e8f0', 'size': 28}}
            ))
            fig_gauge.update_layout(**PLOTLY_TEMPLATE, height=300,
                                     margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
