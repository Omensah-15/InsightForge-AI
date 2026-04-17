import os
import io
import warnings
import traceback
import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import requests

warnings.filterwarnings("ignore")
load_dotenv()

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightForge AI",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.03em;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #e2e8f0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 10px 20px;
    background: transparent;
    border: none;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.stTabs [aria-selected="true"] {
    color: #0f172a !important;
    border-bottom: 2px solid #0f172a;
}

.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 16px 20px;
}

.metric-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}

.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #0f172a;
}

.cluster-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 3px;
    margin-right: 6px;
}

.chat-user {
    background: #0f172a;
    color: #f8fafc;
    padding: 12px 16px;
    border-radius: 6px 6px 0 6px;
    margin: 8px 0;
    font-size: 0.9rem;
    max-width: 80%;
    margin-left: auto;
}

.chat-ai {
    background: #f1f5f9;
    color: #0f172a;
    padding: 12px 16px;
    border-radius: 6px 6px 6px 0;
    margin: 8px 0;
    font-size: 0.9rem;
    max-width: 85%;
    border-left: 3px solid #0f172a;
}

.insight-box {
    background: #fafafa;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #0f172a;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin: 6px 0;
    font-size: 0.88rem;
}

.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-radius: 4px;
    border: 2px solid #0f172a;
    background: #0f172a;
    color: white;
    padding: 8px 20px;
}

.stButton > button:hover {
    background: white;
    color: #0f172a;
}

.stSelectbox label, .stSlider label, .stMultiSelect label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
}

.code-output {
    background: #0f172a;
    color: #94a3b8;
    padding: 16px;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    white-space: pre-wrap;
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)


# ─── AI Client ───────────────────────────────────────────────────────────────
def call_ai(messages: list, system_prompt: str = "") -> str:
    """Call OpenRouter API (free models available)."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return "AI is not available at the moment. Please try again later."

    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [{"role": "system", "content": system_prompt}] + messages if system_prompt else messages,
        "max_tokens": 1200,
        "temperature": 0.3,
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://insightforge.app",
                "X-Title": "InsightForge AI",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "Request timed out. Check your connection and try again."
    except Exception as e:
        return f"AI call failed: {str(e)}"


def get_dataset_context(df: pd.DataFrame, cluster_col: str = None) -> str:
    """Build a compact dataset context string for the AI."""
    schema_lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        schema_lines.append(f"  - {col} ({dtype}, {n_unique} unique values)")

    sample = df.head(5).to_string(index=False)

    ctx = f"""Dataset: {df.shape[0]} rows x {df.shape[1]} columns
Columns:
{chr(10).join(schema_lines)}

Sample rows:
{sample}
"""
    if cluster_col and cluster_col in df.columns:
        cluster_summary = df.groupby(cluster_col).size().to_dict()
        ctx += f"\nCluster sizes: {cluster_summary}"

    return ctx


# ─── Data Processing ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_profile(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def detect_column_types(df: pd.DataFrame):
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    text = [c for c in categorical if df[c].str.len().mean() > 30 if df[c].dtype == object]
    categorical = [c for c in categorical if c not in text]
    return numerical, categorical, text


@st.cache_data
def preprocess(df: pd.DataFrame, numerical_cols: list, categorical_cols: list):
    df_proc = df.copy()

    # Fill missing values
    for col in numerical_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())

    for col in categorical_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna(df_proc[col].mode()[0] if not df_proc[col].mode().empty else "unknown")

    # Encode categoricals
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df_proc.columns:
            df_proc[col + "_enc"] = le.fit_transform(df_proc[col].astype(str))

    encoded_cats = [c + "_enc" for c in categorical_cols if c in df_proc.columns]
    feature_cols = numerical_cols + encoded_cats

    feature_df = df_proc[feature_cols].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)
    scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=df.index)

    return scaled_df, feature_cols


@st.cache_data
def run_kmeans(scaled_df: pd.DataFrame, n_clusters: int):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_df)
    score = silhouette_score(scaled_df, labels) if n_clusters > 1 else 0.0
    return labels, score, km.inertia_


@st.cache_data
def run_dbscan(scaled_df: pd.DataFrame, eps: float, min_samples: int):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(scaled_df)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    score = silhouette_score(scaled_df, labels) if n_clusters > 1 else 0.0
    return labels, n_clusters, score


@st.cache_data
def compute_elbow(scaled_df: pd.DataFrame, max_k: int = 10):
    inertias = []
    scores = []
    ks = list(range(2, min(max_k + 1, len(scaled_df))))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(scaled_df)
        inertias.append(km.inertia_)
        scores.append(silhouette_score(scaled_df, lbl))
    return ks, inertias, scores


@st.cache_data
def compute_pca(scaled_df: pd.DataFrame):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled_df)
    return coords, pca.explained_variance_ratio_


# ─── Visualization helpers ───────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold


def plot_elbow(ks, inertias, scores):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia",
                             line=dict(color="#0f172a", width=2),
                             marker=dict(size=7, color="#0f172a")), secondary_y=False)
    fig.add_trace(go.Scatter(x=ks, y=scores, mode="lines+markers", name="Silhouette",
                             line=dict(color="#ef4444", width=2, dash="dot"),
                             marker=dict(size=7, color="#ef4444")), secondary_y=True)
    fig.update_layout(
        title="Elbow Curve + Silhouette Score",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia",
        yaxis2_title="Silhouette Score",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="IBM Plex Mono",
        height=380,
        legend=dict(x=0.7, y=0.95),
    )
    fig.update_xaxes(gridcolor="#f1f5f9")
    fig.update_yaxes(gridcolor="#f1f5f9")
    return fig


def plot_pca_clusters(coords, labels, hover_data=None):
    df_plot = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                             "Cluster": [str(l) for l in labels]})
    fig = px.scatter(df_plot, x="PC1", y="PC2", color="Cluster",
                     color_discrete_sequence=PALETTE,
                     title="PCA Cluster Projection",
                     labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"})
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      font_family="IBM Plex Mono", height=420)
    fig.update_xaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
    return fig


def plot_cluster_sizes(labels):
    counts = pd.Series(labels).value_counts().sort_index().reset_index()
    counts.columns = ["Cluster", "Count"]
    counts["Cluster"] = counts["Cluster"].astype(str)
    fig = px.bar(counts, x="Cluster", y="Count", color="Cluster",
                 color_discrete_sequence=PALETTE,
                 title="Cluster Size Distribution")
    fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                      font_family="IBM Plex Mono", height=320)
    fig.update_xaxes(gridcolor="#f1f5f9")
    fig.update_yaxes(gridcolor="#f1f5f9")
    return fig


def plot_correlation(df: pd.DataFrame, numerical_cols: list):
    corr = df[numerical_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Feature Correlation Heatmap")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      font_family="IBM Plex Mono", height=420)
    return fig


def plot_feature_distributions(df: pd.DataFrame, numerical_cols: list, cluster_col: str):
    cols_to_plot = numerical_cols[:6]
    if not cols_to_plot:
        return None
    n = len(cols_to_plot)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=cols_to_plot)
    for i, col in enumerate(cols_to_plot):
        row = i // ncols + 1
        c = i % ncols + 1
        for cluster in sorted(df[cluster_col].unique()):
            subset = df[df[cluster_col] == cluster][col].dropna()
            fig.add_trace(go.Histogram(x=subset, name=f"Cluster {cluster}",
                                       opacity=0.6,
                                       marker_color=PALETTE[int(cluster) % len(PALETTE)],
                                       showlegend=(i == 0)),
                          row=row, col=c)
    fig.update_layout(barmode="overlay", plot_bgcolor="white", paper_bgcolor="white",
                      font_family="IBM Plex Mono", height=300 * nrows,
                      title="Feature Distributions by Cluster")
    return fig


# ─── Auto insights ───────────────────────────────────────────────────────────
def generate_rule_based_insights(df: pd.DataFrame, cluster_col: str, numerical_cols: list) -> list:
    insights = []
    if not numerical_cols:
        return ["No numerical columns found for insight generation."]

    overall_means = df[numerical_cols].mean()
    for cluster in sorted(df[cluster_col].unique()):
        subset = df[df[cluster_col] == cluster]
        size = len(subset)
        pct = round(100 * size / len(df), 1)
        parts = [f"Cluster {cluster} ({size} records, {pct}%):"]
        for col in numerical_cols[:5]:
            cluster_mean = subset[col].mean()
            overall_mean = overall_means[col]
            if overall_mean == 0:
                continue
            diff_pct = (cluster_mean - overall_mean) / abs(overall_mean) * 100
            if abs(diff_pct) > 15:
                direction = "above" if diff_pct > 0 else "below"
                parts.append(f"{col} is {abs(diff_pct):.0f}% {direction} average ({cluster_mean:.2f} vs {overall_mean:.2f})")
        insights.append(" | ".join(parts))

    return insights


# ─── Code execution sandbox ──────────────────────────────────────────────────
ALLOWED_IMPORTS = {"pandas", "pd", "numpy", "np", "plotly", "px", "go",
                   "plotly.express", "plotly.graph_objects"}

BLOCKED_TOKENS = ["os.", "sys.", "subprocess", "open(", "exec(", "eval(",
                  "__import__", "importlib", "shutil", "socket", "requests"]


def sanitize_code(code: str) -> tuple[bool, str]:
    for token in BLOCKED_TOKENS:
        if token in code:
            return False, f"Blocked token detected: {token}"
    return True, ""


def execute_ai_code(code: str, df: pd.DataFrame):
    safe, reason = sanitize_code(code)
    if not safe:
        return None, f"Code blocked: {reason}"

    # Strip markdown fences if present
    code = code.replace("```python", "").replace("```", "").strip()

    local_ns = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "make_subplots": make_subplots,
    }

    try:
        exec(compile(code, "<ai_code>", "exec"), {"__builtins__": {}}, local_ns)
    except Exception as e:
        return None, f"Execution error: {traceback.format_exc(limit=3)}"

    fig = local_ns.get("fig", None)
    output = local_ns.get("output", None)
    return fig, None


# ─── Session state init ──────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clustered" not in st.session_state:
    st.session_state.df_clustered = None
if "cluster_labels" not in st.session_state:
    st.session_state.cluster_labels = None
if "scaled_df" not in st.session_state:
    st.session_state.scaled_df = None
if "numerical_cols" not in st.session_state:
    st.session_state.numerical_cols = []
if "categorical_cols" not in st.session_state:
    st.session_state.categorical_cols = []


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## InsightForge AI")
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_raw = load_and_profile(uploaded)
        st.session_state.df = df_raw
        numerical_cols, categorical_cols, text_cols = detect_column_types(df_raw)
        st.session_state.numerical_cols = numerical_cols
        st.session_state.categorical_cols = categorical_cols

    st.markdown("---")
    st.markdown("**Clustering Model**")
    model_choice = st.selectbox("Algorithm", ["KMeans", "DBSCAN"], label_visibility="collapsed")

    if model_choice == "KMeans":
        auto_k = st.checkbox("Auto-select k", value=True)
        if auto_k:
            k_val = None
        else:
            k_val = st.slider("Number of clusters", 2, 15, 4)
    else:
        eps_val = st.slider("DBSCAN eps", 0.1, 5.0, 0.5, 0.1)
        min_s = st.slider("Min samples", 2, 20, 5)

    st.markdown("---")
    st.markdown("**Feature Selection**")
    if st.session_state.numerical_cols:
        selected_num = st.multiselect(
            "Numerical features",
            options=st.session_state.numerical_cols,
            default=st.session_state.numerical_cols[:8],
        )
    else:
        selected_num = []

    run_btn = st.button("Run Segmentation", use_container_width=True)


# ─── Main area ───────────────────────────────────────────────────────────────
st.markdown("# InsightForge AI")
st.markdown("Customer segmentation and AI-powered analysis")
st.markdown("---")

if st.session_state.df is None:
    st.info("Upload a CSV file in the sidebar to get started.")
    st.stop()

df = st.session_state.df
numerical_cols = st.session_state.numerical_cols
categorical_cols = st.session_state.categorical_cols

# ─── Run clustering ───────────────────────────────────────────────────────────
if run_btn and selected_num:
    with st.spinner("Preprocessing data..."):
        scaled_df, feature_cols = preprocess(df, selected_num, categorical_cols)
        st.session_state.scaled_df = scaled_df

    if model_choice == "KMeans":
        with st.spinner("Computing elbow curve..."):
            ks, inertias, sil_scores = compute_elbow(scaled_df)
            best_k = ks[np.argmax(sil_scores)] if auto_k else k_val

        with st.spinner(f"Running KMeans (k={best_k})..."):
            labels, sil, inertia = run_kmeans(scaled_df, best_k)

    else:
        with st.spinner("Running DBSCAN..."):
            labels, n_clusters_found, sil = run_dbscan(scaled_df, eps_val, min_s)

    st.session_state.cluster_labels = labels
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels
    st.session_state.df_clustered = df_clustered
    st.success(f"Segmentation complete. {len(set(labels)) - (1 if -1 in labels else 0)} clusters found.")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Clustering", "Visualizations", "AI Chat"])


# ─── TAB 1: Overview ─────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Rows", f"{df.shape[0]:,}"),
        ("Columns", str(df.shape[1])),
        ("Numerical", str(len(numerical_cols))),
        ("Categorical", str(len(categorical_cols))),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Data Preview")
    st.dataframe(df.head(50), use_container_width=True, height=300)

    st.markdown("### Schema")
    schema_df = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Non-null": [df[c].notna().sum() for c in df.columns],
        "Null %": [f"{df[c].isna().mean()*100:.1f}%" for c in df.columns],
        "Unique": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(schema_df, use_container_width=True)

    if numerical_cols:
        st.markdown("### Descriptive Statistics")
        st.dataframe(df[numerical_cols].describe().round(3), use_container_width=True)


# ─── TAB 2: Clustering ───────────────────────────────────────────────────────
with tab2:
    if st.session_state.cluster_labels is None:
        st.info("Run segmentation from the sidebar to see results here.")
    else:
        labels = st.session_state.cluster_labels
        df_clustered = st.session_state.df_clustered
        scaled_df = st.session_state.scaled_df

        unique_clusters = sorted(set(labels))
        n_clusters = len([c for c in unique_clusters if c != -1])
        sil_final = silhouette_score(scaled_df, labels) if n_clusters > 1 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""<div class="metric-card"><div class="label">Clusters Found</div><div class="value">{n_clusters}</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card"><div class="label">Silhouette Score</div><div class="value">{sil_final:.3f}</div></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card"><div class="label">Total Records</div><div class="value">{len(df_clustered):,}</div></div>""", unsafe_allow_html=True)

        st.markdown("### Cluster Profiles")
        if selected_num:
            profile = df_clustered.groupby("Cluster")[selected_num].mean().round(3)
            st.dataframe(profile, use_container_width=True)

        st.markdown("### Cluster Sizes")
        st.plotly_chart(plot_cluster_sizes(labels), use_container_width=True)

        if model_choice == "KMeans" and 'ks' in dir():
            st.markdown("### Elbow Curve")
            st.plotly_chart(plot_elbow(ks, inertias, sil_scores), use_container_width=True)

        st.markdown("### Auto-Generated Insights")
        if selected_num:
            insights = generate_rule_based_insights(df_clustered, "Cluster", selected_num)
            for insight in insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        st.markdown("### Clustered Data")
        st.dataframe(df_clustered, use_container_width=True, height=350)

        csv_out = df_clustered.to_csv(index=False).encode()
        st.download_button("Download Clustered CSV", csv_out, "clustered_data.csv", "text/csv")


# ─── TAB 3: Visualizations ──────────────────────────────────────────────────
with tab3:
    if st.session_state.cluster_labels is None:
        st.info("Run segmentation first to unlock visualizations.")
    else:
        df_clustered = st.session_state.df_clustered
        scaled_df = st.session_state.scaled_df
        labels = st.session_state.cluster_labels

        st.markdown("### PCA Cluster Projection")
        pca_coords, explained = compute_pca(scaled_df)
        st.plotly_chart(plot_pca_clusters(pca_coords, labels), use_container_width=True)
        st.caption(f"PC1 explains {explained[0]*100:.1f}% variance, PC2 explains {explained[1]*100:.1f}% variance")

        if numerical_cols:
            st.markdown("### Feature Distributions by Cluster")
            dist_fig = plot_feature_distributions(df_clustered, selected_num[:6], "Cluster")
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)

            st.markdown("### Correlation Heatmap")
            if len(selected_num) > 1:
                st.plotly_chart(plot_correlation(df, selected_num), use_container_width=True)

        if len(selected_num) >= 2:
            st.markdown("### Custom Scatter")
            col_a, col_b = st.columns(2)
            x_col = col_a.selectbox("X axis", selected_num, index=0)
            y_col = col_b.selectbox("Y axis", selected_num, index=min(1, len(selected_num)-1))
            fig_scatter = px.scatter(df_clustered, x=x_col, y=y_col,
                                     color=df_clustered["Cluster"].astype(str),
                                     color_discrete_sequence=PALETTE,
                                     title=f"{x_col} vs {y_col}")
            fig_scatter.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                      font_family="IBM Plex Mono")
            st.plotly_chart(fig_scatter, use_container_width=True)


# ─── TAB 4: AI Chat ──────────────────────────────────────────────────────────
with tab4:
    st.markdown("### AI Chat Assistant")
    st.markdown("Ask questions about your data or request visualizations. Examples:")
    st.markdown("- *Which cluster has the highest average income?*")
    st.markdown("- *Show a bar chart of cluster size*")
    st.markdown("- *Plot income vs spending score colored by cluster*")
    st.markdown("---")

    # Render history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            content = msg["content"]
            if msg.get("is_code"):
                st.markdown('<div class="chat-ai">Generated code:</div>', unsafe_allow_html=True)
                st.code(content, language="python")
                if msg.get("fig"):
                    st.plotly_chart(msg["fig"], use_container_width=True)
                if msg.get("error"):
                    st.error(msg["error"])
            else:
                st.markdown(f'<div class="chat-ai">{content}</div>', unsafe_allow_html=True)

    # Input area
    user_input = st.chat_input("Ask about your data or request a chart...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        working_df = st.session_state.df_clustered if st.session_state.df_clustered is not None else df
        ctx = get_dataset_context(working_df, "Cluster" if st.session_state.df_clustered is not None else None)

        viz_keywords = ["plot", "chart", "graph", "show me", "visualize", "draw", "bar", "scatter", "histogram", "heatmap"]
        wants_viz = any(kw in user_input.lower() for kw in viz_keywords)

        if wants_viz:
            system_prompt = textwrap.dedent(f"""
                You are a data analyst writing Python code for data visualization.
                The dataset is available as a pandas DataFrame called `df`.
                Always store the final Plotly figure in a variable called `fig`.

                Dataset context:
                {ctx}

                Rules:
                - Return ONLY valid Python code. No explanations, no markdown, no comments unless inline.
                - Use only: pandas (pd), numpy (np), plotly.express (px), plotly.graph_objects (go).
                - Do not import anything. All libraries are pre-imported.
                - The cluster column is called 'Cluster' if it exists.
                - The final figure must be stored in `fig`.
            """).strip()
        else:
            system_prompt = textwrap.dedent(f"""
                You are a helpful data analyst assistant.
                Answer the user's question about the dataset in plain English.
                Be concise and specific. Use numbers from the dataset when relevant.

                Dataset context:
                {ctx}
            """).strip()

        messages = [{"role": "user", "content": user_input}]

        with st.spinner("Thinking..."):
            response = call_ai(messages, system_prompt)

        if wants_viz:
            fig_result, exec_error = execute_ai_code(response, working_df)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "is_code": True,
                "fig": fig_result,
                "error": exec_error,
            })
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "is_code": False,
            })

        st.rerun()
