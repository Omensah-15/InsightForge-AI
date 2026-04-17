import os
import time
import warnings
import traceback
import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import requests

warnings.filterwarnings("ignore")


# ─── Load API key: Streamlit Secrets first, then environment variable ─────────
def get_api_key() -> str:
    try:
        key = st.secrets.get("OPENROUTER_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY", "")


# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightForge AI",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.03em; }
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 2px solid #e2e8f0; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 600;
    padding: 10px 20px; background: transparent; border: none;
    color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em;
}
.stTabs [aria-selected="true"] { color: #0f172a !important; border-bottom: 2px solid #0f172a; }
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 6px; padding: 16px 20px;
}
.metric-card .label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 600;
    color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;
}
.metric-card .value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #0f172a; }
.chat-user {
    background: #0f172a; color: #f8fafc; padding: 12px 16px;
    border-radius: 6px 6px 0 6px; margin: 8px 0; font-size: 0.9rem;
    max-width: 80%; margin-left: auto;
}
.chat-ai {
    background: #f1f5f9; color: #0f172a; padding: 12px 16px;
    border-radius: 6px 6px 6px 0; margin: 8px 0; font-size: 0.9rem;
    max-width: 85%; border-left: 3px solid #0f172a;
}
.insight-box {
    background: #fafafa; border: 1px solid #e2e8f0; border-left: 4px solid #0f172a;
    padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 6px 0; font-size: 0.88rem;
}
.stButton > button {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; border-radius: 4px;
    border: 2px solid #0f172a; background: #0f172a; color: white; padding: 8px 20px;
}
.stButton > button:hover { background: white; color: #0f172a; }
.stSelectbox label, .stSlider label, .stMultiSelect label, .stCheckbox label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; color: #64748b;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
DEFAULTS = {
    "chat_history": [],
    "df": None,
    "df_clustered": None,
    "cluster_labels": None,
    "scaled_df": None,
    "numerical_cols": [],
    "categorical_cols": [],
    "selected_num": [],
    "clustering_done": False,
    "ks": None,
    "inertias": None,
    "sil_scores": None,
    "_file_hash": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_all():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def reset_clustering():
    for k in ["df_clustered", "cluster_labels", "scaled_df",
              "clustering_done", "ks", "inertias", "sil_scores", "chat_history"]:
        st.session_state[k] = DEFAULTS[k]


# ─── Free models — full current list, tried in order ─────────────────────────
FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-4b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "openai/gpt-oss-20b:free",
    "arcee-ai/trinity-large-preview:free",
    "minimax/minimax-m2.5:free",
]


def call_ai(messages: list, system_prompt: str = "") -> tuple:
    """
    Returns (text, is_error).
    Tries each free model in order. Skips 429 rate-limited models.
    Retries once with a short delay before moving on.
    """
    api_key = get_api_key()
    if not api_key:
        return (
            "The AI assistant is not configured yet. "
            "Add your OpenRouter API key in Streamlit Cloud → Manage app → Settings → Secrets.",
            True,
        )

    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://insightforge.app",
        "X-Title": "InsightForge AI",
    }

    rate_limited = 0
    for model in FREE_MODELS:
        for attempt in range(2):  # try each model twice before moving on
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": full_messages,
                        "max_tokens": 1200,
                        "temperature": 0.3,
                    },
                    timeout=45,
                )

                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    if content:
                        return content, False

                elif resp.status_code == 429:
                    rate_limited += 1
                    if attempt == 0:
                        time.sleep(2)  # brief wait then retry same model once
                    break  # move to next model after retry

                elif resp.status_code in (401, 403):
                    return (
                        "API key rejected. Check that your OpenRouter key is correct "
                        "in Streamlit Secrets.",
                        True,
                    )

                else:
                    break  # non-retryable error, try next model

            except requests.exceptions.Timeout:
                break  # move to next model
            except Exception:
                break

    if rate_limited >= len(FREE_MODELS):
        return (
            "All AI models are currently busy due to high demand on free tier. "
            "Please wait 30 seconds and try again.",
            True,
        )

    return (
        "The AI could not generate a response right now. "
        "Please try rephrasing your question or try again in a moment.",
        True,
    )


def get_dataset_context(df: pd.DataFrame, cluster_col: str = None) -> str:
    lines = [f"  - {c} ({df[c].dtype}, {df[c].nunique()} unique)" for c in df.columns]
    ctx = (
        f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns\n"
        f"Columns:\n" + "\n".join(lines) +
        f"\n\nSample rows:\n{df.head(5).to_string(index=False)}\n"
    )
    if cluster_col and cluster_col in df.columns:
        ctx += f"\nCluster sizes: {df.groupby(cluster_col).size().to_dict()}"
    return ctx


# ─── Data processing ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(file_bytes))


def detect_column_types(df: pd.DataFrame):
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numerical, categorical


@st.cache_data(show_spinner=False)
def preprocess(df_json: str, numerical_cols: tuple, categorical_cols: tuple):
    import io
    df = pd.read_json(io.StringIO(df_json))
    df_proc = df.copy()
    for col in numerical_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna(
                df_proc[col].mode()[0] if not df_proc[col].mode().empty else "unknown"
            )
            df_proc[col + "_enc"] = le.fit_transform(df_proc[col].astype(str))
    encoded_cats = [c + "_enc" for c in categorical_cols if c in df_proc.columns]
    feature_cols = list(numerical_cols) + encoded_cats
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_proc[feature_cols])
    scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=df.index)
    return scaled_df, feature_cols


@st.cache_data(show_spinner=False)
def run_kmeans(scaled_json: str, n_clusters: int):
    import io
    scaled_df = pd.read_json(io.StringIO(scaled_json))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_df)
    score = silhouette_score(scaled_df, labels) if n_clusters > 1 else 0.0
    return labels.tolist(), float(score), float(km.inertia_)


@st.cache_data(show_spinner=False)
def run_dbscan(scaled_json: str, eps: float, min_samples: int):
    import io
    scaled_df = pd.read_json(io.StringIO(scaled_json))
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(scaled_df)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    score = silhouette_score(scaled_df, labels) if n_clusters > 1 else 0.0
    return labels.tolist(), n_clusters, float(score)


@st.cache_data(show_spinner=False)
def compute_elbow(scaled_json: str, max_k: int = 10):
    import io
    scaled_df = pd.read_json(io.StringIO(scaled_json))
    ks, inertias, scores = [], [], []
    for k in range(2, min(max_k + 1, len(scaled_df))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(scaled_df)
        ks.append(k)
        inertias.append(float(km.inertia_))
        scores.append(float(silhouette_score(scaled_df, lbl)))
    return ks, inertias, scores


@st.cache_data(show_spinner=False)
def compute_pca(scaled_json: str):
    import io
    scaled_df = pd.read_json(io.StringIO(scaled_json))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled_df)
    return coords.tolist(), pca.explained_variance_ratio_.tolist()


# ─── Visualisation helpers ────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold


def plot_elbow(ks, inertias, scores):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia",
                             line=dict(color="#0f172a", width=2), marker=dict(size=7)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=ks, y=scores, mode="lines+markers", name="Silhouette",
                             line=dict(color="#ef4444", width=2, dash="dot"),
                             marker=dict(size=7, color="#ef4444")),
                  secondary_y=True)
    fig.update_layout(title="Elbow Curve + Silhouette Score",
                      xaxis_title="k", yaxis_title="Inertia",
                      yaxis2_title="Silhouette Score",
                      plot_bgcolor="white", paper_bgcolor="white",
                      font_family="IBM Plex Mono", height=380,
                      legend=dict(x=0.7, y=0.95))
    fig.update_xaxes(gridcolor="#f1f5f9")
    fig.update_yaxes(gridcolor="#f1f5f9")
    return fig


def plot_pca_clusters(coords, labels):
    arr = np.array(coords)
    df_plot = pd.DataFrame({"PC1": arr[:, 0], "PC2": arr[:, 1],
                             "Cluster": [str(l) for l in labels]})
    fig = px.scatter(df_plot, x="PC1", y="PC2", color="Cluster",
                     color_discrete_sequence=PALETTE, title="PCA Cluster Projection")
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
                 color_discrete_sequence=PALETTE, title="Cluster Size Distribution")
    fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                      font_family="IBM Plex Mono", height=320)
    fig.update_xaxes(gridcolor="#f1f5f9")
    fig.update_yaxes(gridcolor="#f1f5f9")
    return fig


def plot_correlation(df: pd.DataFrame, cols: list):
    corr = df[cols].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      font_family="IBM Plex Mono", height=420)
    return fig


def plot_feature_distributions(df: pd.DataFrame, cols: list, cluster_col: str):
    valid = [c for c in cols if c in df.columns][:6]
    if not valid:
        return None
    n = len(valid)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=valid)
    for i, col in enumerate(valid):
        row, c = i // ncols + 1, i % ncols + 1
        for cluster in sorted(df[cluster_col].unique()):
            subset = df[df[cluster_col] == cluster][col].dropna()
            fig.add_trace(go.Histogram(x=subset, name=f"Cluster {cluster}", opacity=0.6,
                                       marker_color=PALETTE[int(cluster) % len(PALETTE)],
                                       showlegend=(i == 0)),
                          row=row, col=c)
    fig.update_layout(barmode="overlay", plot_bgcolor="white", paper_bgcolor="white",
                      font_family="IBM Plex Mono", height=300 * nrows,
                      title="Feature Distributions by Cluster")
    return fig


# ─── Auto insights ────────────────────────────────────────────────────────────
def generate_rule_based_insights(df: pd.DataFrame, cluster_col: str, cols: list) -> list:
    valid = [c for c in cols if c in df.columns]
    if not valid:
        return ["No numerical columns available for insight generation."]
    overall_means = df[valid].mean()
    insights = []
    for cluster in sorted(df[cluster_col].unique()):
        subset = df[df[cluster_col] == cluster]
        size = len(subset)
        pct = round(100 * size / len(df), 1)
        parts = [f"Cluster {cluster} ({size} records, {pct}%)"]
        for col in valid[:5]:
            cm = subset[col].mean()
            om = overall_means[col]
            if om == 0:
                continue
            diff = (cm - om) / abs(om) * 100
            if abs(diff) > 15:
                parts.append(
                    f"{col} is {abs(diff):.0f}% {'above' if diff > 0 else 'below'} "
                    f"average ({cm:.2f} vs {om:.2f})"
                )
        insights.append(" | ".join(parts))
    return insights


# ─── Code sandbox ─────────────────────────────────────────────────────────────
BLOCKED_TOKENS = ["os.", "sys.", "subprocess", "open(", "exec(", "eval(",
                  "__import__", "importlib", "shutil", "socket", "requests"]


def execute_ai_code(code: str, df: pd.DataFrame) -> tuple:
    """Returns (fig_or_None, error_str_or_None)."""
    for fence in ["```python", "```"]:
        code = code.replace(fence, "")
    code = code.strip()
    if not code:
        return None, "AI returned empty code."
    for token in BLOCKED_TOKENS:
        if token in code:
            return None, f"Code blocked: contains '{token}'."
    local_ns = {"df": df.copy(), "pd": pd, "np": np,
                 "px": px, "go": go, "make_subplots": make_subplots}
    try:
        exec(compile(code, "<ai_code>", "exec"), {"__builtins__": {}}, local_ns)
    except Exception:
        return None, f"Execution error:\n{traceback.format_exc(limit=4)}"
    fig = local_ns.get("fig", None)
    if fig is None:
        return None, "Code ran but `fig` was not defined. Ask again with more detail."
    return fig, None


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## InsightForge AI")
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        file_bytes = uploaded.read()
        new_hash = hash(file_bytes)
        if st.session_state._file_hash != new_hash:
            reset_all()
            st.session_state._file_hash = new_hash
            with st.spinner("Loading data..."):
                df_raw = load_csv(file_bytes)
            st.session_state.df = df_raw
            num_cols, cat_cols = detect_column_types(df_raw)
            st.session_state.numerical_cols = num_cols
            st.session_state.categorical_cols = cat_cols
            st.session_state.selected_num = num_cols[:8]

    if st.session_state.df is not None:
        col_a, col_b = st.columns(2)
        if col_a.button("Clear Data", use_container_width=True):
            reset_all()
            st.rerun()
        if col_b.button("Reset Clusters", use_container_width=True):
            reset_clustering()
            st.rerun()

    st.markdown("---")
    st.markdown("**Clustering Model**")
    model_choice = st.selectbox("Algorithm", ["KMeans", "DBSCAN"],
                                label_visibility="collapsed")

    k_val = None
    eps_val, min_s = 0.5, 5
    if model_choice == "KMeans":
        auto_k = st.checkbox("Auto-select k (recommended)", value=True)
        if not auto_k:
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
            default=st.session_state.selected_num,
        )
        st.session_state.selected_num = selected_num
    else:
        selected_num = []

    st.markdown("")
    run_btn = st.button("Run Segmentation", use_container_width=True)


# ─── Main ────────────────────────────────────────────────────────────────────
st.markdown("# InsightForge AI")
st.markdown("Customer segmentation and AI-powered data analysis")
st.markdown("---")

if st.session_state.df is None:
    st.info("Upload a CSV file from the sidebar to get started.")
    st.stop()

df = st.session_state.df
numerical_cols = st.session_state.numerical_cols
categorical_cols = st.session_state.categorical_cols

# ─── Run clustering ───────────────────────────────────────────────────────────
if run_btn:
    if not selected_num:
        st.sidebar.error("Select at least one numerical feature first.")
    else:
        with st.spinner("Preprocessing..."):
            df_json = df.to_json()
            scaled_df, feature_cols = preprocess(
                df_json, tuple(selected_num), tuple(categorical_cols)
            )
        scaled_json = scaled_df.to_json()
        st.session_state.scaled_df = scaled_df

        if model_choice == "KMeans":
            with st.spinner("Computing elbow curve..."):
                ks, inertias, sil_scores = compute_elbow(scaled_json)
            st.session_state.ks = ks
            st.session_state.inertias = inertias
            st.session_state.sil_scores = sil_scores
            best_k = ks[int(np.argmax(sil_scores))] if (auto_k and ks) else (k_val or 3)
            with st.spinner(f"Running KMeans (k={best_k})..."):
                labels, sil, inertia = run_kmeans(scaled_json, best_k)
        else:
            with st.spinner("Running DBSCAN..."):
                labels, n_found, sil = run_dbscan(scaled_json, eps_val, min_s)

        df_clustered = df.copy()
        df_clustered["Cluster"] = labels
        st.session_state.cluster_labels = labels
        st.session_state.df_clustered = df_clustered
        st.session_state.clustering_done = True
        n_found = len(set(labels)) - (1 if -1 in labels else 0)
        st.success(f"Segmentation complete — {n_found} clusters found.")


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Clustering", "Visualizations", "AI Chat"])


# ═══ TAB 1: Overview ═════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, value) in zip([c1, c2, c3, c4], [
        ("Rows", f"{df.shape[0]:,}"),
        ("Columns", str(df.shape[1])),
        ("Numerical", str(len(numerical_cols))),
        ("Categorical", str(len(categorical_cols))),
    ]):
        col.markdown(
            f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Data Preview")
    st.dataframe(df.head(100), use_container_width=True, height=320)

    st.markdown("### Schema")
    st.dataframe(pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Non-null": [int(df[c].notna().sum()) for c in df.columns],
        "Null %": [f"{df[c].isna().mean()*100:.1f}%" for c in df.columns],
        "Unique": [int(df[c].nunique()) for c in df.columns],
    }), use_container_width=True)

    if numerical_cols:
        st.markdown("### Descriptive Statistics")
        st.dataframe(df[numerical_cols].describe().round(3), use_container_width=True)


# ═══ TAB 2: Clustering ════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.clustering_done:
        st.info("Select features in the sidebar and click Run Segmentation.")
    else:
        labels = st.session_state.cluster_labels
        df_clustered = st.session_state.df_clustered
        scaled_df = st.session_state.scaled_df

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        sil_final = silhouette_score(scaled_df, labels) if n_clusters > 1 else 0.0

        c1, c2, c3 = st.columns(3)
        for col, (label, value) in zip([c1, c2, c3], [
            ("Clusters Found", str(n_clusters)),
            ("Silhouette Score", f"{sil_final:.3f}"),
            ("Total Records", f"{len(df_clustered):,}"),
        ]):
            col.markdown(
                f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("### Cluster Sizes")
        st.plotly_chart(plot_cluster_sizes(labels), use_container_width=True)

        valid_num = [c for c in selected_num if c in df_clustered.columns]
        if valid_num:
            st.markdown("### Cluster Profiles (mean values per cluster)")
            profile = df_clustered.groupby("Cluster")[valid_num].mean().round(3)
            st.dataframe(profile, use_container_width=True)

        if model_choice == "KMeans" and st.session_state.ks:
            st.markdown("### Elbow Curve")
            st.plotly_chart(
                plot_elbow(st.session_state.ks, st.session_state.inertias,
                           st.session_state.sil_scores),
                use_container_width=True,
            )

        st.markdown("### Auto-Generated Insights")
        for insight in generate_rule_based_insights(df_clustered, "Cluster", selected_num):
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        st.markdown("### Clustered Data")
        st.dataframe(df_clustered, use_container_width=True, height=350)
        st.download_button(
            "Download Clustered CSV",
            df_clustered.to_csv(index=False).encode(),
            "clustered_data.csv", "text/csv",
        )


# ═══ TAB 3: Visualizations ════════════════════════════════════════════════════
with tab3:
    if not st.session_state.clustering_done:
        st.info("Run segmentation first to unlock visualizations.")
    else:
        df_clustered = st.session_state.df_clustered
        scaled_df = st.session_state.scaled_df
        labels = st.session_state.cluster_labels
        valid_num = [c for c in selected_num if c in df_clustered.columns]

        st.markdown("### PCA Cluster Projection")
        pca_coords, explained = compute_pca(scaled_df.to_json())
        st.plotly_chart(plot_pca_clusters(pca_coords, labels), use_container_width=True)
        st.caption(
            f"PC1 explains {explained[0]*100:.1f}% variance — "
            f"PC2 explains {explained[1]*100:.1f}% variance"
        )

        if valid_num:
            st.markdown("### Feature Distributions by Cluster")
            dist_fig = plot_feature_distributions(df_clustered, valid_num, "Cluster")
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)

            if len(valid_num) > 1:
                st.markdown("### Correlation Heatmap")
                st.plotly_chart(plot_correlation(df_clustered, valid_num),
                                use_container_width=True)

        if len(valid_num) >= 2:
            st.markdown("### Custom Scatter")
            col_a, col_b = st.columns(2)
            x_col = col_a.selectbox("X axis", valid_num, index=0, key="sx")
            y_col = col_b.selectbox("Y axis", valid_num,
                                    index=min(1, len(valid_num) - 1), key="sy")
            fig_sc = px.scatter(df_clustered, x=x_col, y=y_col,
                                color=df_clustered["Cluster"].astype(str),
                                color_discrete_sequence=PALETTE,
                                title=f"{x_col} vs {y_col}")
            fig_sc.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                 font_family="IBM Plex Mono")
            st.plotly_chart(fig_sc, use_container_width=True)


# ═══ TAB 4: AI Chat ══════════════════════════════════════════════════════════
with tab4:
    st.markdown("### AI Chat Assistant")

    if not get_api_key():
        st.error(
            "No API key found. "
            "Go to your Streamlit Cloud app → Settings → Secrets and add:\n\n"
            "```\nOPENROUTER_API_KEY = \"sk-or-v1-your-key-here\"\n```\n\n"
            "Get a free key at openrouter.ai (no credit card required)."
        )
        st.stop()

    working_df = (
        st.session_state.df_clustered
        if st.session_state.df_clustered is not None else df
    )
    cluster_col = "Cluster" if st.session_state.df_clustered is not None else None

    st.markdown(
        "Ask questions about your data or request visualizations.\n\n"
        "Examples: *Which cluster has the highest income?* — "
        "*Show a bar chart of cluster sizes* — "
        "*Plot income vs spending colored by cluster*"
    )
    st.markdown("---")

    if st.session_state.chat_history:
        if st.button("Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

    # Render chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            if msg.get("is_error"):
                st.error(msg["content"])
            elif msg.get("is_code"):
                st.markdown('<div class="chat-ai">Generated and executed code:</div>',
                            unsafe_allow_html=True)
                st.code(msg["content"], language="python")
                if msg.get("fig") is not None:
                    st.plotly_chart(msg["fig"], use_container_width=True)
                if msg.get("exec_error"):
                    st.warning(f"Chart note: {msg['exec_error']}")
            else:
                st.markdown(f'<div class="chat-ai">{msg["content"]}</div>',
                            unsafe_allow_html=True)

    user_input = st.chat_input("Ask about your data or request a chart...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        ctx = get_dataset_context(working_df, cluster_col)

        viz_keywords = ["plot", "chart", "graph", "show", "visualize", "draw",
                        "bar", "scatter", "histogram", "heatmap", "distribution", "figure"]
        wants_viz = any(kw in user_input.lower() for kw in viz_keywords)

        if wants_viz:
            system_prompt = textwrap.dedent(f"""
                You are a Python data analyst. Return ONLY valid Python code — no prose, no markdown, no explanations.
                The dataset is a pandas DataFrame named `df`.
                The final Plotly figure MUST be stored in a variable called `fig`.

                Dataset context:
                {ctx}

                Rules:
                - Return only Python code, nothing else.
                - Only use pre-imported names: df, pd, np, px, go, make_subplots.
                - Do not import anything. Do not use os, sys, open, exec, eval, requests.
                - The cluster column is called 'Cluster' if present.
                - Always assign the chart to `fig`.
                - Use px for simple charts. Use go / make_subplots only if needed.
                - Apply: fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_family='IBM Plex Mono')
            """).strip()
        else:
            system_prompt = textwrap.dedent(f"""
                You are a helpful, concise data analyst assistant.
                Answer the user's question in plain English using actual numbers where relevant.
                Do not generate code unless explicitly asked.

                Dataset context:
                {ctx}
            """).strip()

        with st.spinner("Thinking..."):
            response, is_error = call_ai(
                [{"role": "user", "content": user_input}], system_prompt
            )

        if is_error:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response, "is_error": True}
            )
        elif wants_viz:
            fig_result, exec_error = execute_ai_code(response, working_df)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "is_code": True,
                "fig": fig_result,
                "exec_error": exec_error,
            })
        else:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response, "is_code": False}
            )

        st.rerun()
