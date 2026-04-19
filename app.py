import os
import re
import warnings
import traceback
import textwrap
import io

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

# ─── API key ──────────────────────────────────────────────────────────────────
def get_api_key() -> str:
    try:
        key = st.secrets.get("OPENROUTER_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY", "")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightForge AI",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
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
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 16px 20px;
}
.metric-card .label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 600;
    color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #0f172a;
}
.chat-user {
    background: #0f172a; color: #f8fafc; padding: 12px 16px;
    border-radius: 6px 6px 0 6px; margin: 8px 0 8px auto;
    font-size: 0.9rem; max-width: 80%;
}
.chat-ai {
    background: #f1f5f9; color: #0f172a; padding: 12px 16px;
    border-radius: 6px 6px 6px 0; margin: 8px 0; font-size: 0.9rem;
    max-width: 85%; border-left: 3px solid #0f172a; line-height: 1.6;
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

# ─── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "chat_history": [],
    "df": None,
    "df_edited": None,
    "df_clustered": None,
    "cluster_labels": None,
    "scaled_df": None,
    "numerical_cols": [],
    "categorical_cols": [],
    "text_cols": [],
    "selected_num": [],
    "clustering_done": False,
    "ks": None,
    "inertias": None,
    "sil_scores": None,
    "auto_visuals": [],
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


def apply_edits_to_pipeline():
    """Recompute column types and auto-visuals after data edits."""
    active_df = st.session_state.df_edited if st.session_state.df_edited is not None else st.session_state.df
    if active_df is None:
        return
    num_cols, cat_cols, txt_cols = detect_column_types(active_df)
    st.session_state.numerical_cols = num_cols
    st.session_state.categorical_cols = cat_cols
    st.session_state.text_cols = txt_cols
    st.session_state.selected_num = num_cols[:8]
    st.session_state.auto_visuals = generate_auto_visuals(active_df, num_cols, cat_cols)
    reset_clustering()


# ─── AI models ────────────────────────────────────────────────────────────────
FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-3-4b-it:free",
    "arcee-ai/trinity-large-preview:free",
]


def call_ai(messages: list, system_prompt: str = "") -> tuple:
    """Returns (text, is_error). Tries all free models, skips 429s."""
    api_key = get_api_key()
    if not api_key:
        return (
            "AI is not configured. Add your OpenRouter API key in "
            "Streamlit Cloud → Manage app → Settings → Secrets.",
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
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={"model": model, "messages": full_messages,
                      "max_tokens": 1400, "temperature": 0.2},
                timeout=50,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content:
                    return content, False
            elif resp.status_code == 429:
                rate_limited += 1
                continue
            elif resp.status_code in (401, 403):
                return "API key is invalid. Please check your OpenRouter key in Streamlit Secrets.", True
        except requests.exceptions.Timeout:
            continue
        except Exception:
            continue

    if rate_limited >= len(FREE_MODELS):
        return "All AI models are busy right now. Please wait a moment and try again.", True
    return "Could not get a response. Please try rephrasing your question.", True


# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Strip column name whitespace
    df.columns = [c.strip() for c in df.columns]
    # Strip string values
    for col in df.select_dtypes(include=["object","string"]).columns:
        df[col] = df[col].astype(str).str.strip()
    # Parse date columns
    for col in df.columns:
        if df[col].dtype == object and any(
            kw in col.lower() for kw in ["date", "time", "datetime", "timestamp"]
        ):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


def detect_column_types(df: pd.DataFrame):
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if df[c].nunique() <= 50
    ]
    text = [
        c for c in df.select_dtypes(include=["object"]).columns
        if df[c].nunique() > 50 or (df[c].str.len().mean() > 20)
    ]
    # Remove overlap
    text = [c for c in text if c not in categorical]
    return numerical, categorical, text


def get_dataset_context(df: pd.DataFrame, cluster_col: str = None) -> str:
    """Build rich context string for the AI."""
    cols_info = []
    for c in df.columns:
        dtype = str(df[c].dtype)
        n_unique = df[c].nunique()
        if df[c].dtype == object and n_unique <= 20:
            sample_vals = df[c].dropna().unique()[:6].tolist()
            cols_info.append(f"  - '{c}' (text, {n_unique} unique values: {sample_vals})")
        elif pd.api.types.is_numeric_dtype(df[c]):
            cols_info.append(
                f"  - '{c}' (numeric, min={df[c].min():.2f}, max={df[c].max():.2f}, mean={df[c].mean():.2f})"
            )
        else:
            cols_info.append(f"  - '{c}' ({dtype}, {n_unique} unique)")

    sample_rows = df.head(3).to_string(index=False)
    ctx = (
        f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns\n"
        f"Columns (use EXACT names in code):\n" + "\n".join(cols_info) +
        f"\n\nSample rows:\n{sample_rows}"
    )
    if cluster_col and cluster_col in df.columns:
        sizes = df.groupby(cluster_col).size().to_dict()
        ctx += f"\n\nCluster sizes: {sizes}"
    return ctx


# ─── Auto visuals engine ──────────────────────────────────────────────────────
def generate_auto_visuals(df: pd.DataFrame, numerical_cols: list,
                          categorical_cols: list) -> list:
    """
    Automatically generate meaningful charts based on what columns exist.
    Returns a list of plotly figures with titles.
    """
    visuals = []
    PALETTE = px.colors.qualitative.Bold
    layout = dict(plot_bgcolor="white", paper_bgcolor="white",
                  font_family="IBM Plex Mono", height=380)

    # 1. Categorical distribution — pick most interesting cat col
    for col in categorical_cols[:2]:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "Count"]
        fig = px.bar(counts.head(15), x=col, y="Count", color=col,
                     color_discrete_sequence=PALETTE,
                     title=f"Distribution of {col}")
        fig.update_layout(**layout, showlegend=False)
        fig.update_xaxes(gridcolor="#f1f5f9")
        fig.update_yaxes(gridcolor="#f1f5f9")
        visuals.append((f"Distribution of {col}", fig))

    # 2. Numerical distributions
    for col in numerical_cols[:3]:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}",
                           color_discrete_sequence=[PALETTE[2]])
        fig.update_layout(**layout)
        fig.update_xaxes(gridcolor="#f1f5f9")
        fig.update_yaxes(gridcolor="#f1f5f9")
        visuals.append((f"Distribution of {col}", fig))

    # 3. Categorical vs numerical — e.g. Sentiment vs Confidence Score
    if categorical_cols and numerical_cols:
        cat_col = categorical_cols[0]
        num_col = numerical_cols[0]
        fig = px.box(df, x=cat_col, y=num_col, color=cat_col,
                     color_discrete_sequence=PALETTE,
                     title=f"{num_col} by {cat_col}")
        fig.update_layout(**layout, showlegend=False)
        fig.update_xaxes(gridcolor="#f1f5f9")
        fig.update_yaxes(gridcolor="#f1f5f9")
        visuals.append((f"{num_col} by {cat_col}", fig))

    # 4. Second categorical vs numerical
    if len(categorical_cols) > 1 and numerical_cols:
        cat_col = categorical_cols[1]
        num_col = numerical_cols[0]
        counts = df.groupby(cat_col)[num_col].mean().reset_index()
        fig = px.bar(counts, x=cat_col, y=num_col, color=cat_col,
                     color_discrete_sequence=PALETTE,
                     title=f"Average {num_col} by {cat_col}")
        fig.update_layout(**layout, showlegend=False)
        fig.update_xaxes(gridcolor="#f1f5f9")
        fig.update_yaxes(gridcolor="#f1f5f9")
        visuals.append((f"Average {num_col} by {cat_col}", fig))

    # 5. Scatter of two numericals
    if len(numerical_cols) >= 2:
        fig = px.scatter(df, x=numerical_cols[0], y=numerical_cols[1],
                         color=categorical_cols[0] if categorical_cols else None,
                         color_discrete_sequence=PALETTE,
                         title=f"{numerical_cols[0]} vs {numerical_cols[1]}")
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.update_layout(**layout)
        fig.update_xaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
        fig.update_yaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
        visuals.append((f"{numerical_cols[0]} vs {numerical_cols[1]}", fig))

    # 6. Correlation heatmap if enough numericals
    if len(numerical_cols) >= 3:
        corr = df[numerical_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="Feature Correlation Heatmap")
        fig.update_layout(**layout)
        visuals.append(("Correlation Heatmap", fig))

    return visuals


# ─── Preprocessing & clustering ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def preprocess(df_json: str, numerical_cols: tuple, categorical_cols: tuple):
    df = pd.read_json(io.StringIO(df_json))
    df_proc = df.copy()
    for col in numerical_cols:
        if col in df_proc.columns:
            df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")
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
    if not feature_cols:
        return pd.DataFrame(), []
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_proc[feature_cols])
    scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=df.index)
    return scaled_df, feature_cols


@st.cache_data(show_spinner=False)
def run_kmeans(scaled_json: str, n_clusters: int):
    scaled_df = pd.read_json(io.StringIO(scaled_json))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_df)
    score = silhouette_score(scaled_df, labels) if n_clusters > 1 else 0.0
    return labels.tolist(), float(score), float(km.inertia_)


@st.cache_data(show_spinner=False)
def run_dbscan(scaled_json: str, eps: float, min_samples: int):
    scaled_df = pd.read_json(io.StringIO(scaled_json))
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(scaled_df)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    score = silhouette_score(scaled_df, labels) if n_clusters > 1 else 0.0
    return labels.tolist(), n_clusters, float(score)


@st.cache_data(show_spinner=False)
def compute_elbow(scaled_json: str, max_k: int = 10):
    scaled_df = pd.read_json(io.StringIO(scaled_json))
    ks, inertias, scores = [], [], []
    limit = min(max_k + 1, len(scaled_df))
    for k in range(2, limit):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(scaled_df)
        ks.append(k)
        inertias.append(float(km.inertia_))
        scores.append(float(silhouette_score(scaled_df, lbl)))
    return ks, inertias, scores


@st.cache_data(show_spinner=False)
def compute_pca(scaled_json: str):
    scaled_df = pd.read_json(io.StringIO(scaled_json))
    n = min(2, scaled_df.shape[1])
    pca = PCA(n_components=n, random_state=42)
    coords = pca.fit_transform(scaled_df)
    return coords.tolist(), pca.explained_variance_ratio_.tolist()


# ─── Visualization helpers ────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold
BASE_LAYOUT = dict(plot_bgcolor="white", paper_bgcolor="white", font_family="IBM Plex Mono")


def apply_base(fig, height=380):
    fig.update_layout(**BASE_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
    return fig


def plot_elbow(ks, inertias, scores):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia",
                             line=dict(color="#0f172a", width=2), marker=dict(size=7)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=ks, y=scores, mode="lines+markers", name="Silhouette",
                             line=dict(color="#ef4444", width=2, dash="dot"),
                             marker=dict(size=7, color="#ef4444")),
                  secondary_y=True)
    fig.update_layout(title="Elbow Curve + Silhouette Score", xaxis_title="k",
                      yaxis_title="Inertia", yaxis2_title="Silhouette Score",
                      legend=dict(x=0.7, y=0.95), **BASE_LAYOUT, height=380)
    return fig


def plot_pca_clusters(coords, labels):
    arr = np.array(coords)
    df_plot = pd.DataFrame({"PC1": arr[:, 0], "PC2": arr[:, 1],
                             "Cluster": [str(l) for l in labels]})
    fig = px.scatter(df_plot, x="PC1", y="PC2", color="Cluster",
                     color_discrete_sequence=PALETTE, title="PCA Cluster Projection")
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    return apply_base(fig, 420)


def plot_cluster_sizes(labels):
    counts = pd.Series(labels).value_counts().sort_index().reset_index()
    counts.columns = ["Cluster", "Count"]
    counts["Cluster"] = counts["Cluster"].astype(str)
    fig = px.bar(counts, x="Cluster", y="Count", color="Cluster",
                 color_discrete_sequence=PALETTE, title="Cluster Size Distribution")
    fig.update_layout(showlegend=False)
    return apply_base(fig, 320)


def plot_correlation(df, cols):
    corr = df[cols].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
    return apply_base(fig, 420)


def plot_feature_distributions(df, cols, cluster_col):
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
    fig.update_layout(barmode="overlay", title="Feature Distributions by Cluster",
                      **BASE_LAYOUT, height=300 * nrows)
    return fig


def generate_rule_based_insights(df, cluster_col, cols):
    valid = [c for c in cols if c in df.columns]
    if not valid:
        return []
    overall_means = df[valid].mean()
    insights = []
    for cluster in sorted(df[cluster_col].unique()):
        subset = df[df[cluster_col] == cluster]
        size = len(subset)
        pct = round(100 * size / len(df), 1)
        parts = [f"Cluster {cluster} ({size} records, {pct}%)"]
        for col in valid[:5]:
            cm, om = subset[col].mean(), overall_means[col]
            if om == 0:
                continue
            diff = (cm - om) / abs(om) * 100
            if abs(diff) > 15:
                direction = "above" if diff > 0 else "below"
                parts.append(f"{col} is {abs(diff):.0f}% {direction} average ({cm:.2f} vs {om:.2f})")
        insights.append(" | ".join(parts))
    return insights


# ─── Code sandbox ─────────────────────────────────────────────────────────────
BLOCKED = ["os.", "sys.", "subprocess", "open(", "__import__",
           "importlib", "shutil", "socket", "requests", "eval(", "exec("]


def clean_ai_code(code: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    code = re.sub(r"```(?:python)?", "", code)
    code = code.replace("```", "").strip()
    return code


def execute_ai_code(code: str, df: pd.DataFrame) -> tuple:
    """Execute AI-generated code safely. Returns (fig, error_message)."""
    code = clean_ai_code(code)
    if not code:
        return None, "No executable code was returned."

    for token in BLOCKED:
        if token in code:
            return None, f"Code was blocked for safety reasons."

    # Always use clean column names
    df_exec = df.copy()
    df_exec.columns = [c.strip() for c in df_exec.columns]

    local_ns = {
        "df": df_exec, "pd": pd, "np": np,
        "px": px, "go": go, "make_subplots": make_subplots,
    }

    try:
        exec(compile(code, "<ai_code>", "exec"), {"__builtins__": {}}, local_ns)
    except ValueError as e:
        err = str(e)
        if "not the name of a column" in err or "Expected one of" in err:
            cols = df_exec.columns.tolist()
            return None, (
                f"The chart could not be generated because a column name did not match. "
                f"Your columns are: {cols}. Please try asking again."
            )
        return None, f"Chart error: {err}"
    except KeyError as e:
        cols = df_exec.columns.tolist()
        return None, f"Column {e} not found. Your columns are: {cols}."
    except Exception as e:
        return None, f"Chart could not be generated: {str(e)}"

    fig = local_ns.get("fig", None)
    if fig is None:
        return None, "The AI did not produce a chart. Try rephrasing your request."
    return fig, None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
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
            with st.spinner("Loading..."):
                df_raw = load_csv(file_bytes)
            num_cols, cat_cols, txt_cols = detect_column_types(df_raw)
            st.session_state.df = df_raw
            st.session_state.numerical_cols = num_cols
            st.session_state.categorical_cols = cat_cols
            st.session_state.text_cols = txt_cols
            st.session_state.selected_num = num_cols[:8]
            # Generate auto visuals immediately on load
            st.session_state.auto_visuals = generate_auto_visuals(df_raw, num_cols, cat_cols)

        # Recovery: re-detect if lost
        if st.session_state.df is not None and not st.session_state.numerical_cols and not st.session_state.categorical_cols:
            num_cols, cat_cols, txt_cols = detect_column_types(st.session_state.df)
            st.session_state.numerical_cols = num_cols
            st.session_state.categorical_cols = cat_cols
            st.session_state.text_cols = txt_cols
            st.session_state.selected_num = num_cols[:8]
            if not st.session_state.auto_visuals:
                st.session_state.auto_visuals = generate_auto_visuals(
                    st.session_state.df, num_cols, cat_cols)

    if st.session_state.df is not None:
        ca, cb = st.columns(2)
        if ca.button("Clear Data", use_container_width=True):
            reset_all()
            st.rerun()
        if cb.button("Reset Segments", use_container_width=True):
            reset_clustering()
            st.rerun()

    st.markdown("---")
    st.markdown("**Segmentation Settings**")
    model_choice = st.selectbox(
        "Grouping method",
        ["Auto (recommended)", "Density-based"],
        label_visibility="collapsed",
        help="Auto groups records by similarity. Density-based finds natural clusters of any shape.",
    )

    k_val, eps_val, min_s = None, 0.5, 5
    if model_choice == "Auto (recommended)":
        auto_k = st.checkbox("Find number of groups automatically", value=True)
        if not auto_k:
            k_val = st.slider("Number of groups", 2, 15, 4)
    else:
        eps_val = st.slider("Sensitivity", 0.1, 5.0, 0.5, 0.1,
                            help="How close records need to be to form a group.")
        min_s = st.slider("Minimum group size", 2, 20, 5)

    st.markdown("---")
    st.markdown("**Fields to Analyse**")
    st.caption("Select the numerical fields to base segmentation on.")
    num_options = st.session_state.numerical_cols
    if num_options:
        valid_defaults = [c for c in st.session_state.selected_num if c in num_options]
        if not valid_defaults:
            valid_defaults = num_options[:8]
        selected_num = st.multiselect(
            "Fields",
            options=num_options,
            default=valid_defaults,
            label_visibility="collapsed",
        )
        st.session_state.selected_num = selected_num
    else:
        selected_num = []
        if st.session_state.df is not None:
            st.caption("No numerical fields detected. Segmentation requires numerical data.")

    st.markdown("")
    run_btn = st.button("Run Segmentation", use_container_width=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("# InsightForge AI")
st.markdown("Customer segmentation and data analysis")
st.markdown("---")

if st.session_state.df is None:
    st.info("Upload a CSV file from the sidebar to get started.")
    st.stop()

df = st.session_state.df_edited if st.session_state.df_edited is not None else st.session_state.df
numerical_cols = st.session_state.numerical_cols
categorical_cols = st.session_state.categorical_cols

# ─── Run segmentation ─────────────────────────────────────────────────────────
if run_btn:
    if not selected_num:
        st.sidebar.error("Select at least one numerical feature.")
    else:
        with st.spinner("Preprocessing data..."):
            df_json = df.to_json()
            scaled_df, feature_cols = preprocess(
                df_json, tuple(selected_num), tuple(categorical_cols)
            )

        if scaled_df.empty:
            st.error("Could not build a feature matrix. Make sure you have valid numerical columns.")
        else:
            scaled_json = scaled_df.to_json()
            st.session_state.scaled_df = scaled_df

            if model_choice == "Auto (recommended)":
                with st.spinner("Finding optimal groups..."):
                    ks, inertias, sil_scores = compute_elbow(scaled_json)
                st.session_state.ks = ks
                st.session_state.inertias = inertias
                st.session_state.sil_scores = sil_scores
                best_k = ks[int(np.argmax(sil_scores))] if (auto_k and ks) else (k_val or 3)
                with st.spinner(f"Analysing groups..."):
                    labels, sil, inertia = run_kmeans(scaled_json, best_k)
            else:
                with st.spinner("Analysing groups..."):
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


# ═══ TAB 1: Overview ══════════════════════════════════════════════════════════
with tab1:
    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, value) in zip([c1, c2, c3, c4], [
        ("Records", f"{df.shape[0]:,}"),
        ("Fields", str(df.shape[1])),
        ("Numerical", str(len(numerical_cols))),
        ("Categorical", str(len(categorical_cols))),
    ]):
        col.markdown(
            f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Data Preview ──────────────────────────────────────────────────────────
    is_edited = st.session_state.df_edited is not None

    preview_label = "Data Preview"
    if is_edited:
        preview_label += "  —  edited"
    st.markdown(f"### {preview_label}")

    if is_edited:
        st.caption("You are working on an edited version of the original file.")

    st.dataframe(df.head(200), use_container_width=True, height=340)

    if is_edited:
        st.download_button(
            "Download edited data",
            df.to_csv(index=False).encode(),
            "edited_data.csv",
            "text/csv",
        )

    # ── Data Editor ───────────────────────────────────────────────────────────
    with st.expander("Data Actions", expanded=False):
        st.caption("Clean, reshape, or filter your data. Every change applies across all tabs and resets segmentation.")

        active_df = (
            st.session_state.df_edited
            if st.session_state.df_edited is not None
            else st.session_state.df
        )

        ea1, ea2 = st.columns([2, 1])
        action = ea1.selectbox(
            "Action",
            [
                "Select an action...",
                "Remove duplicate rows",
                "Drop rows with missing values",
                "Fill missing — numeric (mean)",
                "Fill missing — numeric (median)",
                "Fill missing — numeric (zero)",
                "Fill missing — text (blank)",
                "Rename a column",
                "Drop a column",
                "Change column type",
                "Filter rows by value",
                "Reset all edits",
            ],
            key="edit_action",
        )

        if action == "Select an action...":
            st.caption("Choose an action from the dropdown to see options.")

        elif action == "Remove duplicate rows":
            n_dups = int(active_df.duplicated().sum())
            st.caption(f"{n_dups} duplicate rows found.")
            if st.button("Remove duplicates", key="apply_dedup"):
                result = active_df.drop_duplicates().reset_index(drop=True)
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success(f"Removed {n_dups} duplicate rows.")
                st.rerun()

        elif action == "Drop rows with missing values":
            n_missing = int(active_df.isnull().any(axis=1).sum())
            st.caption(f"{n_missing} rows contain at least one missing value.")
            if st.button("Drop rows", key="apply_dropna"):
                result = active_df.dropna().reset_index(drop=True)
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success(f"Removed {n_missing} rows with missing values.")
                st.rerun()

        elif action == "Fill missing — numeric (mean)":
            cols_with_missing = [c for c in active_df.select_dtypes(include=[np.number]).columns
                                 if active_df[c].isna().any()]
            st.caption(f"Applies to: {cols_with_missing if cols_with_missing else 'no numeric columns have missing values'}")
            if st.button("Fill with mean", key="apply_fill_mean"):
                result = active_df.copy()
                for c in result.select_dtypes(include=[np.number]).columns:
                    result[c] = result[c].fillna(result[c].mean())
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success("Filled missing numeric values with each column's mean.")
                st.rerun()

        elif action == "Fill missing — numeric (median)":
            if st.button("Fill with median", key="apply_fill_median"):
                result = active_df.copy()
                for c in result.select_dtypes(include=[np.number]).columns:
                    result[c] = result[c].fillna(result[c].median())
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success("Filled missing numeric values with each column's median.")
                st.rerun()

        elif action == "Fill missing — numeric (zero)":
            if st.button("Fill with zero", key="apply_fill_zero"):
                result = active_df.copy()
                for c in result.select_dtypes(include=[np.number]).columns:
                    result[c] = result[c].fillna(0)
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success("Filled missing numeric values with 0.")
                st.rerun()

        elif action == "Fill missing — text (blank)":
            if st.button("Fill with blank", key="apply_fill_blank"):
                result = active_df.copy()
                for c in result.select_dtypes(include=["object", "string"]).columns:
                    result[c] = result[c].fillna("")
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success("Filled missing text values with blank.")
                st.rerun()

        elif action == "Rename a column":
            rb1, rb2 = st.columns(2)
            col_to_rename = rb1.selectbox("Column to rename", active_df.columns.tolist(), key="rename_col")
            new_name = rb2.text_input("New name", value=col_to_rename, key="rename_val")
            if st.button("Rename", key="apply_rename"):
                new_name = new_name.strip()
                if new_name and new_name != col_to_rename:
                    result = active_df.rename(columns={col_to_rename: new_name})
                    st.session_state.df_edited = result
                    apply_edits_to_pipeline()
                    st.success(f"Renamed '{col_to_rename}' to '{new_name}'.")
                    st.rerun()
                else:
                    st.warning("Enter a different name.")

        elif action == "Drop a column":
            col_to_drop = st.selectbox("Column to drop", active_df.columns.tolist(), key="drop_col")
            st.caption(f"This will permanently remove '{col_to_drop}' from the working data.")
            if st.button("Drop column", key="apply_drop_col"):
                result = active_df.drop(columns=[col_to_drop])
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success(f"Removed column '{col_to_drop}'.")
                st.rerun()

        elif action == "Change column type":
            ct1, ct2 = st.columns(2)
            col_to_cast = ct1.selectbox("Column", active_df.columns.tolist(), key="cast_col")
            target_type = ct2.selectbox("Convert to", ["Numeric", "Text", "Date/Time"], key="cast_type")
            st.caption(f"Current type: {active_df[col_to_cast].dtype}")
            if st.button("Convert", key="apply_cast"):
                result = active_df.copy()
                try:
                    if target_type == "Numeric":
                        result[col_to_cast] = pd.to_numeric(result[col_to_cast], errors="coerce")
                    elif target_type == "Text":
                        result[col_to_cast] = result[col_to_cast].astype(str)
                    elif target_type == "Date/Time":
                        result[col_to_cast] = pd.to_datetime(result[col_to_cast], errors="coerce")
                    st.session_state.df_edited = result
                    apply_edits_to_pipeline()
                    st.success(f"Converted '{col_to_cast}' to {target_type}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Conversion failed: {e}")

        elif action == "Filter rows by value":
            filter_col = st.selectbox("Filter by field", active_df.columns.tolist(), key="filter_col")
            unique_vals = active_df[filter_col].dropna().unique().tolist()
            if len(unique_vals) <= 100:
                keep_vals = st.multiselect(
                    "Keep rows where value is",
                    options=unique_vals,
                    default=unique_vals,
                    key="filter_vals",
                )
                st.caption(f"Keeping {len(keep_vals)} of {len(unique_vals)} values.")
                if st.button("Apply filter", key="apply_filter"):
                    result = active_df[active_df[filter_col].isin(keep_vals)].reset_index(drop=True)
                    st.session_state.df_edited = result
                    apply_edits_to_pipeline()
                    st.success(f"Filtered to {len(result):,} rows.")
                    st.rerun()
            else:
                st.caption("Too many unique values for multi-select. Use AI Chat to filter with a custom condition.")

        elif action == "Reset all edits":
            st.caption("This will discard all changes and restore the original uploaded data.")
            if st.button("Reset to original", key="apply_reset"):
                st.session_state.df_edited = None
                apply_edits_to_pipeline()
                st.success("Data restored to original.")
                st.rerun()

    # ── Schema ────────────────────────────────────────────────────────────────
    st.markdown("### Field Summary")
    schema_rows = []
    for c in df.columns:
        n_null = int(df[c].isna().sum())
        schema_rows.append({
            "Field": c,
            "Type": str(df[c].dtype),
            "Non-null": int(df[c].notna().sum()),
            "Missing": n_null,
            "Missing %": f"{df[c].isna().mean()*100:.1f}%",
            "Unique values": int(df[c].nunique()),
        })
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

    # ── Stats ─────────────────────────────────────────────────────────────────
    if numerical_cols:
        st.markdown("### Summary Statistics")
        st.dataframe(df[numerical_cols].describe().round(3), use_container_width=True)


# ═══ TAB 2: Clustering ════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.clustering_done:
        if not numerical_cols:
            st.warning(
                "This dataset does not contain numerical fields. "
                "Segmentation works by comparing numerical values across records. "
                "Try the Visualizations tab to explore your data."
            )
        else:
            st.markdown("### Customer Segmentation")
            st.markdown(
                "Segmentation groups your records into distinct profiles based on "
                "patterns in the data. Each group shares similar characteristics — "
                "for example, high-value customers, occasional buyers, or at-risk accounts."
            )
            st.info("Choose your fields in the sidebar and click Run Segmentation to begin.")
    else:
        labels = st.session_state.cluster_labels
        df_clustered = st.session_state.df_clustered
        scaled_df = st.session_state.scaled_df

        n_groups = len(set(labels)) - (1 if -1 in labels else 0)
        sil_final = silhouette_score(scaled_df, labels) if n_groups > 1 else 0.0
        separation_pct = round(sil_final * 100)

        # Plain-language separation description
        if sil_final >= 0.7:
            separation_label = "Very well separated"
        elif sil_final >= 0.5:
            separation_label = "Well separated"
        elif sil_final >= 0.3:
            separation_label = "Moderately separated"
        else:
            separation_label = "Overlapping groups"

        st.markdown("### Segmentation Results")

        c1, c2, c3, c4 = st.columns(4)
        for col, (label, value) in zip([c1, c2, c3, c4], [
            ("Groups Found", str(n_groups)),
            ("Total Records", f"{len(df_clustered):,}"),
            ("Group Separation", separation_label),
            ("Fields Used", str(len(selected_num))),
        ]):
            col.markdown(
                f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value" style="font-size:1.1rem">{value}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Group size chart with plain labels
        st.markdown("### How Records Are Distributed Across Groups")
        size_data = pd.Series(labels).value_counts().sort_index().reset_index()
        size_data.columns = ["Group", "Records"]
        size_data["Group"] = size_data["Group"].apply(lambda x: f"Group {x + 1}")
        size_data["Share"] = (size_data["Records"] / size_data["Records"].sum() * 100).round(1)
        size_data["Label"] = size_data.apply(
            lambda r: f"{r['Records']:,} records ({r['Share']}%)", axis=1
        )
        fig_sizes = px.bar(
            size_data, x="Group", y="Records", color="Group",
            text="Label",
            color_discrete_sequence=PALETTE,
            title="Records per Group",
        )
        fig_sizes.update_traces(textposition="outside")
        fig_sizes.update_layout(showlegend=False, **BASE_LAYOUT, height=360)
        fig_sizes.update_xaxes(gridcolor="#f1f5f9")
        fig_sizes.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_sizes, use_container_width=True)

        # Group profiles in plain language
        valid_num = [c for c in selected_num if c in df_clustered.columns]
        if valid_num:
            st.markdown("### Group Profiles")
            st.caption(
                "Average values for each field across groups. "
                "Use this to understand what makes each group different."
            )
            profile_raw = df_clustered.groupby("Cluster")[valid_num].mean().round(2)
            profile_raw.index = [f"Group {i + 1}" for i in profile_raw.index]
            profile_raw.index.name = "Group"
            st.dataframe(profile_raw, use_container_width=True)

            # Highlight what is distinctive per group
            st.markdown("### What Makes Each Group Distinct")
            overall_means = df_clustered[valid_num].mean()
            for cluster in sorted(set(labels)):
                subset = df_clustered[df_clustered["Cluster"] == cluster]
                size = len(subset)
                pct = round(100 * size / len(df_clustered), 1)
                group_label = f"Group {cluster + 1}"
                differences = []
                for col in valid_num[:6]:
                    cm = subset[col].mean()
                    om = overall_means[col]
                    if om == 0:
                        continue
                    diff = (cm - om) / abs(om) * 100
                    if abs(diff) > 15:
                        direction = "higher" if diff > 0 else "lower"
                        differences.append(
                            f"{col} is {abs(diff):.0f}% {direction} than average ({cm:.2f} vs {om:.2f})"
                        )
                if differences:
                    summary = f"{group_label} — {size:,} records ({pct}%): " + "; ".join(differences)
                else:
                    summary = f"{group_label} — {size:,} records ({pct}%): Close to average across all fields."
                st.markdown(
                    f'<div class="insight-box">{summary}</div>',
                    unsafe_allow_html=True,
                )

        # Group scatter plot using PCA — with plain language
        if st.session_state.scaled_df is not None:
            st.markdown("### Visual Map of Groups")
            st.caption(
                "Each dot represents one record. Dots close together share similar "
                "characteristics. Colour shows which group each record belongs to."
            )
            pca_coords, explained = compute_pca(scaled_df.to_json())
            arr = np.array(pca_coords)
            df_pca = pd.DataFrame({
                "Dimension 1": arr[:, 0],
                "Dimension 2": arr[:, 1],
                "Group": [f"Group {l + 1}" for l in labels],
            })
            fig_pca = px.scatter(
                df_pca, x="Dimension 1", y="Dimension 2", color="Group",
                color_discrete_sequence=PALETTE,
                title="Customer Map",
            )
            fig_pca.update_traces(marker=dict(size=7, opacity=0.75))
            fig_pca = apply_base(fig_pca, 460)
            st.plotly_chart(fig_pca, use_container_width=True)

        if model_choice == "Auto (recommended)" and st.session_state.ks:
            with st.expander("How the number of groups was chosen"):
                st.caption(
                    "The app tested different numbers of groups and measured how well-separated "
                    "they were. The number with the highest separation score was selected automatically."
                )
                st.plotly_chart(
                    plot_elbow(
                        st.session_state.ks,
                        st.session_state.inertias,
                        st.session_state.sil_scores,
                    ),
                    use_container_width=True,
                )

        st.markdown("### Full Segmented Dataset")
        display_df = df_clustered.copy()
        display_df["Group"] = display_df["Cluster"].apply(lambda x: f"Group {x + 1}")
        display_df = display_df.drop(columns=["Cluster"])
        st.dataframe(display_df, use_container_width=True, height=350)
        st.download_button(
            "Download Segmented Data",
            display_df.to_csv(index=False).encode(),
            "segmented_data.csv",
            "text/csv",
        )


# ═══ TAB 3: Visualizations ════════════════════════════════════════════════════
with tab3:
    if df is None:
        st.info("Upload a CSV file to start building charts.")
        st.stop()

    all_cols = df.columns.tolist()
    num_cols_viz = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols_viz = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df[c])]

    # ── Chart Builder ────────────────────────────────────────────────────────
    st.markdown("### Chart Builder")
    st.caption("Select a chart type and configure the fields below. The chart updates instantly.")

    CHART_TYPES = [
        "Bar Chart",
        "Line Chart",
        "Scatter Plot",
        "Histogram",
        "Box Plot",
        "Pie Chart",
        "Area Chart",
        "Heatmap (Correlation)",
    ]

    AGGREGATIONS = ["Sum", "Average", "Count", "Count Distinct", "Min", "Max", "Median"]

    # Controls row
    cc1, cc2, cc3 = st.columns([1, 1, 1])
    chart_type = cc1.selectbox("Chart Type", CHART_TYPES, key="vz_chart_type")
    color_col = cc3.selectbox(
        "Colour by (optional)",
        ["None"] + cat_cols_viz,
        key="vz_color",
    )

    # Field selectors change based on chart type
    x_col, y_col, agg_func = None, None, "Count"

    if chart_type == "Histogram":
        x_col = cc2.selectbox("Field", num_cols_viz or all_cols, key="vz_x")
        bins = st.slider("Number of bins", 5, 100, 30, key="vz_bins")

    elif chart_type == "Pie Chart":
        x_col = cc2.selectbox("Category (slices)", cat_cols_viz or all_cols, key="vz_x")
        if num_cols_viz:
            y_col = st.selectbox(
                "Value field (optional — leave blank to count records)",
                ["Count records"] + num_cols_viz,
                key="vz_y_pie",
            )
            if y_col == "Count records":
                y_col = None
            agg_func = st.selectbox("Aggregation", AGGREGATIONS, key="vz_agg_pie") if y_col else "Count"
        else:
            y_col = None

    elif chart_type == "Heatmap (Correlation)":
        selected_for_corr = st.multiselect(
            "Select numerical fields for correlation",
            options=num_cols_viz,
            default=num_cols_viz[:8],
            key="vz_corr_cols",
        )

    elif chart_type == "Scatter Plot":
        fc1, fc2, fc3 = st.columns(3)
        x_col = fc1.selectbox("X axis", num_cols_viz or all_cols, key="vz_x")
        y_col = fc2.selectbox(
            "Y axis",
            [c for c in (num_cols_viz or all_cols) if c != x_col] or all_cols,
            key="vz_y",
        )
        size_col = fc3.selectbox("Size by (optional)", ["None"] + num_cols_viz, key="vz_size")

    else:
        # Bar, Line, Box, Area — all follow x + y + aggregation pattern
        fc1, fc2, fc3 = st.columns(3)

        if chart_type in ["Box Plot"]:
            x_col = fc1.selectbox("Category (X axis)", cat_cols_viz or all_cols, key="vz_x")
            y_col = fc2.selectbox("Value (Y axis)", num_cols_viz or all_cols, key="vz_y")
        elif chart_type in ["Line Chart", "Area Chart"]:
            x_col = fc1.selectbox("X axis", all_cols, key="vz_x")
            y_col = fc2.selectbox("Y axis", num_cols_viz or all_cols, key="vz_y")
            agg_func = fc3.selectbox("Aggregation", AGGREGATIONS, key="vz_agg")
        else:
            # Bar chart — most flexible
            x_col = fc1.selectbox("X axis (Category or Field)", all_cols, key="vz_x")
            y_col = fc2.selectbox(
                "Y axis (Value — leave as Count to count records)",
                ["Count records"] + num_cols_viz,
                key="vz_y_bar",
            )
            if y_col == "Count records":
                y_col = None
                agg_func = "Count"
            else:
                agg_func = fc3.selectbox("Aggregation", AGGREGATIONS, key="vz_agg")

    # ── Apply aggregation and build chart ─────────────────────────────────────
    def apply_aggregation(df, x_col, y_col, agg_func, color_col=None):
        """Aggregate df by x_col (and optionally color_col) using agg_func on y_col."""
        group_cols = [x_col]
        if color_col and color_col != "None" and color_col in df.columns:
            group_cols.append(color_col)

        if y_col is None or agg_func == "Count":
            agg_df = df.groupby(group_cols).size().reset_index(name="Count")
            return agg_df, "Count"

        agg_map = {
            "Sum": "sum",
            "Average": "mean",
            "Min": "min",
            "Max": "max",
            "Median": "median",
            "Count Distinct": "nunique",
            "Count": "count",
        }
        agg_df = df.groupby(group_cols)[y_col].agg(agg_map[agg_func]).reset_index()
        agg_df.columns = group_cols + [f"{agg_func} of {y_col}"]
        return agg_df, f"{agg_func} of {y_col}"

    # Build and display chart
    chart_error = None
    fig_vz = None

    try:
        color_val = color_col if color_col != "None" and color_col in df.columns else None

        if chart_type == "Histogram":
            fig_vz = px.histogram(
                df, x=x_col,
                nbins=bins,
                color=color_val,
                color_discrete_sequence=PALETTE,
                title=f"Distribution of {x_col}",
            )

        elif chart_type == "Pie Chart":
            if y_col and agg_func != "Count":
                agg_map_pie = {"Sum": "sum", "Average": "mean", "Count": "count",
                               "Min": "min", "Max": "max", "Median": "median",
                               "Count Distinct": "nunique"}
                pie_data = df.groupby(x_col)[y_col].agg(agg_map_pie[agg_func]).reset_index()
                pie_data.columns = [x_col, "Value"]
            else:
                pie_data = df[x_col].value_counts().reset_index()
                pie_data.columns = [x_col, "Value"]
            fig_vz = px.pie(
                pie_data, names=x_col, values="Value",
                color_discrete_sequence=PALETTE,
                title=f"{x_col} breakdown",
            )

        elif chart_type == "Scatter Plot":
            size_val = size_col if size_col != "None" else None
            fig_vz = px.scatter(
                df, x=x_col, y=y_col,
                color=color_val,
                size=size_val,
                color_discrete_sequence=PALETTE,
                title=f"{x_col} vs {y_col}",
                opacity=0.7,
            )
            fig_vz.update_traces(marker=dict(size=8 if not size_val else None))

        elif chart_type == "Box Plot":
            fig_vz = px.box(
                df, x=x_col, y=y_col,
                color=color_val or x_col,
                color_discrete_sequence=PALETTE,
                title=f"{y_col} by {x_col}",
            )

        elif chart_type == "Heatmap (Correlation)":
            if len(selected_for_corr) < 2:
                chart_error = "Select at least 2 numerical fields to generate a correlation heatmap."
            else:
                corr = df[selected_for_corr].corr()
                fig_vz = px.imshow(
                    corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap",
                )

        elif chart_type == "Line Chart":
            agg_df, y_label = apply_aggregation(df, x_col, y_col, agg_func, color_val)
            line_color = color_val if color_val and color_val in agg_df.columns else None
            fig_vz = px.line(
                agg_df, x=x_col, y=y_label,
                color=line_color,
                color_discrete_sequence=PALETTE,
                title=f"{y_label} by {x_col}",
                markers=True,
            )

        elif chart_type == "Area Chart":
            agg_df, y_label = apply_aggregation(df, x_col, y_col, agg_func, color_val)
            area_color = color_val if color_val and color_val in agg_df.columns else None
            fig_vz = px.area(
                agg_df, x=x_col, y=y_label,
                color=area_color,
                color_discrete_sequence=PALETTE,
                title=f"{y_label} by {x_col}",
            )

        else:
            # Bar Chart
            agg_df, y_label = apply_aggregation(df, x_col, y_col, agg_func, color_val)
            bar_color = color_val if color_val and color_val in agg_df.columns else x_col
            if bar_color not in agg_df.columns:
                bar_color = None
            sort_vals = agg_df.sort_values(y_label, ascending=False)
            fig_vz = px.bar(
                sort_vals, x=x_col, y=y_label,
                color=bar_color,
                color_discrete_sequence=PALETTE,
                title=f"{y_label} by {x_col}",
                text=y_label,
            )
            fig_vz.update_traces(texttemplate="%{text:,.1f}", textposition="outside")

    except Exception as e:
        chart_error = str(e)

    if chart_error:
        st.warning(f"Could not build chart: {chart_error}")
    elif fig_vz is not None:
        fig_vz = apply_base(fig_vz, 460)
        if chart_type not in ["Pie Chart", "Heatmap (Correlation)"]:
            fig_vz.update_xaxes(gridcolor="#f1f5f9")
            fig_vz.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_vz, use_container_width=True)

    st.markdown("---")

    # ── Auto Overview Charts ──────────────────────────────────────────────────
    auto_visuals = st.session_state.auto_visuals
    if auto_visuals:
        st.markdown("### Auto-Generated Overview")
        st.caption("These charts are built automatically from your data on upload.")
        for title, fig_auto in auto_visuals:
            st.plotly_chart(fig_auto, use_container_width=True)
    elif not auto_visuals and df is not None:
        st.info("Upload a CSV file to see automatic overview charts.")


# ═══ TAB 4: AI Chat ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown("### AI Chat Assistant")

    if not get_api_key():
        st.error(
            "No API key configured. In Streamlit Cloud go to: "
            "Manage app (bottom-right) → three-dot menu → Settings → Secrets tab, "
            "and add:\n\n`OPENROUTER_API_KEY = \"your-key-here\"`\n\n"
            "Get a free key at openrouter.ai"
        )
        st.stop()

    working_df = (
        st.session_state.df_clustered
        if st.session_state.df_clustered is not None
        else (st.session_state.df_edited if st.session_state.df_edited is not None else df)
    )
    cluster_col = "Cluster" if st.session_state.df_clustered is not None else None
    exact_cols = [c.strip() for c in working_df.columns.tolist()]

    st.markdown(
        "Ask questions about your data or request a chart.\n\n"
        "- *What patterns do you see in this data?*\n"
        "- *Show a bar chart of Sentiment by Location*\n"
        "- *Which location has the highest average confidence score?*"
    )
    st.markdown("---")

    if st.session_state.chat_history:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Render history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            if msg.get("is_error"):
                st.warning(msg["content"])
            elif msg.get("is_code"):
                # Show the AI's text response cleanly
                if msg.get("text_response"):
                    st.markdown(
                        f'<div class="chat-ai">{msg["text_response"]}</div>',
                        unsafe_allow_html=True,
                    )
                # Show chart if successful
                if msg.get("fig") is not None:
                    st.plotly_chart(msg["fig"], use_container_width=True)
                # Show clean error if chart failed
                elif msg.get("exec_error"):
                    st.markdown(
                        f'<div class="chat-ai">{msg["exec_error"]}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f'<div class="chat-ai">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    # Chat input
    user_input = st.chat_input("Ask a question or request a chart...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        ctx = get_dataset_context(working_df, cluster_col)

        viz_keywords = [
            "plot", "chart", "graph", "show", "visualize", "draw",
            "bar", "scatter", "histogram", "heatmap", "distribution",
            "figure", "map", "pie", "line",
        ]
        wants_viz = any(kw in user_input.lower() for kw in viz_keywords)

        if wants_viz:
            system_prompt = textwrap.dedent(f"""
                You are a data analyst. The user wants a chart.
                First write ONE sentence describing what you are plotting (no code in this sentence).
                Then on a new line write ONLY the Python code block.

                The dataset is a pandas DataFrame called `df`.
                The final Plotly figure MUST be stored in a variable called `fig`.

                Dataset context:
                {ctx}

                EXACT column names — copy character for character, including spaces:
                {exact_cols}

                Code rules:
                - Only use: df, pd, np, px, go, make_subplots (all pre-imported).
                - Do NOT import anything.
                - Do NOT use os, sys, open, eval, exec, requests.
                - Column names must exactly match the list above.
                - Always store the chart in `fig`.
                - Apply to every chart:
                  fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_family='IBM Plex Mono')

                Format your response as:
                [One sentence description]

                ```python
                [code here]
                ```
            """).strip()
        else:
            system_prompt = textwrap.dedent(f"""
                You are a concise, friendly data analyst.
                Answer the user's question in plain English.
                Use actual numbers and column names from the dataset where relevant.
                Keep your answer to 3-5 sentences maximum.
                Do not generate code.

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
            # Separate the text description from the code block
            code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
            if code_match:
                code_only = code_match.group(1).strip()
                text_only = response[:response.find("```")].strip()
            else:
                # No code fences — treat entire response as code
                code_only = response.strip()
                text_only = "Here is the chart:"

            fig_result, exec_error = execute_ai_code(code_only, working_df)
            st.session_state.chat_history.append({
                "role": "assistant",
                "is_code": True,
                "text_response": text_only,
                "content": code_only,
                "fig": fig_result,
                "exec_error": exec_error,
            })
        else:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response, "is_code": False}
            )

        st.rerun()
