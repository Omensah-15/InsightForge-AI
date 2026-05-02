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

# ── API key ───────────────────────────────────────────────────────────────────
def get_api_key() -> str:
    try:
        key = st.secrets.get("OPENROUTER_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY", "")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightForge AI",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.25rem;
}

/* ── Main area ── */
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1400px;
}

/* ── App header ── */
.app-header {
    margin-bottom: 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #21262d;
}
.app-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.02em;
}
.app-subtitle {
    font-size: 0.9rem;
    color: #8b949e;
    margin: 0;
    font-weight: 400;
}

/* ── Sidebar title ── */
.sidebar-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -0.01em;
    margin-bottom: 0.25rem;
}
.sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6e7681;
    margin: 1.25rem 0 0.5rem 0;
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #388bfd; }
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6e7681;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.65rem;
    font-weight: 700;
    color: #f0f6fc;
    line-height: 1;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.0rem;
    font-weight: 600;
    color: #f0f6fc;
    margin: 1.5rem 0 0.5rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #21262d;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.65rem 1.25rem;
    background: transparent;
    border: none;
    color: #6e7681;
    border-bottom: 2px solid transparent;
    transition: color 0.2s;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #c9d1d9; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem; }

/* ── Buttons ── */
.stButton > button {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-radius: 6px;
    border: 1px solid #388bfd;
    background: #388bfd;
    color: #ffffff;
    padding: 0.5rem 1.1rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: #2f7ae5;
    border-color: #2f7ae5;
    color: #ffffff;
}

/* Primary action button variant */
.run-btn .stButton > button {
    background: #238636;
    border-color: #2ea043;
    font-size: 0.8rem;
    padding: 0.65rem 1.25rem;
}
.run-btn .stButton > button:hover { background: #2ea043; }

/* Secondary buttons */
.sec-btn .stButton > button {
    background: transparent;
    border-color: #30363d;
    color: #8b949e;
}
.sec-btn .stButton > button:hover {
    background: #21262d;
    border-color: #8b949e;
    color: #c9d1d9;
}

/* ── Form labels ── */
.stSelectbox label, .stSlider label, .stMultiSelect label,
.stCheckbox label, .stTextInput label, .stFileUploader label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6e7681 !important;
}

/* ── Select, input boxes ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
}
.stSelectbox > div > div:hover,
.stMultiSelect > div > div:hover {
    border-color: #388bfd !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: #388bfd;
}

/* ── Info / success / error boxes ── */
.stAlert {
    border-radius: 6px;
    border-left: 3px solid;
}

/* ── Chat bubbles ── */
.chat-user {
    background: #1f6feb;
    color: #ffffff;
    padding: 0.75rem 1rem;
    border-radius: 12px 12px 2px 12px;
    margin: 0.6rem 0 0.6rem 15%;
    font-size: 0.9rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(31,111,235,0.3);
}
.chat-ai {
    background: #161b22;
    color: #e6edf3;
    padding: 0.75rem 1rem;
    border-radius: 12px 12px 12px 2px;
    margin: 0.6rem 15% 0.6rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
    border: 1px solid #21262d;
    border-left: 3px solid #388bfd;
}

/* ── Insight boxes ── */
.insight-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #388bfd;
    padding: 0.75rem 1rem;
    border-radius: 0 6px 6px 0;
    margin: 0.5rem 0;
    font-size: 0.875rem;
    color: #c9d1d9;
    line-height: 1.5;
}
.insight-box.positive { border-left-color: #3fb950; }
.insight-box.warning  { border-left-color: #d29922; }

/* ── Group profile cards ── */
.group-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.6rem 0;
    transition: border-color 0.2s;
}
.group-card:hover { border-color: #388bfd; }
.group-card-header {
    font-weight: 600;
    font-size: 0.95rem;
    color: #f0f6fc;
    margin-bottom: 0.35rem;
}
.group-card-sub {
    font-size: 0.82rem;
    color: #8b949e;
}

/* ── Result stat boxes ── */
.result-stat {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.result-stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6e7681;
    margin-bottom: 0.3rem;
}
.result-stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    color: #58a6ff;
}

/* ── Dividers ── */
hr { border-color: #21262d !important; margin: 1rem 0; }

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid #21262d;
    border-radius: 6px;
    overflow: hidden;
}

/* ── Caption ── */
.stCaption { color: #6e7681 !important; font-size: 0.8rem !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
    color: #c9d1d9 !important;
    font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6e7681; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 8px;
    padding: 0.5rem;
}
[data-testid="stFileUploader"]:hover { border-color: #388bfd; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #388bfd !important; }

/* ── Download button ── */
.stDownloadButton > button {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    background: transparent;
    border: 1px solid #30363d;
    color: #8b949e;
    border-radius: 6px;
    padding: 0.45rem 1rem;
    transition: all 0.2s;
}
.stDownloadButton > button:hover {
    border-color: #388bfd;
    color: #58a6ff;
    background: rgba(56,139,253,0.1);
}

/* ── Chat input ── */
.stChatInput > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
.stChatInput input {
    color: #e6edf3 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
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


# ── AI models ─────────────────────────────────────────────────────────────────
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
    api_key = get_api_key()
    if not api_key:
        return (
            "AI is not configured. Add your OpenRouter API key in "
            "Streamlit Cloud > Manage app > Settings > Secrets.",
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


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
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
    text = [c for c in text if c not in categorical]
    return numerical, categorical, text


def get_dataset_context(df: pd.DataFrame, cluster_col: str = None) -> str:
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


# ── Auto visuals ──────────────────────────────────────────────────────────────
PALETTE = ["#388bfd", "#3fb950", "#d29922", "#bc8cff", "#f78166", "#39c5cf", "#ffa657", "#ff7b72"]

def chart_layout(height=380):
    return dict(
        plot_bgcolor="#161b22",
        paper_bgcolor="#0d1117",
        font_family="JetBrains Mono",
        font_color="#c9d1d9",
        height=height,
        margin=dict(l=16, r=16, t=40, b=16),
    )


def apply_base(fig, height=380):
    fig.update_layout(**chart_layout(height))
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#30363d",
                     linecolor="#30363d", tickcolor="#6e7681")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#30363d",
                     linecolor="#30363d", tickcolor="#6e7681")
    return fig


def generate_auto_visuals(df: pd.DataFrame, numerical_cols: list,
                          categorical_cols: list) -> list:
    visuals = []

    for col in categorical_cols[:2]:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "Count"]
        fig = px.bar(counts.head(15), x=col, y="Count", color=col,
                     color_discrete_sequence=PALETTE,
                     title=f"Distribution of {col}")
        fig.update_layout(**chart_layout(), showlegend=False)
        fig.update_xaxes(gridcolor="#21262d")
        fig.update_yaxes(gridcolor="#21262d")
        visuals.append((f"Distribution of {col}", fig))

    for col in numerical_cols[:3]:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}",
                           color_discrete_sequence=[PALETTE[0]])
        fig.update_layout(**chart_layout())
        fig.update_xaxes(gridcolor="#21262d")
        fig.update_yaxes(gridcolor="#21262d")
        visuals.append((f"Distribution of {col}", fig))

    if categorical_cols and numerical_cols:
        cat_col = categorical_cols[0]
        num_col = numerical_cols[0]
        fig = px.box(df, x=cat_col, y=num_col, color=cat_col,
                     color_discrete_sequence=PALETTE,
                     title=f"{num_col} by {cat_col}")
        fig.update_layout(**chart_layout(), showlegend=False)
        visuals.append((f"{num_col} by {cat_col}", fig))

    if len(categorical_cols) > 1 and numerical_cols:
        cat_col = categorical_cols[1]
        num_col = numerical_cols[0]
        counts = df.groupby(cat_col)[num_col].mean().reset_index()
        fig = px.bar(counts, x=cat_col, y=num_col, color=cat_col,
                     color_discrete_sequence=PALETTE,
                     title=f"Average {num_col} by {cat_col}")
        fig.update_layout(**chart_layout(), showlegend=False)
        visuals.append((f"Average {num_col} by {cat_col}", fig))

    if len(numerical_cols) >= 2:
        fig = px.scatter(df, x=numerical_cols[0], y=numerical_cols[1],
                         color=categorical_cols[0] if categorical_cols else None,
                         color_discrete_sequence=PALETTE,
                         title=f"{numerical_cols[0]} vs {numerical_cols[1]}")
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.update_layout(**chart_layout())
        visuals.append((f"{numerical_cols[0]} vs {numerical_cols[1]}", fig))

    if len(numerical_cols) >= 3:
        corr = df[numerical_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="Blues",
                        title="Feature Correlation Heatmap")
        fig.update_layout(**chart_layout())
        visuals.append(("Correlation Heatmap", fig))

    return visuals


# ── Preprocessing & clustering ────────────────────────────────────────────────
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


def plot_elbow(ks, inertias, scores):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia",
                             line=dict(color="#388bfd", width=2), marker=dict(size=7)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=ks, y=scores, mode="lines+markers", name="Silhouette",
                             line=dict(color="#3fb950", width=2, dash="dot"),
                             marker=dict(size=7, color="#3fb950")),
                  secondary_y=True)
    fig.update_layout(title="Elbow Curve + Silhouette Score", xaxis_title="k",
                      yaxis_title="Inertia", yaxis2_title="Silhouette Score",
                      legend=dict(x=0.7, y=0.95, bgcolor="rgba(0,0,0,0)"),
                      **chart_layout(380))
    fig.update_xaxes(gridcolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d")
    return fig


# ── Code sandbox ──────────────────────────────────────────────────────────────
BLOCKED = ["os.", "sys.", "subprocess", "open(", "__import__",
           "importlib", "shutil", "socket", "requests", "eval(", "exec("]


def clean_ai_code(code: str) -> str:
    code = re.sub(r"```(?:python)?", "", code)
    code = code.replace("```", "").strip()
    return code


def execute_ai_code(code: str, df: pd.DataFrame) -> tuple:
    code = clean_ai_code(code)
    if not code:
        return None, "No executable code was returned."
    for token in BLOCKED:
        if token in code:
            return None, "Code was blocked for safety reasons."
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
            return None, f"Column name mismatch. Your columns are: {cols}."
        return None, f"Chart error: {err}"
    except KeyError as e:
        cols = df_exec.columns.tolist()
        return None, f"Column {e} not found. Your columns are: {cols}."
    except Exception as e:
        return None, f"Chart could not be generated: {str(e)}"
    fig = local_ns.get("fig", None)
    if fig is None:
        return None, "The AI did not produce a chart. Try rephrasing your request."
    # Apply dark theme to AI-generated charts
    apply_base(fig)
    return fig, None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">InsightForge AI</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sidebar-section">Data Source</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        file_bytes = uploaded.read()
        new_hash = hash(file_bytes)
        if st.session_state._file_hash != new_hash:
            reset_all()
            st.session_state._file_hash = new_hash
            with st.spinner("Loading data..."):
                df_raw = load_csv(file_bytes)
            num_cols, cat_cols, txt_cols = detect_column_types(df_raw)
            st.session_state.df = df_raw
            st.session_state.numerical_cols = num_cols
            st.session_state.categorical_cols = cat_cols
            st.session_state.text_cols = txt_cols
            st.session_state.selected_num = num_cols[:8]
            st.session_state.auto_visuals = generate_auto_visuals(df_raw, num_cols, cat_cols)

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
        st.markdown('<div class="sidebar-section">Data Actions</div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            st.markdown('<div class="sec-btn">', unsafe_allow_html=True)
            if st.button("Clear Data", use_container_width=True):
                reset_all()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with cb:
            st.markdown('<div class="sec-btn">', unsafe_allow_html=True)
            if st.button("Reset Segments", use_container_width=True):
                reset_clustering()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Segmentation Settings</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Grouping Method",
        ["Auto (recommended)", "Density-based"],
        help="Auto groups records by similarity. Density-based finds natural clusters of any shape.",
    )

    k_val, eps_val, min_s, auto_k = None, 0.5, 5, True
    if model_choice == "Auto (recommended)":
        auto_k = st.checkbox("Find number of groups automatically", value=True)
        if not auto_k:
            k_val = st.slider("Number of groups", 2, 15, 4)
    else:
        eps_val = st.slider("Sensitivity", 0.1, 5.0, 0.5, 0.1,
                            help="How close records need to be to form a group.")
        min_s = st.slider("Minimum group size", 2, 20, 5)

    st.markdown('<div class="sidebar-section">Fields to Analyse</div>', unsafe_allow_html=True)
    num_options = st.session_state.numerical_cols
    selected_num = []
    if num_options:
        valid_defaults = [c for c in st.session_state.selected_num if c in num_options]
        if not valid_defaults:
            valid_defaults = num_options[:8]
        selected_num = st.multiselect(
            "Numerical Fields",
            options=num_options,
            default=valid_defaults,
            label_visibility="collapsed",
        )
        st.session_state.selected_num = selected_num
    elif st.session_state.df is not None:
        st.caption("No numerical fields detected.")

    st.markdown("")
    st.markdown('<div class="run-btn">', unsafe_allow_html=True)
    run_btn = st.button("Run Segmentation", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">InsightForge AI</div>
  <div class="app-subtitle">Customer segmentation and data analysis</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.df is None:
    st.info("Upload a CSV file from the sidebar to get started.")
    st.stop()

df = st.session_state.df_edited if st.session_state.df_edited is not None else st.session_state.df
numerical_cols = st.session_state.numerical_cols
categorical_cols = st.session_state.categorical_cols

# ── Run segmentation ──────────────────────────────────────────────────────────
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
            st.success(f"Segmentation complete — {n_found} groups found.")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Segmentation", "Visualizations", "AI Chat"])


# ════════════════ TAB 1: Overview ═══════════════════════════════════════════
with tab1:
    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, value) in zip([c1, c2, c3, c4], [
        ("Total Records", f"{df.shape[0]:,}"),
        ("Fields", str(df.shape[1])),
        ("Numerical", str(len(numerical_cols))),
        ("Categorical", str(len(categorical_cols))),
    ]):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    is_edited = st.session_state.df_edited is not None
    preview_label = "Data Preview" + ("  —  edited" if is_edited else "")
    st.markdown(f'<div class="section-header">{preview_label}</div>', unsafe_allow_html=True)
    if is_edited:
        st.caption("Working on an edited version of the original file.")

    st.dataframe(df.head(200), use_container_width=True, height=320)

    if is_edited:
        st.download_button(
            "Download Edited Data",
            df.to_csv(index=False).encode(),
            "edited_data.csv",
            "text/csv",
        )

    # Data actions expander
    with st.expander("Data Actions", expanded=False):
        st.caption("Clean, reshape, or filter your data. Changes apply across all tabs and reset segmentation.")
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

    # Field summary
    st.markdown('<div class="section-header">Field Summary</div>', unsafe_allow_html=True)
    schema_rows = []
    for c in df.columns:
        n_null = int(df[c].isna().sum())
        schema_rows.append({
            "Field": c,
            "Type": str(df[c].dtype),
            "Non-null": int(df[c].notna().sum()),
            "Missing": n_null,
            "Missing %": f"{df[c].isna().mean()*100:.1f}%",
            "Unique Values": int(df[c].nunique()),
        })
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

    if numerical_cols:
        st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df[numerical_cols].describe().round(3), use_container_width=True)


# ════════════════ TAB 2: Segmentation ═══════════════════════════════════════
with tab2:
    if not st.session_state.clustering_done:
        if not numerical_cols:
            st.warning(
                "This dataset does not contain numerical fields. "
                "Segmentation works by comparing numerical values across records. "
                "Try the Visualizations tab to explore your data."
            )
        else:
            st.markdown('<div class="section-header">Customer Segmentation</div>', unsafe_allow_html=True)
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

        if sil_final >= 0.7:
            separation_label = "Very well separated"
        elif sil_final >= 0.5:
            separation_label = "Well separated"
        elif sil_final >= 0.3:
            separation_label = "Moderately separated"
        else:
            separation_label = "Overlapping groups"

        st.markdown('<div class="section-header">Segmentation Results</div>', unsafe_allow_html=True)

        # Result stats
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, value) in zip([c1, c2, c3, c4], [
            ("Groups Found", str(n_groups)),
            ("Total Records", f"{len(df_clustered):,}"),
            ("Group Separation", separation_label),
            ("Fields Used", str(len(selected_num))),
        ]):
            col.markdown(
                f'<div class="result-stat">'
                f'<div class="result-stat-label">{label}</div>'
                f'<div class="result-stat-value">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Group distribution chart
        st.markdown('<div class="section-header">Records per Group</div>', unsafe_allow_html=True)
        size_data = pd.Series(labels).value_counts().sort_index().reset_index()
        size_data.columns = ["Group", "Records"]
        size_data["Group"] = size_data["Group"].apply(lambda x: f"Group {x + 1}")
        size_data["Share"] = (size_data["Records"] / size_data["Records"].sum() * 100).round(1)
        size_data["Label"] = size_data.apply(lambda r: f"{r['Records']:,} ({r['Share']}%)", axis=1)
        fig_sizes = px.bar(
            size_data, x="Group", y="Records", color="Group",
            text="Label", color_discrete_sequence=PALETTE,
            title="Records per Group",
        )
        fig_sizes.update_traces(textposition="outside")
        fig_sizes.update_layout(**chart_layout(360), showlegend=False)
        fig_sizes.update_xaxes(gridcolor="#21262d")
        fig_sizes.update_yaxes(gridcolor="#21262d")
        st.plotly_chart(fig_sizes, use_container_width=True)

        # Group profiles
        valid_num = [c for c in selected_num if c in df_clustered.columns]
        if valid_num:
            st.markdown('<div class="section-header">Group Profiles</div>', unsafe_allow_html=True)
            st.caption("Average values for each field across groups.")
            profile_raw = df_clustered.groupby("Cluster")[valid_num].mean().round(2)
            profile_raw.index = [f"Group {i + 1}" for i in profile_raw.index]
            profile_raw.index.name = "Group"
            st.dataframe(profile_raw, use_container_width=True)

            st.markdown('<div class="section-header">What Makes Each Group Distinct</div>', unsafe_allow_html=True)
            overall_means = df_clustered[valid_num].mean()
            group_colors = ["positive", "", "warning", ""]
            for idx, cluster in enumerate(sorted(set(labels))):
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
                    summary = f"<strong>{group_label}</strong> — {size:,} records ({pct}%): " + "; ".join(differences)
                else:
                    summary = f"<strong>{group_label}</strong> — {size:,} records ({pct}%): Close to average across all fields."
                color_class = group_colors[idx % len(group_colors)]
                st.markdown(
                    f'<div class="insight-box {color_class}">{summary}</div>',
                    unsafe_allow_html=True,
                )

        # PCA scatter
        if st.session_state.scaled_df is not None:
            st.markdown('<div class="section-header">Visual Map of Groups</div>', unsafe_allow_html=True)
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
                    plot_elbow(st.session_state.ks, st.session_state.inertias, st.session_state.sil_scores),
                    use_container_width=True,
                )

        st.markdown('<div class="section-header">Full Segmented Dataset</div>', unsafe_allow_html=True)
        display_df = df_clustered.copy()
        display_df["Group"] = display_df["Cluster"].apply(lambda x: f"Group {x + 1}")
        display_df = display_df.drop(columns=["Cluster"])
        st.dataframe(display_df, use_container_width=True, height=320)
        st.download_button(
            "Download Segmented Data",
            display_df.to_csv(index=False).encode(),
            "segmented_data.csv",
            "text/csv",
        )


# ════════════════ TAB 3: Visualizations ══════════════════════════════════════
with tab3:
    all_cols = df.columns.tolist()
    num_cols_viz = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols_viz = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df[c])]

    st.markdown('<div class="section-header">Chart Builder</div>', unsafe_allow_html=True)
    st.caption("Select a chart type and configure the fields below.")

    CHART_TYPES = [
        "Bar Chart", "Line Chart", "Scatter Plot", "Histogram",
        "Box Plot", "Pie Chart", "Area Chart", "Heatmap (Correlation)",
    ]
    AGGREGATIONS = ["Sum", "Average", "Count", "Count Distinct", "Min", "Max", "Median"]

    cc1, cc2, cc3 = st.columns([1, 1, 1])
    chart_type = cc1.selectbox("Chart Type", CHART_TYPES, key="vz_chart_type")
    color_col = cc3.selectbox("Colour by (optional)", ["None"] + cat_cols_viz, key="vz_color")

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
        fc1, fc2, fc3 = st.columns(3)
        if chart_type == "Box Plot":
            x_col = fc1.selectbox("Category (X axis)", cat_cols_viz or all_cols, key="vz_x")
            y_col = fc2.selectbox("Value (Y axis)", num_cols_viz or all_cols, key="vz_y")
        elif chart_type in ["Line Chart", "Area Chart"]:
            x_col = fc1.selectbox("X axis", all_cols, key="vz_x")
            y_col = fc2.selectbox("Y axis", num_cols_viz or all_cols, key="vz_y")
            agg_func = fc3.selectbox("Aggregation", AGGREGATIONS, key="vz_agg")
        else:
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

    def apply_aggregation(df, x_col, y_col, agg_func, color_col=None):
        group_cols = [x_col]
        if color_col and color_col != "None" and color_col in df.columns:
            group_cols.append(color_col)
        if y_col is None or agg_func == "Count":
            agg_df = df.groupby(group_cols).size().reset_index(name="Count")
            return agg_df, "Count"
        agg_map = {
            "Sum": "sum", "Average": "mean", "Min": "min",
            "Max": "max", "Median": "median", "Count Distinct": "nunique", "Count": "count",
        }
        agg_df = df.groupby(group_cols)[y_col].agg(agg_map[agg_func]).reset_index()
        agg_df.columns = group_cols + [f"{agg_func} of {y_col}"]
        return agg_df, f"{agg_func} of {y_col}"

    chart_error = None
    fig_vz = None

    try:
        color_val = color_col if color_col != "None" and color_col in df.columns else None

        if chart_type == "Histogram":
            fig_vz = px.histogram(df, x=x_col, nbins=bins, color=color_val,
                                   color_discrete_sequence=PALETTE,
                                   title=f"Distribution of {x_col}")

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
            fig_vz = px.pie(pie_data, names=x_col, values="Value",
                             color_discrete_sequence=PALETTE, title=f"{x_col} breakdown")

        elif chart_type == "Scatter Plot":
            size_val = size_col if size_col != "None" else None
            fig_vz = px.scatter(df, x=x_col, y=y_col, color=color_val, size=size_val,
                                 color_discrete_sequence=PALETTE,
                                 title=f"{x_col} vs {y_col}", opacity=0.7)
            fig_vz.update_traces(marker=dict(size=8 if not size_val else None))

        elif chart_type == "Box Plot":
            fig_vz = px.box(df, x=x_col, y=y_col, color=color_val or x_col,
                             color_discrete_sequence=PALETTE, title=f"{y_col} by {x_col}")

        elif chart_type == "Heatmap (Correlation)":
            if len(selected_for_corr) < 2:
                chart_error = "Select at least 2 numerical fields to generate a correlation heatmap."
            else:
                corr = df[selected_for_corr].corr()
                fig_vz = px.imshow(corr, text_auto=".2f", aspect="auto",
                                    color_continuous_scale="Blues", title="Correlation Heatmap")

        elif chart_type == "Line Chart":
            agg_df, y_label = apply_aggregation(df, x_col, y_col, agg_func, color_val)
            line_color = color_val if color_val and color_val in agg_df.columns else None
            fig_vz = px.line(agg_df, x=x_col, y=y_label, color=line_color,
                              color_discrete_sequence=PALETTE,
                              title=f"{y_label} by {x_col}", markers=True)

        elif chart_type == "Area Chart":
            agg_df, y_label = apply_aggregation(df, x_col, y_col, agg_func, color_val)
            area_color = color_val if color_val and color_val in agg_df.columns else None
            fig_vz = px.area(agg_df, x=x_col, y=y_label, color=area_color,
                              color_discrete_sequence=PALETTE, title=f"{y_label} by {x_col}")

        else:
            agg_df, y_label = apply_aggregation(df, x_col, y_col, agg_func, color_val)
            bar_color = color_val if color_val and color_val in agg_df.columns else x_col
            if bar_color not in agg_df.columns:
                bar_color = None
            sort_vals = agg_df.sort_values(y_label, ascending=False)
            fig_vz = px.bar(sort_vals, x=x_col, y=y_label, color=bar_color,
                             color_discrete_sequence=PALETTE,
                             title=f"{y_label} by {x_col}", text=y_label)
            fig_vz.update_traces(texttemplate="%{text:,.1f}", textposition="outside")

    except Exception as e:
        chart_error = str(e)

    if chart_error:
        st.warning(f"Could not build chart: {chart_error}")
    elif fig_vz is not None:
        fig_vz = apply_base(fig_vz, 460)
        st.plotly_chart(fig_vz, use_container_width=True)

    st.markdown("---")

    # Auto-generated overview
    auto_visuals = st.session_state.auto_visuals
    if auto_visuals:
        st.markdown('<div class="section-header">Auto-Generated Overview</div>', unsafe_allow_html=True)
        st.caption("These charts are built automatically from your data on upload.")
        for title, fig_auto in auto_visuals:
            st.plotly_chart(fig_auto, use_container_width=True)


# ════════════════ TAB 4: AI Chat ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">AI Chat Assistant</div>', unsafe_allow_html=True)

    if not get_api_key():
        st.error(
            "No API key configured. In Streamlit Cloud go to: "
            "Manage app > three-dot menu > Settings > Secrets tab, "
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

    st.caption("Ask questions about your data or request a chart.")
    st.markdown(
        '<div class="insight-box" style="margin-bottom:1rem; font-size:0.82rem; color:#8b949e;">'
        'Examples: &nbsp; "What patterns do you see in this data?" &nbsp;|&nbsp; '
        '"Show a bar chart of Sentiment by Location" &nbsp;|&nbsp; '
        '"Which location has the highest average confidence score?"'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.chat_history:
        st.markdown('<div class="sec-btn" style="max-width:120px">', unsafe_allow_html=True)
        if st.button("Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

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
                if msg.get("text_response"):
                    st.markdown(
                        f'<div class="chat-ai">{msg["text_response"]}</div>',
                        unsafe_allow_html=True,
                    )
                if msg.get("fig") is not None:
                    st.plotly_chart(msg["fig"], use_container_width=True)
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
                - Apply dark theme to every chart:
                  fig.update_layout(plot_bgcolor='#161b22', paper_bgcolor='#0d1117',
                                    font_color='#c9d1d9', font_family='JetBrains Mono')
                  fig.update_xaxes(gridcolor='#21262d', zerolinecolor='#30363d')
                  fig.update_yaxes(gridcolor='#21262d', zerolinecolor='#30363d')

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
            code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
            if code_match:
                code_only = code_match.group(1).strip()
                text_only = response[:response.find("```")].strip()
            else:
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
