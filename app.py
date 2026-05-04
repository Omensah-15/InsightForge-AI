import os
import re
import warnings
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
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS — InsightForge Industrial Design System ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
  --surface:       #fcf9f8;
  --surface-dim:   #dcd9d9;
  --surface-card:  #ffffff;
  --border:        #e2e8f0;
  --border-strong: #cbd5e1;
  --primary:       #0F62FE;
  --primary-dark:  #0043ce;
  --text:          #0f172a;
  --text-mid:      #475569;
  --text-dim:      #94a3b8;
  --mono:          'IBM Plex Mono', monospace;
  --sans:          'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
  font-family: var(--sans);
  background-color: var(--surface) !important;
  color: var(--text);
}

.main .block-container {
  padding: 0 2rem 3rem 2rem !important;
  max-width: 1400px;
}

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background-color: #0f172a !important;
  border-right: none !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stMarkdown h2 {
  font-family: var(--mono) !important;
  font-size: 0.9rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: #ffffff !important;
  margin-bottom: 0 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
  font-family: var(--mono) !important;
  font-size: 0.68rem !important;
  color: #475569 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  margin: 0 !important;
}
[data-testid="stSidebar"] hr {
  border-color: #1e293b !important;
  margin: 0.75rem 0 !important;
}
[data-testid="stSidebar"] label {
  font-family: var(--mono) !important;
  font-size: 0.65rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: #64748b !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
  background: #1e293b !important;
  border: 1px solid #334155 !important;
  border-radius: 2px !important;
}
[data-testid="stSidebar"] .stButton > button {
  background: #0F62FE !important;
  border: 2px solid #0F62FE !important;
  color: white !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  border-radius: 2px !important;
  width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: #0043ce !important;
  border-color: #0043ce !important;
}

/* ── Page Header ──────────────────────────────────────────── */
.if-header {
  background: #0f172a;
  margin: -1rem -2rem 0 -2rem;
  padding: 1.4rem 2rem 1.2rem 2rem;
  border-bottom: 3px solid #0F62FE;
  display: flex;
  align-items: flex-end;
  gap: 1.25rem;
  margin-bottom: 1.5rem;
}
.if-wordmark {
  font-family: var(--mono);
  font-size: 1.35rem;
  font-weight: 600;
  color: #ffffff;
  letter-spacing: -0.03em;
  line-height: 1;
}
.if-wordmark span { color: #0F62FE; }
.if-tagline {
  font-family: var(--mono);
  font-size: 0.66rem;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  padding-bottom: 0.1rem;
}
.if-badge {
  margin-left: auto;
  font-family: var(--mono);
  font-size: 0.6rem;
  color: #334155;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  border: 1px solid #334155;
  padding: 3px 8px;
  border-radius: 2px;
}

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 0;
  border-bottom: 2px solid var(--border);
  background: transparent;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--mono) !important;
  font-size: 0.69rem !important;
  font-weight: 600 !important;
  padding: 10px 22px !important;
  background: transparent !important;
  border: none !important;
  color: var(--text-dim) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -2px !important;
}
.stTabs [aria-selected="true"] {
  color: var(--text) !important;
  border-bottom: 2px solid var(--primary) !important;
}

/* ── Section label ────────────────────────────────────────── */
.if-section {
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
  margin: 1.5rem 0 1rem 0;
}

/* ── Metric Grid ──────────────────────────────────────────── */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  margin-bottom: 1.5rem;
}
.metric-card {
  background: var(--surface-card);
  padding: 1.25rem 1.5rem;
  position: relative;
}
.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--border);
}
.metric-card.accent::before { background: var(--primary); }
.metric-card .m-label {
  font-family: var(--mono);
  font-size: 0.6rem;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 0.4rem;
}
.metric-card .m-value {
  font-family: var(--mono);
  font-size: 2rem;
  font-weight: 600;
  color: var(--text);
  line-height: 1;
}
.metric-card .m-sub {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--text-dim);
  margin-top: 0.2rem;
}

/* ── Pills ────────────────────────────────────────────────── */
.pill {
  display: inline-block;
  font-family: var(--mono);
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 2px;
}
.pill-blue   { background: #dbeafe; color: #1d4ed8; }
.pill-green  { background: #dcfce7; color: #15803d; }
.pill-yellow { background: #fef9c3; color: #a16207; }
.pill-red    { background: #fee2e2; color: #b91c1c; }
.pill-gray   { background: #f1f5f9; color: #475569; }

/* ── Insight Boxes ────────────────────────────────────────── */
.insight-box {
  background: var(--surface-card);
  border: 1px solid var(--border);
  border-left: 4px solid var(--primary);
  padding: 0.875rem 1.125rem;
  margin: 0.5rem 0;
}
.insight-box .ib-label {
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--primary);
  margin-bottom: 0.2rem;
}
.insight-box .ib-text {
  color: var(--text-mid);
  font-size: 0.82rem;
  line-height: 1.6;
}

/* ── Chat ─────────────────────────────────────────────────── */
.chat-row-user {
  display: flex;
  justify-content: flex-end;
  margin: 0.75rem 0;
}
.chat-row-ai {
  display: flex;
  justify-content: flex-start;
  margin: 0.75rem 0;
}
.chat-bubble-user {
  background: var(--text);
  color: #f1f5f9;
  padding: 0.7rem 1rem;
  border-radius: 2px 2px 0 2px;
  font-size: 0.87rem;
  max-width: 78%;
  font-family: var(--mono);
  line-height: 1.55;
}
.chat-bubble-ai {
  background: var(--surface-card);
  color: var(--text);
  padding: 0.7rem 1rem;
  border-radius: 2px 2px 2px 0;
  font-size: 0.87rem;
  max-width: 82%;
  border: 1px solid var(--border);
  border-left: 3px solid var(--primary);
  line-height: 1.6;
}
.chat-label {
  font-family: var(--mono);
  font-size: 0.56rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text-dim);
  margin-bottom: 0.2rem;
}
.chat-hint {
  background: #f8fafc;
  border: 1px solid var(--border);
  padding: 0.7rem 1rem;
  margin-bottom: 1rem;
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--text-dim);
  line-height: 1.6;
}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
  font-family: var(--mono) !important;
  font-size: 0.7rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  border-radius: 2px !important;
  padding: 0.5rem 1.2rem !important;
  transition: all 0.12s !important;
  border: 2px solid var(--text) !important;
  background: var(--text) !important;
  color: white !important;
}
.stButton > button:hover {
  background: white !important;
  color: var(--text) !important;
}
.stDownloadButton > button {
  font-family: var(--mono) !important;
  font-size: 0.7rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  border-radius: 2px !important;
  border: 2px solid var(--border-strong) !important;
  background: transparent !important;
  color: var(--text) !important;
}

/* ── Form labels ──────────────────────────────────────────── */
.stSelectbox label, .stSlider label,
.stMultiSelect label, .stCheckbox label,
.stFileUploader label, .stTextInput label {
  font-family: var(--mono) !important;
  font-size: 0.65rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: var(--text-mid) !important;
}

/* ── Expander ─────────────────────────────────────────────── */
.streamlit-expanderHeader {
  font-family: var(--mono) !important;
  font-size: 0.7rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
}

/* ── Dataframe ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
}

/* ── Caption ──────────────────────────────────────────────── */
.stCaption {
  font-family: var(--mono) !important;
  font-size: 0.68rem !important;
  color: var(--text-dim) !important;
}

hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1.25rem 0 !important;
}

/* ── Analysis config panel ────────────────────────────────── */
.analysis-panel {
  background: #f8fafc;
  border: 1px solid var(--border);
  padding: 1rem 1.25rem 0.25rem 1.25rem;
  margin-bottom: 0.75rem;
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
    active = st.session_state.df_edited if st.session_state.df_edited is not None else st.session_state.df
    if active is None:
        return
    num_cols, cat_cols, txt_cols = detect_column_types(active)
    st.session_state.numerical_cols = num_cols
    st.session_state.categorical_cols = cat_cols
    st.session_state.text_cols = txt_cols
    st.session_state.selected_num = num_cols[:8]
    st.session_state.auto_visuals = generate_auto_visuals(active, num_cols, cat_cols)
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
]


def call_ai(messages: list, system_prompt: str = "") -> tuple:
    api_key = get_api_key()
    if not api_key:
        return ("AI not configured. Add OPENROUTER_API_KEY in Streamlit Secrets.", True)
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
            elif resp.status_code in (401, 403):
                return "Invalid API key. Check your OpenRouter key.", True
        except Exception:
            continue
    if rate_limited >= len(FREE_MODELS):
        return "All AI models are busy. Please wait and try again.", True
    return "Could not get a response. Please try rephrasing.", True


# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
    for col in df.columns:
        if df[col].dtype == object and any(kw in col.lower()
           for kw in ["date", "time", "datetime", "timestamp"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


def detect_column_types(df: pd.DataFrame):
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.select_dtypes(include=["object", "category"]).columns
                   if df[c].nunique() <= 50]
    text = [c for c in df.select_dtypes(include=["object"]).columns
            if df[c].nunique() > 50 or (df[c].str.len().mean() > 20)]
    text = [c for c in text if c not in categorical]
    return numerical, categorical, text


def get_dataset_context(df, cluster_col=None):
    cols_info = []
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique() <= 20:
            sample_vals = df[c].dropna().unique()[:6].tolist()
            cols_info.append(f"  - '{c}' (text, {df[c].nunique()} unique: {sample_vals})")
        elif pd.api.types.is_numeric_dtype(df[c]):
            cols_info.append(
                f"  - '{c}' (numeric, min={df[c].min():.2f}, max={df[c].max():.2f}, mean={df[c].mean():.2f})")
        else:
            cols_info.append(f"  - '{c}' ({df[c].dtype}, {df[c].nunique()} unique)")
    sample = df.head(3).to_string(index=False)
    ctx = (f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns\nColumns:\n"
           + "\n".join(cols_info) + f"\n\nSample rows:\n{sample}")
    if cluster_col and cluster_col in df.columns:
        ctx += f"\n\nCluster sizes: {df.groupby(cluster_col).size().to_dict()}"
    return ctx


# ─── Charts & Palette ─────────────────────────────────────────────────────────
PALETTE = ["#0F62FE", "#00B4A2", "#FF6B2B", "#8B5CF6",
           "#EC4899", "#F59E0B", "#10B981", "#6366F1"]
BASE_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font_family="IBM Plex Mono", font_color="#0f172a",
    margin=dict(t=36, b=40, l=40, r=20),
)


def apply_base(fig, height=380):
    fig.update_layout(**BASE_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", linecolor="#e2e8f0")
    return fig


def generate_auto_visuals(df, numerical_cols, categorical_cols):
    visuals = []
    for col in categorical_cols[:2]:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "Count"]
        fig = px.bar(counts.head(15), x=col, y="Count", color=col,
                     color_discrete_sequence=PALETTE, title=f"Distribution — {col}")
        fig.update_layout(**BASE_LAYOUT, height=340, showlegend=False)
        fig.update_xaxes(gridcolor="#f1f5f9")
        fig.update_yaxes(gridcolor="#f1f5f9")
        visuals.append((f"Distribution — {col}", fig))
    for col in numerical_cols[:3]:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution — {col}",
                           color_discrete_sequence=[PALETTE[0]])
        fig.update_layout(**BASE_LAYOUT, height=340)
        fig.update_xaxes(gridcolor="#f1f5f9")
        fig.update_yaxes(gridcolor="#f1f5f9")
        visuals.append((f"Distribution — {col}", fig))
    if categorical_cols and numerical_cols:
        cat_col, num_col = categorical_cols[0], numerical_cols[0]
        fig = px.box(df, x=cat_col, y=num_col, color=cat_col,
                     color_discrete_sequence=PALETTE, title=f"{num_col} by {cat_col}")
        fig.update_layout(**BASE_LAYOUT, height=340, showlegend=False)
        fig.update_xaxes(gridcolor="#f1f5f9")
        fig.update_yaxes(gridcolor="#f1f5f9")
        visuals.append((f"{num_col} by {cat_col}", fig))
    if len(numerical_cols) >= 2:
        fig = px.scatter(df, x=numerical_cols[0], y=numerical_cols[1],
                         color=categorical_cols[0] if categorical_cols else None,
                         color_discrete_sequence=PALETTE,
                         title=f"{numerical_cols[0]} vs {numerical_cols[1]}")
        fig.update_traces(marker=dict(size=6, opacity=0.65))
        fig.update_layout(**BASE_LAYOUT, height=360)
        fig.update_xaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
        fig.update_yaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
        visuals.append((f"{numerical_cols[0]} vs {numerical_cols[1]}", fig))
    if len(numerical_cols) >= 3:
        corr = df[numerical_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", title="Correlation Heatmap")
        fig.update_layout(**BASE_LAYOUT, height=400)
        visuals.append(("Correlation Heatmap", fig))
    return visuals


# ─── Preprocessing & Clustering ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def preprocess(df_json, numerical_cols, categorical_cols):
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
                df_proc[col].mode()[0] if not df_proc[col].mode().empty else "unknown")
            df_proc[col + "_enc"] = le.fit_transform(df_proc[col].astype(str))
    encoded_cats = [c + "_enc" for c in categorical_cols if c in df_proc.columns]
    feature_cols = list(numerical_cols) + encoded_cats
    if not feature_cols:
        return pd.DataFrame(), []
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_proc[feature_cols])
    return pd.DataFrame(scaled, columns=feature_cols, index=df.index), feature_cols


@st.cache_data(show_spinner=False)
def run_kmeans(scaled_json, n_clusters):
    scaled_df = pd.read_json(io.StringIO(scaled_json))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_df)
    score = silhouette_score(scaled_df, labels) if n_clusters > 1 else 0.0
    return labels.tolist(), float(score), float(km.inertia_)


@st.cache_data(show_spinner=False)
def run_dbscan(scaled_json, eps, min_samples):
    scaled_df = pd.read_json(io.StringIO(scaled_json))
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(scaled_df)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    score = silhouette_score(scaled_df, labels) if n_clusters > 1 else 0.0
    return labels.tolist(), n_clusters, float(score)


@st.cache_data(show_spinner=False)
def compute_elbow(scaled_json, max_k=10):
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
def compute_pca(scaled_json):
    scaled_df = pd.read_json(io.StringIO(scaled_json))
    n = min(2, scaled_df.shape[1])
    pca = PCA(n_components=n, random_state=42)
    coords = pca.fit_transform(scaled_df)
    return coords.tolist(), pca.explained_variance_ratio_.tolist()


def plot_elbow(ks, inertias, scores):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia",
                             line=dict(color="#0f172a", width=2), marker=dict(size=7)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=ks, y=scores, mode="lines+markers", name="Silhouette",
                             line=dict(color="#0F62FE", width=2, dash="dot"),
                             marker=dict(size=7, color="#0F62FE")),
                  secondary_y=True)
    fig.update_layout(title="Elbow Curve + Silhouette Score", xaxis_title="k",
                      yaxis_title="Inertia", yaxis2_title="Silhouette Score",
                      legend=dict(x=0.7, y=0.95), **BASE_LAYOUT, height=340)
    return fig


# ─── Code sandbox ─────────────────────────────────────────────────────────────
BLOCKED = ["os.", "sys.", "subprocess", "open(", "__import__",
           "importlib", "shutil", "socket", "requests", "eval(", "exec("]


def clean_ai_code(code):
    code = re.sub(r"```(?:python)?", "", code)
    return code.replace("```", "").strip()


def execute_ai_code(code, df):
    code = clean_ai_code(code)
    if not code:
        return None, "No executable code returned."
    for token in BLOCKED:
        if token in code:
            return None, "Code blocked for safety reasons."
    df_exec = df.copy()
    df_exec.columns = [c.strip() for c in df_exec.columns]
    local_ns = {"df": df_exec, "pd": pd, "np": np,
                "px": px, "go": go, "make_subplots": make_subplots}
    try:
        exec(compile(code, "<ai_code>", "exec"), {"__builtins__": {}}, local_ns)
    except ValueError as e:
        err = str(e)
        if "not the name of a column" in err or "Expected one of" in err:
            return None, f"Column name mismatch. Available: {df_exec.columns.tolist()}"
        return None, f"Chart error: {err}"
    except KeyError as e:
        return None, f"Column {e} not found. Available: {df_exec.columns.tolist()}"
    except Exception as e:
        return None, f"Chart could not be generated: {str(e)}"
    fig = local_ns.get("fig")
    if fig is None:
        return None, "No chart produced. Try rephrasing your request."
    return fig, None


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## InsightForge AI")
    st.markdown("Segmentation platform")
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
            st.session_state.auto_visuals = generate_auto_visuals(df_raw, num_cols, cat_cols)

        if (st.session_state.df is not None
                and not st.session_state.numerical_cols
                and not st.session_state.categorical_cols):
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
        if ca.button("Clear", use_container_width=True):
            reset_all()
            st.rerun()
        if cb.button("Reset Segs", use_container_width=True):
            reset_clustering()
            st.rerun()

    st.markdown("---")
    st.markdown("**Segmentation**")
    model_choice = st.selectbox("Method", ["Auto (K-Means)", "Density-based (DBSCAN)"])

    k_val, eps_val, min_s = None, 0.5, 5
    if model_choice == "Auto (K-Means)":
        auto_k = st.checkbox("Auto-detect group count", value=True)
        if not auto_k:
            k_val = st.slider("Number of groups", 2, 15, 4)
    else:
        eps_val = st.slider("Sensitivity (eps)", 0.1, 5.0, 0.5, 0.1)
        min_s = st.slider("Min group size", 2, 20, 5)

    st.markdown("---")
    st.markdown("**Feature Selection**")
    num_options = st.session_state.numerical_cols
    selected_num = []
    if num_options:
        valid_defaults = [c for c in st.session_state.selected_num if c in num_options]
        if not valid_defaults:
            valid_defaults = num_options[:8]
        selected_num = st.multiselect("Numerical fields", options=num_options,
                                      default=valid_defaults)
        st.session_state.selected_num = selected_num
    elif st.session_state.df is not None:
        st.caption("No numerical fields detected.")

    st.markdown("")
    run_btn = st.button("Run Segmentation", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="if-header">
  <div>
    <div class="if-wordmark">Insight<span>Forge</span> AI</div>
    <div class="if-tagline">Customer segmentation &amp; data analysis</div>
  </div>
  <div class="if-badge">v2.0 &nbsp;&bull;&nbsp; Industrial</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.df is None:
    st.markdown("""
    <div style="border:1px solid #e2e8f0;border-left:4px solid #0F62FE;
                padding:1.75rem 2rem;background:white;margin-top:1rem;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;font-weight:600;
                  text-transform:uppercase;letter-spacing:0.12em;color:#94a3b8;margin-bottom:0.5rem;">
        Getting Started
      </div>
      <div style="font-size:0.95rem;color:#0f172a;font-weight:500;font-family:'IBM Plex Sans',sans-serif;
                  margin-bottom:0.4rem;">
        Upload a CSV file from the sidebar to begin analysis.
      </div>
      <div style="font-size:0.82rem;color:#64748b;font-family:'IBM Plex Sans',sans-serif;">
        Column types are detected automatically. The platform will generate overview charts, 
        prepare features for segmentation, and enable the AI chat assistant.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df_edited if st.session_state.df_edited is not None else st.session_state.df
numerical_cols = st.session_state.numerical_cols
categorical_cols = st.session_state.categorical_cols

# ─── Run segmentation ─────────────────────────────────────────────────────────
if run_btn:
    if not selected_num:
        st.sidebar.error("Select at least one numerical feature.")
    else:
        with st.spinner("Preprocessing..."):
            scaled_df, feature_cols = preprocess(df.to_json(), tuple(selected_num), tuple(categorical_cols))
        if scaled_df.empty:
            st.error("Could not build feature matrix. Check numerical columns.")
        else:
            scaled_json = scaled_df.to_json()
            st.session_state.scaled_df = scaled_df
            if model_choice == "Auto (K-Means)":
                with st.spinner("Computing elbow curve..."):
                    ks, inertias, sil_scores = compute_elbow(scaled_json)
                st.session_state.ks = ks
                st.session_state.inertias = inertias
                st.session_state.sil_scores = sil_scores
                best_k = ks[int(np.argmax(sil_scores))] if (auto_k and ks) else (k_val or 3)
                with st.spinner(f"Running K-Means (k={best_k})..."):
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
            st.success(f"Segmentation complete — {n_found} groups found.")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Segments", "Analysis", "AI Chat"])


# ══════════ TAB 1: OVERVIEW ═══════════════════════════════════════════════════
with tab1:
    is_edited = st.session_state.df_edited is not None
    edited_tag = (' &nbsp;<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.58rem;'
                  'font-weight:600;letter-spacing:0.08em;text-transform:uppercase;padding:2px 6px;'
                  'border-radius:2px;background:#dbeafe;color:#1d4ed8;">Edited</span>'
                  if is_edited else "")

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card accent">
        <div class="m-label">Total Records</div>
        <div class="m-value">{df.shape[0]:,}</div>
        <div class="m-sub">rows</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Fields</div>
        <div class="m-value">{df.shape[1]}</div>
        <div class="m-sub">columns</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Numerical</div>
        <div class="m-value">{len(numerical_cols)}</div>
        <div class="m-sub">numeric fields</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Categorical</div>
        <div class="m-value">{len(categorical_cols)}</div>
        <div class="m-sub">category fields</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="if-section">Data Preview{edited_tag}</div>', unsafe_allow_html=True)
    if is_edited:
        st.caption("Working on an edited version of the original dataset.")
    st.dataframe(df.head(200), use_container_width=True, height=310)
    if is_edited:
        st.download_button("Download Edited CSV", df.to_csv(index=False).encode(),
                           "edited_data.csv", "text/csv")

    with st.expander("Data Actions", expanded=False):
        st.caption("Clean and reshape data. Changes propagate across all tabs and reset segmentation.")
        active_df = st.session_state.df_edited if st.session_state.df_edited is not None else st.session_state.df
        ea1, _ = st.columns([2, 1])
        action = ea1.selectbox("Action", [
            "Select an action...",
            "Remove duplicate rows", "Drop rows with missing values",
            "Fill missing — numeric (mean)", "Fill missing — numeric (median)",
            "Fill missing — numeric (zero)", "Fill missing — text (blank)",
            "Rename a column", "Drop a column", "Change column type",
            "Filter rows by value", "Reset all edits",
        ], key="edit_action")

        if action == "Remove duplicate rows":
            n_dups = int(active_df.duplicated().sum())
            st.caption(f"{n_dups} duplicate rows detected.")
            if st.button("Remove duplicates"):
                result = active_df.drop_duplicates().reset_index(drop=True)
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success(f"Removed {n_dups} duplicate rows.")
                st.rerun()

        elif action == "Drop rows with missing values":
            n_missing = int(active_df.isnull().any(axis=1).sum())
            st.caption(f"{n_missing} rows have at least one missing value.")
            if st.button("Drop rows"):
                result = active_df.dropna().reset_index(drop=True)
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success(f"Removed {n_missing} rows.")
                st.rerun()

        elif action == "Fill missing — numeric (mean)":
            if st.button("Fill with mean"):
                result = active_df.copy()
                for c in result.select_dtypes(include=[np.number]).columns:
                    result[c] = result[c].fillna(result[c].mean())
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success("Filled missing numeric values with mean.")
                st.rerun()

        elif action == "Fill missing — numeric (median)":
            if st.button("Fill with median"):
                result = active_df.copy()
                for c in result.select_dtypes(include=[np.number]).columns:
                    result[c] = result[c].fillna(result[c].median())
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success("Filled missing numeric values with median.")
                st.rerun()

        elif action == "Fill missing — numeric (zero)":
            if st.button("Fill with zero"):
                result = active_df.copy()
                for c in result.select_dtypes(include=[np.number]).columns:
                    result[c] = result[c].fillna(0)
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success("Filled missing numeric values with 0.")
                st.rerun()

        elif action == "Fill missing — text (blank)":
            if st.button("Fill with blank"):
                result = active_df.copy()
                for c in result.select_dtypes(include=["object", "string"]).columns:
                    result[c] = result[c].fillna("")
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success("Filled missing text values with blank.")
                st.rerun()

        elif action == "Rename a column":
            rb1, rb2 = st.columns(2)
            col_to_rename = rb1.selectbox("Column", active_df.columns.tolist(), key="rename_col")
            new_name = rb2.text_input("New name", value=col_to_rename, key="rename_val")
            if st.button("Rename"):
                new_name = new_name.strip()
                if new_name and new_name != col_to_rename:
                    result = active_df.rename(columns={col_to_rename: new_name})
                    st.session_state.df_edited = result
                    apply_edits_to_pipeline()
                    st.success(f"Renamed '{col_to_rename}' → '{new_name}'.")
                    st.rerun()
                else:
                    st.warning("Enter a different name.")

        elif action == "Drop a column":
            col_to_drop = st.selectbox("Column to drop", active_df.columns.tolist(), key="drop_col")
            if st.button("Drop column"):
                result = active_df.drop(columns=[col_to_drop])
                st.session_state.df_edited = result
                apply_edits_to_pipeline()
                st.success(f"Removed '{col_to_drop}'.")
                st.rerun()

        elif action == "Change column type":
            ct1, ct2 = st.columns(2)
            col_to_cast = ct1.selectbox("Column", active_df.columns.tolist(), key="cast_col")
            target_type = ct2.selectbox("Convert to", ["Numeric", "Text", "Date/Time"], key="cast_type")
            st.caption(f"Current type: {active_df[col_to_cast].dtype}")
            if st.button("Convert"):
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
                keep_vals = st.multiselect("Keep where value is", options=unique_vals,
                                           default=unique_vals, key="filter_vals")
                if st.button("Apply filter"):
                    result = active_df[active_df[filter_col].isin(keep_vals)].reset_index(drop=True)
                    st.session_state.df_edited = result
                    apply_edits_to_pipeline()
                    st.success(f"Filtered to {len(result):,} rows.")
                    st.rerun()
            else:
                st.caption("Too many unique values. Use AI Chat to filter with a custom condition.")

        elif action == "Reset all edits":
            st.caption("Discard all changes and restore the original uploaded data.")
            if st.button("Reset to original"):
                st.session_state.df_edited = None
                apply_edits_to_pipeline()
                st.success("Restored to original data.")
                st.rerun()

    st.markdown('<div class="if-section">Field Summary</div>', unsafe_allow_html=True)
    schema_rows = []
    for c in df.columns:
        schema_rows.append({
            "Field": c, "Type": str(df[c].dtype),
            "Non-null": int(df[c].notna().sum()),
            "Missing": int(df[c].isna().sum()),
            "Missing %": f"{df[c].isna().mean()*100:.1f}%",
            "Unique": int(df[c].nunique()),
        })
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

    if numerical_cols:
        st.markdown('<div class="if-section">Summary Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df[numerical_cols].describe().round(3), use_container_width=True)


# ══════════ TAB 2: SEGMENTS ═══════════════════════════════════════════════════
with tab2:
    if not st.session_state.clustering_done:
        if not numerical_cols:
            st.warning("No numerical fields detected. Segmentation requires numerical data.")
        else:
            st.markdown("""
            <div style="border:1px solid #e2e8f0;border-left:4px solid #0F62FE;
                        padding:1.5rem 1.75rem;background:white;margin-top:0.5rem;">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;font-weight:600;
                          text-transform:uppercase;letter-spacing:0.12em;color:#94a3b8;margin-bottom:0.4rem;">
                Ready
              </div>
              <div style="font-size:0.92rem;color:#0f172a;font-weight:500;
                          font-family:'IBM Plex Sans',sans-serif;margin-bottom:0.3rem;">
                Configure and run segmentation from the sidebar.
              </div>
              <div style="font-size:0.8rem;color:#64748b;font-family:'IBM Plex Sans',sans-serif;">
                Select numerical features, choose a method (K-Means or DBSCAN), then click 
                Run Segmentation. Results will appear here.
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        labels = st.session_state.cluster_labels
        df_clustered = st.session_state.df_clustered
        scaled_df = st.session_state.scaled_df

        n_groups = len(set(labels)) - (1 if -1 in labels else 0)
        sil_final = silhouette_score(scaled_df, labels) if n_groups > 1 else 0.0

        if sil_final >= 0.7:
            sep_label, sep_pill = "Very well separated", "pill-green"
        elif sil_final >= 0.5:
            sep_label, sep_pill = "Well separated", "pill-blue"
        elif sil_final >= 0.3:
            sep_label, sep_pill = "Moderately separated", "pill-yellow"
        else:
            sep_label, sep_pill = "Overlapping", "pill-red"

        sep_html = (f'<span class="pill {sep_pill}" style="font-family:\'IBM Plex Mono\',monospace;'
                    f'font-size:0.68rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;'
                    f'padding:3px 9px;border-radius:2px;">{sep_label}</span>')

        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-card accent">
            <div class="m-label">Groups Found</div>
            <div class="m-value">{n_groups}</div>
            <div class="m-sub">distinct segments</div>
          </div>
          <div class="metric-card">
            <div class="m-label">Group Separation</div>
            <div class="m-value" style="font-size:1rem;padding-top:0.5rem;">{sep_html}</div>
            <div class="m-sub">silhouette: {sil_final:.3f}</div>
          </div>
          <div class="metric-card">
            <div class="m-label">Total Records</div>
            <div class="m-value">{len(df_clustered):,}</div>
            <div class="m-sub">segmented</div>
          </div>
          <div class="metric-card">
            <div class="m-label">Features Used</div>
            <div class="m-value">{len(selected_num)}</div>
            <div class="m-sub">numerical fields</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Group distribution
        st.markdown('<div class="if-section">Records per Group</div>', unsafe_allow_html=True)
        size_data = pd.Series(labels).value_counts().sort_index().reset_index()
        size_data.columns = ["Group", "Records"]
        size_data["Group"] = size_data["Group"].apply(lambda x: f"Group {x + 1}")
        size_data["Share"] = (size_data["Records"] / size_data["Records"].sum() * 100).round(1)
        size_data["Label"] = size_data.apply(lambda r: f"{r['Records']:,}  ({r['Share']}%)", axis=1)
        fig_sizes = px.bar(size_data, x="Group", y="Records", color="Group",
                           text="Label", color_discrete_sequence=PALETTE)
        fig_sizes.update_traces(textposition="outside")
        fig_sizes.update_layout(**BASE_LAYOUT, height=320, showlegend=False)
        fig_sizes.update_xaxes(gridcolor="#f1f5f9")
        fig_sizes.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_sizes, use_container_width=True)

        # PCA map
        if st.session_state.scaled_df is not None:
            st.markdown('<div class="if-section">Customer Map — PCA Projection</div>', unsafe_allow_html=True)
            st.caption("Each point is one record. Proximity indicates similarity. Colour indicates group.")
            pca_coords, _ = compute_pca(scaled_df.to_json())
            arr = np.array(pca_coords)
            df_pca = pd.DataFrame({
                "Dimension 1": arr[:, 0],
                "Dimension 2": arr[:, 1],
                "Group": [f"Group {l + 1}" for l in labels],
            })
            fig_pca = px.scatter(df_pca, x="Dimension 1", y="Dimension 2", color="Group",
                                 color_discrete_sequence=PALETTE)
            fig_pca.update_traces(marker=dict(size=7, opacity=0.72))
            fig_pca = apply_base(fig_pca, 440)
            st.plotly_chart(fig_pca, use_container_width=True)

        # Group profiles
        valid_num = [c for c in selected_num if c in df_clustered.columns]
        if valid_num:
            st.markdown('<div class="if-section">Group Profiles — Mean Values</div>', unsafe_allow_html=True)
            profile_raw = df_clustered.groupby("Cluster")[valid_num].mean().round(2)
            profile_raw.index = [f"Group {i + 1}" for i in profile_raw.index]
            profile_raw.index.name = "Group"
            st.dataframe(profile_raw, use_container_width=True)

            st.markdown('<div class="if-section">Key Distinctions</div>', unsafe_allow_html=True)
            overall_means = df_clustered[valid_num].mean()
            for cluster in sorted(set(labels)):
                subset = df_clustered[df_clustered["Cluster"] == cluster]
                size = len(subset)
                pct = round(100 * size / len(df_clustered), 1)
                diffs = []
                for col in valid_num[:6]:
                    cm, om = subset[col].mean(), overall_means[col]
                    if om == 0:
                        continue
                    diff = (cm - om) / abs(om) * 100
                    if abs(diff) > 15:
                        direction = "higher" if diff > 0 else "lower"
                        diffs.append(f"{col}: {abs(diff):.0f}% {direction} ({cm:.2f} vs avg {om:.2f})")
                detail = " &nbsp;&bull;&nbsp; ".join(diffs) if diffs else "Close to average across all fields."
                st.markdown(f"""
                <div class="insight-box">
                  <div class="ib-label">Group {cluster + 1} &mdash; {size:,} records ({pct}%)</div>
                  <div class="ib-text">{detail}</div>
                </div>
                """, unsafe_allow_html=True)

        if model_choice == "Auto (K-Means)" and st.session_state.ks:
            with st.expander("Group count selection — Elbow curve + Silhouette"):
                st.plotly_chart(plot_elbow(
                    st.session_state.ks, st.session_state.inertias, st.session_state.sil_scores
                ), use_container_width=True)

        st.markdown('<div class="if-section">Full Segmented Dataset</div>', unsafe_allow_html=True)
        display_df = df_clustered.copy()
        display_df["Group"] = display_df["Cluster"].apply(lambda x: f"Group {x + 1}")
        display_df = display_df.drop(columns=["Cluster"])
        st.dataframe(display_df, use_container_width=True, height=310)
        st.download_button("Download Segmented CSV",
                           display_df.to_csv(index=False).encode(),
                           "segmented_data.csv", "text/csv")


# ══════════ TAB 3: ANALYSIS ═══════════════════════════════════════════════════
with tab3:
    all_cols = df.columns.tolist()
    num_cols_viz = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols_viz = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df[c])]

    st.markdown('<div class="if-section">Chart Builder</div>', unsafe_allow_html=True)

    CHART_TYPES = ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram",
                   "Box Plot", "Pie Chart", "Area Chart", "Heatmap (Correlation)"]
    AGGREGATIONS = ["Sum", "Average", "Count", "Count Distinct", "Min", "Max", "Median"]

    st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns([1, 1, 1])
    chart_type = cc1.selectbox("Chart Type", CHART_TYPES, key="vz_chart_type")
    color_col = cc3.selectbox("Colour by", ["None"] + cat_cols_viz, key="vz_color")
    st.markdown("</div>", unsafe_allow_html=True)

    x_col, y_col, agg_func = None, None, "Count"

    if chart_type == "Histogram":
        x_col = cc2.selectbox("Field", num_cols_viz or all_cols, key="vz_x")
        bins = st.slider("Bins", 5, 100, 30, key="vz_bins")

    elif chart_type == "Pie Chart":
        x_col = cc2.selectbox("Category (slices)", cat_cols_viz or all_cols, key="vz_x")
        if num_cols_viz:
            y_col = st.selectbox("Value field", ["Count records"] + num_cols_viz, key="vz_y_pie")
            if y_col == "Count records":
                y_col = None
            agg_func = st.selectbox("Aggregation", AGGREGATIONS, key="vz_agg_pie") if y_col else "Count"
        else:
            y_col = None

    elif chart_type == "Heatmap (Correlation)":
        selected_for_corr = st.multiselect("Fields for correlation", options=num_cols_viz,
                                            default=num_cols_viz[:8], key="vz_corr_cols")

    elif chart_type == "Scatter Plot":
        fc1, fc2, fc3 = st.columns(3)
        x_col = fc1.selectbox("X axis", num_cols_viz or all_cols, key="vz_x")
        y_col = fc2.selectbox("Y axis", [c for c in (num_cols_viz or all_cols)
                                          if c != x_col] or all_cols, key="vz_y")
        size_col = fc3.selectbox("Size by", ["None"] + num_cols_viz, key="vz_size")

    else:
        fc1, fc2, fc3 = st.columns(3)
        if chart_type == "Box Plot":
            x_col = fc1.selectbox("Category (X)", cat_cols_viz or all_cols, key="vz_x")
            y_col = fc2.selectbox("Value (Y)", num_cols_viz or all_cols, key="vz_y")
        elif chart_type in ["Line Chart", "Area Chart"]:
            x_col = fc1.selectbox("X axis", all_cols, key="vz_x")
            y_col = fc2.selectbox("Y axis", num_cols_viz or all_cols, key="vz_y")
            agg_func = fc3.selectbox("Aggregation", AGGREGATIONS, key="vz_agg")
        else:
            x_col = fc1.selectbox("X axis", all_cols, key="vz_x")
            y_col = fc2.selectbox("Y axis", ["Count records"] + num_cols_viz, key="vz_y_bar")
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
        agg_map = {"Sum": "sum", "Average": "mean", "Min": "min", "Max": "max",
                   "Median": "median", "Count Distinct": "nunique", "Count": "count"}
        agg_df = df.groupby(group_cols)[y_col].agg(agg_map[agg_func]).reset_index()
        agg_df.columns = group_cols + [f"{agg_func} of {y_col}"]
        return agg_df, f"{agg_func} of {y_col}"

    chart_error, fig_vz = None, None
    try:
        color_val = color_col if color_col != "None" and color_col in df.columns else None
        if chart_type == "Histogram":
            fig_vz = px.histogram(df, x=x_col, nbins=bins, color=color_val,
                                  color_discrete_sequence=PALETTE, title=f"Distribution — {x_col}")
        elif chart_type == "Pie Chart":
            if y_col and agg_func != "Count":
                agg_map_pie = {"Sum": "sum", "Average": "mean", "Count": "count", "Min": "min",
                               "Max": "max", "Median": "median", "Count Distinct": "nunique"}
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
                chart_error = "Select at least 2 fields."
            else:
                corr = df[selected_for_corr].corr()
                fig_vz = px.imshow(corr, text_auto=".2f", aspect="auto",
                                   color_continuous_scale="RdBu_r", title="Correlation Heatmap")
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
        if chart_type not in ["Pie Chart", "Heatmap (Correlation)"]:
            fig_vz.update_xaxes(gridcolor="#f1f5f9")
            fig_vz.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_vz, use_container_width=True)

    auto_visuals = st.session_state.auto_visuals
    if auto_visuals:
        st.markdown('<div class="if-section">Auto-Generated Overview</div>', unsafe_allow_html=True)
        st.caption("Generated automatically on upload from detected column patterns.")
        for title, fig_auto in auto_visuals:
            st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;'
                        f'font-weight:600;color:#475569;margin:1rem 0 0.25rem 0;">{title}</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_auto, use_container_width=True)


# ══════════ TAB 4: AI CHAT ════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="if-section">AI Chat Assistant</div>', unsafe_allow_html=True)

    if not get_api_key():
        st.error(
            "No API key configured. Go to Manage App → Settings → Secrets and add:\n\n"
            "`OPENROUTER_API_KEY = \"your-key-here\"`\n\n"
            "Get a free key at openrouter.ai"
        )
        st.stop()

    working_df = (
        st.session_state.df_clustered if st.session_state.df_clustered is not None
        else (st.session_state.df_edited if st.session_state.df_edited is not None else df)
    )
    cluster_col = "Cluster" if st.session_state.df_clustered is not None else None
    exact_cols = [c.strip() for c in working_df.columns.tolist()]

    st.markdown("""
    <div class="chat-hint">
      Ask questions about your data or request charts.
      Examples: "What patterns do you see?" &bull; "Show a bar chart of Sentiment by Location" &bull;
      "Which group has the highest average spend?"
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.chat_history:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-row-user">
              <div>
                <div class="chat-label" style="text-align:right;">You</div>
                <div class="chat-bubble-user">{msg["content"]}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if msg.get("is_error"):
                st.warning(msg["content"])
            elif msg.get("is_code"):
                st.markdown(f"""
                <div class="chat-row-ai">
                  <div style="max-width:82%;">
                    <div class="chat-label">InsightForge AI</div>
                    <div class="chat-bubble-ai">{msg.get("text_response", "Here is the chart:")}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                if msg.get("fig") is not None:
                    st.plotly_chart(msg["fig"], use_container_width=True)
                elif msg.get("exec_error"):
                    st.markdown(f"""
                    <div class="chat-row-ai">
                      <div style="max-width:82%;">
                        <div class="chat-label">InsightForge AI</div>
                        <div class="chat-bubble-ai" style="border-left-color:#dc2626;">
                          {msg["exec_error"]}
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-row-ai">
                  <div style="max-width:82%;">
                    <div class="chat-label">InsightForge AI</div>
                    <div class="chat-bubble-ai">{msg["content"]}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    user_input = st.chat_input("Ask a question or request a chart...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        ctx = get_dataset_context(working_df, cluster_col)

        viz_keywords = ["plot", "chart", "graph", "show", "visualize", "draw",
                        "bar", "scatter", "histogram", "heatmap", "distribution",
                        "figure", "map", "pie", "line"]
        wants_viz = any(kw in user_input.lower() for kw in viz_keywords)

        if wants_viz:
            system_prompt = textwrap.dedent(f"""
                You are a data analyst. The user wants a chart.
                First write ONE sentence describing what you are plotting.
                Then write ONLY the Python code block.

                The dataset is a pandas DataFrame called `df`.
                The final Plotly figure MUST be stored in `fig`.

                Dataset context:
                {ctx}

                EXACT column names (copy verbatim):
                {exact_cols}

                Code rules:
                - Only use: df, pd, np, px, go, make_subplots (all pre-imported).
                - Do NOT import anything.
                - Column names must exactly match the list above.
                - Always store the chart in `fig`.
                - Apply: fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                  font_family='IBM Plex Mono', font_color='#0f172a')

                Format:
                [One sentence description]

                ```python
                [code here]
                ```
            """).strip()
        else:
            system_prompt = textwrap.dedent(f"""
                You are a concise data analyst. Answer in plain English using actual 
                numbers from the dataset. 3-5 sentences max. No code.

                Dataset context:
                {ctx}
            """).strip()

        with st.spinner("Thinking..."):
            response, is_error = call_ai([{"role": "user", "content": user_input}], system_prompt)

        if is_error:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response, "is_error": True})
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
                "role": "assistant", "is_code": True,
                "text_response": text_only, "content": code_only,
                "fig": fig_result, "exec_error": exec_error,
            })
        else:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response, "is_code": False})

        st.rerun()
