# InsightForge AI
 
A production-ready application for automated customer segmentation and AI-powered data analysis. Upload any CSV, run unsupervised machine learning, and interrogate your data through a natural language chat interface.
 
---
 
## Features
 
- Automatic column type detection (numerical, categorical, text)
- KMeans and DBSCAN clustering with auto-selected optimal k via silhouette scoring
- PCA projection, correlation heatmap, and per-cluster feature distribution charts
- Rule-based insight generation per cluster
- AI chat assistant that answers questions and generates Plotly visualizations on demand
- Sandboxed code execution for AI-generated charts
---
 
## Stack
 
- [Streamlit](https://streamlit.io) — UI framework
- [scikit-learn](https://scikit-learn.org) — KMeans, DBSCAN, PCA, preprocessing
- [Plotly](https://plotly.com/python) — interactive visualizations
- [OpenRouter](https://openrouter.ai) — free AI API (Mistral 7B by default)
---
