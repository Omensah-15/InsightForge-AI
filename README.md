# InsightForge AI

Most businesses sit on customer data they never fully use. InsightForge AI changes that. Upload any customer dataset and the app automatically segments your customers into distinct groups based on their behavior, spending, and demographics.

Ask questions in plain English. Get charts on demand. Understand your customers in minutes.

**Live App:** [insightforge-ai](https://insightforge-ai-3s9iwssbrkg3uf598xq5om.streamlit.app/)

---

## What It Does

- Automatically detects numerical, categorical, and text fields in any CSV
- Segments customers into groups using machine learning, with automatic selection of the optimal number of groups
- Generates charts and visual summaries the moment data is uploaded
- Provides an interactive chart builder — select field, chart type, and aggregation
- Includes a data editing panel for cleaning and reshaping data before analysis
- AI chat assistant answers questions about your data and generates charts on request

---

## Stack

- [Streamlit](https://streamlit.io) — application framework
- [scikit-learn](https://scikit-learn.org) — segmentation, dimensionality reduction, preprocessing
- [Plotly](https://plotly.com/python) — interactive charts
- [OpenRouter](https://openrouter.ai) — AI inference API

---

## Local Setup

```bash
git clone https://github.com/your-username/insightforge-ai.git
cd insightforge-ai

pip install -r requirements.txt

cp .env.example .env
# Paste your OpenRouter API key into .env

streamlit run app.py
```

---

## API Key

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Free keys are available at [openrouter.ai](https://openrouter.ai). No credit card required.

For Streamlit Cloud deployment, add the key under app Settings → Secrets:

```toml
OPENROUTER_API_KEY = "sk-or-v1-your-key-here"
```

---

## Usage

1. Upload a CSV file from the sidebar
2. Select the fields to base segmentation on and click Run Segmentation
3. Explore the Overview, Clustering, Visualizations, and AI Chat tabs
4. Use Data Actions in the Overview tab to clean or reshape data before analysis

---

## Author

**Mensah Obed**

[heavenzlebron7@gmail.com](mailto:heavenzlebron7@gmail.com) · [LinkedIn](https://www.linkedin.com/in/obed-mensah-87001237b)
