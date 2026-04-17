# InsightForge AI
 
Most businesses collect customer data, purchase history, demographics, spending habits, and never fully use it. InsightForge AI changes that. Upload your customer dataset and the app automatically groups your customers into meaningful segments based on their behavior and spending patterns with no coding and no data science team required.

You can then ask questions about your customers in plain English. Which group spends the most? Show me income vs spending by segment. The AI answers instantly, complete with charts.

It turns raw customer data into clear segments and actionable insights in minutes.

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

## Local Setup
 
```bash
git clone https://github.com/your-username/segmentiq.git
cd segmentiq
 
pip install -r requirements.txt
 
cp .env.example .env
# Add your OpenRouter API key to .env
 
streamlit run app.py
```
 
---
 
## Environment Variable
 
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```
 
Get a free key at [openrouter.ai](https://openrouter.ai). No credit card required. The app uses `mistralai/mistral-7b-instruct:free` by default.
 
The key can also be pasted directly into the sidebar at runtime without any file changes.
 
---

## Usage
 
1. Upload a CSV file from the sidebar
2. Select features and a clustering algorithm
3. Click **Run Segmentation**
4. Explore results across the Overview, Clustering, Visualizations, and AI Chat tabs

---

## Author

**Mensah Obed**
[Email](mailto:heavenzlebron7@gmail.com) · [LinkedIn](https://www.linkedin.com/in/obed-mensah-87001237b)
