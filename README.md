# 🚚 Supply Chain Delay Predictor

> An end-to-end MLOps pipeline that predicts whether an e-commerce shipment will be delayed, with AI-powered explanations and actionable suggestions.

**Live Demo:** https://supply-chain-delay-predictor.onrender.com  
**API Docs:** https://supply-chain-delay-predictor.onrender.com/docs

---

## Architecture
Dataset (Kaggle)
↓
EDA & Feature Engineering (Jupyter Notebook)
↓
Model Training & Comparison (5 Models)
↓
Experiment Tracking (MLflow)
↓
REST API (FastAPI)
↓
AI Explanation Layer (Gemini/Fallback)
↓
Frontend UI (HTML/CSS/JS)
↓
Deployment (Render)
---

## Features

- 📊 **Exploratory Data Analysis** — 4 charts covering all features with business insights
- 🤖 **5 ML Models Compared** — Logistic Regression, Random Forest, XGBoost, Gradient Boosting, Voting Classifier
- 🎯 **Final Model: Gradient Boosting** — Selected for highest recall (86%) on delayed shipments
- 📈 **MLflow Tracking** — All experiments logged, versioned and comparable
- ⚡ **REST API** — FastAPI with /predict, /summarize and /health endpoints
- 🧠 **AI Explanation Layer** — Natural language explanations with actionable suggestions
- 🌐 **Live Deployment** — Publicly accessible on Render

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Frontend UI |
| GET | /health | Health check |
| GET | /docs | Interactive API documentation |
| POST | /predict | Predict shipment delay |
| POST | /summarize | Summarize patterns across multiple shipments |

---

## Model Performance

| Model | Accuracy | Recall (Delayed) | F1 Score |
|-------|----------|-----------------|----------|
| Logistic Regression | 68.73% | 81% | 0.68 |
| Random Forest | 67.73% | 73% | 0.65 |
| XGBoost | 66.00% | 65% | 0.61 |
| **Gradient Boosting** | **68.18%** | **86%** | **0.69** |
| Voting Classifier | 69.05% | 84% | 0.69 |

**Why Gradient Boosting?**  
In supply chain, missing a delay (false negative) is costlier than a false alarm. Gradient Boosting achieves the highest recall of 86% — meaning it correctly identifies 86 out of 100 actual delays.

---

## Key Findings from EDA

- **Discount offered** is the strongest predictor (56.5% feature importance)
- Delayed shipments average only **5.55% discount** vs **18.66%** for on-time
- Delayed shipments are **heavier on average** (4168gms vs 3272gms)
- **Customer care calls** strongly correlate with delays (34% at 2 calls vs 48% at 7 calls)

---

## Project Structure
supply-chain-delay-predictor/
├── notebook.ipynb          ← EDA, feature engineering, model training
├── main.py                 ← FastAPI application
├── mlflow_tracking.py      ← MLflow experiment tracking
├── index.html              ← Frontend UI
├── requirements.txt        ← Dependencies
├── runtime.txt             ← Python version for deployment
├── Procfile                ← Deployment configuration
├── model/
│   ├── final_model.pkl     ← Gradient Boosting (production model)
│   ├── gb_model.pkl        ← Gradient Boosting
│   └── scaler.pkl          ← Feature scaler
├── eda_plot1.png           ← Delay by shipment mode & warehouse
├── eda_plot2.png           ← Weight & discount vs delay
├── eda_plot3.png           ← Customer care calls & prior purchases
└── eda_plot4.png           ← Feature importance chart
---

## Local Setup

```bash
# Clone the repo
git clone https://github.com/SINCHANA2809/supply-chain-delay-predictor.git
cd supply-chain-delay-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn main:app --reload

# Open in browser
http://127.0.0.1:8000
```

---

## Example API Request

```json
POST /predict
{
  "warehouse_block": "F",
  "mode_of_shipment": "Ship",
  "customer_care_calls": 6,
  "customer_rating": 2,
  "cost_of_product": 250,
  "prior_purchases": 3,
  "product_importance": "Low",
  "gender": "M",
  "discount_offered": 5,
  "weight_in_gms": 4500
}
```

## Example Response

```json
{
  "prediction": 0,
  "status": "DELAYED",
  "confidence": 57.54,
  "explanation": "This shipment is predicted DELAYED due to very low discount (5%), high product weight (4500gms). Recommended action: Increase discount offered to incentivise faster processing.",
  "input_received": {...}
}
```

---

## Future Improvements

- Integration of regional holiday calendars
- Real-time weather flags for Ship mode
- Carrier performance history
- Delay duration estimation (regression layer)
- Enhanced UI with dark/light mode

---

## Dataset

E-Commerce Shipping Dataset — [Kaggle](https://www.kaggle.com/datasets/prachi13/customer-analytics)

---

## Team

**Event:** The Hitchhiker's Guide to MLOps  
**Team:** Team Hedwig 
**Members:** Sinchana S
             Brunda S
             Sneha Nayak