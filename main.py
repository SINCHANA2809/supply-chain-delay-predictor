from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

# Load model and scaler
model = joblib.load('model/final_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Feature column names - must match training order
X_columns = [
    'Warehouse_block', 'Mode_of_Shipment', 'Customer_care_calls',
    'Customer_rating', 'Cost_of_the_Product', 'Prior_purchases',
    'Product_importance', 'Gender', 'Discount_offered', 'Weight_in_gms',
    'weight_category', 'discount_category', 'cost_per_gram'
]

# Initialize FastAPI app
app = FastAPI(
    title="Supply Chain Delay Predictor",
    description="Predicts whether a shipment will be delayed",
    version="1.0.0"
)

# Allow frontend to talk to API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input structure
class ShipmentInput(BaseModel):
    warehouse_block: str
    mode_of_shipment: str
    customer_care_calls: int
    customer_rating: int
    cost_of_product: int
    prior_purchases: int
    product_importance: str
    gender: str
    discount_offered: int
    weight_in_gms: int

# Encoding maps
WAREHOUSE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}
SHIPMENT_MAP = {'Flight': 0, 'Road': 1, 'Ship': 2}
IMPORTANCE_MAP = {'High': 0, 'Low': 1, 'Medium': 2}
GENDER_MAP = {'F': 0, 'M': 1}

def encode_input(data: ShipmentInput):
    warehouse = WAREHOUSE_MAP.get(data.warehouse_block.upper(), 0)
    shipment = SHIPMENT_MAP.get(data.mode_of_shipment.capitalize(), 2)
    importance = IMPORTANCE_MAP.get(data.product_importance.capitalize(), 1)
    gender = GENDER_MAP.get(data.gender.upper(), 1)
    discount_cat = 0 if data.discount_offered <= 10 else (1 if data.discount_offered <= 30 else 2)
    weight_cat = 0 if data.weight_in_gms <= 2000 else (1 if data.weight_in_gms <= 4000 else 2)
    cost_per_gram = round(data.cost_of_product / data.weight_in_gms, 4)
    return [
        warehouse, shipment, data.customer_care_calls, data.customer_rating,
        data.cost_of_product, data.prior_purchases, importance, gender,
        data.discount_offered, data.weight_in_gms, weight_cat, discount_cat, cost_per_gram
    ]

def get_ai_explanation(data: ShipmentInput, prediction: int, confidence: float):
    try:
        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        status = "DELAYED" if prediction == 0 else "ON TIME"
        prompt = f"""You are a supply chain expert. A shipment has been analyzed:

Shipment Details:
- Warehouse Block: {data.warehouse_block}
- Mode of Shipment: {data.mode_of_shipment}
- Customer Care Calls: {data.customer_care_calls}
- Customer Rating: {data.customer_rating}
- Product Cost: {data.cost_of_product}
- Prior Purchases: {data.prior_purchases}
- Product Importance: {data.product_importance}
- Discount Offered: {data.discount_offered}%
- Weight: {data.weight_in_gms}gms

Prediction: {status} (Confidence: {confidence:.1f}%)

In exactly 2-3 sentences explain why this shipment is predicted to be {status}.
Then suggest exactly 1 specific operational action to improve delivery performance.
Be concise and practical."""

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        return response.text

    except Exception as e:
        print(f"Gemini error: {e}")
        reasons = []
        actions = []

        if data.discount_offered < 10:
            reasons.append(f"very low discount ({data.discount_offered}%)")
            actions.append("increase discount offered to incentivise faster processing")
        if data.weight_in_gms > 4000:
            reasons.append(f"high product weight ({data.weight_in_gms}gms)")
            actions.append("consider splitting heavy shipments")
        if data.customer_care_calls > 5:
            reasons.append(f"high customer care calls ({data.customer_care_calls})")
            actions.append("proactively communicate shipment status to reduce calls")
        if data.mode_of_shipment.lower() == "ship":
            reasons.append("Ship mode which has higher delay risk")
            actions.append("consider switching to Flight mode for urgent deliveries")
        if data.warehouse_block.upper() == "F":
            reasons.append("Warehouse F which shows higher delay patterns")
            actions.append("prioritise dispatch from Warehouse F")

        if prediction == 0:
            reason_text = ", ".join(reasons) if reasons else "multiple risk factors"
            action_text = actions[0] if actions else "prioritise dispatch within 24 hours"
            return (f"This shipment is predicted DELAYED due to {reason_text}. "
                    f"Recommended action: {action_text.capitalize()}.")
        else:
            return (f"This shipment is predicted ON TIME. "
                    f"Positive indicators: discount of {data.discount_offered}% "
                    f"and weight of {data.weight_in_gms}gms are within safe ranges. "
                    f"Continue current handling procedures.")
@app.get("/")
def home():
    return FileResponse('index.html')

@app.get("/health")
def health():
    return {"status": "healthy", "model": "Calibrated Gradient Boosting"}

@app.get("/test-gemini")
def test_gemini():
    try:
        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents="Say hello in one sentence"
        )
        return {"status": "success", "response": response.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/predict")
def predict(data: ShipmentInput):
    try:
        features = encode_input(data)
        features_array = pd.DataFrame([features], columns=X_columns)
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        confidence = max(probability) * 100
        explanation = get_ai_explanation(data, prediction, confidence)
        return {
            "prediction": int(prediction),
            "status": "ON TIME" if prediction == 1 else "DELAYED",
            "confidence": round(confidence, 2),
            "explanation": explanation,
            "input_received": data.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))