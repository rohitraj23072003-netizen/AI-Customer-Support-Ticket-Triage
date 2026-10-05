import csv, os
from datetime import datetime, timezone
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

CATEGORY_MODEL = "models/category_model.joblib"
URGENCY_MODEL = "models/urgency_model.joblib"
LOG_FILE = "logs/low_confidence_predictions.csv"
CONFIDENCE_THRESHOLD = 0.60
QUEUE_MAP = {"billing":"Billing Support Queue","technical":"Technical Support Queue","account":"Account Support Queue","product":"Product Support Queue"}
app = FastAPI(title="AI Customer Support Ticket Triage")
category_model = None
urgency_model = None

class Ticket(BaseModel):
    ticket_text: str = Field(min_length=3, max_length=5000)

def load_models():
    global category_model, urgency_model
    if not (os.path.exists(CATEGORY_MODEL) and os.path.exists(URGENCY_MODEL)):
        raise RuntimeError("Models not found. Run: python train.py")
    category_model = joblib.load(CATEGORY_MODEL)
    urgency_model = joblib.load(URGENCY_MODEL)

def ensure_log():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp","ticket_text","category","category_confidence","urgency","urgency_confidence","queue"])

def predict_ticket(text):
    if category_model is None or urgency_model is None:
        load_models()
    cat_probs = category_model.predict_proba([text])[0]
    urg_probs = urgency_model.predict_proba([text])[0]
    cat_idx, urg_idx = cat_probs.argmax(), urg_probs.argmax()
    category = category_model.classes_[cat_idx]
    urgency = urgency_model.classes_[urg_idx]
    category_conf, urgency_conf = float(cat_probs[cat_idx]), float(urg_probs[urg_idx])
    confidence = min(category_conf, urgency_conf)
    queue = QUEUE_MAP.get(category, "General Support Queue")
    human_review = confidence < CONFIDENCE_THRESHOLD
    if human_review:
        ensure_log()
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), text, category, round(category_conf,4), urgency, round(urgency_conf,4), queue])
    return {"category":category,"urgency":urgency,"category_confidence":round(category_conf,4),"urgency_confidence":round(urgency_conf,4),"overall_confidence":round(confidence,4),"support_queue":queue,"human_review_required":human_review}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return '''<!DOCTYPE html><html><head><meta charset="utf-8"><title>AI Customer Support Ticket Triage</title><style>body{font-family:Arial;max-width:850px;margin:40px auto;padding:0 20px}textarea{width:100%;height:150px;padding:12px;font-size:16px}button{padding:12px 20px;margin-top:12px}.card{margin-top:20px;padding:18px;border:1px solid #ddd;border-radius:10px;background:#fafafa}</style></head><body><h1>AI Customer Support Ticket Triage</h1><p>Enter an incoming support ticket. The AI predicts category, urgency, and support queue.</p><textarea id="ticket" placeholder="Example: My payment failed and I cannot complete my order"></textarea><br><button onclick="predict()">Predict & Route</button><div id="result" class="card" style="display:none"></div><script>async function predict(){const t=document.getElementById("ticket").value;const r=document.getElementById("result");r.style.display="block";r.textContent="Predicting...";const x=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({ticket_text:t})});const d=await x.json();r.innerHTML="<pre>"+JSON.stringify(d,null,2)+"</pre>"}</script></body></html>'''

@app.post("/predict")
def predict(ticket: Ticket):
    return predict_ticket(ticket.ticket_text)
