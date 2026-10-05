# AI Customer Support Ticket Triage

## Short description
Build an AI system that reads incoming support tickets, predicts their category and urgency, and routes them to the appropriate support queue.

## How to approach

### 1. Collect historical ticket text, category, priority, and resolution data.
The project uses `data/historical_tickets.csv` as historical data with ticket text, category, priority, and resolution.

### 2. Clean text and create labels such as billing, technical, account, or product.
The training code cleans text and uses billing, technical, account, and product category labels. Urgency labels are low, medium, and high.

### 3. Create text features using TF-IDF or sentence embeddings.
TF-IDF unigram and bigram features are implemented.

### 4. Train a multiclass classifier and a separate urgency model.
Two separate Logistic Regression models are trained: one category model and one urgency model.

### 5. Evaluate with F1-score, confusion matrix, and precision/recall by class.
The training program prints macro F1-score, classification report with precision/recall/F1 by class, and confusion matrices for both models.

### 6. Expose the model through a small API/dashboard and log low-confidence predictions for human review.
FastAPI provides a `/predict` endpoint and browser dashboard. Predictions below 0.60 overall confidence are logged to `logs/low_confidence_predictions.csv` and marked for human review.

## Run
```bash
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
```
Open `http://127.0.0.1:8000`.
