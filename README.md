# AI Customer Support Ticket Triage

This project follows the six steps shown in the provided specification without changing the requested approach.

1. Collect historical ticket text, category, priority, and resolution data.
2. Clean text and create labels such as billing, technical, account, or product.
3. Create text features using TF-IDF or sentence embeddings.
4. Train a multiclass classifier and a separate urgency model.
5. Evaluate with F1-score, confusion matrix, and precision/recall by class.
6. Expose the model through a small API/dashboard and log low-confidence predictions for human review.

Run:
`pip install -r requirements.txt`
`python train.py`
`uvicorn app:app --reload`
