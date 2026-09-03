#!/bin/bash
set -e
echo "🛡️ PayGuard AI — Adaptive Risk Manager"
echo "========================================"

if [ ! -d ".venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv .venv
fi

echo "📦 Installing dependencies..."
.venv/bin/pip install -q -r requirements.txt

if [ ! -f "backend/xgb_model.joblib" ]; then
  echo "🗂  Generating synthetic dataset (5,000 txns)..."
  .venv/bin/python data/generate_data.py

  echo "🤖 Training XGBoost & Logistic Regression models..."
  .venv/bin/python backend/train.py
fi

echo ""
echo "🚀 Starting PayGuard AI server at http://localhost:8000"
echo "   Open your browser to view the Live Risk Dashboard"
echo "   Press Ctrl+C to stop"
echo ""
.venv/bin/python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
