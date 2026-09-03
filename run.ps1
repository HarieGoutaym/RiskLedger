# Run script for Windows PowerShell — PayGuard AI Risk Manager
Write-Host "🛡️ PayGuard AI — Adaptive Risk Manager" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor DarkGray

# 1. Create venv if missing
if (-not (Test-Path ".venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# 2. Activate and install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
.\.venv\Scripts\pip install -q -r requirements.txt

# 3. Generate data & train model if missing
if (-not (Test-Path "backend\xgb_model.joblib")) {
    Write-Host "🗂  Generating synthetic dataset (5,000 txns)..." -ForegroundColor Yellow
    .\.venv\Scripts\python data\generate_data.py

    Write-Host "🤖 Training XGBoost & Logistic Regression models..." -ForegroundColor Yellow
    .\.venv\Scripts\python backend\train.py
}

# 4. Start server
Write-Host ""
Write-Host "🚀 Starting PayGuard AI server at http://localhost:8000" -ForegroundColor Green
Write-Host "   Open your browser to view the Live Risk Dashboard" -ForegroundColor DarkGray
Write-Host "   Press Ctrl+C to stop" -ForegroundColor DarkGray
Write-Host ""
.\.venv\Scripts\python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
