# 📘 User Manual — Crypto Risk Manager Pro

## 🎯 Overview
Crypto Risk Manager Pro is a **“Prudent Person” risk management and asset analysis program** designed for cryptocurrency portfolios. Powered by **Comparative Bayesian Quantile Regression Analysis (CBQRA)**, it blends institutional-grade rigor with beginner-friendly onboarding.

---

## 🛡️ Philosophy: Capital Preservation First
- Built for **risk-averse investors** prioritizing capital preservation
- Focused on **realistic ROI** — no “get rich quick” hype
- Institutional-grade analysis made accessible to everyone
- Power users can tweak backend parameters, but beware: non-prudent ratios are outside the philosophy

---

## 📊 Tabs & Features
### 1. Risk Dashboard
- Portfolio configuration: set portfolio value & risk tolerance
- Position sizing: optimized allocations with stop-loss/take-profit
- Active monitoring: real-time alerts with dismissal options
- Performance metrics: Sharpe ratio, volatility, max drawdown

### 2. CBQRA Analysis
- Bayesian forecasting with MCMC sampling
- Multi-asset analysis (1–10+ cryptos)
- Automatic professional chart generation
- Real-time progress tracking

### 3. Backtesting
- Historical simulation against market data
- Rebalancing strategies (daily, weekly, monthly)
- Risk-adjusted performance analytics
- Scenario testing under different market conditions

---

## 🛠️ Technical Architecture
- **Core Stack**:
  - `MultiCryptoBQRAnalysis()` → Bayesian quantile regression
  - `CryptoMonteCarlo()` → Portfolio simulation engine
  - `AdvancedVisualizations()` → Chart generation
  - `RiskMonitor()` → Real-time alerts

- **Risk Framework Flow**:
  Portfolio Config → Bayesian Analysis → Risk Monitoring → Position Recommendations

---

## 🚀 Quick Start
```bash
git clone https://github.com/jaichai/crm_cbqra.git
cd crypto-risk-manager
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run crm_cbqra.py
