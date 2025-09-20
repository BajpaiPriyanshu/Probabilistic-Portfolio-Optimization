# Probabilistic Portfolio Optimization

## 📌 Project Description  
This project extends the traditional **mean–variance portfolio optimization** framework by incorporating uncertainty in return estimates using probability distributions. It implements **Bayesian portfolio optimization** and **robust optimization techniques**, comparing their performance against classical Markowitz optimization. The goal is to create more reliable and stable portfolios in the presence of estimation risk.

---

## 🚀 Features  
- Model asset returns using probability distributions instead of fixed point estimates.  
- Implement Bayesian portfolio optimization to account for parameter uncertainty.  
- Apply robust optimization techniques to reduce sensitivity to estimation errors.  
- Compare Bayesian, robust, and classical portfolios via backtesting.  
- Visualize efficient frontiers, allocation heatmaps, and risk-return tradeoffs.  

---

## 🛠️ Libraries Used  
- **pandas** – data handling and preprocessing  
- **yfinance** – fetching historical financial data  
- **numpy** – numerical computations  
- **scipy** – statistical distributions & optimization methods  
- **matplotlib** – visualizations and performance plots  

---

---

## ⚙️ Installation  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/yourusername/probabilistic-portfolio-optimization.git
cd probabilistic-portfolio-optimization
pip install -r requirements.txt
