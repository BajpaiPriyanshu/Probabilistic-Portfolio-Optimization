"""
Probabilistic Portfolio Optimization Project
==========================================

This project extends traditional mean-variance optimization by incorporating uncertainty 
in return estimates using probability distributions. It implements Bayesian portfolio 
optimization, robust optimization techniques, and compares performance against classical approaches.

Author: AI Assistant
Libraries: pandas, yfinance, numpy, scipy, matplotlib
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import ttest_ind, wishart
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class PortfolioOptimizer:
    """Portfolio optimization class implementing various optimization techniques"""
    
    def __init__(self, tickers, start_date='2020-01-01', end_date='2024-01-01'):
        self.tickers = tickers
        print("Downloading data...")
        
        # Download data with better error handling
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            # Extract adjusted close prices
            if len(tickers) == 1:
                if isinstance(data.columns, pd.MultiIndex):
                    self.prices = data['Adj Close'].to_frame()
                    self.prices.columns = tickers
                else:
                    self.prices = data[['Adj Close']].copy()
                    self.prices.columns = tickers
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    self.prices = data['Adj Close'].copy()
                else:
                    # Fallback for different data structure
                    self.prices = data.copy()
            
            # Clean data
            self.prices = self.prices.dropna()
            self.returns = self.prices.pct_change().dropna()
            self.mean_returns = self.returns.mean()
            self.cov_matrix = self.returns.cov()
            
            print(f"Data loaded: {len(self.returns)} days, {len(tickers)} assets")
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            # Create sample data for demonstration
            np.random.seed(42)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')[:1000]
            sample_data = {}
            for ticker in tickers:
                prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
                sample_data[ticker] = prices
            
            self.prices = pd.DataFrame(sample_data, index=dates)
            self.returns = self.prices.pct_change().dropna()
            self.mean_returns = self.returns.mean()
            self.cov_matrix = self.returns.cov()
            print(f"Using sample data: {len(self.returns)} days, {len(tickers)} assets")
    
    def _portfolio_stats(self, weights, rf=0.02):
        """Calculate portfolio statistics"""
        ret = np.sum(self.mean_returns * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        sharpe = (ret - rf) / vol
        return ret, vol, sharpe
    
    def classical_optimization(self, rf=0.02):
        """Classical Markowitz optimization"""
        n = len(self.tickers)
        
        def neg_sharpe(w):
            return -self._portfolio_stats(w, rf)[2]
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n))
        result = minimize(neg_sharpe, np.ones(n)/n, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x, self._portfolio_stats(result.x, rf)
    
    def bayesian_optimization(self, n_sim=200, rf=0.02):
        """Bayesian portfolio optimization with parameter uncertainty"""
        n = len(self.tickers)
        n_obs = len(self.returns)
        
        # Bayesian parameters (simplified approach)
        mu_sample = self.mean_returns.values
        cov_sample = self.cov_matrix.values
        
        optimal_weights = []
        
        print(f"Running Bayesian optimization with {n_sim} simulations...")
        
        for i in range(n_sim):
            try:
                # Add uncertainty to parameters
                # For mean returns: add noise based on standard error
                se_mu = np.sqrt(np.diag(cov_sample)) / np.sqrt(n_obs)
                mu_bayes = mu_sample + np.random.normal(0, se_mu)
                
                # For covariance: add small perturbation
                noise_factor = 0.1
                cov_noise = np.random.normal(0, noise_factor * np.abs(cov_sample))
                cov_bayes = cov_sample + cov_noise
                
                # Ensure covariance matrix is positive definite
                eigenvals, eigenvecs = np.linalg.eig(cov_bayes)
                eigenvals = np.maximum(eigenvals, 1e-6)  # Ensure positive eigenvalues
                cov_bayes = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                
                # Optimize for this sample
                def neg_sharpe_bayes(w):
                    ret = np.sum(mu_bayes * w) * 252
                    vol = np.sqrt(np.dot(w.T, np.dot(cov_bayes * 252, w)))
                    if vol == 0:
                        return 1000  # Large penalty for zero volatility
                    return -(ret - rf) / vol
                
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                bounds = tuple((0, 1) for _ in range(n))
                
                result = minimize(neg_sharpe_bayes, np.ones(n)/n, method='SLSQP',
                                bounds=bounds, constraints=constraints,
                                options={'maxiter': 100})
                
                if result.success and np.all(result.x >= 0) and np.abs(np.sum(result.x) - 1) < 1e-6:
                    optimal_weights.append(result.x)
                    
            except Exception as e:
                continue
        
        if len(optimal_weights) == 0:
            print("Bayesian optimization failed, using classical weights")
            return self.classical_optimization(rf)
        
        bayes_weights = np.mean(optimal_weights, axis=0)
        # Normalize weights to ensure they sum to 1
        bayes_weights = bayes_weights / np.sum(bayes_weights)
        
        return bayes_weights, self._portfolio_stats(bayes_weights, rf)
    
    def robust_optimization(self, uncertainty=0.1, rf=0.02):
        """Robust optimization with uncertainty sets"""
        n = len(self.tickers)
        mu_nom = self.mean_returns.values
        cov_nom = self.cov_matrix.values
        
        def robust_sharpe(w):
            # Worst-case return (conservative estimate)
            worst_ret = np.sum((mu_nom - uncertainty * np.abs(mu_nom)) * w) * 252
            worst_vol = np.sqrt(np.dot(w.T, np.dot(cov_nom * (1 + uncertainty) * 252, w)))
            if worst_vol == 0:
                return 1000
            return -(worst_ret - rf) / worst_vol
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n))
        result = minimize(robust_sharpe, np.ones(n)/n, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x, self._portfolio_stats(result.x, rf)
    
    def black_litterman(self, tau=0.05, rf=0.02):
        """Black-Litterman optimization"""
        n = len(self.tickers)
        # Equal market weights assumption
        market_weights = np.ones(n) / n
        
        # Implied returns
        risk_aversion = 3.0
        implied_returns = risk_aversion * np.dot(self.cov_matrix * 252, market_weights)
        
        # BL without views (simplified)
        bl_mu = implied_returns
        bl_cov = tau * self.cov_matrix * 252
        
        def bl_objective(w):
            ret = np.sum(bl_mu * w)
            var = np.dot(w.T, np.dot(bl_cov, w))
            return var - ret / risk_aversion
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n))
        result = minimize(bl_objective, market_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x, self._portfolio_stats(result.x, rf)
    
    def efficient_frontier(self, n_points=30):
        """Generate efficient frontier"""
        n = len(self.tickers)
        min_ret = self.mean_returns.min() * 252
        max_ret = self.mean_returns.max() * 252
        target_returns = np.linspace(min_ret * 1.5, max_ret * 0.7, n_points)
        
        frontier_vol = []
        frontier_ret = []
        
        for target in target_returns:
            try:
                def portfolio_vol(w):
                    return np.dot(w.T, np.dot(self.cov_matrix * 252, w))
                
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) * 252 - target}
                ]
                bounds = tuple((0, 1) for _ in range(n))
                
                result = minimize(portfolio_vol, np.ones(n)/n, method='SLSQP',
                                bounds=bounds, constraints=constraints)
                
                if result.success:
                    frontier_ret.append(target)
                    frontier_vol.append(np.sqrt(result.fun))
            except:
                continue
        
        return np.array(frontier_ret), np.array(frontier_vol)

def run_analysis():
    """Run comprehensive portfolio optimization analysis"""
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']
    optimizer = PortfolioOptimizer(tickers)
    
    print("\n" + "="*50)
    print("RUNNING OPTIMIZATION METHODS")
    print("="*50)
    
    # Run all optimization methods
    print("1. Classical optimization...")
    classical_w, classical_stats = optimizer.classical_optimization()
    
    print("2. Bayesian optimization...")
    bayesian_w, bayesian_stats = optimizer.bayesian_optimization()
    
    print("3. Robust optimization...")
    robust_w, robust_stats = optimizer.robust_optimization()
    
    print("4. Black-Litterman optimization...")
    bl_w, bl_stats = optimizer.black_litterman()
    
    # Results summary
    methods = ['Classical', 'Bayesian', 'Robust', 'Black-Litterman']
    results = [classical_stats, bayesian_stats, robust_stats, bl_stats]
    weights = [classical_w, bayesian_w, robust_w, bl_w]
    
    print(f"\n{'Method':<15} {'Return':<8} {'Risk':<8} {'Sharpe':<8}")
    print("-" * 45)
    for method, (ret, risk, sharpe) in zip(methods, results):
        print(f"{method:<15} {ret:.3f}    {risk:.3f}    {sharpe:.3f}")
    
    # Generate efficient frontier
    print("5. Generating efficient frontier...")
    ef_ret, ef_vol = optimizer.efficient_frontier()
    
    # Create visualizations
    create_plots(optimizer, methods, results, weights, ef_ret, ef_vol)
    
    # Performance analysis
    analyze_performance(optimizer, methods, weights)
    
    return optimizer, results, weights

def create_plots(optimizer, methods, results, weights, ef_ret, ef_vol):
    """Create three comprehensive visualization figures"""
    
    # Figure 1: Portfolio Analysis
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Efficient Frontier
    if len(ef_ret) > 0:
        ax1.plot(ef_vol, ef_ret, 'b-', linewidth=2, label='Efficient Frontier')
    
    colors = ['red', 'green', 'orange', 'purple']
    
    for i, (method, (ret, vol, _)) in enumerate(zip(methods, results)):
        ax1.scatter(vol, ret, color=colors[i], s=100, label=method, alpha=0.8)
    
    ax1.set_xlabel('Risk (Volatility)')
    ax1.set_ylabel('Expected Return')
    ax1.set_title('Efficient Frontier & Portfolio Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Portfolio Weights
    x_pos = np.arange(len(optimizer.tickers))
    width = 0.2
    
    for i, (method, w) in enumerate(zip(methods, weights)):
        ax2.bar(x_pos + i*width, w, width, label=method, alpha=0.8, color=colors[i])
    
    ax2.set_xlabel('Assets')
    ax2.set_ylabel('Weight')
    ax2.set_title('Portfolio Weights Comparison')
    ax2.set_xticks(x_pos + width * 1.5)
    ax2.set_xticklabels(optimizer.tickers, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Risk-Return Scatter
    rets = [r[0] for r in results]
    risks = [r[1] for r in results]
    sharpes = [r[2] for r in results]
    
    scatter = ax3.scatter(risks, rets, c=sharpes, s=[abs(s)*500 for s in sharpes], 
                         alpha=0.7, cmap='viridis')
    
    for i, method in enumerate(methods):
        ax3.annotate(method, (risks[i], rets[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Risk')
    ax3.set_ylabel('Return')
    ax3.set_title('Risk-Return Performance')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Sharpe Ratio')
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Risk Analysis
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Portfolio concentration
    concentrations = []
    for i, (method, w) in enumerate(zip(methods, weights)):
        herfindahl = np.sum(w**2)
        effective_assets = 1 / herfindahl
        concentrations.append(effective_assets)
        ax1.bar(i, effective_assets, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Portfolio Method')
    ax1.set_ylabel('Effective Number of Assets')
    ax1.set_title('Portfolio Concentration')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Return distributions (sample)
    sample_size = min(500, len(optimizer.returns))
    sample_returns = optimizer.returns.sample(n=sample_size)
    for i, ticker in enumerate(optimizer.tickers[:4]):  # Show first 4 assets
        ax2.hist(sample_returns[ticker] * np.sqrt(252), bins=20, alpha=0.5, 
                label=ticker, density=True)
    
    ax2.set_xlabel('Annualized Returns')
    ax2.set_ylabel('Density')
    ax2.set_title('Asset Return Distributions (Sample)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Correlation heatmap
    corr = optimizer.returns.corr()
    im = ax3.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(optimizer.tickers)))
    ax3.set_yticks(range(len(optimizer.tickers)))
    ax3.set_xticklabels(optimizer.tickers, rotation=45)
    ax3.set_yticklabels(optimizer.tickers)
    ax3.set_title('Asset Correlations')
    
    for i in range(len(optimizer.tickers)):
        for j in range(len(optimizer.tickers)):
            color = "black" if abs(corr.iloc[i,j]) < 0.5 else "white"
            ax3.text(j, i, f'{corr.iloc[i,j]:.2f}', ha="center", va="center", 
                    color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax3)
    plt.tight_layout()
    plt.show()
    
    # Figure 3: Performance Analysis
    fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Sharpe ratio comparison
    sharpe_ratios = [r[2] for r in results]
    bars = ax1.bar(methods, sharpe_ratios, color=colors, alpha=0.8)
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Sharpe Ratio Comparison')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    for bar, ratio in zip(bars, sharpe_ratios):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ratio:.3f}', ha='center', va='bottom')
    
    # Risk contribution analysis (for classical portfolio)
    classical_weights = weights[0]
    cov_annual = optimizer.cov_matrix * 252
    portfolio_var = np.dot(classical_weights.T, np.dot(cov_annual, classical_weights))
    marginal_contrib = np.dot(cov_annual, classical_weights)
    risk_contrib = classical_weights * marginal_contrib / portfolio_var
    
    # Only show non-zero contributions
    non_zero_contrib = risk_contrib[risk_contrib > 0.01]
    non_zero_tickers = [optimizer.tickers[i] for i, contrib in enumerate(risk_contrib) if contrib > 0.01]
    
    if len(non_zero_contrib) > 0:
        ax2.pie(non_zero_contrib, labels=non_zero_tickers, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Risk Contribution (Classical Portfolio)')
    else:
        ax2.text(0.5, 0.5, 'Risk contribution\ntoo dispersed\nto visualize', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Risk Contribution (Classical Portfolio)')
    
    # Scenario analysis
    scenarios = {
        'Market Crash': np.array([-0.3, -0.25, -0.35, -0.4, -0.2, -0.45]),
        'Bull Market': np.array([0.25, 0.3, 0.2, 0.35, 0.4, 0.45])
    }
    
    scenario_results = {}
    for scenario, returns in scenarios.items():
        scenario_results[scenario] = []
        for w in weights:
            scenario_return = np.sum(w * returns)
            scenario_results[scenario].append(scenario_return)
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    crash_returns = scenario_results['Market Crash']
    bull_returns = scenario_results['Bull Market']
    
    ax3.bar(x_pos - width/2, crash_returns, width, label='Market Crash', 
            alpha=0.8, color='red')
    ax3.bar(x_pos + width/2, bull_returns, width, label='Bull Market', 
            alpha=0.8, color='green')
    
    ax3.set_xlabel('Portfolio Method')
    ax3.set_ylabel('Scenario Return')
    ax3.set_title('Scenario Analysis')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_performance(optimizer, methods, weights):
    """Analyze portfolio performance with bootstrap simulation"""
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Bootstrap simulation
    n_sim = 500  # Reduced for faster execution
    holding_period = min(252, len(optimizer.returns))
    
    results = {}
    print("Running bootstrap simulation...")
    
    for i, (method, w) in enumerate(zip(methods, weights)):
        portfolio_returns = []
        
        for _ in range(n_sim):
            try:
                sample = optimizer.returns.sample(n=holding_period, replace=True)
                daily_returns = np.sum(sample.values * w, axis=1)
                annual_return = np.prod(1 + daily_returns) - 1
                portfolio_returns.append(annual_return)
            except:
                continue
        
        if len(portfolio_returns) > 0:
            results[method] = {
                'mean': np.mean(portfolio_returns),
                'std': np.std(portfolio_returns),
                'var_95': np.percentile(portfolio_returns, 5),
                'returns': portfolio_returns
            }
    
    print(f"\nBootstrap Results ({n_sim} simulations):")
    print(f"{'Method':<15} {'Mean':<8} {'Std':<8} {'95% VaR':<8}")
    print("-" * 45)
    
    for method, res in results.items():
        print(f"{method:<15} {res['mean']:.3f}    {res['std']:.3f}    {res['var_95']:.3f}")
    
    # Statistical tests
    print(f"\nPairwise t-tests (p-values):")
    method_list = list(results.keys())
    for i in range(len(method_list)):
        for j in range(i+1, len(method_list)):
            try:
                _, p_val = ttest_ind(results[method_list[i]]['returns'], 
                                   results[method_list[j]]['returns'])
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                print(f"{method_list[i]} vs {method_list[j]}: {p_val:.4f} {sig}")
            except:
                print(f"{method_list[i]} vs {method_list[j]}: N/A")

if __name__ == "__main__":
    print("PROBABILISTIC PORTFOLIO OPTIMIZATION")
    print("="*50)
    
    optimizer, results, weights = run_analysis()
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print("\nKey Features Implemented:")
    print("• Classical Markowitz Optimization")
    print("• Bayesian Portfolio Optimization")
    print("• Robust Optimization with Uncertainty")
    print("• Black-Litterman Model")
    print("• Efficient Frontier Generation")
    print("• Bootstrap Performance Analysis")
    print("• Statistical Significance Testing")
    print("• 9 Comprehensive Visualizations")