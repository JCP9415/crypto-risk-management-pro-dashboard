import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CryptoMonteCarlo:
    """
    Advanced Monte Carlo simulator for crypto portfolio risk analysis
    FIXED: Removed internal seed management to respect external profile-specific seeding
    """

    def __init__(self, analyzer, crypto_data: Dict, precomputed_correlation: Optional[pd.DataFrame] = None):
        self.analyzer = analyzer
        self.crypto_data = crypto_data
        self.crypto_names = list(crypto_data.keys())

        # Validate crypto_data
        if not self.crypto_names:
            raise ValueError("crypto_data dictionary is empty")

        # SP500 benchmark data (annual returns ~7-10%, volatility ~15-18%)
        self.sp500_annual_return = 0.08  # 8% historical average
        self.sp500_annual_vol = 0.16     # 16% volatility

        # Student's t-distribution parameter for fat tails (nu)
        self.T_DOF = 4

        # FIXED: NO SEED SET HERE - external caller (final.py) sets profile-specific seeds
        print("✓ Random seed management delegated to caller (profile-specific seeding)")

        # FIXED: Use precomputed correlation from CBQRA if available (single source of truth)
        if precomputed_correlation is not None:
            print("✓ Using precomputed correlation matrix from CBQRA")
            self.correlation_matrix = precomputed_correlation
            self._validate_correlation_matrix()
        else:
            print("⚠️ No precomputed correlation available, calculating from historical data")
            self.correlation_matrix = self._calculate_correlation_matrix()

    def _validate_correlation_matrix(self):
        """Validate and fix correlation matrix to ensure it's mathematically valid"""
        corr = self.correlation_matrix

        # Check dimensions
        if corr.shape[0] != len(self.crypto_names) or corr.shape[1] != len(self.crypto_names):
            raise ValueError(f"Correlation matrix dimensions {corr.shape} don't match number of cryptos {len(self.crypto_names)}")

        # Check diagonal is 1.0
        diag = np.diag(corr.values)
        if not np.allclose(diag, 1.0, atol=1e-6):
            print(f"⚠️ Correlation matrix diagonal not 1.0, normalizing...")
            d = np.sqrt(np.abs(np.diag(corr.values)))
            corr = pd.DataFrame(
                corr.values / np.outer(d, d),
                index=corr.index,
                columns=corr.columns
            )

        # Check symmetry
        if not np.allclose(corr.values, corr.values.T, atol=1e-6):
            print("⚠️ Correlation matrix not symmetric, symmetrizing...")
            corr = pd.DataFrame(
                (corr.values + corr.values.T) / 2,
                index=corr.index,
                columns=corr.columns
            )

        # FIXED: Proper positive definite correction using nearest PSD matrix
        eigenvalues = np.linalg.eigvals(corr.values)
        min_eig = np.min(eigenvalues)

        if min_eig < 1e-8:  # Not positive definite
            print(f"⚠️ Correlation matrix not positive definite (min eigenvalue: {min_eig:.6f})")
            print("   Applying nearest positive semi-definite matrix correction...")

            # Eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(corr.values)

            # Set negative eigenvalues to small positive value
            eigvals[eigvals < 1e-8] = 1e-8

            # Reconstruct matrix
            corr_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Re-normalize to correlation matrix (diagonal = 1)
            d = np.sqrt(np.diag(corr_fixed))
            corr_fixed = corr_fixed / np.outer(d, d)

            # Ensure symmetry after reconstruction
            corr_fixed = (corr_fixed + corr_fixed.T) / 2

            corr = pd.DataFrame(corr_fixed, index=corr.index, columns=corr.columns)

            # Verify fix
            new_eigvals = np.linalg.eigvals(corr.values)
            print(f"   ✓ Fixed: min eigenvalue now {np.min(new_eigvals):.6f}")

        self.correlation_matrix = corr

    def _calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix from historical price data with enhanced error handling
        FIXED: Consistent with CBQRA methodology, proper validation
        """
        try:
            # Extract price data for all cryptos (using Price column for consistency)
            price_data = {}
            for crypto in self.crypto_names:
                if crypto in self.analyzer.data_dict:
                    df = self.analyzer.data_dict[crypto].copy()
                    df = df.set_index('Date')

                    # FIXED: Validate price data
                    if 'Price' not in df.columns:
                        raise ValueError(f"Price column not found for {crypto}")

                    prices = df['Price']

                    # FIXED: Check for zero or negative prices
                    if (prices <= 0).any():
                        print(f"⚠️ {crypto} has zero/negative prices, filtering...")
                        prices = prices[prices > 0]

                    if len(prices) < 30:
                        raise ValueError(f"{crypto} has insufficient valid price data: {len(prices)} points")

                    price_data[crypto] = prices
                else:
                    raise ValueError(f"Crypto {crypto} not found in analyzer data")

            # Create combined DataFrame
            combined_prices = pd.DataFrame(price_data)

            if combined_prices.empty:
                raise ValueError("No price data available for correlation calculation")

            # FIXED: Check data alignment and coverage
            null_counts = combined_prices.isnull().sum()
            total_rows = len(combined_prices)

            for crypto in self.crypto_names:
                null_pct = (null_counts[crypto] / total_rows) * 100
                if null_pct > 50:
                    print(f"⚠️ {crypto} has {null_pct:.1f}% missing data in common date range")

            # FIXED: Replace deprecated fillna(method='ffill') with ffill()
            combined_prices = combined_prices.ffill(limit=3)
            combined_prices = combined_prices.dropna()

            if len(combined_prices) < 30:
                raise ValueError(f"Insufficient overlapping data: only {len(combined_prices)} days available (need at least 30)")

            print(f"✓ Using {len(combined_prices)} days of overlapping price data")

            # Calculate returns and correlation matrix
            returns = combined_prices.pct_change().dropna()

            if len(returns) < 30:
                raise ValueError(f"Insufficient return data for correlation: {len(returns)} days")

            correlation_matrix = returns.corr()

            # Validate correlation matrix
            if correlation_matrix.isna().any().any():
                print("⚠️ Correlation matrix has NaN values, using pairwise complete observations")
                # Recalculate with pairwise deletion
                correlation_matrix = returns.corr(method='pearson', min_periods=30)

                # If still NaN, use fallback
                if correlation_matrix.isna().any().any():
                    avg_corr = 0.5  # Conservative fallback
                    correlation_matrix = correlation_matrix.fillna(avg_corr)
                    print(f"⚠️ Filled remaining missing correlations with {avg_corr}")

            # Validate and fix if needed
            self.correlation_matrix = correlation_matrix
            self._validate_correlation_matrix()

            return self.correlation_matrix

        except Exception as e:
            print(f"❌ Error calculating correlation matrix: {e}")
            print("   Using fallback correlation estimates from crypto_data...")

            # Fallback: use correlations from crypto_data
            n = len(self.crypto_names)
            fallback_corr = np.eye(n)
            for i, crypto1 in enumerate(self.crypto_names):
                for j, crypto2 in enumerate(self.crypto_names):
                    if i != j:
                        # Use average of both cryptos' correlation estimates
                        corr1 = self.crypto_data[crypto1].get('correlation', 0.5)
                        corr2 = self.crypto_data[crypto2].get('correlation', 0.5)
                        corr = (corr1 + corr2) / 2
                        fallback_corr[i, j] = corr
                        fallback_corr[j, i] = corr

            correlation_matrix = pd.DataFrame(fallback_corr, index=self.crypto_names, columns=self.crypto_names)
            self.correlation_matrix = correlation_matrix
            self._validate_correlation_matrix()

            return self.correlation_matrix

    def simulate_asset_returns(self, days: int, n_simulations: int) -> np.ndarray:
        """
        Simulate correlated asset returns using multivariate T-distribution noise.

        CRITICAL FIX: This function NO LONGER sets np.random.seed()!
        The seed must be set externally by the caller (final.py) before calling this method.
        This allows profile-specific seeding to work correctly.

        Args:
            days: Number of trading days to simulate
            n_simulations: Number of simulation paths

        Returns:
            Array of shape (n_simulations, days, n_assets) with daily returns
        """
        # FIXED: Validate inputs
        if days <= 0:
            raise ValueError(f"days must be positive, got {days}")
        if n_simulations <= 0:
            raise ValueError(f"n_simulations must be positive, got {n_simulations}")

        n_assets = len(self.crypto_names)

        # Get expected returns and volatilities
        expected_returns = []
        volatilities = []

        for crypto in self.crypto_names:
            data = self.crypto_data[crypto]
            # Convert annual metrics to daily
            ann_vol = data.get('volatility', 0.8)
            ann_return = data.get('expected_return', 0.0)

            # FIXED: Validate volatility is positive
            if ann_vol <= 0:
                print(f"⚠️ {crypto} has invalid volatility {ann_vol}, using 0.8")
                ann_vol = 0.8

            daily_vol = ann_vol / np.sqrt(365)  # Crypto markets: 365 days/year
            daily_return = ann_return / 365

            volatilities.append(daily_vol)
            expected_returns.append(daily_return)

        # Convert to arrays
        mu = np.array(expected_returns)
        sigma = np.array(volatilities)

        # FIXED: Validate volatilities are positive
        if np.any(sigma <= 0):
            raise ValueError(f"Volatilities must be positive, got: {sigma}")

        # Create covariance matrix from correlation matrix and volatilities
        cov_matrix = np.outer(sigma, sigma) * self.correlation_matrix.values

        # FIXED: Validate covariance matrix before use
        try:
            np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            print("❌ Covariance matrix is not positive definite after correlation fix")
            print("   Applying additional regularization...")
            # Add small ridge to diagonal
            ridge = 1e-6
            cov_matrix += np.eye(n_assets) * ridge
            try:
                np.linalg.cholesky(cov_matrix)
                print(f"   ✓ Fixed with ridge regularization: {ridge}")
            except np.linalg.LinAlgError:
                raise ValueError("Covariance matrix cannot be made positive definite. Check correlation matrix and volatility inputs.")

        # Generate Correlated Student's T-Distributed Noise
        nu = self.T_DOF

        if nu <= 2:
            print("⚠️ T_DOF <= 2. Using Normal distribution for noise.")
            random_noise = np.random.multivariate_normal(
                np.zeros(n_assets), self.correlation_matrix.values, size=(n_simulations, days)
            )
        else:
            # Standardized T-Noise with unit variance
            t_scaling_factor = np.sqrt((nu - 2.0) / nu)

            # Generate independent t-distributed noise
            independent_t_noise = stats.t.rvs(df=nu, size=(n_simulations * days, n_assets)) * t_scaling_factor

            # Apply Cholesky decomposition to introduce correlation
            try:
                L = np.linalg.cholesky(self.correlation_matrix.values)
            except np.linalg.LinAlgError:
                raise ValueError("Failed to compute Cholesky decomposition of correlation matrix (should not happen after validation)")

            # Correlated T-Noise
            random_noise = np.dot(independent_t_noise, L.T)

            # Reshape back to (n_simulations, days, n_assets)
            random_noise = random_noise.reshape((n_simulations, days, n_assets))

        # Calculate daily returns using the GBM framework with T-noise
        daily_vol_term = sigma * random_noise
        daily_returns = mu + daily_vol_term

        return daily_returns

    def simulate_portfolio_paths(self, allocations: Dict, initial_capital: float,
                               days: int, n_simulations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate portfolio value paths using Monte Carlo
        FIXED: Added comprehensive validation

        Args:
            allocations: Dictionary of asset allocations
            initial_capital: Starting portfolio value
            days: Number of trading days to simulate
            n_simulations: Number of simulation paths

        Returns:
            Tuple of (portfolio_paths, asset_paths)
        """
        # FIXED: Validate inputs
        if not allocations:
            raise ValueError("Allocations dictionary is empty")

        if initial_capital <= 0:
            raise ValueError(f"Initial capital must be positive, got {initial_capital}")

        total_alloc = sum(allocations.values())
        if total_alloc == 0:
            raise ValueError("Total allocation is zero")

        if abs(total_alloc - 1.0) > 0.01:
            print(f"⚠️ Allocations sum to {total_alloc:.4f}, normalizing to 1.0")
            allocations = {k: v/total_alloc for k, v in allocations.items()}

        # FIXED: Validate all allocations are non-negative
        if any(v < 0 for v in allocations.values()):
            raise ValueError(f"Allocations must be non-negative, got: {allocations}")

        n_assets = len(self.crypto_names)

        # Convert allocations to array in correct order
        alloc_array = np.array([allocations.get(crypto, 0.0) for crypto in self.crypto_names])

        if np.sum(alloc_array) == 0:
            raise ValueError("No valid allocations found for specified cryptos")

        # Simulate asset returns (uses external seed set by caller)
        asset_returns = self.simulate_asset_returns(days, n_simulations)

        # Calculate asset price paths
        asset_prices = np.zeros((n_simulations, days + 1, n_assets))
        asset_prices[:, 0, :] = 1.0  # Normalized starting prices

        for t in range(days):
            asset_prices[:, t + 1, :] = asset_prices[:, t, :] * (1 + asset_returns[:, t, :])

        # Calculate portfolio values
        portfolio_values = np.zeros((n_simulations, days + 1))
        portfolio_values[:, 0] = initial_capital

        for t in range(days):
            # Calculate portfolio return for this period
            portfolio_return = np.sum(asset_returns[:, t, :] * alloc_array, axis=1)
            portfolio_values[:, t + 1] = portfolio_values[:, t] * (1 + portfolio_return)

        return portfolio_values, asset_prices

    def generate_metrics(self, portfolio_paths: np.ndarray, initial_capital: float) -> Tuple[Dict, np.ndarray]:
        """
        Generate performance and risk metrics from simulation results
        FIXED: Added validation

        Args:
            portfolio_paths: Array of portfolio value paths
            initial_capital: Starting portfolio value

        Returns:
            Tuple of (metrics_dict, final_values)
        """
        # FIXED: Validate inputs
        if initial_capital <= 0:
            raise ValueError(f"Initial capital must be positive, got {initial_capital}")

        if portfolio_paths.size == 0:
            raise ValueError("Portfolio paths array is empty")

        final_values = portfolio_paths[:, -1]

        # FIXED: Check for invalid final values
        if np.any(final_values <= 0):
            n_invalid = np.sum(final_values <= 0)
            print(f"⚠️ {n_invalid} simulations resulted in zero/negative portfolio value")
            # Filter out invalid paths for metrics calculation
            valid_mask = final_values > 0
            if np.sum(valid_mask) < len(final_values) * 0.1:  # Less than 10% valid
                raise ValueError("Too many simulations resulted in portfolio collapse")
            final_values = final_values[valid_mask]
            portfolio_paths = portfolio_paths[valid_mask]

        total_returns = (final_values / initial_capital - 1.0) * 100

        # Basic metrics
        expected_return = np.mean(total_returns)
        return_std = np.std(total_returns)
        best_case = np.max(total_returns)
        worst_case = np.min(total_returns)

        # Risk metrics
        var_95 = np.percentile(total_returns, 5)  # 5th percentile = 95% VaR
        cvar_95 = total_returns[total_returns <= var_95].mean()

        # Probability metrics
        probability_positive = (total_returns > 0).mean() * 100

        # Compare against SP500
        sp500_final = initial_capital * (1 + self.sp500_annual_return)
        probability_beating_sp500 = (final_values > sp500_final).mean() * 100

        # Final value statistics
        median_final_value = np.median(final_values)
        final_value_std = np.std(final_values)

        metrics = {
            'expected_return': expected_return,
            'return_std': return_std,
            'best_case': best_case,
            'worst_case': worst_case,
            'var_95': var_95,
            'cvar_95': cvar_95 if not np.isnan(cvar_95) else var_95,
            'probability_positive': probability_positive,
            'probability_beating_sp500': probability_beating_sp500,
            'median_final_value': median_final_value,
            'final_value_std': final_value_std
        }

        return metrics, final_values

    def plot_monte_carlo_results(self, portfolio_paths: np.ndarray, initial_capital: float,
                               metrics: Dict, n_paths_to_plot: int = 100) -> plt.Figure:
        """
        Create comprehensive visualization of Monte Carlo results

        Args:
            portfolio_paths: Array of portfolio value paths
            initial_capital: Starting portfolio value
            metrics: Dictionary of performance metrics
            n_paths_to_plot: Number of individual paths to display

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        final_values = portfolio_paths[:, -1]
        total_returns = (final_values / initial_capital - 1.0) * 100

        # Plot 1: Portfolio paths with confidence intervals
        ax1 = axes[0, 0]
        days = portfolio_paths.shape[1] - 1

        # Calculate percentiles
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = np.percentile(portfolio_paths, percentiles, axis=0)

        # Plot a subset of individual paths
        for i in range(min(n_paths_to_plot, len(portfolio_paths))):
            ax1.plot(range(days + 1), portfolio_paths[i], alpha=0.1, color='blue', linewidth=0.5)

        # Plot percentiles
        colors = ['red', 'orange', 'green', 'orange', 'red']
        linestyles = ['--', '--', '-', '--', '--']

        for i, p in enumerate(percentiles):
            ax1.plot(range(days + 1), percentile_values[i],
                    color=colors[i], linestyle=linestyles[i],
                    label=f'{p}th percentile', linewidth=2)

        ax1.axhline(initial_capital, color='black', linestyle='-', linewidth=2, label='Initial Capital')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Monte Carlo Simulation: Portfolio Paths')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Distribution of final values
        ax2 = axes[0, 1]
        ax2.hist(final_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(initial_capital, color='red', linestyle='--', linewidth=2, label='Initial Capital')
        ax2.axvline(metrics['median_final_value'], color='green', linestyle='-', linewidth=2, label='Median Final Value')
        ax2.set_xlabel('Final Portfolio Value ($)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Final Portfolio Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Distribution of total returns
        ax3 = axes[1, 0]
        ax3.hist(total_returns, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax3.axvline(metrics['expected_return'], color='blue', linestyle='--', linewidth=2, label='Expected Return')
        ax3.axvline(metrics['var_95'], color='red', linestyle='--', linewidth=2, label='95% VaR')
        ax3.set_xlabel('Total Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Total Returns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Risk-return scatter
        ax4 = axes[1, 1]

        # Calculate rolling returns and volatilities for each path
        path_returns = (portfolio_paths[:, 1:] / portfolio_paths[:, :-1]) - 1
        annual_returns = np.mean(path_returns, axis=1) * 365 * 100
        annual_volatilities = np.std(path_returns, axis=1) * np.sqrt(365) * 100

        scatter = ax4.scatter(annual_volatilities, annual_returns,
                             c=final_values, alpha=0.6, cmap='viridis')
        ax4.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax4.axvline(np.mean(annual_volatilities), color='red', linestyle='--', linewidth=1, alpha=0.7)

        # Add colorbar
        plt.colorbar(scatter, ax=ax4, label='Final Value ($)')

        ax4.set_xlabel('Annualized Volatility (%)')
        ax4.set_ylabel('Annualized Return (%)')
        ax4.set_title('Risk-Return Scatter by Simulation Path')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def stress_test(self, allocations: Dict, initial_capital: float, days: int,
                   n_simulations: int = 1000, stress_scenarios: Dict = None) -> Dict:
        """
        Run stress testing under various market conditions

        Args:
            allocations: Dictionary of asset allocations
            initial_capital: Starting portfolio value
            days: Number of trading days to simulate
            n_simulations: Number of simulation paths
            stress_scenarios: Dictionary of stress scenarios with multiplier factors

        Returns:
            Dictionary of stress test results
        """
        if stress_scenarios is None:
            stress_scenarios = {
                'normal': 1.0,
                'high_volatility': 1.5,
                'bear_market': 0.7,
                'crypto_winter': 0.5,
                'flash_crash': 0.3
            }

        results = {}

        for scenario, multiplier in stress_scenarios.items():
            # Modify crypto data for stress scenario
            stressed_data = {}

            for crypto in self.crypto_names:
                stressed_data[crypto] = self.crypto_data[crypto].copy()

                if scenario == 'high_volatility':
                    stressed_data[crypto]['volatility'] *= 1.5
                elif scenario == 'bear_market':
                    stressed_data[crypto]['expected_return'] *= 0.7
                elif scenario == 'crypto_winter':
                    stressed_data[crypto]['expected_return'] *= 0.5
                    stressed_data[crypto]['volatility'] *= 2.0
                elif scenario == 'flash_crash':
                    stressed_data[crypto]['expected_return'] *= 0.3
                    stressed_data[crypto]['volatility'] *= 3.0

            # Create temporary simulator with stressed data (reuse correlation matrix)
            temp_simulator = CryptoMonteCarlo(
                self.analyzer,
                stressed_data,
                precomputed_correlation=self.correlation_matrix
            )

            # Run simulation
            portfolio_paths, _ = temp_simulator.simulate_portfolio_paths(
                allocations, initial_capital, days, n_simulations
            )

            # Calculate metrics
            metrics, _ = temp_simulator.generate_metrics(portfolio_paths, initial_capital)
            results[scenario] = metrics

        return results
