#!/usr/bin/env python3
"""
CRYPTO RISK MANAGEMENT & CBQRA DASHBOARD - OPERATION STABILITY v3.1
Mission: Full-screen images + reliable downloads with NO width validation issues
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import tempfile
import shutil
from pathlib import Path as PPath
from itertools import combinations
from contextlib import contextmanager

# === GRACEFUL IMPORTS WITH FALLBACKS ===
try:
    from multi_crypto_bqr import MultiCryptoBQRAnalysis
    BQR_AVAILABLE = True
except ImportError:
    MultiCryptoBQRAnalysis = None
    BQR_AVAILABLE = False

try:
    from advanced_visualizations import AdvancedCryptoVisualizations
    VIZ_AVAILABLE = True
except ImportError:
    AdvancedCryptoVisualizations = None
    VIZ_AVAILABLE = False

try:
    from monte_carlo_simulator import CryptoMonteCarlo
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    CryptoMonteCarlo = None
    MONTE_CARLO_AVAILABLE = False

try:
    from glossary import GLOSSARY
    if not isinstance(GLOSSARY, dict) or len(GLOSSARY) == 0:
        raise ImportError("Empty glossary")
except ImportError:
    GLOSSARY = {
        "Sharpe Ratio": "Risk-adjusted return measure (higher is better)",
        "Volatility": "Standard deviation of returns (measure of risk)",
        "Max Drawdown": "Maximum peak-to-trough decline",
        "Value at Risk (VaR)": "Worst-case loss at a given confidence level",
        "Bayesian Quantile Regression (BQR)": "Advanced statistical method for forecasting",
        "Monte Carlo Simulation": "Random sampling technique for forecasting",
        "Kelly Criterion": "Optimal bet sizing formula"
    }

try:
    from thefuzz import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

# === STARTUP DIAGNOSTICS ===
print("\n" + "="*60)
print("🔧 CRYPTO DASHBOARD - OPERATION STABILITY v3.1")
print("="*60)
print(f"✅ BQR Analysis: {BQR_AVAILABLE}")
print(f"✅ Visualizations: {VIZ_AVAILABLE}")
print(f"✅ Monte Carlo: {MONTE_CARLO_AVAILABLE}")
print(f"✅ Glossary: {len(GLOSSARY)} terms")
print(f"✅ Fuzzy Search: {FUZZY_AVAILABLE}")
print("="*60 + "\n")

# === CONFIGURATION ===
PROFILE_SEEDS = {'conservative': 42, 'moderate': 123, 'aggressive': 789}

RISK_THRESHOLDS = {
    'max_correlation': 0.90,
    'volatility_multiplier': 2.5,
    'speculative_allocation_max': 0.15,
    'daily_loss_limit': 0.40,
    'flash_crash_assets': 3,
    'flash_crash_drop': 0.15,
    'rebalance_drift': 0.05
}

SPECULATIVE_PATTERNS = ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'MEME']

BASE_CONFIG = {
    'quantiles': [0.05, 0.5, 0.95],
    'forecast_days': 30,
    'rolling_window': 30,
    'output_dir': 'crypto_analysis_results',
    'dpi': 150,
    'mcmc_samples': 1000,
    'mcmc_tune': 1000,
    'mcmc_target_accept': 0.95,
    'mcmc_cores': 4
}

# === DATA STRUCTURES ===
crypto_data = {
    "XRP": {"volatility": 0.89, "correlation": 0.644, "sharpe": 1.2, "max_drawdown": 45, "beta": 1.1, "expected_return": 0.0},
    "XLM": {"volatility": 0.92, "correlation": 0.644, "sharpe": 0.8, "max_drawdown": 52, "beta": 0.95, "expected_return": 0.0},
    "XMR": {"volatility": 0.78, "correlation": 0.455, "sharpe": 0.3, "max_drawdown": 38, "beta": 0.75, "expected_return": 0.0},
    "TRX": {"volatility": 1.15, "correlation": 0.408, "sharpe": 2.1, "max_drawdown": 68, "beta": 1.35, "expected_return": 0.0},
    "DOGE": {"volatility": 1.85, "correlation": 0.306, "sharpe": 1.5, "max_drawdown": 89, "beta": 1.65, "expected_return": 0.0, "speculative": True}
}

risk_profiles = {
    "conservative": {"max_position": 0.15, "max_portfolio_vol": 0.50, "stop_loss": 0.10, "take_profit": 0.25, "kelly_scale": 0.65},
    "moderate": {"max_position": 0.25, "max_portfolio_vol": 0.75, "stop_loss": 0.15, "take_profit": 0.40, "kelly_scale": 1.00},
    "aggressive": {"max_position": 0.35, "max_portfolio_vol": 1.00, "stop_loss": 0.20, "take_profit": 0.60, "kelly_scale": 1.35}
}

# === HELPER FUNCTIONS ===
def is_speculative_asset(crypto_name):
    """Check if asset matches speculative patterns"""
    return any(pattern in crypto_name.upper() for pattern in SPECULATIVE_PATTERNS)

def fuzzy_search_glossary(search_term, glossary_dict, min_score=70, max_results=10):
    """Robust fuzzy search with fallback"""
    if not search_term or len(search_term.strip()) < 2:
        return []

    try:
        if FUZZY_AVAILABLE:
            matches = process.extract(search_term, list(glossary_dict.keys()),
                                    limit=max_results, scorer=fuzz.partial_ratio)
            return [(m[0], m[1], glossary_dict[m[0]]) for m in matches if m[1] >= min_score]
        else:
            results = []
            for term, definition in glossary_dict.items():
                if search_term.lower() in term.lower():
                    results.append((term, 100, definition))
            return results[:max_results]
    except Exception as e:
        print(f"Search error: {e}")
        return []

def calculate_position_size(crypto, risk_tolerance):
    """Calculate position size using Kelly Criterion"""
    if crypto not in crypto_data:
        crypto_data[crypto] = {
            "volatility": 1.0, "correlation": 0.5, "sharpe": 0.5,
            "max_drawdown": 50, "beta": 1.0, "expected_return": 0.0,
            "speculative": is_speculative_asset(crypto)
        }

    data = crypto_data[crypto]
    profile = risk_profiles[risk_tolerance]

    if data.get("expected_return", 0) > 0:
        kelly_fraction_raw = max(0, min(1, data["expected_return"] / (data["volatility"] ** 2)))
    else:
        kelly_fraction_raw = 0.25

    kelly_fraction_scaled = kelly_fraction_raw * profile["kelly_scale"]
    corr_adjustment = max(0.05, 1 - data.get("correlation", 0))
    vol_adjustment = (1.0 / max(0.01, data.get("volatility", 1.0))) * corr_adjustment
    base_allocation = profile["max_position"] * vol_adjustment * kelly_fraction_scaled

    if data.get("speculative", False):
        base_allocation = min(base_allocation, RISK_THRESHOLDS['speculative_allocation_max'])

    return min(base_allocation, profile["max_position"])

def check_risk_violations(allocations, correlation_matrix=None, current_cryptos=None):
    """Check for risk violations"""
    violations = []

    if not allocations or not current_cryptos:
        return violations

    if correlation_matrix is not None:
        max_corr = 0
        corr_pair = None
        try:
            corr_df = pd.DataFrame(correlation_matrix, index=current_cryptos, columns=current_cryptos)
            for i, crypto1 in enumerate(current_cryptos):
                for j, crypto2 in enumerate(current_cryptos):
                    if i < j:
                        corr_val = abs(corr_df.loc[crypto1, crypto2])
                        if corr_val > max_corr:
                            max_corr = corr_val
                            corr_pair = (crypto1, crypto2)

            if max_corr > RISK_THRESHOLDS['max_correlation']:
                violations.append({
                    'type': 'HIGH_CORRELATION',
                    'severity': 'WARNING',
                    'message': f"⚠️ Correlation between {corr_pair[0]} and {corr_pair[1]} is {max_corr:.2f}",
                    'recommendation': "Reduce combined exposure by 20%"
                })
        except Exception as e:
            print(f"Correlation check error: {e}")

    for crypto, allocation in allocations.items():
        if is_speculative_asset(crypto) and allocation > RISK_THRESHOLDS['speculative_allocation_max']:
            violations.append({
                'type': 'SPECULATIVE_OVERWEIGHT',
                'severity': 'CRITICAL',
                'message': f"🚨 {crypto} allocation is {allocation*100:.1f}% (max: 15%)",
                'recommendation': f"Reduce {crypto} to ≤15% with 25% trailing stop-loss"
            })

    return violations

def display_risk_warnings(violations):
    """Display risk warnings"""
    if not violations:
        return

    if 'dismissed_warnings' not in st.session_state:
        st.session_state['dismissed_warnings'] = set()

    critical = [v for v in violations if v['severity'] == 'CRITICAL']
    warnings = [v for v in violations if v['severity'] == 'WARNING']

    for v in critical:
        v_id = f"{v['type']}_{v['message']}"
        if v_id not in st.session_state['dismissed_warnings']:
            st.error(f"🚨 **{v['message']}**")
            st.warning(f"**Action Required**: {v['recommendation']}")

    for v in warnings:
        v_id = f"{v['type']}_{v['message']}"
        if v_id not in st.session_state['dismissed_warnings']:
            with st.expander(f"⚠️ {v['type']}", expanded=True):
                st.warning(v['message'])
                st.info(f"**Recommendation**: {v['recommendation']}")

@contextmanager
def sidebar_lockdown():
    """Context manager to disable sidebar during MCMC"""
    if 'sidebar_locked' not in st.session_state:
        st.session_state['sidebar_locked'] = False

    original_state = st.session_state['sidebar_locked']
    try:
        st.session_state['sidebar_locked'] = True
        yield
    finally:
        st.session_state['sidebar_locked'] = original_state

def nuclear_flush():
    """
    TACTICAL NUKE: Complete state annihilation with temp directory cleanup
    """
    print("🚨 NUCLEAR FLUSH INITIATED")

    # 1. Clean temp directory
    if 'temp_dir' in st.session_state:
        temp_dir = st.session_state['temp_dir']
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"✅ Nuked temp directory: {temp_dir}")
        except Exception as e:
            print(f"⚠️ Temp cleanup warning: {e}")

    # 2. Annihilate all session state
    keys_to_delete = list(st.session_state.keys())
    for key in keys_to_delete:
        try:
            del st.session_state[key]
        except:
            pass

    # 3. Rebuild critical infrastructure
    required_states = {
        'portfolio_state': {
            'allocations': None,
            'last_updated': None,
            'cbqra_completed': False,
            'risk_tolerance': 'moderate',
            'cbqra_running': False,
            'locked_profile': None
        },
        'forecasts': None,
        'analyzer': None,
        'correlation_matrix': None,
        'uploader_key': 0,
        'monte_carlo_cache': None,
        'monte_carlo_toggle': False,
        'backtest_cache': None,
        'dismissed_warnings': set(),
        'warning_confirmation_count': {},
        'last_allocations': None,
        'uploaded_file_names': None,
        'sidebar_locked': False,
        'use_uploaded': 'Use Default Files',
        'uploaded_files': None,
        'crypto_names_from_upload': []
    }

    for key, default_value in required_states.items():
        st.session_state[key] = default_value

    print("✅ NUCLEAR FLUSH COMPLETE - All clear")
    return True

def safe_cbqra_wrapper(config, risk_profile):
    """
    CRITICAL: Wrapper that ALWAYS clears cbqra_running flag
    """
    try:
        st.session_state['portfolio_state']['cbqra_running'] = True
        st.session_state['portfolio_state']['cbqra_completed'] = False

        result = run_cbqra_analysis(config, risk_profile)

        st.session_state['portfolio_state']['cbqra_running'] = False

        if result is not None:
            st.session_state['portfolio_state']['cbqra_completed'] = True
        else:
            st.session_state['portfolio_state']['cbqra_completed'] = False

        return result

    except Exception as e:
        st.session_state['portfolio_state']['cbqra_running'] = False
        st.session_state['portfolio_state']['cbqra_completed'] = False
        st.error(f"❌ CBQRA failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

    finally:
        if st.session_state['portfolio_state'].get('cbqra_running'):
            st.session_state['portfolio_state']['cbqra_running'] = False

def run_cbqra_analysis(config, risk_profile):
    """Main CBQRA analysis engine"""
    if not BQR_AVAILABLE:
        st.error("❌ CBQRA unavailable - required modules not found")
        return None

    output_dir = config['output_dir']
    PPath(output_dir).mkdir(parents=True, exist_ok=True)
    # 🔥 ADD THIS BLOCK HERE 🔥
    # Clean old pairwise comparisons to prevent ghost data
    try:
        old_pairwise = [f for f in os.listdir(output_dir) if f.startswith('pairwise_') and f.endswith('.png')]
        if old_pairwise:
            st.info(f"🗑️ Cleaning {len(old_pairwise)} old pairwise comparison(s)...")
            for f in old_pairwise:
                os.remove(os.path.join(output_dir, f))
    except Exception as e:
        st.warning(f"⚠️ Couldn't clean old pairwise files: {e}")
    # 🔥 END OF FIX 🔥

    csv_files = [item['file'] for item in config['crypto_data']]
    crypto_names = [item['name'] for item in config['crypto_data']]

    missing = [f for f in csv_files if not os.path.exists(f)]
    if missing:
        st.error(f"❌ Missing files: {', '.join(missing)}")
        return None

    try:
        analyzer = MultiCryptoBQRAnalysis(
            csv_files=csv_files,
            crypto_names=crypto_names,
            quantiles=config['quantiles']
        )

        st.info("🚀 Active MCMC sampling (check terminal)")
        analyzer.run_full_analysis(
            samples=config['mcmc_samples'],
            tune=config['mcmc_tune'],
            cores=config['mcmc_cores']
        )

        st.session_state['correlation_matrix'] = analyzer.correlation_matrix
        st.session_state['analyzer'] = analyzer

        if VIZ_AVAILABLE:
            try:
                advanced_viz = AdvancedCryptoVisualizations(analyzer)
                advanced_viz.generate_all_advanced_visualizations()
                st.success("✅ Advanced visualizations generated")
            except Exception as e:
                st.warning(f"Visualizations partially failed: {e}")

        metrics_file = os.path.join(output_dir, 'performance_metrics_multi.csv')
        if os.path.exists(metrics_file):
            try:
                metrics_df = pd.read_csv(metrics_file)
                for _, row in metrics_df.iterrows():
                    crypto = row['Crypto']
                    if crypto not in crypto_data:
                        crypto_data[crypto] = {
                            "volatility": 1.0, "correlation": 0.5, "sharpe": 0.5,
                            "max_drawdown": 50, "beta": 1.0, "expected_return": 0.0,
                            "speculative": is_speculative_asset(crypto)
                        }

                    if 'Ann. Volatility (%)' in metrics_df.columns:
                        vol = float(row['Ann. Volatility (%)']) / 100.0
                        if vol > 0:
                            crypto_data[crypto]['volatility'] = vol

                    if 'Sharpe Ratio' in metrics_df.columns:
                        crypto_data[crypto]['sharpe'] = float(row['Sharpe Ratio'])

                    if 'Max Drawdown (%)' in metrics_df.columns:
                        crypto_data[crypto]['max_drawdown'] = abs(float(row['Max Drawdown (%)']))
            except Exception as e:
                st.warning(f"Metrics update failed: {e}")

        forecasts = {}
        for crypto in crypto_names:
            try:
                last_date = analyzer.data_dict[crypto]['Date'].max()
                forecast_rows = analyzer.pred_df[
                    (analyzer.pred_df['Crypto'] == crypto) &
                    (analyzer.pred_df['Date'] == last_date)
                ]

                if not forecast_rows.empty:
                    prices = analyzer.data_dict[crypto]['Price']
                    daily_returns = prices.pct_change().dropna()
                    hist_annual = daily_returns.mean() * 365

                    b50 = analyzer.trace_dict[f"{crypto}_q0.5"].posterior["beta"].values.mean().item()
                    bqr_annual = b50 * 365

                    blended = 0.7 * hist_annual + 0.3 * bqr_annual

                    forecasts[crypto] = {
                        'Q0.05 (%)': forecast_rows[forecast_rows['Quantile'] == 0.05]['Estimate'].iloc[0],
                        'Q0.5 (%)': forecast_rows[forecast_rows['Quantile'] == 0.5]['Estimate'].iloc[0],
                        'Q0.95 (%)': forecast_rows[forecast_rows['Quantile'] == 0.95]['Estimate'].iloc[0]
                    }
                    crypto_data[crypto]['expected_return'] = blended
            except Exception as e:
                st.warning(f"Forecast failed for {crypto}: {e}")

        st.session_state['forecasts'] = forecasts

        new_allocations = {c: calculate_position_size(c, risk_profile) for c in crypto_names}
        total = sum(new_allocations.values())
        if total > 0:
            new_allocations = {c: alloc / total for c, alloc in new_allocations.items()}

        st.session_state['last_allocations'] = new_allocations.copy()

        st.session_state['portfolio_state'].update({
            'allocations': new_allocations,
            'last_updated': pd.Timestamp.now(),
            'locked_profile': risk_profile
        })

        # Clear Monte Carlo cache to force fresh simulation with new data
        st.session_state['monte_carlo_cache'] = None
        st.session_state['backtest_cache'] = None

        # If Monte Carlo toggle was on, turn it off to avoid confusion
        if 'monte_carlo_toggle' in st.session_state:
            st.session_state['monte_carlo_toggle'] = False

        st.success(f"✅ CBQRA completed with {risk_profile.upper()} profile")
        return True

    except Exception as e:
        st.error(f"❌ CBQRA failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def parse_backtest_period(period_str, data_dict):
    """Convert period string to dates"""
    end_date = min(data_dict[c]['Date'].max() for c in data_dict)

    period_map = {
        "Last 30 Days": 30,
        "Last 90 Days": 90,
        "Last 180 Days": 180,
        "Last Year": 365
    }

    if period_str in period_map:
        start_date = end_date - pd.Timedelta(days=period_map[period_str])
    else:
        start_date = max(data_dict[c]['Date'].min() for c in data_dict)

    return start_date, end_date

def get_rebalance_dates(start_date, end_date, frequency):
    """Generate rebalancing dates"""
    if frequency == "Daily":
        return pd.date_range(start_date, end_date, freq='D')
    elif frequency == "Weekly":
        return pd.date_range(start_date, end_date, freq='W-MON')
    else:
        return pd.date_range(start_date, end_date, freq='MS')

def run_portfolio_backtest(analyzer, allocations, initial_capital, start_date, end_date, rebalance_freq, risk_profile):
    """Run portfolio backtest"""
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive")
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")

    price_data = {}
    date_ranges = {}

    for crypto in allocations.keys():
        if crypto not in analyzer.data_dict:
            raise ValueError(f"Crypto {crypto} not found")

        df = analyzer.data_dict[crypto].copy()
        if 'Price' not in df.columns:
            raise ValueError(f"Price column not found for {crypto}")

        df = df[df['Price'] > 0]
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        if len(df) < 2:
            raise ValueError(f"Insufficient data for {crypto}")

        price_data[crypto] = df[['Date', 'Price']].set_index('Date')
        date_ranges[crypto] = (df['Date'].min(), df['Date'].max())

    common_start = max(dr[0] for dr in date_ranges.values())
    common_end = min(dr[1] for dr in date_ranges.values())

    if common_start >= common_end:
        raise ValueError("Insufficient date overlap")

    aligned_prices = pd.DataFrame({crypto: price_data[crypto]['Price'] for crypto in allocations.keys()})
    aligned_prices = aligned_prices.ffill(limit=3).dropna()

    if len(aligned_prices) < 2:
        raise ValueError("Insufficient aligned data")

    portfolio_values = []
    holdings = {}
    cash = initial_capital

    rebalance_dates = get_rebalance_dates(aligned_prices.index[0], aligned_prices.index[-1], rebalance_freq)
    rebalance_dates = [d for d in rebalance_dates if d in aligned_prices.index]
    if not rebalance_dates:
        rebalance_dates = [aligned_prices.index[0]]

    for date in aligned_prices.index:
        if date in rebalance_dates:
            for crypto in holdings:
                cash += holdings[crypto] * aligned_prices.loc[date, crypto]
            holdings = {}

            for crypto, allocation in allocations.items():
                target_value = cash * allocation
                price = aligned_prices.loc[date, crypto]
                if price > 0:
                    shares = target_value / price
                    holdings[crypto] = shares
                    cash -= target_value

            if cash < -0.01:
                cash = 0

        portfolio_value = cash
        for crypto, shares in holdings.items():
            price = aligned_prices.loc[date, crypto]
            portfolio_value += shares * price

        portfolio_values.append({'Date': date, 'Value': portfolio_value, 'Cash': cash})

    results_df = pd.DataFrame(portfolio_values).dropna(subset=['Value'])

    if len(results_df) < 2:
        raise ValueError("Insufficient portfolio data")

    returns = results_df['Value'].pct_change().dropna()
    final_value = results_df['Value'].iloc[-1]
    total_return = (final_value / initial_capital - 1.0) * 100

    days = (results_df['Date'].iloc[-1] - results_df['Date'].iloc[0]).days
    if days < 1:
        raise ValueError("Insufficient time period")

    ann_return = ((final_value / initial_capital) ** (365.25 / days) - 1) * 100
    volatility = returns.std() * np.sqrt(365) if len(returns) > 1 else 0
    sharpe = (ann_return / 100) / volatility if volatility > 0 else 0

    cummax = results_df['Value'].expanding().max()
    drawdown = (results_df['Value'] - cummax) / cummax
    max_drawdown = drawdown.min() * 100

    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0

    return {
        'results_df': results_df,
        'total_return': total_return,
        'ann_return': ann_return,
        'volatility': volatility * 100,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'final_value': final_value
    }

def save_visualization_to_disk(fig, filename, output_dir='crypto_analysis_results'):
    """Save matplotlib figure"""
    PPath(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    return filepath

# === STREAMLIT APP CONFIGURATION ===
st.set_page_config(
    page_title="Crypto Risk Manager Pro",
    layout="wide",
    page_icon="📊"
)

# === SESSION STATE INITIALIZATION ===
required_states = {
    'portfolio_state': {
        'allocations': None,
        'last_updated': None,
        'cbqra_completed': False,
        'risk_tolerance': 'moderate',
        'cbqra_running': False,
        'locked_profile': None
    },
    'forecasts': None,
    'analyzer': None,
    'correlation_matrix': None,
    'uploader_key': 0,
    'monte_carlo_cache': None,
    'monte_carlo_toggle': False,
    'backtest_cache': None,
    'dismissed_warnings': set(),
    'warning_confirmation_count': {},
    'last_allocations': None,
    'uploaded_file_names': None,
    'sidebar_locked': False,
    'use_uploaded': 'Use Default Files',
    'uploaded_files': None,
    'crypto_names_from_upload': []
}

for key, default_value in required_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# === ENHANCED SIDEBAR WITH LOCKDOWN ===
with st.sidebar:
    # LOCKDOWN CHECK
    is_locked = st.session_state.get('portfolio_state', {}).get('cbqra_running', False)

    if is_locked:
        st.error("🔒 SIDEBAR LOCKED")
        st.warning("⚙️ MCMC analysis in progress")
        st.info("Sidebar controls disabled to prevent reload")

        # Emergency stop only
        if st.button("🛑 EMERGENCY STOP", type="primary"):
            st.session_state['portfolio_state']['cbqra_running'] = False
            st.session_state['portfolio_state']['cbqra_completed'] = False
            st.warning("⚠️ Analysis interrupted!")
            st.rerun()
    else:
        # NORMAL SIDEBAR OPERATIONS
        st.markdown("---")
        st.header("📚 Smart Glossary")

        if GLOSSARY:
            st.caption(f"📖 {len(GLOSSARY)} terms available")
            search_term = st.text_input(
                "🔍 Search glossary:",
                placeholder="e.g., sharpe ratio, volatility...",
                key="sidebar_glossary_search"
            )

            if search_term:
                matches = fuzzy_search_glossary(search_term, GLOSSARY)
                if matches:
                    st.success(f"🎯 Found {len(matches)} matches:")
                    for term, score, definition in matches:
                        emoji = "🟢" if score >= 90 else "🟡" if score >= 80 else "🟠"
                        with st.expander(f"{emoji} {term} ({score}%)", expanded=(score >= 90)):
                            st.markdown(f"**Definition:** {definition}")
                else:
                    st.warning(f"No matches for '{search_term}'")
            else:
                selected = st.selectbox("Browse terms:", [""] + sorted(GLOSSARY.keys()))
                if selected:
                    st.markdown(f"**{selected}:** {GLOSSARY[selected]}")

        # FILE UPLOAD SECTION
        st.markdown("---")
        st.header("📂 Data Source")

        use_uploaded = st.radio(
            "Choose data source:",
            ["Use Default Files", "Upload CSV Files"],
            index=0 if st.session_state['use_uploaded'] == "Use Default Files" else 1
        )

        # Update session state
        st.session_state['use_uploaded'] = use_uploaded

        uploaded_files = None
        crypto_names_from_upload = []

        if use_uploaded == "Upload CSV Files":
            st.info("Upload CSV files with Date and Price columns")
            uploaded_files = st.file_uploader(
                "Select CSV files",
                type=['csv'],
                accept_multiple_files=True,
                key=f"file_uploader_{st.session_state.get('uploader_key', 0)}"
            )

            # Update session state
            st.session_state['uploaded_files'] = uploaded_files

            if uploaded_files:
                # Track current file set
                current_file_names = sorted([f.name for f in uploaded_files])

                # Detect changes
                if st.session_state.get('uploaded_file_names') != current_file_names:
                    st.info("🔄 New files detected - clearing previous analysis")
                    st.session_state['uploaded_file_names'] = current_file_names

                    # Clear analysis but preserve file upload state
                    st.session_state['portfolio_state']['cbqra_completed'] = False
                    st.session_state['portfolio_state']['cbqra_running'] = False
                    st.session_state['forecasts'] = None
                    st.session_state['analyzer'] = None
                    st.session_state['correlation_matrix'] = None
                    st.session_state['monte_carlo_cache'] = None
                    st.session_state['backtest_cache'] = None

                unique_files = {}
                for file in uploaded_files:
                    ticker = file.name.split('_')[0].split('.')[0].upper()
                    if ticker not in unique_files:
                        unique_files[ticker] = file

                crypto_names_from_upload = list(unique_files.keys())
                st.session_state['crypto_names_from_upload'] = crypto_names_from_upload
                st.success(f"✅ {len(crypto_names_from_upload)} files loaded")
                for crypto in crypto_names_from_upload:
                    st.write(f"• {crypto}")
            else:
                st.warning("⏳ No files uploaded")
                st.session_state['crypto_names_from_upload'] = []
        else:
            st.info("Using default dataset (XRP, XLM, XMR, TRX, DOGE)")
            st.success("✅ 5 default files loaded")
            st.session_state['crypto_names_from_upload'] = []

        # SYSTEM CONTROL - ENHANCED
        st.markdown("---")
        st.subheader("🔧 System Control")

        if st.button("🗑️ NUCLEAR FLUSH", type="secondary", help="Complete system reset"):
            if nuclear_flush():
                st.success("✅ NUCLEAR FLUSH COMPLETE!")
                st.info("Page will reload in 1 second...")
                time.sleep(1)
                st.rerun()

        with st.expander("🎯 Selective Reset", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Clear Analysis", key="clear_analysis"):
                    st.session_state['portfolio_state']['cbqra_completed'] = False
                    st.session_state['portfolio_state']['cbqra_running'] = False
                    st.session_state['forecasts'] = None
                    st.session_state['analyzer'] = None
                    st.success("✅ Cleared!")
                    st.rerun()

            with col2:
                if st.button("Reset Warnings", key="reset_warnings"):
                    st.session_state['dismissed_warnings'] = set()
                    st.success("✅ Reset!")
                    st.rerun()

            if st.button("Reset Monte Carlo", key="reset_mc"):
                if 'monte_carlo_toggle' in st.session_state:
                    st.session_state['monte_carlo_toggle'] = False
                st.session_state['monte_carlo_cache'] = None
                st.success("✅ MC Reset!")
                st.rerun()

            if st.button("Clear Uploads", key="clear_uploads"):
                # Clear temp directory
                if 'temp_dir' in st.session_state:
                    temp_dir = st.session_state['temp_dir']
                    try:
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir, ignore_errors=True)
                    except:
                        pass
                    del st.session_state['temp_dir']

                # Reset upload tracking
                st.session_state['uploaded_file_names'] = None
                st.session_state['uploader_key'] = st.session_state.get('uploader_key', 0) + 1
                st.success("✅ Uploads cleared!")
                st.rerun()

# === DYNAMIC CONFIG ===
# Get values from session state
use_uploaded = st.session_state.get('use_uploaded', 'Use Default Files')
uploaded_files = st.session_state.get('uploaded_files', None)
crypto_names_from_upload = st.session_state.get('crypto_names_from_upload', [])

if use_uploaded == "Upload CSV Files" and uploaded_files:
    if 'temp_dir' not in st.session_state:
        st.session_state['temp_dir'] = tempfile.mkdtemp()

    temp_dir = st.session_state['temp_dir']
    crypto_data_list = []

    for file in uploaded_files:
        ticker = file.name.split('_')[0].split('.')[0].upper()
        temp_path = os.path.join(temp_dir, file.name)
        with open(temp_path, 'wb') as f:
            f.write(file.getbuffer())
        crypto_data_list.append({'file': temp_path, 'name': ticker})

    CONFIG = {**BASE_CONFIG, 'crypto_data': crypto_data_list}
else:
    CONFIG = {
        **BASE_CONFIG,
        'crypto_data': [
            {'file': 'xrp_2017-09-13_2025-10-14.csv', 'name': 'XRP'},
            {'file': 'xlm_2017-09-13_2025-10-14.csv', 'name': 'XLM'},
            {'file': 'xmr_2017-09-13_2025-10-14.csv', 'name': 'XMR'},
            {'file': 'trx_2017-09-13_2025-10-14.csv', 'name': 'TRX'},
            {'file': 'doge_2017-09-13_2025-10-14.csv', 'name': 'DOGE'},
        ]
    }

# === MAIN TABS ===
tab1, tab2, tab3 = st.tabs(["🎯 Risk Dashboard", "🧠 CBQRA", "📈 Backtesting"])

# === TAB 1: RISK DASHBOARD ===
with tab1:
    st.header("📊 Portfolio Risk Dashboard")
    st.info("💡 **Pro Tip**: Check the Smart Glossary in the sidebar for term definitions")

    if st.checkbox("📘 Show Learning Center"):
        with st.expander("🎓 Crypto Quant Education Hub", expanded=True):
            search_term = st.text_input("🔍 Search learning center:", key="edu_search")

            if search_term:
                matches = fuzzy_search_glossary(search_term, GLOSSARY, min_score=60)
                if matches:
                    st.success(f"🎯 Found {len(matches)} relevant concepts:")
                    for term, score, definition in matches:
                        with st.expander(f"📖 {term} (relevance: {score}%)", expanded=False):
                            st.markdown(f"**Definition:** {definition}")
                else:
                    st.info("Try different keywords")

            categories = {
                "📈 Risk & Return Metrics": ["Sharpe Ratio", "Max Drawdown", "Volatility", "Value at Risk (VaR)"],
                "🔮 Simulation & Forecasting": ["Monte Carlo Simulation", "Bayesian Quantile Regression (BQR)"],
                "⚖️ Portfolio Construction": ["Kelly Criterion"],
            }

            for category, terms in categories.items():
                st.markdown(f"### {category}")
                for term in terms:
                    if term in GLOSSARY:
                        st.markdown(f"**{term}**: {GLOSSARY[term]}")
                st.markdown("---")

    st.subheader("💰 Portfolio Configuration")
    portfolio_value = st.number_input("Portfolio Value ($)", value=10000.0, min_value=100.0, step=1000.0)

    pstate = st.session_state['portfolio_state']

    st.subheader("🎯 Risk Tolerance Selection")

    if pstate.get('cbqra_completed') and pstate.get('locked_profile'):
        st.warning(f"🔒 **Analysis locked to {pstate['locked_profile'].upper()} profile**. Re-run CBQRA to change.")

    risk_tolerance = st.radio(
        "Select your risk profile:",
        ["conservative", "moderate", "aggressive"],
        index=["conservative", "moderate", "aggressive"].index(pstate.get('risk_tolerance', 'moderate'))
    )

    if risk_tolerance != pstate.get('risk_tolerance'):
        st.session_state['portfolio_state']['risk_tolerance'] = risk_tolerance

        if pstate.get('cbqra_completed') and pstate.get('locked_profile') == risk_tolerance:
            current_cryptos = [item['name'] for item in CONFIG['crypto_data']]
            new_allocations = {crypto: calculate_position_size(crypto, risk_tolerance) for crypto in current_cryptos}
            total = sum(new_allocations.values())
            if total > 0:
                new_allocations = {c: alloc / total for c, alloc in new_allocations.items()}
            st.session_state['portfolio_state']['allocations'] = new_allocations
            st.success(f"🔄 Allocations updated for {risk_tolerance.upper()} profile")

    current_cryptos = [item['name'] for item in CONFIG['crypto_data']]

    if pstate['allocations'] is None and current_cryptos:
        allocations = {crypto: calculate_position_size(crypto, risk_tolerance) for crypto in current_cryptos}
        total = sum(allocations.values())
        if total > 0:
            allocations = {crypto: alloc / total for crypto, alloc in allocations.items()}
    else:
        allocations = pstate['allocations'] if pstate['allocations'] else {}

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk Profile", risk_tolerance.upper())
    with col2:
        st.metric("Assets in Portfolio", f"{len(current_cryptos)}")

    if pstate.get('cbqra_completed'):
        locked_profile = pstate.get('locked_profile', 'unknown')
        if locked_profile == risk_tolerance:
            st.success(f"✅ Dashboard synced with **{risk_tolerance.upper()}** analysis")
        else:
            st.error(f"❌ **MISMATCH**: Showing {risk_tolerance.upper()}, analysis used {locked_profile.upper()}")
    else:
        st.info(f"📊 Ready for analysis with **{risk_tolerance.upper()}** profile")

    if current_cryptos and allocations:
        portfolio_metrics = {
            "weighted_vol": sum(allocations.get(crypto, 0) * crypto_data.get(crypto, {}).get("volatility", 1.0) for crypto in current_cryptos),
            "weighted_sharpe": sum(allocations.get(crypto, 0) * crypto_data.get(crypto, {}).get("sharpe", 0.5) for crypto in current_cryptos),
            "max_drawdown": max([allocations.get(crypto, 0) * crypto_data.get(crypto, {}).get("max_drawdown", 50) for crypto in current_cryptos] + [0])
        }

        st.subheader("📈 Portfolio Health Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Volatility", f"{portfolio_metrics['weighted_vol'] * 100:.1f}%")
        col2.metric("Sharpe Ratio", f"{portfolio_metrics['weighted_sharpe']:.2f}")
        col3.metric("Worst Case Drawdown", f"{portfolio_metrics['max_drawdown']:.1f}%")

        st.subheader("💼 Recommended Positions")
        for crypto in current_cryptos:
            allocation = allocations.get(crypto, 0)
            position_value = portfolio_value * allocation
            profile = risk_profiles[risk_tolerance]
            stop_loss = position_value * (1 - profile["stop_loss"])
            take_profit = position_value * (1 + profile["take_profit"])

            label = f"{crypto}: {allocation*100:.1f}% (${position_value:,.0f})"
            if is_speculative_asset(crypto):
                label = f"⚠️ {label} [SPECULATIVE]"

            with st.expander(label):
                col_sl, col_tp = st.columns(2)
                with col_sl:
                    st.markdown(f"🔴 **Stop Loss:** ${stop_loss:,.0f}")
                with col_tp:
                    st.markdown(f"🟢 **Take Profit:** ${take_profit:,.0f}")

                if is_speculative_asset(crypto):
                    st.warning(f"⚠️ **{crypto} is high-risk**. Use 25% trailing stop-loss.")

        violations = check_risk_violations(allocations, st.session_state.get('correlation_matrix'), current_cryptos)

        if violations:
            st.markdown("---")
            st.subheader("🚨 Active Risk Monitoring")
            display_risk_warnings(violations)

    if st.session_state['forecasts']:
        st.subheader("🔮 BQR Trend Forecasts")
        forecast_df = pd.DataFrame(st.session_state['forecasts']).T
        st.dataframe(forecast_df.style.format("{:.2f}"), width='stretcch')
    else:
        st.info("📈 Run CBQRA analysis to get Bayesian trend forecasts")

    profile_match = pstate.get('locked_profile') == pstate.get('risk_tolerance') if pstate.get('cbqra_completed') else True

    # MONTE CARLO SECTION - FIXED CONDITIONAL RENDERING
    st.markdown("---")
    st.subheader("🎲 Monte Carlo Simulations")

    if not pstate.get('cbqra_completed'):
        # PRE-CBQRA STATE - Show disabled message
        st.warning("⏳ **Monte Carlo functionality disabled**")
        st.info("📊 To enable Monte Carlo simulations, you must first run CBQRA analysis in the 'CBQRA' tab.")

        # Show preview of what will be available
        with st.expander("📖 What Monte Carlo Simulations Provide"):
            st.markdown("""
            Once CBQRA analysis is complete, Monte Carlo simulations will allow you to:

            - **Project portfolio performance** using thousands of random scenarios
            - **Visualize probability distributions** of potential outcomes
            - **Calculate risk metrics** like Value at Risk (VaR) and Conditional VaR
            - **Run stress tests** simulating market crash scenarios
            - **Estimate probability** of beating market benchmarks

            **Configuration options:**
            - Simulation count: 500 / 1000 / 2000 paths
            - Time horizons: 3 months / 6 months / 1 year / 2 years
            - Risk-adjusted random seed based on your profile
            """)

    elif not profile_match:
        # PROFILE MISMATCH STATE
        st.error("❌ **Cannot run Monte Carlo with mismatched profiles!**")
        st.warning(f"Current: **{risk_tolerance.upper()}** | Analysis: **{pstate.get('locked_profile', 'unknown').upper()}**")
        st.info("Please re-run CBQRA with the current risk profile to enable Monte Carlo simulations.")

    elif not MONTE_CARLO_AVAILABLE:
        # MODULE NOT AVAILABLE
        st.error("❌ **Monte Carlo module not found**")
        st.info("The 'monte_carlo_simulator.py' module is required for this functionality.")

    else:
        # FULLY ENABLED STATE - Profile matches and CBQRA completed
        st.success(f"✅ Monte Carlo ready | Profile: **{risk_tolerance.upper()}**")

        if st.checkbox("Show Monte Carlo Projections", key="monte_carlo_toggle"):
            st.info("💡 Monte Carlo simulations project potential portfolio performance using random sampling")

            col1, col2 = st.columns(2)
            with col1:
                n_simulations = st.selectbox("Number of Simulations", [500, 1000, 2000], index=1)
            with col2:
                time_horizon = st.selectbox("Time Horizon", ["3 months", "6 months", "1 year", "2 years"], index=2)

            days_map = {"3 months": 90, "6 months": 180, "1 year": 365, "2 years": 730}
            days = days_map[time_horizon]

            if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
                with st.spinner(f"Running {n_simulations} Monte Carlo simulations..."):
                    try:
                        if st.session_state['analyzer'] is None:
                            st.error("❌ Analyzer not found. Run CBQRA first.")
                        else:
                            # Get current crypto data for Monte Carlo
                            current_crypto_data = {}
                            for crypto in allocations.keys():
                                if crypto in crypto_data:
                                    current_crypto_data[crypto] = crypto_data[crypto]
                                else:
                                    # Fallback data if crypto not in main dataset
                                    current_crypto_data[crypto] = {
                                        "volatility": 1.0,
                                        "correlation": 0.5,
                                        "sharpe": 0.5,
                                        "max_drawdown": 50,
                                        "beta": 1.0,
                                        "expected_return": 0.0
                                    }

                            profile_seed = PROFILE_SEEDS.get(risk_tolerance, 42)
                            np.random.seed(profile_seed)

                            # Initialize and run Monte Carlo
                            mc = CryptoMonteCarlo(
                                st.session_state['analyzer'],
                                current_crypto_data,
                                precomputed_correlation=st.session_state['correlation_matrix']
                            )

                            portfolio_paths, asset_paths = mc.simulate_portfolio_paths(
                                allocations=allocations,
                                initial_capital=portfolio_value,
                                days=days,
                                n_simulations=n_simulations
                            )

                            metrics, final_values = mc.generate_metrics(portfolio_paths, portfolio_value)

                            # CACHE RESULTS IN SESSION STATE
                            st.session_state['monte_carlo_cache'] = {
                                'portfolio_paths': portfolio_paths,
                                'asset_paths': asset_paths,
                                'metrics': metrics,
                                'final_values': final_values,
                                'mc_instance': mc,
                                'config': {
                                    'n_simulations': n_simulations,
                                    'time_horizon': time_horizon,
                                    'days': days,
                                    'risk_tolerance': risk_tolerance,
                                    'portfolio_value': portfolio_value
                                }
                            }

                            st.success(f"✅ Simulation completed: {n_simulations} paths over {days} days")
                            st.rerun()  # Rerun to display cached results

                    except Exception as e:
                        st.error(f"❌ Monte Carlo simulation failed: {e}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())

            # DISPLAY CACHED RESULTS IF AVAILABLE
            if st.session_state.get('monte_carlo_cache'):
                cache = st.session_state['monte_carlo_cache']
                metrics = cache['metrics']

                st.markdown("---")
                st.subheader("📊 Monte Carlo Results")

                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Expected Return", f"{metrics['expected_return']:.1f}%")
                col2.metric("Best Case", f"{metrics['best_case']:.1f}%")
                col3.metric("Worst Case", f"{metrics['worst_case']:.1f}%")
                col4.metric("95% VaR", f"{metrics['var_95']:.1f}%")

                col5, col6, col7 = st.columns(3)
                col5.metric("Probability Positive", f"{metrics['probability_positive']:.1f}%")
                col6.metric("Beat SP500 Chance", f"{metrics['probability_beating_sp500']:.1f}%")
                col7.metric("Median Final Value", f"${metrics['median_final_value']:,.0f}")

                # Display Monte Carlo plot# Display Monte Carlo plot
                st.subheader("📈 Simulation Visualization")
                try:
                    mc = cache['mc_instance']
                    fig = mc.plot_monte_carlo_results(
                        cache['portfolio_paths'],
                        cache['config']['portfolio_value'],
                        metrics
                    )
                    # Display without fullscreen button
                    st.pyplot(fig)
                    st.caption("💡 To enlarge: Right-click → 'Open image in new tab' → [Enter]")

                    # Save and provide download
                    config = cache['config']
                    mc_filename = f"monte_carlo_{config['risk_tolerance']}_{config['time_horizon'].replace(' ', '_')}_{config['n_simulations']}sims.png"
                    save_visualization_to_disk(fig, mc_filename)

                    # ✅ DOWNLOAD USING RAW BYTES - No width validation
                    with open(f"crypto_analysis_results/{mc_filename}", "rb") as f:
                        img_bytes = f.read()
                        st.download_button(
                            label="📥 Download Monte Carlo Chart",
                            data=img_bytes,
                            file_name=mc_filename,
                            mime="image/png",
                            key="dl_monte_carlo"
                        )
                except Exception as e:
                    st.error(f"Error displaying plot: {e}")

                # Risk analysis metrics
                st.subheader("📊 Risk Analysis")
                col8, col9, col10 = st.columns(3)
                col8.metric("Return Volatility", f"{metrics['return_std']:.1f}%")
                col9.metric("Conditional VaR (CVaR)", f"{metrics['cvar_95']:.1f}%")
                col10.metric("Final Value Std Dev", f"${metrics['final_value_std']:,.0f}")

                # STRESS TESTING
                if st.checkbox("🔬 Include Stress Testing", key="stress_test_toggle"):
                    st.subheader("⚠️ Stress Test Scenarios")

                    st.info("""
                    **What is Stress Testing?**

                    Stress testing shows how your portfolio would perform during major market crashes.
                    These scenarios apply historical crisis drawdowns to your current Monte Carlo projections.

                    **Scenarios Based On:**
                    - **2008 Financial Crisis**: -50% market decline
                    - **2020 COVID Crash**: -35% sudden drop
                    - **Bear Market**: -20% sustained decline
                    - **Mild Correction**: -10% typical pullback
                    """)

                    stress_scenarios = {
                        "2008 Financial Crisis (-50%)": 0.5,
                        "2020 COVID Crash (-35%)": 0.65,
                        "Bear Market (-20%)": 0.8,
                        "Mild Correction (-10%)": 0.9
                    }

                    final_values = cache['final_values']
                    stress_results = []
                    for scenario, multiplier in stress_scenarios.items():
                        stressed_final = final_values * multiplier
                        stressed_return = (np.mean(stressed_final) / cache['config']['portfolio_value'] - 1) * 100
                        stressed_median = np.median(stressed_final)

                        stress_results.append({
                            "Scenario": scenario,
                            "Expected Return": f"{stressed_return:.2f}%",
                            "Median Final Value": f"${stressed_median:,.0f}",
                            "Worst Case": f"${np.min(stressed_final):,.0f}"
                        })

                    stress_df = pd.DataFrame(stress_results)
                    st.dataframe(stress_df, width='stretch')

                    st.markdown("---")
                    st.subheader("💡 Stress Test Interpretation")

                    worst_scenario = stress_results[0]
                    st.warning(f"""
                    **Key Takeaways:**
                    - In a 2008-style crisis, expect returns around **{worst_scenario['Expected Return']}**
                    - Your worst-case portfolio value could be **{worst_scenario['Worst Case']}**
                    - Use these scenarios to set appropriate stop-losses
                    - Consider keeping 20-30% cash reserves for crisis buying opportunities
                    """)

                    stress_csv = os.path.join('crypto_analysis_results', f'stress_test_{cache["config"]["risk_tolerance"]}.csv')
                    stress_df.to_csv(stress_csv, index=False)

                    # ✅ DOWNLOAD CSV USING RAW BYTES
                    with open(stress_csv, "rb") as f:
                        csv_bytes = f.read()
                        st.download_button(
                            label="📥 Download Stress Test Results",
                            data=csv_bytes,
                            file_name=f"stress_test_{cache['config']['risk_tolerance']}.csv",
                            mime="text/csv",
                            key="dl_stress_test"
                        )

# === TAB 2: CBQRA ANALYSIS ===
with tab2:
    st.header("🧠 CBQRA Engine")

    pstate = st.session_state['portfolio_state']

    if pstate.get('cbqra_running') and not pstate.get('cbqra_completed'):
        st.error("🚨 **ANALYSIS IN PROGRESS** - Don't close this tab!")
        st.warning("⚙️ **MCMC Sampling Running** - Check terminal for progress")

        with st.expander("📊 Monitoring Instructions", expanded=True):
            st.markdown("""
            ### Real-Time Progress Monitoring

            **Where to look:**
            - **Terminal/Console** where you ran `streamlit run complete8.py`
            - **VS Code**: View → Terminal
            - **PyCharm**: Run/Debug console at bottom
            - **Streamlit.app**: click '<Manage app' at bottom, right

            **Expected output:**
            ```
            Sampling 4 chains for 1,000 tune and 1,000 draw iterations...
            Progress | Draws | Divergences | Step size | Speed
            ████████ | 2000  | 0           | 0.559     | 79.54 draws/s
            ```

            **⏳ Expected duration:** 1-3 minutes per asset
            **🔄 Page auto-updates when complete**
            """)

    elif pstate.get('cbqra_completed'):
        st.success("✅ Analysis completed successfully!")

        # ENHANCED VISUALIZATION ORGANIZATION
        output_dir = CONFIG['output_dir']
        st.subheader("📈 Analysis Visualizations")

        # Get all PNG files in the output directory
        all_png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')] if os.path.exists(output_dir) else []

        # Create categorized visualization lists
        main_viz_files = []
        pairwise_viz_files = []

        # Main advanced visualizations
        main_viz = [
            ('correlation_matrix_heatmap.png', 'Correlation Matrix Heatmap'),
            ('rolling_correlation_heatmap.png', 'Rolling Correlation Heatmap'),
            ('volatility_comparison.png', 'Volatility Comparison'),
            ('performance_dashboard.png', 'Performance Dashboard'),
            ('return_distributions.png', 'Return Distributions'),
            ('cumulative_returns_comparison.png', 'Cumulative Returns'),
            ('risk_return_scatter.png', 'Risk-Return Scatter'),
            ('drawdown_comparison.png', 'Drawdown Comparison'),
            ('multi_crypto_correlation_matrix.png', 'Multi-Crypto Correlation'),
            ('multi_asset_summary.png', 'Multi-Asset Summary'),
            ('forecast_comparison.png', 'Forecast Comparison')
        ]

        # Add main visualizations that exist
        for file, title in main_viz:
            if file in all_png_files:
                main_viz_files.append((file, title))

        # Add pairwise comparisons to separate list
        pairwise_files = [f for f in all_png_files if f.startswith('pairwise_')]
        for pair_file in pairwise_files:
            # Convert filename to readable title
            crypto_pair = pair_file.replace('pairwise_', '').replace('.png', '').replace('_vs_', ' vs ')
            pairwise_viz_files.append((pair_file, crypto_pair))

            # Sort for consistent display
            main_viz_files.sort()
            pairwise_viz_files.sort()

        # ===== DISPLAY MAIN VISUALIZATIONS =====
        if main_viz_files:
            st.subheader("📊 Main Visualizations")
            st.success(f"✅ Found {len(main_viz_files)} main visualizations")

            # Display in 2-column grid
            cols = st.columns(2)
            for idx, (viz_file, viz_title) in enumerate(main_viz_files):
                file_path = os.path.join(output_dir, viz_file)
                with cols[idx % 2]:
                    try:
                        # Display without fullscreen button
                        st.image(file_path, caption=viz_title)
                        st.caption("💡 To enlarge: Right-click → 'Open image in new tab' → [Enter]")

                        # ✅ DOWNLOAD USING RAW BYTES - No width validation
                        with open(file_path, "rb") as f:
                            img_bytes = f.read()
                            st.download_button(
                                label=f"📥 Download {viz_title}",
                                data=img_bytes,
                                file_name=viz_file,
                                mime="image/png",
                                key=f"dl_main_{viz_file}"
                            )
                    except Exception as e:
                        st.error(f"Error loading {viz_file}: {e}")
        else:
             st.warning("⏳ No main visualizations found")

        # ===== DISPLAY PAIRWISE COMPARISONS =====
        if pairwise_viz_files:
            st.markdown("---")
            st.subheader("🔍 Pairwise Asset Comparisons")
            st.info(f"🎯 Found {len(pairwise_viz_files)} pairwise comparisons")

            # Dropdown selector
            pairwise_options = {title: (file, title) for file, title in pairwise_viz_files}
            selected_pairwise = st.selectbox(
                "Select pairwise comparison to view:",
                options=list(pairwise_options.keys()),
                index=0,
                key="pairwise_selector"
            )
            if selected_pairwise:
                selected_file, selected_title = pairwise_options[selected_pairwise]
                file_path = os.path.join(output_dir, selected_file)

                try:
                    # Display without fullscreen button
                    st.image(file_path, caption=f"Pairwise Comparison: {selected_title}")
                    st.caption("💡 To enlarge: Right-click → 'Open image in new tab' → [Enter]")

                    # ✅ DOWNLOAD USING RAW BYTES - No width validation
                    with open(file_path, "rb") as f:
                        img_bytes = f.read()
                        st.download_button(
                            label=f"📥 Download {selected_title} Comparison",
                            data=img_bytes,
                            file_name=selected_file,
                            mime="image/png",
                            key=f"dl_pairwise_{selected_file}"
                        )
                except Exception as e:
                    st.error(f"Error loading {selected_file}: {e}")

            # Optional: Show all pairwise files list
            with st.expander("📋 View All Pairwise Comparisons List", expanded=False):
                st.write("All available pairwise comparisons:")
                for file, title in pairwise_viz_files:
                    st.write(f"• {title}")
        else:
            st.info("🔍 No pairwise comparisons found")

        # ===== METRICS TABLE =====
        metrics_file = os.path.join(output_dir, 'performance_metrics_multi.csv')
        if os.path.exists(metrics_file):
            st.markdown("---")
            st.subheader("📊 Performance Metrics")
            metrics_df = pd.read_csv(metrics_file)
            st.dataframe(metrics_df, width='stretch')

            csv_data = metrics_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Metrics CSV",
                csv_data,
                "performance_metrics.csv",
                "text/csv",
                key="dl_metrics"
            )

    if use_uploaded == "Upload CSV Files" and uploaded_files:
        st.info(f"📂 Using {len(crypto_names_from_upload)} uploaded files")
    elif use_uploaded == "Upload CSV Files" and not uploaded_files:
        st.warning("⚠️ No files uploaded yet")
    else:
        st.info("📂 Using default dataset (XRP, XLM, XMR, TRX, DOGE)")

    st.warning("⚠️ Total time required for completing Markov Chain Monte Carlo simulations depends solely on hardware capability and number of assets being analyzed.")

    current_risk = st.session_state['portfolio_state']['risk_tolerance']
    st.info(f"🎯 Current Risk Profile: **{current_risk.upper()}**")

    if pstate.get('cbqra_completed') and pstate.get('locked_profile') != current_risk:
        st.error(f"⚠️ **Profile mismatch**: Analysis used {pstate['locked_profile'].upper()}, current is {current_risk.upper()}")
        st.info("Running new analysis will use current profile and clear previous results.")

    can_run = True
    if use_uploaded == "Upload CSV Files" and not uploaded_files:
        can_run = False
        st.error("⚠️ Upload CSV files first")

    button_disabled = not can_run or (pstate.get('cbqra_running') and not pstate.get('cbqra_completed'))

    if st.button("🚀 Run CBQRA Analysis", type="primary", disabled=button_disabled):
        if not BQR_AVAILABLE:
            st.error("❌ CBQRA unavailable - required modules not found")
        else:
            st.session_state['portfolio_state']['cbqra_running'] = True
            st.session_state['portfolio_state']['cbqra_completed'] = False
            st.rerun()

    if pstate.get('cbqra_running') and not pstate.get('cbqra_completed'):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(30):
            progress_bar.progress(i + 1)
            status_text.text(f"⚙️ Initializing Bayesian models... {i+1}%")
            time.sleep(0.05)

        status_text.text("🔬 Running MCMC chains...")

        with st.spinner("Running Bayesian simulations puts an extra load on your CPU. We recommend keeping non-critical, background processes to a minimum during active MCMC ops."):
            results = safe_cbqra_wrapper(CONFIG, current_risk)

        if results:
            for i in range(30, 100):
                progress_bar.progress(i + 1)
                status_text.text(f"✅ Generating visualizations... {i+1}%")
                time.sleep(0.01)

        progress_bar.empty()
        status_text.empty()

        if results:
            st.success("✅ Analysis complete! Updating dashboard...")
            st.rerun()
        else:
            st.error("❌ Analysis failed. Check error messages above.")

# === TAB 3: BACKTESTING ===
with tab3:
    st.header("📆 Backtesting Results")

    pstate = st.session_state['portfolio_state']

    if not pstate['cbqra_completed']:
        st.info("📊 Run CBQRA analysis first to enable backtesting.")
        st.markdown(f"""
        ### What is Backtesting?

        Test your portfolio strategy against historical data to see how it would have performed.

        **Features:**
        - Historical performance simulation
        - Risk-adjusted returns analysis
        - Drawdown scenarios
        - Configurable rebalancing

        **Current profile**: **{pstate['risk_tolerance'].upper()}**
        """)
    else:
        if not profile_match:
            st.error("❌ **Cannot run backtest with mismatched profiles!**")
            st.warning(f"Current: **{pstate['risk_tolerance'].upper()}** | Analysis: **{pstate.get('locked_profile', 'unknown').upper()}**")
            st.info("Please re-run CBQRA or switch profiles in Dashboard tab.")
        else:
            st.success(f"Backtesting ready | Using **{pstate['risk_tolerance'].upper()}** profile")

            col1, col2, col3 = st.columns(3)
            with col1:
                backtest_period = st.selectbox("Backtest Period",
                    ["Last 30 Days", "Last 90 Days", "Last 180 Days", "Last Year", "All Available Data"],
                    index=3)
            with col2:
                rebalance_freq = st.selectbox("Rebalancing Frequency",
                    ["Daily", "Weekly", "Monthly"],
                    index=1)
            with col3:
                initial_capital = st.number_input("Initial Capital ($)",
                    min_value=100.0, max_value=1000000.0, value=10000.0, step=1000.0)

            if st.button("Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    try:
                        if st.session_state['analyzer'] is None:
                            st.error("❌ Analyzer not found. Run CBQRA first.")
                        else:
                            analyzer = st.session_state['analyzer']
                            current_allocations = pstate['allocations']

                            start_date, end_date = parse_backtest_period(backtest_period, analyzer.data_dict)
                            profile = risk_profiles[pstate['risk_tolerance']]

                            results = run_portfolio_backtest(
                                analyzer, current_allocations, initial_capital,
                                start_date, end_date, rebalance_freq, profile
                            )

                            if results:
                                st.subheader(f"Backtest Results — {pstate['risk_tolerance'].upper()} Profile")

                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Total Return", f"{results['total_return']:.2f}%",
                                          f"{results['ann_return']:.2f}% ann.")
                                col2.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")
                                col3.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
                                col4.metric("Win Rate", f"{results['win_rate']:.1f}%")

                                col5, col6 = st.columns(2)
                                col5.metric("Initial Capital", f"${initial_capital:,.2f}")
                                col6.metric("Final Value", f"${results['final_value']:,.2f}",
                                          f"${results['final_value'] - initial_capital:+,.2f}")

                                st.subheader("Portfolio Value Over Time")
                                fig, ax = plt.subplots(figsize=(12, 6))
                                ax.plot(results['results_df']['Date'], results['results_df']['Value'],
                                       linewidth=2, color='#1f77b4')
                                ax.axhline(initial_capital, color='red', linestyle='--',
                                          alpha=0.5, label='Initial Capital')
                                ax.fill_between(results['results_df']['Date'],
                                               results['results_df']['Value'],
                                               initial_capital, alpha=0.3,
                                               color='green' if results['total_return'] > 0 else 'red')
                                ax.set_xlabel("Date")
                                ax.set_ylabel("Portfolio Value ($)")
                                ax.set_title(f"Portfolio Performance ({backtest_period}, {rebalance_freq} Rebalancing)")
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                plt.xticks(rotation=45)
                                plt.tight_layout()

                                # Display without fullscreen button
                                st.pyplot(fig)
                                st.caption("💡 To enlarge: Right-click → 'Open image in new tab' → [Enter]")

                                bt_filename = f"backtest_{pstate['risk_tolerance']}_{backtest_period.replace(' ', '_')}.png"
                                save_visualization_to_disk(fig, bt_filename)
                                # ✅ DOWNLOAD USING RAW BYTES - No width validation
                                with open(f"crypto_analysis_results/{bt_filename}", "rb") as f:
                                    img_bytes = f.read()
                                    st.download_button(
                                        label="📥 Download Backtest Chart",
                                        data=img_bytes,
                                        file_name=bt_filename,
                                        mime="image/png",
                                        key="dl_backtest"
                                    )

                                st.subheader("Portfolio Allocation Used")
                                alloc_df = pd.DataFrame([
                                    {"Asset": k, "Allocation": f"{v*100:.2f}%",
                                     "Value": f"${initial_capital*v:,.2f}"}
                                    for k, v in current_allocations.items()
                                ])
                                st.dataframe(alloc_df, width='stretch')

                                actual_start = results['results_df']['Date'].iloc[0]
                                actual_end = results['results_df']['Date'].iloc[-1]
                                st.success(f"Backtest completed: {len(results['results_df'])} days from {actual_start.date()} to {actual_end.date()}")

                    except Exception as e:
                        st.error(f"❌ Backtest failed: {e}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())

# === FOOTER ===
st.markdown("---")
st.markdown("**🚀 Crypto Risk Manager Pro - Operation Stability v3.1** | Full-screen images + reliable downloads | Nuclear flush operational")
