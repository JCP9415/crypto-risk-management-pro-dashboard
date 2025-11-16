import os
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")


class MultiCryptoBQRAnalysis:
    """
    Bayesian Quantile Regression Analysis for multiple cryptocurrencies (Fixed & Enhanced)

    Our Data Wrangler handles:
        • Mild to moderate data inconsistencies
        • Missing/malformed dates and price columns
        • Multiple CSV formats
        • Zero/negative price filtering

    It CAN'T handle:
        • Extreme data quality issues (missing 50%+ of dates)
        • Multiple currencies in one file
        • Intraday data with complex timezone issues
        • Corporate actions (splits, dividends) without adjustment

    For large-scale analyses (>5 assets), consider reducing `samples` or using `del trace_dict[key]` after prediction.
    """

    def __init__(self, csv_files, crypto_names, quantiles=None, output_dir="crypto_analysis_results"):
        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]
        self.csv_files = list(csv_files)
        self.crypto_names = list(crypto_names)
        self.quantiles = list(quantiles)
        self.data_dict = {}
        self.trace_dict = {}
        self.pred_df_list = []
        self.pred_df = pd.DataFrame()
        self.output_dir = output_dir
        self.correlation_matrix = None
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"\nInitializing MultiCryptoBQRAnalysis for: {', '.join(self.crypto_names)}")

    # ---------- Robust CSV loading ----------
    def load_crypto_csv(self, filename, crypto_name):
        """
        Try a sequence of robust strategies to load CSV data. Prefer pyarrow engine when available.
        Returns a DataFrame with at least ['Date', 'Price'] columns (if present).
        """
        strategies = []

        # Attempt: pandas with pyarrow engine (fast, LTS-preferred) – only if installed
        try:
            strategies.append(("pyarrow_engine", lambda: pd.read_csv(
                filename,
                engine="pyarrow",
                parse_dates=["Date"],
                usecols=lambda c: c.lower() in {"date", "price", "close", "open", "timestamp", "adjclose", "adj_close", "last"},
            )))
        except Exception:
            pass

        # Standard pandas CSV read with parse_dates and dtype coercion
        strategies.append(("pandas_standard", lambda: pd.read_csv(filename, parse_dates=["Date"])))

        # Fallback: robust python engine with on_bad_lines='skip'
        strategies.append(("python_skip_bad_lines", lambda: pd.read_csv(
            filename, engine="python", on_bad_lines="skip", parse_dates=["Date"]
        )))

        # Final fallback: manual line parsing
        def manual():
            rows = []
            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                header = f.readline()
                for line in f:
                    parts = [p.strip() for p in line.strip().split(",")]
                    if not parts:
                        continue
                    date_token = None
                    price_token = None
                    for p in parts:
                        if ("-" in p or "/" in p or ":" in p) and date_token is None:
                            date_token = p
                        try:
                            # Robust float check, handling potential commas
                            float(p.replace(",", ""))
                            if price_token is None:
                                price_token = p
                        except Exception:
                            continue
                    if date_token and price_token:
                        rows.append([date_token, price_token])
            df = pd.DataFrame(rows, columns=["Date", "Price"])
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            return df

        strategies.append(("manual", manual))

        last_err = None
        for name, loader in strategies:
            try:
                df = loader()
                if df is None or df.shape[0] == 0:
                    raise ValueError("Empty dataframe")

                # Normalize column names
                cols = {c.lower(): c for c in df.columns}

                # 1. Search for standard 'Date' column names
                if "date" in cols:
                    df.rename(columns={cols["date"]: "Date"}, inplace=True)
                elif "timestamp" in cols:
                    df.rename(columns={cols["timestamp"]: "Date"}, inplace=True)

                # 2. Robust Date Column Inference (The Data Wrangler Enhancement)
                if "Date" not in df.columns:
                    for col in df.columns:
                        try:
                            # Try to parse a sample of values as dates
                            sample_size = min(50, len(df))
                            date_sample = df[col].head(sample_size).apply(lambda x: pd.to_datetime(x, errors='coerce'))
                            success_rate = date_sample.notna().sum() / sample_size
                            if success_rate > 0.7:
                                df.rename(columns={col: "Date"}, inplace=True)
                                print(f"    Inferred '{col}' as Date column.")
                                break
                        except Exception:
                            continue

                # Find price column
                price_col = None
                for candidate in ("price", "close", "adjclose", "adj_close", "last"):
                    if candidate in cols:
                        price_col = cols[candidate]
                        break
                if price_col is None:
                    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "Date"]
                    if numeric_cols:
                        price_col = numeric_cols[-1]
                if price_col:
                    df.rename(columns={price_col: "Price"}, inplace=True)

                # Ensure Date and Price exist
                if "Date" not in df.columns or "Price" not in df.columns:
                    raise ValueError(f"Required columns not found after loading with '{name}'")

                # Coerce types
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
                df = df.dropna(subset=["Date", "Price"])

                # Filter zero/negative prices
                initial_len = len(df)
                df = df[df["Price"] > 0]
                if len(df) < initial_len:
                    print(f"  ⚠️ {crypto_name} had {initial_len - len(df)} invalid prices, filtered out")

                if len(df) < 30:
                    raise ValueError(f"Insufficient data for {crypto_name} after cleaning: {len(df)} rows (need ≥30)")

                df = df.sort_values("Date").reset_index(drop=True)
                print(f"  ✓ Loaded {crypto_name} with strategy '{name}' ({len(df)} rows)")
                return df
            except Exception as e:
                last_err = e
                continue

        raise ValueError(f"Failed to load '{filename}': {last_err}")

    # ---------- Prepare and clean data ----------
    def load_and_process_data(self):
        """Load all CSVs and prepare working DataFrames with Time, Pct and Crypto fields."""
        for fp, name in zip(self.csv_files, self.crypto_names):
            print(f"Loading {name} from {fp} ...")
            df = self.load_crypto_csv(fp, name)

            df = df.sort_values("Date").reset_index(drop=True)

            min_date = df["Date"].min()
            df["Time"] = (df["Date"] - min_date).dt.days.astype(int)

            # Get first valid price with robust fallback
            first_price = df["Price"].iloc[0] if len(df) else None
            if pd.isna(first_price) or first_price <= 0:
                print(f"  ⚠️ {name} has invalid first price, using median instead")
                first_price = df["Price"].median()
                if pd.isna(first_price) or first_price <= 0:
                    first_price = 1.0
                    print(f"  ⚠️ {name} median also invalid, using fallback value 1.0")

            df["Pct"] = (df["Price"] / first_price) * 100.0
            df["Pct"] = df["Pct"].clip(upper=50000)
            df["Crypto"] = name

            self.data_dict[name] = df
            print(f"  ✓ {name}: {len(df)} rows, date range {df['Date'].min().date()} -> {df['Date'].max().date()}")

        if not self.data_dict:
            raise RuntimeError("No data loaded.")
        print(f"\nAll {len(self.crypto_names)} datasets loaded and cleaned.")

    # ---------- Bayesian quantile regression ----------
    def run_bayesian_qr(self, df, crypto, q, draws=1000, tune=1000, cores=1, random_seed=42):
        """Run Bayesian quantile regression with an Asymmetric Laplace likelihood."""
        df = df.reset_index(drop=True)
        n_obs = len(df)

        if n_obs < 30:
            raise ValueError(f"Insufficient observations for {crypto}: {n_obs} (need at least 30)")

        coords = {"obs": np.arange(n_obs)}

        with pm.Model(coords=coords) as model:
            time_shared = getattr(pm, "MutableData", pm.Data)("time", df["Time"].values, dims="obs")
            alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
            beta = pm.Normal("beta", mu=0.0, sigma=10.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)

            mu = alpha + beta * time_shared
            kappa = float(q) / (1.0 - float(q))

            try:
                pm.AsymmetricLaplace("pct", mu=mu, b=sigma, kappa=kappa, observed=df["Pct"].values, dims="obs")
            except Exception:
                pm.Laplace("pct", mu=mu, b=sigma, observed=df["Pct"].values, dims="obs")

            trace = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=0.9,
                cores=max(1, cores),
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=True
            )

        return model, trace

    def fit_all_models(self, samples=1000, tune=1000, cores=1, seed=42, forecast_days=30):
        """Fit models for all cryptos & quantiles and create predictions collection."""
        print(f"\nFitting BQR models for all cryptos and quantiles (forecast: {forecast_days} days)...")
        for crypto in self.crypto_names:
            df = self.data_dict[crypto]
            for q in self.quantiles:
                key = f"{crypto}_q{q}"
                print(f"  - Fitting {key} ...")
                try:
                    model, trace = self.run_bayesian_qr(df, crypto, q, draws=samples, tune=tune, cores=cores, random_seed=seed)
                    self.trace_dict[key] = trace

                    # Predict over a regular time grid
                    time_seq = np.linspace(df["Time"].min(), df["Time"].min() + forecast_days, 200).astype(int)
                    min_date = df["Date"].min()
                    date_seq = min_date + pd.to_timedelta(time_seq, unit="D")

                    alpha_samples = trace.posterior["alpha"].values
                    beta_samples = trace.posterior["beta"].values

                    alpha_flat = alpha_samples.reshape(-1)[:, np.newaxis]
                    beta_flat = beta_samples.reshape(-1)[:, np.newaxis]
                    time_row = time_seq[np.newaxis, :]

                    pred_mu = alpha_flat + beta_flat * time_row

                    pred_mean = pred_mu.mean(axis=0)
                    pred_lower = np.quantile(pred_mu, 0.025, axis=0)
                    pred_upper = np.quantile(pred_mu, 0.975, axis=0)

                    temp_df = pd.DataFrame({
                        "Date": date_seq,
                        "Time": time_seq,
                        "Estimate": pred_mean,
                        "Lower": pred_lower,
                        "Upper": pred_upper,
                        "Quantile": q,
                        "Crypto": crypto
                    })
                    self.pred_df_list.append(temp_df)
                    print(f"    ✓ {key} fitted")
                except Exception as e:
                    print(f"    ✗ {key} failed: {e}")
                    continue

        if self.pred_df_list:
            self.pred_df = pd.concat(self.pred_df_list, ignore_index=True)
        else:
            self.pred_df = pd.DataFrame()
        print("All models fitted and predictions prepared.")

    # ---------- Simple Analysis Summary ----------
    def create_slope_summary_table(self):
        """Extracts and summarizes the mean posterior slope (beta) for all models."""
        summary_data = []
        for key, trace in self.trace_dict.items():
            try:
                crypto, quantile_str = key.rsplit('_q', 1)
                q = float(quantile_str)
                mean_beta = trace.posterior["beta"].mean().item()
                annualized_slope_pct = mean_beta * 365 * 100.0

                summary_data.append({
                    "Crypto": crypto,
                    "Quantile": f"Q{q}",
                    "Mean Posterior Slope (Daily)": f"{mean_beta:.6f}",
                    "Annualized Growth (%)": f"{annualized_slope_pct:.2f}"
                })
            except Exception as e:
                print(f"Skipping summary for {key}: {e}")
                continue

        if not summary_data:
            print("No slope data to summarize.")
            return None

        slope_df = pd.DataFrame(summary_data)
        outp = os.path.join(self.output_dir, "bqr_annualized_slope_summary.csv")
        slope_df.to_csv(outp, index=False)
        print(f"\nSaved BQR Slope Summary Table to {outp}")
        return slope_df

    # ---------- Higher level runner ----------
    def run_full_analysis(self, samples=1000, tune=1000, cores=1, seed=42, forecast_days=30):
        """Load data, fit models, create correlations and basic visuals."""
        self.load_and_process_data()
        self.fit_all_models(samples=samples, tune=tune, cores=cores, seed=seed, forecast_days=forecast_days)

        corr = self.create_correlation_matrix()
        self.correlation_matrix = corr
        slope_summary = self.create_slope_summary_table()

        try:
            self.create_multi_asset_summary()
        except Exception as e:
            print(f"Multi-asset summary failed: {e}")

        try:
            self.generate_all_pairwise_comparisons()
        except Exception as e:
            print(f"Pairwise generation failed: {e}")

        return {"correlation_matrix": corr, "slope_summary": slope_summary}

    # ---------- Visualization functions ----------
    def create_correlation_matrix(self):
        """
        Calculate correlations and save a heatmap and rolling pairwise for first pair.
        Improved NaN handling with warnings.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        returns_dict = {}

        for crypto in self.crypto_names:
            df = self.data_dict[crypto]
            prices = df.set_index("Date")["Price"]

            # Filter any remaining invalid prices
            if (prices <= 0).any():
                print(f"⚠️ {crypto} has invalid prices in correlation calc, filtering...")
                prices = prices[prices > 0]

            returns = prices.pct_change().dropna()
            returns_dict[crypto] = returns

        aligned = pd.DataFrame(returns_dict)
        initial_rows = {crypto: len(returns_dict[crypto]) for crypto in self.crypto_names}
        aligned = aligned.dropna()
        final_rows = len(aligned)

        if final_rows < 30:
            print(f"Warning: Only {final_rows} overlapping days for correlation")

        retention_pct = (final_rows / max(initial_rows.values())) * 100 if initial_rows else 0
        print(f"Correlation computed using {final_rows} overlapping days ({retention_pct:.1f}% retention)")

        # Use min_periods to avoid NaNs where possible
        corr_matrix = aligned.corr(method='pearson', min_periods=20)

        if corr_matrix.isna().any().any():
            print("Warning: Some correlations are NaN due to insufficient overlap. Consider longer history or fewer assets.")

        sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                    vmin=-1, vmax=1, ax=axes[0], cbar_kws={"label": "Correlation"})
        axes[0].set_title("Daily Returns Correlation Matrix", fontsize=12, fontweight="bold")

        if len(self.crypto_names) >= 2:
            c1, c2 = self.crypto_names[0], self.crypto_names[1]
            rolling = aligned[c1].rolling(window=30).corr(aligned[c2])
            if rolling.dropna().shape[0] > 0:
                axes[1].plot(rolling.index, rolling, linewidth=2)
                axes[1].set_title(f"Rolling Correlation: {c1} vs {c2}", fontsize=12, fontweight="bold")
                axes[1].set_ylim(-1, 1)
                axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)
                axes[1].tick_params(axis="x", rotation=45)
            else:
                axes[1].set_title(f"Rolling Correlation: {c1} vs {c2} (Insufficient Data)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        outp = os.path.join(self.output_dir, "multi_crypto_correlation_matrix.png")
        fig.savefig(outp, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved correlation matrix to {outp}")

        return corr_matrix

    def create_pairwise_comparison(self, crypto1, crypto2):
        """Create the pairwise comparison plot (multi-panel)."""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        colors = {crypto1: "tab:blue", crypto2: "tab:orange"}

        ax1 = fig.add_subplot(gs[0, 0])
        for crypto in (crypto1, crypto2):
            data = self.data_dict[crypto]
            pred = self.pred_df[self.pred_df["Crypto"] == crypto]
            ax1.scatter(data["Date"], data["Pct"], alpha=0.2, s=6, label=f"{crypto} Actual", color=colors[crypto])
            median = pred[pred["Quantile"] == 0.5]
            if not median.empty:
                ax1.plot(median["Date"], median["Estimate"], color=colors[crypto], linewidth=2, label=f"{crypto} Median")
            low = pred[pred["Quantile"] == 0.05]
            high = pred[pred["Quantile"] == 0.95]
            if not low.empty and not high.empty and not median.empty:
                ax1.fill_between(median["Date"], low["Estimate"], high["Estimate"],
                                 color=colors[crypto], alpha=0.18, label=f"{crypto} 90% CI")
        ax1.set_ylabel("Percentage Change (%)")
        ax1.set_title(f"{crypto1} vs {crypto2}: BQR Comparison")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        slope_data = []
        for crypto in (crypto1, crypto2):
            for q in self.quantiles:
                key = f"{crypto}_q{q}"
                if key in self.trace_dict:
                    beta = self.trace_dict[key].posterior["beta"].values.flatten()
                    sample_beta = beta[: min(500, beta.size)]
                    slope_data.extend([{"Crypto": crypto, "Quantile": f"Q{q}", "Slope": b * 100.0} for b in sample_beta])
        if slope_data:
            slope_df = pd.DataFrame(slope_data)
            sns.violinplot(data=slope_df, x="Quantile", y="Slope", hue="Crypto", split=True, ax=ax2)
        ax2.set_title("Posterior Slope Distributions by Quantile")
        ax2.axhline(0, color="red", linestyle="--", alpha=0.6)
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        risk_return = []
        for crypto in (crypto1, crypto2):
            try:
                b05 = self.trace_dict[f"{crypto}_q0.05"].posterior["beta"].values.mean()
                b50 = self.trace_dict[f"{crypto}_q0.5"].posterior["beta"].values.mean()
                b95 = self.trace_dict[f"{crypto}_q0.95"].posterior["beta"].values.mean()
                risk_spread = (b95 - b05) * np.sqrt(365) * 100.0
                ret = b50 * 365 * 100.0
                risk_return.append({"Crypto": crypto, "Return": ret, "Risk": risk_spread})
            except Exception:
                continue
        for item in risk_return:
            ax3.scatter(item["Risk"], item["Return"], s=200, label=item["Crypto"])
            ax3.annotate(item["Crypto"], (item["Risk"], item["Return"]))
        ax3.set_xlabel("Annualized Risk (Volatility Spread)")
        ax3.set_ylabel("Annualized Expected Return (%)")
        ax3.set_title("Risk-Return Profile Comparison")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        for crypto in (crypto1, crypto2):
            pred = self.pred_df[self.pred_df["Crypto"] == crypto]
            if pred.empty:
                continue
            grouped = pred.groupby("Date")
            dates = []
            spreads = []
            for date, g in grouped:
                if set(g["Quantile"]) >= {0.05, 0.5, 0.95}:
                    up = g[g["Quantile"] == 0.95]["Estimate"].values[0]
                    low = g[g["Quantile"] == 0.05]["Estimate"].values[0]
                    dates.append(date)
                    spreads.append(up - low)
            if dates:
                ax4.plot(dates, spreads, label=f"{crypto} Uncertainty")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Uncertainty Spread (Q0.95 - Q0.05)")
        ax4.set_title("Evolution of Uncertainty Over Time")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        outp = os.path.join(self.output_dir, f"pairwise_{crypto1}_vs_{crypto2}.png")
        fig.savefig(outp, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved pairwise comparison to {outp}")

    def generate_all_pairwise_comparisons(self):
        """Generate pairwise comparisons for all combinations"""
        pairs = list(combinations(self.crypto_names, 2))
        for a, b in pairs:
            try:
                self.create_pairwise_comparison(a, b)
            except Exception as e:
                print(f"Failed pairwise {a} vs {b}: {e}")

    def create_multi_asset_summary(self):
        """Summary plot combining quantiles for each asset"""
        n = len(self.crypto_names)
        cols = 2
        rows = (n + 1) // cols
        fig = plt.figure(figsize=(18, 5 * rows))
        gs = GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)
        colors = plt.cm.tab10(np.linspace(0, 1, n))
        cmap = dict(zip(self.crypto_names, colors))

        for idx, crypto in enumerate(self.crypto_names):
            r = idx // cols
            c = idx % cols
            ax = fig.add_subplot(gs[r, c])
            df = self.data_dict[crypto]
            pred = self.pred_df[self.pred_df["Crypto"] == crypto]
            ax.scatter(df["Date"], df["Pct"], s=8, alpha=0.25, color=cmap[crypto])
            for q in self.quantiles:
                qpred = pred[pred["Quantile"] == q]
                if not qpred.empty:
                    linestyle = ("--", "-", ":")[self.quantiles.index(q) % 3]
                    ax.plot(qpred["Date"], qpred["Estimate"], linestyle=linestyle, linewidth=2,
                            label=f"Q{q}" if q == 0.5 else None, color=cmap[crypto])
            ax.set_title(f"{crypto} - Predictive Distribution")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        outp = os.path.join(self.output_dir, "multi_asset_summary.png")
        fig.savefig(outp, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved multi-asset summary to {outp}")