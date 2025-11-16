import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

OUT_DIR = "crypto_analysis_results"
os.makedirs(OUT_DIR, exist_ok=True)


class AdvancedCryptoVisualizations:
    def __init__(self, analyzer):
        """analyzer: MultiCryptoBQRAnalysis instance"""
        self.analyzer = analyzer
        self.crypto_names = analyzer.crypto_names
        self.data_dict = analyzer.data_dict
        self.trace_dict = analyzer.trace_dict
        self.pred_df = analyzer.pred_df
        self.output_dir = analyzer.output_dir if hasattr(analyzer, "output_dir") else OUT_DIR

    def plot_3d_returns_space(self):
        if len(self.crypto_names) != 3:
            print("3D plot requires exactly 3 cryptocurrencies")
            return

        returns = {}
        for c in self.crypto_names:
            df = self.data_dict[c]
            # FIXED: Use Price column and validate
            prices = df.set_index("Date")["Price"]
            if (prices <= 0).any():
                print(f"⚠️ {c} has invalid prices in 3D plot, filtering...")
                prices = prices[prices > 0]
            returns[c] = prices.pct_change().dropna()

        aligned = pd.DataFrame(returns)

        # FIXED: Check data quality before plotting
        initial_size = min(len(returns[c]) for c in self.crypto_names)
        aligned = aligned.dropna()
        final_size = len(aligned)

        if final_size < 30:
            print(f"⚠️ Insufficient aligned data for 3D plot: {final_size} points")
            return

        retention = (final_size / initial_size) * 100
        print(f"✓ 3D plot using {final_size} aligned points ({retention:.1f}% retention)")

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")
        n = len(aligned)
        scatter = ax.scatter(
            aligned[self.crypto_names[0]],
            aligned[self.crypto_names[1]],
            aligned[self.crypto_names[2]],
            c=np.arange(n),
            cmap="viridis",
            s=20,
            alpha=0.6
        )
        ax.set_xlabel(f"{self.crypto_names[0]} Returns")
        ax.set_ylabel(f"{self.crypto_names[1]} Returns")
        ax.set_zlabel(f"{self.crypto_names[2]} Returns")
        ax.set_title("3D Returns Space Trajectory")
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label("Time Progression")
        plt.tight_layout(pad=1.0)
        outp = os.path.join(self.output_dir, "3d_returns_space.png")
        fig.savefig(outp, dpi=300, bbox_inches="tight", metadata={"Software": "CBQRA Suite 2025"})
        plt.close(fig)
        print(f"Saved 3D returns space to {outp}")

    def plot_rolling_correlation_heatmap(self, window=30):
        returns = {}
        for c in self.crypto_names:
            df = self.data_dict[c]
            # FIXED: Use Price column and validate
            prices = df.set_index("Date")["Price"]
            if (prices <= 0).any():
                print(f"⚠️ {c} has invalid prices in rolling corr, filtering...")
                prices = prices[prices > 0]
            returns[c] = prices.pct_change().dropna()

        aligned = pd.DataFrame(returns)

        # FIXED: Validate alignment
        initial_size = min(len(returns[c]) for c in self.crypto_names)
        aligned = aligned.dropna()
        final_size = len(aligned)

        if final_size < window * 2:
            print(f"⚠️ Insufficient data for rolling correlation: {final_size} points (need {window*2})")
            return

        pairs = list(combinations(self.crypto_names, 2))
        fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 4 * len(pairs)))
        if len(pairs) == 1:
            axes = [axes]

        for i, (a, b) in enumerate(pairs):
            rolling = aligned[a].rolling(window=window).corr(aligned[b])
            # FIXED: Handle NaN values in rolling correlation
            rolling = rolling.dropna()
            if len(rolling) > 0:
                axes[i].plot(rolling.index, rolling, linewidth=2)
                axes[i].fill_between(rolling.index, 0, rolling, alpha=0.25)
                axes[i].axhline(0, color="red", linestyle="--", alpha=0.5)
                axes[i].set_ylim(-1, 1)
                axes[i].set_title(f"{a} vs {b} - {window}-day Rolling Corr")
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, "Insufficient data",
                           ha='center', va='center', transform=axes[i].transAxes)

        axes[-1].set_xlabel("Date")
        plt.tight_layout(pad=1.0)
        outp = os.path.join(self.output_dir, "rolling_correlation_heatmap.png")
        fig.savefig(outp, dpi=300, bbox_inches="tight", metadata={"Software": "CBQRA Suite 2025"})
        plt.close(fig)
        print(f"Saved rolling correlation heatmap to {outp}")

    def plot_volatility_comparison(self):
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.crypto_names)))
        cmap = dict(zip(self.crypto_names, colors))

        for crypto in self.crypto_names:
            df = self.data_dict[crypto]
            # FIXED: Use Price for returns calculation
            prices = df["Price"]
            if (prices <= 0).any():
                print(f"⚠️ {crypto} has invalid prices in volatility calc, filtering...")
                valid_idx = prices > 0
                prices = prices[valid_idx]
                df = df[valid_idx].reset_index(drop=True)

            returns = prices.pct_change().dropna()
            rolling_vol = returns.rolling(30).std() * np.sqrt(365)

            # FIXED: Properly align dates with rolling volatility
            # Skip first 30 points where rolling window is incomplete
            valid_dates = df["Date"].iloc[30:len(rolling_vol)+30]
            valid_vol = rolling_vol.iloc[29:]

            minlen = min(len(valid_dates), len(valid_vol))
            if minlen > 0:
                axes[0].plot(valid_dates.iloc[:minlen], valid_vol.iloc[:minlen],
                           label=crypto, color=cmap[crypto])

        axes[0].set_title("Rolling 30-Day Volatility Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Volatility distribution
        vol_list = []
        for crypto in self.crypto_names:
            df = self.data_dict[crypto]
            prices = df["Price"]
            if (prices <= 0).any():
                prices = prices[prices > 0]
            returns = prices.pct_change().dropna()
            rolling_vol = returns.rolling(30).std() * np.sqrt(365)
            vol_list.extend([{"Crypto": crypto, "Volatility": v} for v in rolling_vol.dropna()])

        if vol_list:
            vol_df = pd.DataFrame(vol_list)
            sns.violinplot(data=vol_df, x="Crypto", y="Volatility", ax=axes[1])
        else:
            axes[1].text(0.5, 0.5, "No volatility data available",
                       ha='center', va='center', transform=axes[1].transAxes)

        axes[1].set_title("Volatility Distribution Comparison")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout(pad=1.0)
        outp = os.path.join(self.output_dir, "volatility_comparison.png")
        fig.savefig(outp, dpi=300, bbox_inches="tight", metadata={"Software": "CBQRA Suite 2025"})
        plt.close(fig)
        print(f"Saved volatility comparison to {outp}")

    def plot_drawdown_comparison(self):
        n = len(self.crypto_names)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
        if n == 1:
            axes = [axes]
        colors = plt.cm.tab10(np.linspace(0, 1, n))
        max_dds = []

        for i, crypto in enumerate(self.crypto_names):
            df = self.data_dict[crypto]
            # FIXED: Validate Pct column
            pct = df["Pct"]
            if pct.isnull().any():
                print(f"⚠️ {crypto} has null values in Pct column, dropping...")
                pct = pct.dropna()
                df = df[df["Pct"].notnull()].reset_index(drop=True)

            running_max = df["Pct"].expanding().max()
            # FIXED: Handle division by zero
            drawdown = ((df["Pct"] - running_max) / running_max.replace(0, 1)) * 100.0

            axes[i].fill_between(df["Date"], drawdown, 0, alpha=0.25, color=colors[i])
            axes[i].plot(df["Date"], drawdown, color=colors[i])
            axes[i].set_title(f"{crypto} - Drawdown Analysis")
            axes[i].grid(True, alpha=0.3)

            max_dd = drawdown.min()
            max_dds.append({"Crypto": crypto, "Max Drawdown": max_dd})
            axes[i].text(0.02, 0.05, f"Max Drawdown: {max_dd:.1f}%",
                       transform=axes[i].transAxes,
                       bbox=dict(facecolor="white", alpha=0.7))

        axes[-1].set_xlabel("Date")
        plt.tight_layout(pad=1.0)
        outp = os.path.join(self.output_dir, "drawdown_comparison.png")
        fig.savefig(outp, dpi=300, bbox_inches="tight", metadata={"Software": "CBQRA Suite 2025"})
        plt.close(fig)
        print(f"Saved drawdown comparison to {outp}")
        return pd.DataFrame(max_dds)

    def plot_forecast_comparison(self, days_ahead=30):
        n = len(self.crypto_names)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]
        colors = ["red", "blue", "green"][:len(self.analyzer.quantiles)]

        for i, crypto in enumerate(self.crypto_names):
            df = self.data_dict[crypto]
            last_time = df["Time"].max()
            future_times = np.arange(last_time, last_time + days_ahead)
            future_dates = df["Date"].max() + pd.to_timedelta(future_times - last_time, unit="D")

            axes[i].plot(df["Date"], df["Pct"], color="gray", alpha=0.7, label="Historical")

            for j, q in enumerate(self.analyzer.quantiles):
                key = f"{crypto}_q{q}"
                if key not in self.trace_dict:
                    continue

                try:
                    trace = self.trace_dict[key]
                    alpha_samples = trace.posterior["alpha"].values.flatten()
                    beta_samples = trace.posterior["beta"].values.flatten()

                    n_samples = min(1000, alpha_samples.size)
                    preds = []
                    for a_s, b_s in zip(alpha_samples[:n_samples], beta_samples[:n_samples]):
                        preds.append(a_s + b_s * future_times)
                    preds = np.array(preds)

                    mean = preds.mean(axis=0)
                    lo = np.percentile(preds, 2.5, axis=0)
                    hi = np.percentile(preds, 97.5, axis=0)

                    axes[i].plot(future_dates, mean, color=colors[j], linewidth=2, label=f"Q{q} Forecast")
                    axes[i].fill_between(future_dates, lo, hi, alpha=0.15, color=colors[j])
                except Exception as e:
                    print(f"⚠️ Failed to plot forecast for {crypto} Q{q}: {e}")
                    continue

            axes[i].axvline(df["Date"].max(), linestyle="--", color="black")
            axes[i].set_title(f"{crypto} {days_ahead}-Day Forecast")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout(pad=1.0)
        outp = os.path.join(self.output_dir, "forecast_comparison.png")
        fig.savefig(outp, dpi=300, bbox_inches="tight", metadata={"Software": "CBQRA Suite 2025"})
        plt.close(fig)
        print(f"Saved forecast comparison to {outp}")

    def create_performance_dashboard(self):
        metrics = []
        for crypto in self.crypto_names:
            df = self.data_dict[crypto]

            # FIXED: Validate data before calculations
            if len(df) < 2:
                print(f"⚠️ Insufficient data for {crypto} metrics")
                continue

            # FIXED: Handle edge cases in metric calculation
            total_return = df["Pct"].iloc[-1] - 100.0
            days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days

            if days < 1:
                print(f"⚠️ Invalid date range for {crypto}")
                continue

            # FIXED: Prevent division by zero
            final_pct = df["Pct"].iloc[-1]
            if final_pct <= 0:
                ann_return = -100.0
            else:
                ann_return = ((final_pct / 100.0) ** (365.25 / days) - 1.0) * 100

            # Volatility calculation
            prices = df["Price"]
            if (prices <= 0).any():
                prices = prices[prices > 0]
            returns = prices.pct_change().dropna()

            if len(returns) > 1:
                vol = returns.std() * np.sqrt(365)
                sharpe = (ann_return / 100) / vol if vol > 0 else 0.0
            else:
                vol = 0.0
                sharpe = 0.0

            # Drawdown calculation
            running_max = df["Pct"].expanding().max()
            drawdown = ((df["Pct"] - running_max) / running_max.replace(0, 1)) * 100.0
            max_dd = drawdown.min()

            # Extract beta values
            try:
                b05 = self.trace_dict[f"{crypto}_q0.05"].posterior["beta"].mean().item()
                b50 = self.trace_dict[f"{crypto}_q0.5"].posterior["beta"].mean().item()
                b95 = self.trace_dict[f"{crypto}_q0.95"].posterior["beta"].mean().item()
            except Exception:
                b05 = b50 = b95 = 0.0

            metrics.append({
                "Crypto": crypto,
                "Total Return (%)": f"{total_return:.1f}",
                "Ann. Return (%)": f"{ann_return:.1f}",
                "Ann. Volatility (%)": f"{vol * 100:.1f}",
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Max Drawdown (%)": f"{max_dd:.1f}",
                "Q0.05 Slope": f"{b05 * 100:.4f}",
                "Q0.50 Slope": f"{b50 * 100:.4f}",
                "Q0.95 Slope": f"{b95 * 100:.4f}",
                "Risk Spread": f"{(b95 - b05) * 100:.4f}"
            })

        df_metrics = pd.DataFrame(metrics)

        # Save CSV
        outp_csv = os.path.join(self.output_dir, "performance_metrics_multi.csv")
        df_metrics.to_csv(outp_csv, index=False)
        print(f"Saved metrics CSV to {outp_csv}")

        # Create visual table
        fig, ax = plt.subplots(figsize=(16, 2 + 0.5 * len(df_metrics)))
        ax.axis("off")
        table = ax.table(cellText=df_metrics.values, colLabels=df_metrics.columns,
                        loc="center", cellLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        plt.title("Cryptocurrency Performance Metrics Dashboard", fontsize=14)
        plt.tight_layout(pad=1.0)
        outp_img = os.path.join(self.output_dir, "performance_dashboard.png")
        fig.savefig(outp_img, dpi=300, bbox_inches="tight", metadata={"Software": "CBQRA Suite 2025"})
        plt.close(fig)
        print(f"Saved performance dashboard to {outp_img}")
        return df_metrics

    def plot_risk_return_scatter(self):
        data = []
        for crypto in self.crypto_names:
            try:
                median_beta = self.trace_dict[f"{crypto}_q0.5"].posterior["beta"].values.mean()
                upper_beta = self.trace_dict[f"{crypto}_q0.95"].posterior["beta"].values.mean()
                lower_beta = self.trace_dict[f"{crypto}_q0.05"].posterior["beta"].values.mean()
                ann_return = median_beta * 365 * 100.0
                ann_risk = (upper_beta - lower_beta) * np.sqrt(365) * 100.0
            except Exception as e:
                print(f"⚠️ Could not compute risk-return for {crypto}: {e}")
                ann_return = ann_risk = 0.0
            data.append({"Crypto": crypto, "Return": ann_return, "Risk": ann_risk})

        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

        for i, row in df.iterrows():
            ax.scatter(row["Risk"], row["Return"], s=300, color=colors[i], alpha=0.7, edgecolors="k")
            ax.annotate(row["Crypto"], (row["Risk"], row["Return"]),
                       xytext=(10, 10), textcoords="offset points",
                       bbox=dict(facecolor="white", alpha=0.8))

        ax.set_xlabel("Annualized Risk (Volatility Spread)")
        ax.set_ylabel("Annualized Expected Return (%)")
        ax.set_title("Multi-Crypto Risk-Return Profile")
        ax.grid(True, alpha=0.3)
        plt.tight_layout(pad=1.0)
        outp = os.path.join(self.output_dir, "risk_return_scatter.png")
        fig.savefig(outp, dpi=300, bbox_inches="tight", metadata={"Software": "CBQRA Suite 2025"})
        plt.close(fig)
        print(f"Saved risk-return scatter to {outp}")

    def generate_all_advanced_visualizations(self):
        """
        Run all advanced visualization routines in sequence.
        Returns a dictionary of generated artifact paths and key metrics.
        """
        print("\nGenerating advanced visualizations...")

        try:
            self.plot_rolling_correlation_heatmap()
        except Exception as e:
            print(f"✗ Rolling correlation heatmap failed: {e}")

        try:
            self.plot_volatility_comparison()
        except Exception as e:
            print(f"✗ Volatility comparison failed: {e}")

        try:
            self.plot_drawdown_comparison()
        except Exception as e:
            print(f"✗ Drawdown comparison failed: {e}")

        try:
            self.plot_forecast_comparison()
        except Exception as e:
            print(f"✗ Forecast comparison failed: {e}")

        try:
            metrics_df = self.create_performance_dashboard()
        except Exception as e:
            print(f"✗ Performance dashboard failed: {e}")
            metrics_df = None

        try:
            self.plot_risk_return_scatter()
        except Exception as e:
            print(f"✗ Risk-return scatter failed: {e}")

        # Optional 3D visualization if 3 cryptos
        if len(self.crypto_names) == 3:
            try:
                self.plot_3d_returns_space()
            except Exception as e:
                print(f"✗ 3D plot failed: {e}")

        print("✓ All available advanced visualizations completed.")
        return {
            "metrics": metrics_df,
            "output_dir": self.output_dir
        }
