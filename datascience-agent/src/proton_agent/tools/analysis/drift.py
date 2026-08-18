"""
Dataset Drift & Distribution Shift Detection Tool.
Computes two-sample Kolmogorov-Smirnov (KS) tests, Population Stability Index (PSI),
Chi-Square categorical distribution shift, and concept/target drift with automated visualizations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


def _calculate_numeric_psi(ref: np.ndarray, curr: np.ndarray, num_buckets: int | None = None) -> float:
    """Computes Population Stability Index (PSI) for continuous numeric arrays with Laplace smoothing."""
    ref_clean = ref[~np.isnan(ref)]
    curr_clean = curr[~np.isnan(curr)]
    if len(ref_clean) < 5 or len(curr_clean) < 5:
        return 0.0

    n_bins = num_buckets or max(4, min(10, len(ref_clean) // 15))
    quantiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(ref_clean, quantiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    bins[0] = -np.inf
    bins[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bins)
    curr_counts, _ = np.histogram(curr_clean, bins=bins)

    k = len(ref_counts)
    ref_frac = (ref_counts + 0.5) / (len(ref_clean) + 0.5 * k)
    curr_frac = (curr_counts + 0.5) / (len(curr_clean) + 0.5 * k)

    psi_val = np.sum((curr_frac - ref_frac) * np.log(curr_frac / ref_frac))
    return float(round(max(0.0, psi_val), 4))


def _calculate_categorical_psi(ref: pd.Series, curr: pd.Series) -> float:
    """Computes Population Stability Index (PSI) for categorical/discrete series with Laplace smoothing."""
    r_clean = ref.dropna().astype(str)
    c_clean = curr.dropna().astype(str)
    all_cats = list(set(r_clean.unique()) | set(c_clean.unique()))
    if not all_cats:
        return 0.0

    ref_counts = r_clean.value_counts()
    curr_counts = c_clean.value_counts()

    k = len(all_cats)
    ref_total = len(r_clean) + 0.5 * k
    curr_total = len(c_clean) + 0.5 * k

    psi_val = 0.0
    for cat in all_cats:
        r_frac = (ref_counts.get(cat, 0) + 0.5) / ref_total
        c_frac = (curr_counts.get(cat, 0) + 0.5) / curr_total
        psi_val += (c_frac - r_frac) * np.log(c_frac / r_frac)

    return float(round(max(0.0, psi_val), 4))



def _plot_drift_distributions(
    drift_items: List[Dict[str, Any]],
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    save_path: Path,
) -> Optional[str]:
    """Generates an aesthetic distribution comparison plot for the top drifted features."""
    if not drift_items:
        return None

    top_drifted = drift_items[:4]
    n_plots = len(top_drifted)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.2), squeeze=False)
    fig.patch.set_facecolor("#0f172a")

    for idx, item in enumerate(top_drifted):
        ax = axes[0, idx]
        ax.set_facecolor("#1e293b")
        feat = item["feature"]
        psi = item["psi_score"]
        status = item["drift_status"]

        if feat in ref_df.columns and feat in curr_df.columns:
            if item["data_type"] == "numeric":
                r_vals = ref_df[feat].dropna()
                c_vals = curr_df[feat].dropna()
                ax.hist(r_vals, bins=15, density=True, alpha=0.55, color="#38bdf8", label="Baseline")
                ax.hist(c_vals, bins=15, density=True, alpha=0.55, color="#f43f5e", label="Current")
            else:
                r_vc = ref_df[feat].value_counts(normalize=True).head(5)
                c_vc = curr_df[feat].value_counts(normalize=True).head(5)
                cats = list(set(r_vc.index) | set(c_vc.index))[:5]
                x_pos = np.arange(len(cats))
                width = 0.35
                r_heights = [r_vc.get(c, 0.0) for c in cats]
                c_heights = [c_vc.get(c, 0.0) for c in cats]
                ax.bar(x_pos - width / 2, r_heights, width, label="Baseline", color="#38bdf8", alpha=0.7)
                ax.bar(x_pos + width / 2, c_heights, width, label="Current", color="#f43f5e", alpha=0.7)
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(c)[:8] for c in cats], rotation=25, color="#cbd5e1", fontsize=9)

        ax.set_title(f"{feat}\n(PSI: {psi:.3f} • {status})", color="#f8fafc", fontsize=11, fontweight="bold", pad=8)

        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.15, color="#94a3b8")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return str(save_path.resolve())


class DriftDetectionTool(BaseTool):
    """Detects data drift, distribution shift, and concept drift between baseline and current data distributions."""

    name: str = "detect_drift"
    description: str = (
        "Detect data drift and covariate shift between reference and current datasets "
        "using Kolmogorov-Smirnov tests, Population Stability Index (PSI), and Chi-Square analysis."
    )

    def execute(
        self,
        session: SessionState,
        reference_path: str = "",
        current_path: str = "",
        significance_level: float = 0.05,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            # 1. Resolve Reference and Current Datasets
            ref_df: Optional[pd.DataFrame] = None
            curr_df: Optional[pd.DataFrame] = None

            if reference_path and current_path:
                ref_df = pd.read_csv(Path(reference_path).expanduser().resolve())
                curr_df = pd.read_csv(Path(current_path).expanduser().resolve())
            elif session.test_dataset and session.test_dataset.df is not None:
                # Compare active current dataset (training baseline) vs test dataset
                ref_df = session.current_dataset.df if session.current_dataset else None
                curr_df = session.test_dataset.df
            elif session.current_dataset and session.current_dataset.df is not None:
                df = session.current_dataset.df.copy()
                n = len(df)
                if n < 10:
                    return ToolResult(
                        success=False,
                        data={},
                        error="Dataset must contain at least 10 rows to compute statistical distribution drift.",
                    )
                # Split current dataset into temporal baseline (first 50%) and current monitoring (last 50%)
                half = n // 2
                ref_df = df.iloc[:half].reset_index(drop=True)
                curr_df = df.iloc[half:].reset_index(drop=True)
            else:
                return ToolResult(
                    success=False,
                    data={},
                    error="No dataset loaded in session to analyze for drift. Load a dataset first.",
                )

            if ref_df is None or curr_df is None or ref_df.empty or curr_df.empty:
                return ToolResult(
                    success=False,
                    data={},
                    error="Unable to resolve baseline and current datasets for drift analysis.",
                )

            # 2. Analyze Feature Drift
            common_cols = [c for c in ref_df.columns if c in curr_df.columns]
            target_col = session.target_column

            feature_drift_results: List[Dict[str, Any]] = []
            target_drift_result: Optional[Dict[str, Any]] = None
            drifted_count = 0

            for col in common_cols:
                is_target = (col == target_col)
                ref_series = ref_df[col].dropna()
                curr_series = curr_df[col].dropna()

                if ref_series.empty or curr_series.empty:
                    continue

                if pd.api.types.is_numeric_dtype(ref_series) and pd.api.types.is_numeric_dtype(curr_series):
                    # Numeric feature: Two-Sample KS Test + Numeric PSI
                    ks_stat, p_val = stats.ks_2samp(ref_series.values, curr_series.values)
                    psi = _calculate_numeric_psi(ref_series.values, curr_series.values)
                    data_type = "numeric"
                    test_name = "Kolmogorov-Smirnov (KS-2)"
                    stat_val = round(float(ks_stat), 4)
                    p_value = round(float(p_val), 5)
                else:
                    # Categorical feature: Chi-Square Contingency + Categorical PSI
                    data_type = "categorical"
                    test_name = "Chi-Square Contingency"
                    psi = _calculate_categorical_psi(ref_series, curr_series)
                    try:
                        contingency = pd.crosstab(
                            pd.Series(ref_series, name="val"),
                            pd.Series(["ref"] * len(ref_series), name="source")
                        ).join(
                            pd.crosstab(
                                pd.Series(curr_series, name="val"),
                                pd.Series(["curr"] * len(curr_series), name="source")
                            ),
                            how="outer"
                        ).fillna(0)
                        chi2, p_val, _, _ = stats.chi2_contingency(contingency.values)
                        stat_val = round(float(chi2), 4)
                        p_value = round(float(p_val), 5)
                    except Exception:
                        stat_val = 0.0
                        p_value = 1.0

                # Determine Drift Severity & Status
                is_drifted = (psi >= 0.10) or (p_value < significance_level)
                if psi >= 0.25:
                    status = "Significant Drift"
                    action = "Immediate Retraining / Feature Alert"
                    badge = "🔴 Action Required"
                elif psi >= 0.10 or p_value < significance_level:
                    status = "Moderate Drift"
                    action = "Monitor Distribution Shifts"
                    badge = "🟡 Moderate Warning"
                else:
                    status = "Stable"
                    action = "Distribution Stable (No action)"
                    badge = "🟢 Stable"

                if is_drifted and not is_target:
                    drifted_count += 1

                res_item = {
                    "feature": col,
                    "data_type": data_type,
                    "test_used": test_name,
                    "test_statistic": stat_val,
                    "p_value": p_value,
                    "psi_score": psi,
                    "is_drifted": is_drifted,
                    "drift_status": status,
                    "badge": badge,
                    "recommended_action": action,
                }

                if is_target:
                    target_drift_result = res_item
                else:
                    feature_drift_results.append(res_item)

            # Sort features by PSI score descending
            feature_drift_results.sort(key=lambda x: x["psi_score"], reverse=True)

            # 3. Generate Visual Drift Plot
            plot_dir = Path("outputs/plots")
            plot_path = plot_dir / "drift_report.png"
            plot_saved = _plot_drift_distributions(
                feature_drift_results, ref_df, curr_df, plot_path
            )

            high_drift_count = sum(1 for f in feature_drift_results if f["drift_status"] == "Significant Drift")
            mod_drift_count = sum(1 for f in feature_drift_results if f["drift_status"] == "Moderate Drift")
            total_analyzed = len(feature_drift_results)
            overall_drift_detected = (high_drift_count > 0 or mod_drift_count > 0)
            drifted_count = high_drift_count + mod_drift_count

            if high_drift_count >= max(1, total_analyzed // 3) or (high_drift_count + mod_drift_count) >= max(2, int(total_analyzed * 0.5)):
                overall_status = "High Drift Detected"
            elif high_drift_count > 0 or mod_drift_count > 0:
                overall_status = "Moderate Drift Detected"
            else:
                overall_status = "No Significant Drift (Stable)"

            summary_msg = (
                f"Drift analysis complete across {total_analyzed} feature(s): "
                f"{high_drift_count} significant, {mod_drift_count} moderate drift feature(s). Overall Status: {overall_status}."
            )



            return ToolResult(
                success=True,
                data={
                    "drift_detected": overall_drift_detected,
                    "overall_status": overall_status,
                    "drifted_features_count": drifted_count,
                    "total_features_analyzed": total_analyzed,
                    "baseline_samples": len(ref_df),
                    "current_samples": len(curr_df),
                    "feature_drift_metrics": feature_drift_results,
                    "target_drift": target_drift_result,
                    "plot_path": plot_saved,
                    "message": summary_msg,
                },
            )
        except Exception as err:
            return ToolResult(success=False, data={}, error=f"Drift detection failed: {err}")
